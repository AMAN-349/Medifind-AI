from PIL import Image
from hashlib import sha256
from html import escape
from io import BytesIO
import streamlit as st
import pytesseract as pyt
import pandas as pd
import os
import re
import time

from services.rag_service import SERVICE_BUSY_MESSAGE, VERIFIED_REFERENCES, answer_question, get_or_build_vector_store
from services.product_search import search_shopping, sort_listings
from services.report_service import compare_printed_range
from services.vision_service import extract_prescription_candidates, extract_report_results


MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


class ImageValidationError(Exception):
    pass


def extract_medicines(text: str) -> list[tuple[str, str, str]]:
    pattern = r"(Tab|Cap|Syrup|Inj)\.?\s+([A-Za-z]+)\s+(\d+\s?mg)"
    return re.findall(pattern, text)


def normalize_strength(strength: str) -> str:
    return re.sub(r"(?<=\d)\s+(?=[a-z])", "", str(strength).strip().lower())


def combine_candidates(draft: pd.DataFrame, candidates: list[dict[str, str]]) -> pd.DataFrame:
    combined = draft.copy()
    for candidate in candidates:
        name = candidate["name"].strip()
        strength = candidate["strength"].strip()
        form = candidate["form"].strip()
        if not (name or strength or form):
            continue
        if name and strength and form:
            duplicate = (
                (combined["name"].fillna("").str.strip().str.casefold() == name.casefold())
                & (combined["strength"].fillna("").map(normalize_strength) == normalize_strength(strength))
                & (combined["form"].fillna("").str.strip().str.casefold() == form.casefold())
            ).any()
            if duplicate:
                continue
        combined = pd.concat([combined, pd.DataFrame([candidate])], ignore_index=True)
    return combined


def clear_prescription_data() -> None:
    for key in list(st.session_state):
        if key in {"last_file", "medicine_editor_version", "confirmed_medicines", "extraction_warnings"} or key.startswith(
            ("medicines_", "vision_consent_", "shopping_")
        ):
            del st.session_state[key]


def clear_report_data() -> None:
    for key in list(st.session_state):
        if key in {"report_last_file", "report_draft", "report_error", "report_confirmed"} or key.startswith("report_editor_"):
            del st.session_state[key]


def clear_session_data() -> None:
    clear_prescription_data()
    clear_report_data()
    for key in list(st.session_state):
        if key == "question_result" or key.startswith(("prescription_upload_", "report_upload_", "question_text_")):
            del st.session_state[key]
    st.session_state["upload_version"] = st.session_state.get("upload_version", 0) + 1


def require_access() -> None:
    mode = os.environ.get("MEDIFIND_AUTH_MODE", "local").strip().lower()
    if mode == "local":
        if st.config.get_option("server.address") != "127.0.0.1":
            clear_session_data()
            st.error("Local mode requires Streamlit to bind to 127.0.0.1.")
            st.stop()
        return

    if mode != "oidc":
        clear_session_data()
        st.error("Invalid access configuration. Contact the administrator.")
        st.stop()
        return

    allowed_subjects = {
        subject.strip()
        for subject in os.environ.get("MEDIFIND_ALLOWED_SUBJECTS", "").split(",")
        if subject.strip()
    }
    expected_issuer = os.environ.get("MEDIFIND_OIDC_ISSUER", "").strip()
    if not allowed_subjects or not expected_issuer:
        clear_session_data()
        st.error("Access is not configured. Contact the administrator.")
        st.stop()
        return

    user = st.user.to_dict()
    if not user.get("is_logged_in", False):
        clear_session_data()
        st.title("MediFind AI")
        st.info("Sign in to access the prototype.")
        if st.button("Sign in"):
            try:
                st.login()
            except Exception:
                st.error("Sign-in is unavailable. Contact the administrator.")
        st.stop()
        return

    subject = user.get("sub")
    expiry = user.get("exp")
    if (
        not isinstance(subject, str)
        or subject not in allowed_subjects
        or user.get("iss") != expected_issuer
        or type(expiry) not in (int, float)
        or not time.time() < expiry < float("inf")
    ):
        clear_session_data()
        st.error("Your account is not authorized or your session has expired.")
        if st.button("Sign out"):
            st.logout()
        st.stop()
        return

    if st.button("Sign out"):
        clear_session_data()
        st.logout()
        st.stop()


st.set_page_config(page_title="MediFind AI", page_icon=":material/health_and_safety:", layout="centered")
require_access()
st.markdown(
    """
    <style>
    .stApp {
        font-family: "Avenir Next", "Trebuchet MS", sans-serif;
        background: linear-gradient(180deg, #edf5f2 0, #f8faf9 320px, #f8faf9 100%);
    }
    .block-container { max-width: 1060px; padding-top: 4.5rem; padding-bottom: 4rem; }
    .workspace-header { display: flex; align-items: center; gap: 1rem; margin-bottom: 0.4rem; }
    .workspace-mark {
        display: grid; place-items: center; flex: 0 0 3.4rem; height: 3.4rem;
        border-radius: 8px; background: #126d63; color: #ffffff;
        font-family: Georgia, serif; font-size: 2rem; font-weight: 700;
        box-shadow: 4px 4px 0 #eab8a4;
    }
    .workspace-kicker { color: #a45a43; font-size: 0.72rem; font-weight: 800; text-transform: uppercase; }
    .workspace-header h1 { font-family: Georgia, serif !important; font-size: 2.45rem !important; color: #183a37; padding: 0; margin: 0; }
    h2 { font-size: 1.45rem !important; color: #203d3b; }
    h3 { font-size: 1.1rem !important; color: #203d3b; }
    [data-testid="stSidebar"] { background: #e8f1ee; border-right: 1px solid #d6e5df; }
    [data-testid="stTabs"] [role="tablist"] {
        gap: 0.35rem; padding: 0.35rem; background: #e5eeeb; border-radius: 8px;
    }
    [data-testid="stTabs"] [role="tab"] {
        flex: 1 1 0; min-height: 3rem; padding: 0.55rem 0.8rem;
        border-radius: 6px; color: #526a66; justify-content: center;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background: #ffffff; color: #11695f; box-shadow: 0 1px 5px #1a51431a;
    }
    [data-testid="stTabs"] [role="tab"] p { font-weight: 700; }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"],
    [data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }
    [data-testid="stTabs"] [role="tabpanel"] { padding-top: 1.25rem; }
    [data-testid="stFileUploaderDropzone"] {
        background: #ffffff; border: 1px dashed #9cbdb2; border-radius: 8px;
        padding: 1.4rem;
    }
    [data-testid="stAlert"] { border-radius: 6px; }
    [data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button {
        border-radius: 6px; font-weight: 700;
    }
    [data-testid="stDataFrame"] { border-radius: 6px; overflow: hidden; }
    .listing-title { color: #203d3b; font-size: 1.05rem; font-weight: 700; overflow-wrap: anywhere; }
    .listing-source { color: #526a66; font-size: 0.9rem; margin-top: 0.3rem; overflow-wrap: anywhere; }
    .listing-price { color: #126d63; font-size: 1.3rem; font-weight: 800; margin-top: 0.55rem; overflow-wrap: anywhere; }
    @media (max-width: 640px) {
        .block-container { padding-top: 4rem; }
        .workspace-header h1 { font-size: 2rem !important; }
        [data-testid="stTabs"] [role="tablist"] { gap: 0; }
        [data-testid="stTabs"] [role="tab"] { padding: 0.4rem; }
        [data-testid="stTabs"] [role="tab"] p { font-size: 0.8rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="workspace-header"><div class="workspace-mark" aria-hidden="true">M</div>'
    '<div><div class="workspace-kicker">Medicine &amp; report workspace</div>'
    '<h1>MediFind AI</h1></div></div>',
    unsafe_allow_html=True,
)
st.warning(
    "Prototype only: not clinically validated or intended for patient care. "
    "Use synthetic or de-identified images; verified retailer prices are not available."
)
st.sidebar.header("Session")
st.sidebar.button("Clear session", on_click=clear_session_data)
st.sidebar.caption("Clears this app session only; it cannot delete data already sent to third parties.")

review_tab, reference_tab, report_tab = st.tabs(["Prescription review", "Medicine reference", "Report review"])
review_tab.header("Upload a prescription image")
review_tab.warning(
    "Uploading sends the whole image, including personal details, to Google Gemini. "
    "Google may use free-tier API content to improve its products. "
    "By uploading, you agree to this processing; upload only if you have permission."
)
upload_version = st.session_state.setdefault("upload_version", 0)
uploaded_file = review_tab.file_uploader(
    "Upload and send image to Google Gemini", type=["png", "jpg", "jpeg"], key=f"prescription_upload_{upload_version}",
    max_upload_size=10,
)

if uploaded_file is None:
    clear_prescription_data()

if uploaded_file is not None:
    file_id = sha256(uploaded_file.getvalue()).hexdigest()
    if st.session_state.get("last_file") != file_id:
        clear_prescription_data()
    try:
        if len(uploaded_file.getvalue()) > MAX_IMAGE_BYTES:
            raise ImageValidationError("The image must be 10 MB or smaller.")
        image = Image.open(uploaded_file)
        if image.format not in {"PNG", "JPEG"}:
            raise ImageValidationError("Only PNG and JPEG images are supported.")
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ImageValidationError("The image must be 20 million pixels or smaller.")
        review_tab.image(image, caption="Uploaded prescription")
        if "medicines_df" not in st.session_state or st.session_state.get("last_file") != file_id:
            st.session_state["last_file"] = file_id
            st.session_state["medicine_editor_version"] = 0
            st.session_state["extraction_warnings"] = []
            st.session_state.pop("confirmed_medicines", None)
            try:
                matches = extract_medicines(pyt.image_to_string(image))
            except Exception:
                matches = []
                st.session_state["extraction_warnings"].append("Local OCR failed. Review any AI results carefully.")
            rows = [
                {"form": form, "name": med_name, "strength": strength}
                for form, med_name, strength in matches
            ]
            st.session_state["medicines_df"] = pd.DataFrame(rows, columns=["form", "name", "strength"])
            try:
                image_buffer = BytesIO()
                image.convert("RGB").save(image_buffer, format="PNG")
                with review_tab, st.spinner("Reading image with OCR and Gemini..."):
                    candidates = extract_prescription_candidates(image_buffer.getvalue())
                if candidates:
                    st.session_state["medicines_df"] = combine_candidates(st.session_state["medicines_df"], candidates)
            except Exception:
                st.session_state["extraction_warnings"].append(
                    "AI extraction failed. Only any available local OCR results are shown."
                )

        for warning in st.session_state.get("extraction_warnings", []):
            review_tab.warning(warning)

        review_tab.subheader("Review possible medicines")
        review_tab.caption("These are unverified readings, not medical recommendations. Ask a pharmacist or prescriber to check any unclear or conflicting details.")
        editor_version = st.session_state["medicine_editor_version"]
        edited_df = review_tab.data_editor(st.session_state["medicines_df"], num_rows="dynamic", key=f"medicines_{file_id}_{editor_version}")
        st.session_state["medicines_df"] = edited_df

        candidate_names = {str(name).strip().casefold() for name in edited_df["name"].dropna() if str(name).strip()}
        if candidate_names:
            review_tab.subheader("General medicine information")
            review_tab.caption("A reference match does not verify the prescription or whether a medicine is right for a condition.")
            for name in sorted(candidate_names):
                reference = VERIFIED_REFERENCES.get(name)
                if reference is None:
                    review_tab.info(f"Information not found in the reference collection for: {name}")
                    continue
                document = reference[1]
                review_tab.text(f"Possible reading: {name}")
                review_tab.write(document.page_content)
                review_tab.link_button(f"NHS reference: {name}", document.metadata["source"])

        if review_tab.button("Confirm medicines", type="primary"):
            if edited_df.empty:
                review_tab.warning("Add at least one medicine before confirming.")
            elif any(
                pd.isna(row[field]) or not str(row[field]).strip()
                for _, row in edited_df.iterrows()
                for field in ("name", "strength", "form")
            ):
                st.session_state.pop("confirmed_medicines", None)
                review_tab.warning("Every medicine needs a name, strength, and form. Ask a pharmacist or prescriber about unclear details.")
            else:
                st.session_state["confirmed_medicines"] = edited_df.copy()

        confirmed = st.session_state.get("confirmed_medicines")
        if confirmed is not None and confirmed.equals(edited_df):
            review_tab.subheader("Google Shopping demo")
            review_tab.info("No authorized pharmacy data source is connected.")
            review_tab.warning(
                "For synthetic demo data only. Searching sends the selected medicine details to SerpApi, "
                "which may store the query and results. Do not search using real patient information."
            )
            review_tab.caption(
                "These are unverified search results, not confirmed pharmacy offers or medicine matches. "
                "Prices and availability may differ; do not use results for patient care."
            )
            search_options = [f"{row['name']} {row['strength']} {row['form']}" for _, row in confirmed.iterrows()]
            selected_query = review_tab.selectbox("Select a confirmed demo medicine", search_options)
            if st.session_state.get("shopping_selected_query") != selected_query:
                st.session_state["shopping_selected_query"] = selected_query
                st.session_state["shopping_page"] = 0
            if not os.environ.get("SERPAPI_API_KEY", "").strip():
                review_tab.info("Set SERPAPI_API_KEY to enable the Google Shopping demo.")
            elif review_tab.button("Search Google Shopping"):
                st.session_state.pop("shopping_results", None)
                st.session_state["shopping_page"] = 0
                try:
                    with review_tab, st.spinner("Searching Google Shopping..."):
                        listings, retrieved_at = search_shopping(selected_query, os.environ["SERPAPI_API_KEY"])
                    st.session_state["shopping_results"] = (selected_query, listings, retrieved_at)
                except (RuntimeError, ValueError):
                    review_tab.warning("Shopping search failed. Check your API configuration and try again.")
            shopping_results = st.session_state.get("shopping_results")
            if shopping_results and shopping_results[0] == selected_query:
                _, listings, retrieved_at = shopping_results
                review_tab.caption(f"Search retrieved: {retrieved_at}. Prices are not verified.")
                if not listings:
                    review_tab.info("No complete search results were returned.")
                else:
                    sort_order = review_tab.selectbox(
                        "Sort by", ["Relevance", "Price: low to high", "Price: high to low"], key="shopping_sort"
                    )
                    if st.session_state.get("shopping_applied_sort") != sort_order:
                        st.session_state["shopping_applied_sort"] = sort_order
                        st.session_state["shopping_page"] = 0
                    ordered_listings = sort_listings(listings, sort_order)
                    page_count = (len(ordered_listings) + 3) // 4
                    page = min(st.session_state.get("shopping_page", 0), page_count - 1)
                    navigation = review_tab.container(horizontal=True, vertical_alignment="center", key="shopping_navigation")
                    page_changed = False
                    if navigation.button("Previous", icon=":material/chevron_left:", disabled=page == 0):
                        page -= 1
                        page_changed = True
                    if navigation.button("Next", icon=":material/chevron_right:", disabled=page == page_count - 1):
                        page += 1
                        page_changed = True
                    st.session_state["shopping_page"] = page
                    if page_changed:
                        st.rerun()
                    navigation.caption(f"Page {page + 1} of {page_count}  |  {len(ordered_listings)} results")
                    if sort_order != "Relevance":
                        review_tab.caption("Only single INR amounts in a recognized format are sorted; other prices appear last.")
                    for index, listing in enumerate(ordered_listings[page * 4:page * 4 + 4], start=page * 4):
                        card = review_tab.container(border=True, key=f"shopping_card_{index}")
                        card.markdown(
                            f'<div class="listing-title">{escape(str(listing["title"]))}</div>'
                            f'<div class="listing-source">{escape(str(listing["source"]))}</div>'
                            f'<div class="listing-price">{escape(str(listing["price"]))}</div>',
                            unsafe_allow_html=True,
                        )
                        if listing["google_link"]:
                            card.link_button("View Google result", listing["google_link"], key=f"shopping_link_{index}")
    except ImageValidationError as error:
        review_tab.error(str(error))
    except Exception:
        review_tab.error("This file could not be read as an image. Please upload a valid image.")


report_tab.header("Review a medical report")
report_tab.warning(
    "Uploading sends the whole report image, including personal details, to Google Gemini. "
    "Google may use free-tier API content to improve its products. "
    "By uploading, you agree to this processing; upload only if you have permission."
)
report_tab.caption(
    "Prototype only. Results are not a diagnosis. Ultrasound and other narrative findings require a clinician's interpretation."
)
report_file = report_tab.file_uploader(
    "Upload and send report image to Google Gemini", type=["png", "jpg", "jpeg"],
    key=f"report_upload_{upload_version}", max_upload_size=10,
)
if report_file is None:
    clear_report_data()
else:
    report_file_id = sha256(report_file.getvalue()).hexdigest()
    if st.session_state.get("report_last_file") != report_file_id:
        clear_report_data()
    try:
        if len(report_file.getvalue()) > MAX_IMAGE_BYTES:
            raise ImageValidationError("The image must be 10 MB or smaller.")
        report_image = Image.open(report_file)
        if report_image.format not in {"PNG", "JPEG"}:
            raise ImageValidationError("Only PNG and JPEG images are supported.")
        if report_image.width * report_image.height > MAX_IMAGE_PIXELS:
            raise ImageValidationError("The image must be 20 million pixels or smaller.")
        report_tab.image(report_image, caption="Uploaded report")
        if st.session_state.get("report_last_file") != report_file_id:
            st.session_state["report_last_file"] = report_file_id
            st.session_state["report_draft"] = pd.DataFrame(columns=["name", "result", "unit", "reference_range"])
            try:
                image_buffer = BytesIO()
                report_image.convert("RGB").save(image_buffer, format="PNG")
                with report_tab, st.spinner("Reading report..."):
                    results = extract_report_results(image_buffer.getvalue())
                st.session_state["report_draft"] = pd.DataFrame(
                    results, columns=["name", "result", "unit", "reference_range"]
                )
            except Exception:
                st.session_state["report_error"] = "Report extraction failed. You can enter the printed results manually."

        if "report_error" in st.session_state:
            report_tab.warning(st.session_state["report_error"])
        report_tab.subheader("Check extracted results")
        report_tab.caption(
            "Compare every name, result, unit, and reference range with the original report. "
            "Check that each result and range use the same unit. Leave unclear fields blank and ask a healthcare professional for help."
        )
        edited_report = report_tab.data_editor(
            st.session_state["report_draft"], num_rows="dynamic", key=f"report_editor_{report_file_id}"
        )
        st.session_state["report_draft"] = edited_report
        if edited_report.empty:
            report_tab.info("No results are available yet. You can enter the printed results manually.")
        if report_tab.button("Compare checked results", type="primary"):
            if edited_report.empty:
                report_tab.warning("Add at least one result before comparing.")
            else:
                st.session_state["report_confirmed"] = edited_report.copy()
        confirmed_report = st.session_state.get("report_confirmed")
        if confirmed_report is not None and confirmed_report.equals(edited_report):
            chart = confirmed_report.copy()
            chart["comparison"] = [
                compare_printed_range(*(
                    "" if pd.isna(row[field]) else str(row[field])
                    for field in ("name", "result", "unit", "reference_range")
                ))
                for _, row in chart.iterrows()
            ]
            report_tab.subheader("Report chart")
            report_tab.caption(
                "OK means within the entered range only, not necessarily healthy. "
                "Low and high results are outside that range; both need professional review. "
                "Colors do not indicate urgency. This is not a diagnosis."
            )
            status_colors = {
                "OK (within range)": "background-color: #e4f4e8; color: #155e2f",
                "Low": "background-color: #fff2cc; color: #704700",
                "High": "background-color: #fff2cc; color: #704700",
                "Not compared": "background-color: #edf0f2; color: #374151",
            }
            styled_chart = chart.style.map(lambda status: status_colors.get(status, ""), subset=["comparison"])
            report_tab.dataframe(styled_chart, hide_index=True)
    except ImageValidationError as error:
        report_tab.error(str(error))
    except Exception:
        report_tab.error("This file could not be read as an image. Please upload a valid image.")


@st.cache_resource
def get_vector_store():
    return get_or_build_vector_store()


reference_tab.header("Ask a question about your medicines")
reference_tab.caption(
    "Ask a general question such as 'What is amoxicillin?' or "
    "'What is paracetamol used for?'. This is not medical advice; "
    "consult a healthcare professional for personal guidance."
)
with reference_tab.form("medicine_question"):
    question = st.text_input("Your question", key=f"question_text_{upload_version}")
    submitted = st.form_submit_button("Ask", type="primary")

if submitted:
    st.session_state.pop("question_result", None)
    if not question.strip():
        reference_tab.warning("Enter a question first.")
    else:
        try:
            with reference_tab.spinner("Searching verified references..."):
                answer = answer_question(question.strip(), get_vector_store())
            st.session_state["question_result"] = answer
        except Exception as error:
            reference_tab.error(
                f"Could not search the reference data ({type(error).__name__}). "
                "Please try again."
            )

if "question_result" in st.session_state:
    if st.session_state["question_result"] == SERVICE_BUSY_MESSAGE:
        reference_tab.warning(st.session_state["question_result"])
    else:
        reference_tab.write(st.session_state["question_result"])

