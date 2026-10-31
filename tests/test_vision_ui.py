from io import BytesIO
from decimal import Decimal
import runpy
import sys
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image


class VisionUiTests(unittest.TestCase):
    def setUp(self):
        self.image = BytesIO()
        Image.new("RGB", (40, 40), "white").save(self.image, format="PNG")
        self.image.seek(0)
        self.streamlit = MagicMock()
        self.streamlit.sidebar = self.streamlit
        self.report_tab = MagicMock()
        self.report_tab.file_uploader.return_value = None
        self.report_tab.button.return_value = False
        self.report_tab.data_editor.side_effect = lambda data, **kwargs: data.copy()
        self.streamlit.tabs.return_value = (self.streamlit, self.streamlit, self.report_tab)
        self.streamlit.session_state = {}
        self.streamlit.config.get_option.return_value = "127.0.0.1"
        self.streamlit.stop.side_effect = SystemExit
        self.streamlit.user.to_dict.return_value = {}
        self.streamlit.cache_resource.side_effect = lambda function: function
        self.streamlit.file_uploader.return_value = self.image
        self.streamlit.text_input.return_value = ""
        self.streamlit.form_submit_button.return_value = False
        self.streamlit.data_editor.side_effect = lambda data, **kwargs: data.copy()
        self.streamlit.button.return_value = False
        self.ocr = MagicMock()
        self.ocr.image_to_string.return_value = "Tab Paracetamol 500mg"
        self.vision_patch = patch("services.vision_service.extract_prescription_candidates", return_value=[])
        self.vision = self.vision_patch.start()
        self.addCleanup(self.vision_patch.stop)
        self.report_patch = patch("services.vision_service.extract_report_results", return_value=[])
        self.report_vision = self.report_patch.start()
        self.addCleanup(self.report_patch.stop)
        self.shopping_patch = patch("services.product_search.search_shopping", return_value=([], "2026-01-01 00:00 UTC"))
        self.shopping = self.shopping_patch.start()
        self.addCleanup(self.shopping_patch.stop)

    def run_app(self):
        with patch.dict(sys.modules, {"streamlit": self.streamlit, "pytesseract": self.ocr}), patch.dict(
            "os.environ", {"MEDIFIND_AUTH_MODE": "local"}
        ):
            runpy.run_path("app.py")

    def run_with_auth(self, mode, subjects="", issuer="https://issuer.example"):
        with patch.dict(sys.modules, {"streamlit": self.streamlit, "pytesseract": self.ocr}), patch.dict(
            "os.environ", {
                "MEDIFIND_AUTH_MODE": mode,
                "MEDIFIND_ALLOWED_SUBJECTS": subjects,
                "MEDIFIND_OIDC_ISSUER": issuer,
            }
        ):
            runpy.run_path("app.py")

    def test_local_mode_rejects_non_local_bind(self):
        self.streamlit.config.get_option.return_value = "0.0.0.0"
        with self.assertRaises(SystemExit):
            self.run_app()
        self.streamlit.file_uploader.assert_not_called()

    def test_oidc_requires_an_allowlist(self):
        with self.assertRaises(SystemExit):
            self.run_with_auth("oidc")
        self.streamlit.file_uploader.assert_not_called()

    def test_oidc_requires_an_issuer(self):
        with self.assertRaises(SystemExit):
            self.run_with_auth("oidc", "approved-subject", issuer="")
        self.streamlit.file_uploader.assert_not_called()

    def test_oidc_requires_a_login(self):
        with self.assertRaises(SystemExit):
            self.run_with_auth("oidc", "approved-subject")
        self.streamlit.file_uploader.assert_not_called()

    def test_oidc_rejects_unapproved_or_expired_accounts(self):
        for subject, expiry, issuer in [
            ("other-subject", 9999999999, "https://issuer.example"),
            ("approved-subject", 1, "https://issuer.example"),
            ("approved-subject", None, "https://issuer.example"),
            ("approved-subject", 9999999999, "https://other.example"),
        ]:
            with self.subTest(subject=subject, expiry=expiry, issuer=issuer):
                self.streamlit.session_state["question_result"] = "private data"
                self.streamlit.user.to_dict.return_value = {
                    "is_logged_in": True, "sub": subject, "exp": expiry, "iss": issuer
                }
                with self.assertRaises(SystemExit):
                    self.run_with_auth("oidc", "approved-subject")
                self.assertNotIn("question_result", self.streamlit.session_state)
        self.streamlit.file_uploader.assert_not_called()

    def test_oidc_allows_approved_unexpired_account(self):
        self.streamlit.user.to_dict.return_value = {
            "is_logged_in": True, "sub": "approved-subject", "exp": 9999999999, "iss": "https://issuer.example"
        }
        self.run_with_auth("oidc", "approved-subject")
        self.streamlit.file_uploader.assert_called_once()

    def test_unknown_mode_is_denied(self):
        with self.assertRaises(SystemExit):
            self.run_with_auth("typo")
        self.streamlit.file_uploader.assert_not_called()

    @patch("services.rag_service.get_or_build_vector_store")
    @patch("services.rag_service.answer_question", return_value="Reference answer")
    def test_reference_answer_is_rendered_in_reference_tab(self, answer, store):
        reference_tab = MagicMock()
        self.streamlit.tabs.return_value = (self.streamlit, reference_tab, self.report_tab)
        self.streamlit.text_input.return_value = "What is amoxicillin?"
        self.streamlit.form_submit_button.return_value = True
        self.run_app()
        reference_tab.form.assert_called_once_with("medicine_question")
        reference_tab.write.assert_called_once_with("Reference answer")
        self.assertNotIn("Reference answer", [call.args[0] for call in self.streamlit.write.call_args_list])
        answer.assert_called_once_with("What is amoxicillin?", store.return_value)

    def test_upload_runs_both_once_without_checkbox(self):
        self.run_app()
        self.run_app()
        self.vision.assert_called_once()
        self.ocr.image_to_string.assert_called_once()
        self.streamlit.checkbox.assert_not_called()
        self.streamlit.file_uploader.assert_called_with(
            "Upload and send image to Google Gemini", type=["png", "jpg", "jpeg"],
            key="prescription_upload_0", max_upload_size=10,
        )
        self.assertNotIn("Tab Paracetamol 500mg", [call.args[0] for call in self.streamlit.text.call_args_list])
        self.assertEqual(self.streamlit.session_state["medicines_df"]["name"].tolist(), ["Paracetamol"])

    def test_warning_precedes_upload(self):
        self.streamlit.file_uploader.return_value = None
        self.run_app()
        calls = self.streamlit.mock_calls
        warning_index = next(
            index for index, call in enumerate(calls)
            if call[0] == "warning" and "Uploading sends the whole image" in call.args[0]
        )
        upload_index = next(index for index, call in enumerate(calls) if call[0] == "file_uploader")
        self.assertLess(warning_index, upload_index)
        self.vision.assert_not_called()
        self.ocr.image_to_string.assert_not_called()

    def test_ocr_failure_still_attempts_ai(self):
        self.ocr.image_to_string.side_effect = RuntimeError("private error details")
        self.vision.return_value = [{"form": "Tab", "name": "Paracetamol", "strength": "500 mg"}]
        self.run_app()
        self.vision.assert_called_once()
        self.assertEqual(self.streamlit.session_state["medicines_df"]["name"].tolist(), ["Paracetamol"])
        self.streamlit.warning.assert_any_call("Local OCR failed. Review any AI results carefully.")
        self.assertNotIn("private error details", str(self.streamlit.warning.call_args_list))

    def test_removal_and_reupload_processes_image_again(self):
        self.run_app()
        self.streamlit.file_uploader.return_value = None
        self.run_app()
        self.streamlit.file_uploader.return_value = self.image
        self.run_app()
        self.assertEqual(self.vision.call_count, 2)
        self.assertEqual(self.ocr.image_to_string.call_count, 2)

    def test_matching_ai_result_keeps_ocr_draft_without_confirming(self):
        self.vision.return_value = [{"form": "Tab", "name": "Paracetamol", "strength": "500 mg"}]
        self.run_app()
        self.vision.assert_called_once()
        self.assertEqual(self.streamlit.session_state["medicines_df"]["strength"].tolist(), ["500mg"])
        self.assertNotIn("confirmed_medicines", self.streamlit.session_state)
        self.streamlit.button.side_effect = lambda label, **kwargs: label == "Confirm medicines"
        self.run_app()
        self.assertEqual(self.streamlit.session_state["confirmed_medicines"]["strength"].tolist(), ["500mg"])
        self.streamlit.info.assert_any_call(
            "No authorized pharmacy data source is connected."
        )
        self.streamlit.dataframe.assert_not_called()

    def test_shopping_search_requires_confirmation_and_explicit_click(self):
        self.streamlit.selectbox.return_value = "Paracetamol 500mg Tab"
        with patch.dict("os.environ", {"SERPAPI_API_KEY": "test-key"}):
            self.run_app()
            self.shopping.assert_not_called()
            self.streamlit.button.side_effect = lambda label, **kwargs: label == "Confirm medicines"
            self.run_app()
            self.shopping.assert_not_called()
            self.streamlit.button.side_effect = lambda label, **kwargs: label == "Search Google Shopping"
            self.run_app()
            self.shopping.assert_called_once_with("Paracetamol 500mg Tab", "test-key")
            self.streamlit.info.assert_any_call("No complete search results were returned.")

    def test_shopping_accepts_confirmed_medicine_outside_demo_catalog(self):
        self.ocr.image_to_string.return_value = ""
        self.vision.return_value = [{"form": "Tab", "name": "Metformin", "strength": "500mg"}]
        self.streamlit.selectbox.return_value = "Metformin 500mg Tab"
        with patch.dict("os.environ", {"SERPAPI_API_KEY": "test-key"}):
            self.run_app()
            self.streamlit.button.side_effect = lambda label, **kwargs: label == "Confirm medicines"
            self.run_app()
            self.shopping.assert_not_called()
            self.streamlit.button.side_effect = lambda label, **kwargs: label == "Search Google Shopping"
            self.run_app()
            self.shopping.assert_called_once_with("Metformin 500mg Tab", "test-key")
            self.assertNotIn("Catalog match results", [call.args[0] for call in self.streamlit.subheader.call_args_list])

    def test_shopping_search_requires_api_key(self):
        self.streamlit.button.side_effect = lambda label, **kwargs: label == "Confirm medicines"
        with patch.dict("os.environ", {"SERPAPI_API_KEY": ""}):
            self.run_app()
        self.shopping.assert_not_called()
        self.streamlit.info.assert_any_call("Set SERPAPI_API_KEY to enable the Google Shopping demo.")

    def test_shopping_results_are_cleared_with_prescription(self):
        self.streamlit.session_state["shopping_results"] = ("example", [], "timestamp")
        self.streamlit.file_uploader.return_value = None
        self.run_app()
        self.assertNotIn("shopping_results", self.streamlit.session_state)

    def test_shopping_pagination_and_sort_do_not_search_again(self):
        self.shopping.return_value = ([
            {"title": f"Example {index}", "source": "Shop", "price": f"₹{index}",
             "price_inr": Decimal(index), "google_link": ""}
            for index in range(1, 7)
        ], "2026-01-01 00:00 UTC")
        navigation = MagicMock()
        next_clicked = [False]
        navigation.button.side_effect = lambda label, **kwargs: label == "Next" and next_clicked[0]
        self.streamlit.container.side_effect = lambda **kwargs: navigation if kwargs.get("key") == "shopping_navigation" else MagicMock()
        selected_sort = ["Relevance"]
        self.streamlit.selectbox.side_effect = lambda label, *args, **kwargs: (
            "Paracetamol 500mg Tab" if label == "Select a confirmed demo medicine" else selected_sort[0]
        )
        with patch.dict("os.environ", {"SERPAPI_API_KEY": "test-key"}):
            self.streamlit.button.side_effect = lambda label, **kwargs: label == "Confirm medicines"
            self.run_app()
            self.streamlit.button.side_effect = lambda label, **kwargs: label == "Search Google Shopping"
            self.run_app()
            self.assertEqual(self.streamlit.container.call_count, 5)
            navigation.caption.assert_called_with("Page 1 of 2  |  6 results")
            self.streamlit.container.reset_mock()
            next_clicked[0] = True
            self.streamlit.button.side_effect = lambda label, **kwargs: False
            self.run_app()
            self.assertEqual(self.streamlit.container.call_count, 3)
            navigation.caption.assert_called_with("Page 2 of 2  |  6 results")
            self.streamlit.rerun.assert_called_once()
            next_clicked[0] = False
            selected_sort[0] = "Price: high to low"
            self.run_app()
            navigation.caption.assert_called_with("Page 1 of 2  |  6 results")
        self.shopping.assert_called_once()

    def test_conflicting_ai_result_is_kept_separately(self):
        self.vision.return_value = [{"form": "Tab", "name": "Paracetamol", "strength": "250 mg"}]
        self.run_app()
        self.assertEqual(self.streamlit.session_state["medicines_df"]["strength"].tolist(), ["500mg", "250 mg"])
        self.assertNotIn("confirmed_medicines", self.streamlit.session_state)

    def test_candidate_reference_is_general_and_source_linked(self):
        self.streamlit.checkbox.return_value = False
        self.run_app()
        self.streamlit.link_button.assert_called_once_with(
            "NHS reference: paracetamol", "https://www.nhs.uk/medicines/paracetamol-for-adults/"
        )
        self.streamlit.caption.assert_any_call(
            "A reference match does not verify the prescription or whether a medicine is right for a condition."
        )

    def test_unknown_candidate_does_not_get_a_reference(self):
        self.streamlit.checkbox.return_value = False
        self.ocr.image_to_string.return_value = "Tab Unknownmedicine 500mg"
        self.run_app()
        self.streamlit.link_button.assert_not_called()
        self.streamlit.info.assert_any_call("Information not found in the reference collection for: unknownmedicine")

    def test_incomplete_candidate_cannot_be_confirmed(self):
        self.ocr.image_to_string.return_value = ""
        self.vision.return_value = [{"form": "Tab", "name": "Paracetamol", "strength": ""}]
        self.run_app()
        self.streamlit.button.side_effect = lambda label, **kwargs: label == "Confirm medicines"
        self.run_app()
        self.assertNotIn("confirmed_medicines", self.streamlit.session_state)
        self.streamlit.warning.assert_any_call(
            "Every medicine needs a name, strength, and form. Ask a pharmacist or prescriber about unclear details."
        )

    def test_report_chart_requires_review_and_preserves_qualitative_results(self):
        self.report_tab.file_uploader.return_value = self.image
        self.report_vision.return_value = [
            {"name": "Example", "result": "7", "unit": "mg/dL", "reference_range": "4-6"},
            {"name": "Example low", "result": "3", "unit": "mg/dL", "reference_range": "4-6"},
            {"name": "Example within", "result": "5", "unit": "mg/dL", "reference_range": "4-6"},
            {"name": "Ultrasound", "result": "Narrative finding", "unit": "", "reference_range": ""},
        ]
        self.run_app()
        self.report_tab.dataframe.assert_not_called()
        self.report_tab.button.return_value = True
        self.run_app()
        self.report_vision.assert_called_once()
        styled_chart = self.report_tab.dataframe.call_args.args[0]
        self.assertEqual(styled_chart.data["comparison"].tolist(), ["High", "Low", "OK (within range)", "Not compared"])
        chart_html = styled_chart.to_html()
        self.assertIn("background-color: #e4f4e8", chart_html)
        self.assertIn("background-color: #fff2cc", chart_html)
        self.assertIn("background-color: #edf0f2", chart_html)

        self.report_tab.button.return_value = False
        self.report_tab.dataframe.reset_mock()
        def edit_result(data, **kwargs):
            result = data.copy()
            result.loc[0, "result"] = "5"
            return result
        self.report_tab.data_editor.side_effect = edit_result
        self.run_app()
        self.report_tab.dataframe.assert_not_called()

    def test_report_failure_allows_manual_review_without_repeated_send(self):
        self.report_tab.file_uploader.return_value = self.image
        self.report_vision.side_effect = RuntimeError("private error details")
        self.run_app()
        self.run_app()
        self.report_vision.assert_called_once()
        self.report_tab.warning.assert_any_call("Report extraction failed. You can enter the printed results manually.")
        self.assertNotIn("private error details", str(self.report_tab.warning.call_args_list))

    def test_invalid_report_replacement_clears_previous_result(self):
        self.report_tab.file_uploader.return_value = self.image
        self.report_vision.return_value = [{"name": "Example", "result": "5", "unit": "mg/dL", "reference_range": "4-6"}]
        self.run_app()
        self.report_tab.file_uploader.return_value = BytesIO(b"invalid")
        self.run_app()
        self.report_vision.assert_called_once()
        self.assertNotIn("report_draft", self.streamlit.session_state)
        self.report_tab.error.assert_any_call("This file could not be read as an image. Please upload a valid image.")

    def test_clear_session_removes_report_data(self):
        self.report_tab.file_uploader.return_value = self.image
        self.run_app()
        self.streamlit.session_state["report_upload_0"] = b"synthetic report"
        self.streamlit.session_state["report_confirmed"] = "synthetic result"
        clear_call = next(call for call in self.streamlit.button.call_args_list if call.args[0] == "Clear session")
        clear_call.kwargs["on_click"]()
        self.assertEqual(self.streamlit.session_state, {"upload_version": 1})

    def test_failure_preserves_draft(self):
        self.vision.side_effect = RuntimeError("private error details")
        self.run_app()
        self.run_app()
        self.vision.assert_called_once()
        self.assertEqual(self.streamlit.session_state["medicines_df"]["name"].tolist(), ["Paracetamol"])
        self.streamlit.warning.assert_any_call("AI extraction failed. Only any available local OCR results are shown.")
        self.assertNotIn("private error details", str(self.streamlit.warning.call_args_list))

    def test_api_failure_hides_exception_details(self):
        cause = RuntimeError("sensitive response text")
        cause.code = 504
        error = RuntimeError("private error details")
        error.__cause__ = cause
        self.vision.side_effect = error
        self.run_app()
        message = str(self.streamlit.warning.call_args_list)
        self.assertIn("AI extraction failed", message)
        self.assertNotIn("sensitive response text", message)
        self.assertNotIn("private error details", message)

    def test_broken_number_is_preserved_for_review(self):
        self.ocr.image_to_string.return_value = ""
        self.vision.return_value = [{"form": "Tab", "name": "Paracetamol", "strength": "5 00 mg"}]
        self.run_app()
        self.streamlit.button.side_effect = lambda label, **kwargs: label == "Confirm medicines"
        self.run_app()
        self.assertEqual(self.streamlit.session_state["confirmed_medicines"]["strength"].tolist(), ["5 00 mg"])
        self.streamlit.dataframe.assert_not_called()

    def test_clear_session_removes_review_upload_consent_and_question(self):
        self.streamlit.checkbox.return_value = False
        self.run_app()
        state = self.streamlit.session_state
        state.update({
            "prescription_upload_0": b"image",
            "vision_consent_old": True,
            "medicines_old_0": {"edited_rows": {}},
            "confirmed_medicines": "draft",
            "question_text_0": "personal question",
            "question_result": "answer",
        })
        clear_call = next(call for call in self.streamlit.button.call_args_list if call.args[0] == "Clear session")
        clear_call.kwargs["on_click"]()
        self.assertEqual(state, {"upload_version": 1})

    def test_removing_upload_clears_review_and_consent(self):
        self.streamlit.checkbox.return_value = False
        self.run_app()
        self.streamlit.session_state["vision_consent_old"] = True
        self.streamlit.session_state["confirmed_medicines"] = "draft"
        self.streamlit.file_uploader.return_value = None
        self.run_app()
        self.assertEqual(self.streamlit.session_state, {"upload_version": 0})

    def test_invalid_replacement_clears_previous_review(self):
        self.streamlit.checkbox.return_value = False
        self.run_app()
        self.streamlit.session_state["vision_consent_old"] = True
        self.streamlit.session_state["confirmed_medicines"] = "draft"
        self.streamlit.file_uploader.return_value = BytesIO(b"not an image")
        self.run_app()
        self.assertEqual(self.streamlit.session_state, {"upload_version": 0})
        self.streamlit.error.assert_any_call("This file could not be read as an image. Please upload a valid image.")

    def test_oversized_file_is_not_processed(self):
        self.streamlit.file_uploader.return_value = BytesIO(b"x" * (10 * 1024 * 1024 + 1))
        self.run_app()
        self.ocr.image_to_string.assert_not_called()
        self.vision.assert_not_called()
        self.streamlit.error.assert_any_call("The image must be 10 MB or smaller.")

    def test_oversized_dimensions_are_not_processed(self):
        oversized_image = MagicMock(format="PNG", width=5000, height=4001)
        with patch("PIL.Image.open", return_value=oversized_image):
            self.run_app()
        self.ocr.image_to_string.assert_not_called()
        self.vision.assert_not_called()
        self.streamlit.error.assert_any_call("The image must be 20 million pixels or smaller.")

    def test_non_png_or_jpeg_image_is_not_processed(self):
        image = BytesIO()
        Image.new("RGB", (40, 40), "white").save(image, format="GIF")
        image.seek(0)
        self.streamlit.file_uploader.return_value = image
        self.run_app()
        self.ocr.image_to_string.assert_not_called()
        self.vision.assert_not_called()
        self.streamlit.error.assert_any_call("Only PNG and JPEG images are supported.")


if __name__ == "__main__":
    unittest.main()