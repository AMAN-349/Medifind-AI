# MediFind AI

MediFind AI is a learning prototype for reviewing synthetic prescription images, exploring limited medicine references, and comparing transcribed report values with printed ranges. It is not a clinical tool.

For a complete walkthrough of the workflows, AI concepts, modules, data, and tests, open the [Developer Guide](docs/developer-guide.html) in a browser.

## First-time setup

1. Install Python 3.11 or newer, [uv](https://docs.astral.sh/uv/getting-started/installation/), and Tesseract OCR. On macOS with Homebrew, install Tesseract with `brew install tesseract`.
2. Open a terminal in the project folder and run `uv sync` to create the virtual environment and install the dependencies from `uv.lock`.
3. Add `GOOGLE_API_KEY` to a local `.env` file for Gemini. Add `SERPAPI_API_KEY` only if you want to try the optional Google Shopping demo. Use your own keys; never commit or share the file.
4. Run `uv run streamlit run app.py` and open the local address printed in the terminal. Use only synthetic or appropriately de-identified images, and read the privacy disclosure before uploading.
5. Run `uv run python -m unittest discover -s tests -v` to check the automated tests.

The server defaults to `127.0.0.1`, which limits access to the local machine; this is not authentication and can be overridden by configuration.

## Docker and hosted deployment

`Dockerfile` describes a Linux environment with Python, the locked dependencies, and Tesseract. `.dockerignore` keeps local environment files, Streamlit secrets, report samples, and the generated vector store out of the build context. Build the image locally with `docker build -t medifind-ai:local .`. The first build can be large because the Linux lockfile includes machine-learning dependencies.

The image preserves the app's loopback-only local mode; ordinary Docker port publishing does not make a service bound to a container's loopback address reachable. Docker Desktop host networking requires an explicit setting and has not yet been verified for this setup. Do not expose local mode through a proxy, tunnel, or forwarded port, and do not bake credentials into the image. Container environment variables can be visible through Docker inspection; only pass keys at runtime on a trusted machine. A container's generated vector store is temporary unless separately persisted.

Streamlit Community Cloud builds from the GitHub repository rather than this Docker image. It can use `uv.lock` for Python dependencies and `packages.txt` for Tesseract. A hosted deployment is not ready until an OIDC provider, an approved account allowlist, secrets, and the actual deployment configuration have been set up and tested as described under **Access controls**. Do not bypass these checks or deploy for patient use. The new deployment files must be committed and pushed before a cloud build can use them.

Uploading a PNG or JPEG of at most 10 MB and 20 million pixels automatically runs local OCR and sends the entire image to Google Gemini for AI extraction. The disclosure appears before the uploader; by uploading, you agree to this processing. On the Gemini free tier, Google may use submitted content to improve its products. Prescription images can contain sensitive health information; obtain permission and remove personal information when possible. De-identification alone does not guarantee that data is anonymous or appropriate to send to a third party. If Gemini fails, available local OCR results remain available for review.

Use the **Prescription review** and **Medicine reference** tabs to switch between the two workflows. **Clear session**, in the sidebar, removes the current upload, review, consent, question, and answer from the active app session. Removing or replacing an upload also clears its review and consent. This does not delete data previously sent to Google or other copies outside the app; it is not a substitute for a reviewed privacy and retention policy.

## Report review

The **Report review** tab accepts a single PNG or JPEG image (up to 10 MB and 20 million pixels); PDF and multipage uploads are not supported. Uploading automatically sends the whole image to Google Gemini, including any identifying information. Use only synthetic or appropriately de-identified samples for this prototype.

The model transcribes possible test names, results, units, and reference ranges. Check and correct every field against the original report, including whether the result and reference range use the same unit, before selecting **Compare checked results**. The resulting table labels simple numeric results as below, within, or above the two-sided range entered from the report. Missing units or ranges, inequalities, qualitative results, and ambiguous formats are not compared. If extraction fails, the fields can be entered manually. Session clearing removes the current report state but cannot delete data already sent to Google.

AI transcription can be wrong or incomplete. The table is not a diagnosis, an assessment of overall health, or a substitute for a clinician's review. Narrative findings, such as ultrasound reports, are not automatically interpreted; a clinician must interpret them in context.

## Access controls

The default `local` mode is for prototype work on one trusted machine. It has no login and refuses to start the app workflow if Streamlit is bound to an address other than `127.0.0.1`. Do not expose local mode through a reverse proxy, tunnel, or port forwarding: binding to localhost does not prevent those from exposing it.

An experimental `oidc` mode uses Streamlit's OpenID Connect sign-in and permits only explicitly approved subject IDs from one expected issuer. To configure it, an administrator must:

1. Register an application with a trusted OIDC identity provider and configure its exact callback URL. Use HTTPS for a hosted deployment.
2. Provide the provider's `client_id`, `client_secret`, `server_metadata_url`, the `redirect_uri`, and a strong random `cookie_secret` in Streamlit secrets under `[auth]`. See the [Streamlit authentication documentation](https://docs.streamlit.io/develop/concepts/connections/authentication). The local `.streamlit/secrets.toml` file is Git-ignored; never commit secrets.
3. Set `MEDIFIND_AUTH_MODE=oidc`, `MEDIFIND_OIDC_ISSUER` to the exact `iss` claim from that provider, and `MEDIFIND_ALLOWED_SUBJECTS` to a comma-separated list of approved `sub` claims for that provider and client. Obtain these identifiers securely from the identity-provider administrator. Restart the app after changing environment variables.
4. Test login, denied accounts, expiry, logout, and the complete deployment with the actual provider. Require an independent security review before exposing it to others or processing health data.

Missing settings deny access. The app checks issuer, subject, and ID-token expiry when Streamlit reruns the script. It does not continuously revoke an idle session, and Streamlit login cookies can persist for 30 days; signing out does not terminate other already-open sessions. Use a reviewed identity and session-management architecture for production. Browser exception details are hidden, but server logs can still contain details and require separate protection and retention controls.

There is no standalone signup flow. Account enrollment and recovery must be configured with the chosen identity provider, and an administrator must add the new account's subject ID to the allowlist.

OCR and AI candidates are combined without overwriting the existing draft. Exact complete matches are shown once; conflicting and incomplete readings remain separate and unverified. Both OCR and AI may misread or invent text, especially handwriting. Review every field against the prescription and ask a pharmacist about anything unclear. Incomplete entries cannot be confirmed. Confirming the table is required before a demo shopping search, but does not constitute a clinical or pharmacist verification.

The sample catalog is not used to validate medicines or determine availability. No authorized pharmacy data source is connected. The source-checked medical reference collection contains short NHS summaries for nine medicines, shown for exact candidate names as general information. These summaries have not been independently reviewed by a clinician and do not verify the prescription or whether a medicine is appropriate for a condition. Questions are limited to general forms such as "What is amoxicillin?" and "What is paracetamol used for?". Other questions receive an abstention and guidance to consult a healthcare professional. This app does not diagnose, select treatments, or replace a healthcare professional.

## Google Shopping demo

Create a SerpApi account and put `SERPAPI_API_KEY` in the local, Git-ignored `.env` file, then restart the app. Do not share or commit the key. After confirming a synthetic demo medicine, select it and explicitly choose **Search Google Shopping**. The query goes to SerpApi and may be retained by the provider; never use real patient information in this prototype.

The provider's results are unverified Google Shopping search listings, not approved pharmacy offers or confirmed medicine matches. Results can be sorted and browsed four at a time without making additional searches. Price sorting uses only single INR amounts that agree with the provider's numeric price; other prices appear last. Returned prices and seller names may be stale or wrong, and a Google result link is not necessarily a direct retailer link. The timestamp records when this app fetched results, not when a seller updated the price. A result may be missing, and no purchase or treatment decision should be based on it. No direct pharmacy API is connected.

## Data folder

- `data/sample_medicines/` contains the small fictional catalog and retailer-price examples; it is not a real inventory.
- `data/knowledge_base/` contains illustrative, unverified documents, which are excluded from medical answers.
- `data/vector_store/` is the generated search index, including source-checked summaries. It is not the AI model and is excluded from Git.

Real product comparisons require permissioned retailer data, exact product and pack matching, current prices, availability and source timestamps. Do not treat demo data as real listings.

## Before patient use

These are outstanding requirements, not completed reviews or approvals:

- **Pharmacy data:** Obtain a written agreement and API documentation from an authorized pharmacy partner. Confirm permitted use, prescription handling, product identifiers, strength and form, pack size, availability, currency, direct links, location, timestamps, and update frequency. Implement and test the integration against the approved contract; never display fictional examples as live results.
- **Handwriting:** With appropriate permission and privacy safeguards, obtain representative, independently labeled de-identified samples, including illegible entries and different writers. A qualified reviewer should establish acceptance criteria in advance. Evaluate exact medicine, strength, and form matches, missed entries, incorrect entries, and whether illegible text is left unresolved. Separate development and evaluation samples by writer. Synthetic images alone are not clinical validation.
- **Clinical review:** Ask qualified clinicians or pharmacists to review source summaries, extraction and confirmation workflows, abstention behavior, product matching, and safety testing. Document their findings, remediation, and approval before any patient use.
- **Privacy and security:** Obtain an independent assessment of data flows, consent, applicable law, retention and deletion, provider agreements, encryption, logging, backups, incident response, and deployment. Verify that the selected Gemini service and contractual terms are appropriate before transmitting any sensitive data.
- **Access and deployment:** Validate the OIDC gate with the chosen provider and implement independently reviewed session management, encrypted transport, secure secret management, and operational monitoring before making a hosted service available. Localhost binding and session clearing are not substitutes for these controls. Do not put prescription images, API keys, or sensitive evaluation data in Git.

Run the focused checks with `uv run python -m unittest discover -s tests -v`.
