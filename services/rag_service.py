from pathlib import Path
import re

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")
VECTOR_STORE_DIR = Path("data/vector_store")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemini-3.5-flash-lite"
# Below this distance score, a chunk is considered relevant enough to use.
# Calibrated roughly: relevant matches scored ~0.35-0.9, unrelated ~2.0+.
RELEVANCE_THRESHOLD = 1.2
ABSTAIN_MESSAGE = "Information not found in verified database."
SERVICE_BUSY_MESSAGE = "The AI service is temporarily busy or unavailable. Please try again in a few moments."
NHS_PARACETAMOL_SOURCE = "https://www.nhs.uk/medicines/paracetamol-for-adults/"
NHS_PARACETAMOL_ID = "nhs-paracetamol-uses-2026-03-24"
NHS_AMOXICILLIN_SOURCE = "https://www.nhs.uk/medicines/amoxicillin/about-amoxicillin/"
NHS_AMOXICILLIN_ID = "nhs-amoxicillin-overview-2025-06-02"
VERIFIED_REFERENCES = {
    "paracetamol": (NHS_PARACETAMOL_ID, Document(
        page_content="For adults, paracetamol is used to relieve pain, including headaches and toothache, and reduce fever.",
        metadata={"source": NHS_PARACETAMOL_SOURCE, "verification_status": "verified", "document_date": "2026-03-24"},
    )),
    "amoxicillin": (NHS_AMOXICILLIN_ID, Document(
        page_content="Amoxicillin is a penicillin antibiotic used to treat some bacterial infections. It is available only on prescription.",
        metadata={"source": NHS_AMOXICILLIN_SOURCE, "verification_status": "verified", "document_date": "2025-06-02"},
    )),
    "ibuprofen": ("nhs-ibuprofen-adults-2025-08-27", Document(
        page_content="For adults, ibuprofen is a non-steroidal anti-inflammatory medicine used to relieve pain and reduce inflammation and fever.",
        metadata={"source": "https://www.nhs.uk/medicines/ibuprofen-for-adults/", "verification_status": "verified", "document_date": "2025-08-27"},
    )),
    "cetirizine": ("nhs-cetirizine-overview-2025-02-17", Document(
        page_content="Cetirizine is an antihistamine used to relieve allergy symptoms, including hay fever and hives.",
        metadata={"source": "https://www.nhs.uk/medicines/cetirizine/about-cetirizine/", "verification_status": "verified", "document_date": "2025-02-17"},
    )),
    "azithromycin": ("nhs-azithromycin-overview-2026-01-30", Document(
        page_content="Azithromycin is a macrolide antibiotic used to treat some infections. It is available on prescription.",
        metadata={"source": "https://www.nhs.uk/medicines/azithromycin/about-azithromycin/", "verification_status": "verified", "document_date": "2026-01-30"},
    )),
    "metformin": ("nhs-metformin-overview-2026-08-05", Document(
        page_content="Metformin is a prescription medicine used to treat conditions including type 2 diabetes. It helps lower blood sugar.",
        metadata={"source": "https://www.nhs.uk/medicines/metformin/", "verification_status": "verified", "document_date": "2026-08-05"},
    )),
    "amlodipine": ("nhs-amlodipine-overview-2026-06-09", Document(
        page_content="Amlodipine is a prescription calcium-channel blocker used to treat high blood pressure and angina.",
        metadata={"source": "https://www.nhs.uk/medicines/amlodipine/", "verification_status": "verified", "document_date": "2026-06-09"},
    )),
    "atorvastatin": ("nhs-atorvastatin-overview-2026-06-23", Document(
        page_content="Atorvastatin is a prescription statin used to lower high cholesterol and help prevent cardiovascular problems.",
        metadata={"source": "https://www.nhs.uk/medicines/atorvastatin/", "verification_status": "verified", "document_date": "2026-06-23"},
    )),
    "omeprazole": ("nhs-omeprazole-overview-2025-09-17", Document(
        page_content="Omeprazole is a proton pump inhibitor that reduces stomach acid and is used for conditions including heartburn and reflux.",
        metadata={"source": "https://www.nhs.uk/medicines/omeprazole/", "verification_status": "verified", "document_date": "2025-09-17"},
    )),
}
SUPPORTED_QUESTION_MESSAGE = (
    f"{ABSTAIN_MESSAGE} This assistant answers only general questions about "
    "the medicines in its limited reference collection. For personal treatment decisions, ask a "
    "qualified healthcare professional or pharmacist."
)


def load_documents() -> list[Document]:
    documents = []
    for file_path in KNOWLEDGE_BASE_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        documents.append(Document(page_content=text, metadata={"source": file_path.name}))
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_documents(documents)


def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_vector_store() -> Chroma:
    chunks = chunk_documents(load_documents())
    embeddings = get_embedding_model()
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_STORE_DIR),
    )


def load_vector_store() -> Chroma:
    return Chroma(
        persist_directory=str(VECTOR_STORE_DIR),
        embedding_function=get_embedding_model(),
    )


def get_or_build_vector_store() -> Chroma:
    if (VECTOR_STORE_DIR / "chroma.sqlite3").exists():
        store = load_vector_store()
    else:
        store = build_vector_store()
    for document_id, document in VERIFIED_REFERENCES.values():
        if not store.get(ids=[document_id])["ids"]:
            store.add_documents([document], ids=[document_id])
    return store


def retrieve_relevant_chunks(query: str, store: Chroma, k: int = 3) -> list[tuple[Document, float]]:
    return store.similarity_search_with_score(query, k=k)


def retrieve_or_abstain(query: str, store: Chroma, k: int = 3, source: str | None = None) -> list[Document]:
    metadata_filter = {"verification_status": "verified"}
    if source is not None:
        metadata_filter = {"$and": [{"verification_status": "verified"}, {"source": source}]}
    results = store.similarity_search_with_score(
        query, k=k, filter=metadata_filter
    )
    return [doc for doc, score in results if score <= RELEVANCE_THRESHOLD]


def build_prompt(query: str, chunks: list[Document]) -> str:
    context = "\n\n".join(doc.page_content for doc in chunks)
    return (
        "You are a careful medical information assistant. Answer the question "
        "using ONLY the context below. Do not use any outside knowledge. "
        "Do not diagnose, select or recommend a medicine or treatment, "
        "suggest substitutions, or advise changing a prescribed dose. "
        "Treat symptoms and diseases mentioned by the user as unverified. "
        "For personal treatment decisions, direct the user to a qualified "
        "healthcare professional or pharmacist. "
        "If the context does not contain enough information to answer, reply "
        f'exactly with: "{ABSTAIN_MESSAGE}"\n\n'
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, timeout=10, max_retries=0)


def extract_text(content) -> str:
    # Newer Gemini models return content as a list of blocks (text/thoughts/etc.)
    # instead of a plain string, so pull out just the text blocks.
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
    )


def answer_question(query: str, store: Chroma) -> str:
    question = query.strip().lower()
    medicine = "(?:" + "|".join(re.escape(name) for name in VERIFIED_REFERENCES) + ")"
    general_question = (
        rf"(?:what is {medicine}(?: used for)?|"
        rf"what are the uses of {medicine}|"
        rf"tell(?: me)? (?:the )?uses of {medicine})\??"
    )
    if not re.fullmatch(general_question, question):
        return SUPPORTED_QUESTION_MESSAGE

    medicine_name = next(name for name in VERIFIED_REFERENCES if re.search(rf"\b{re.escape(name)}\b", question))
    source = VERIFIED_REFERENCES[medicine_name][1].metadata["source"]
    chunks = [doc for doc in retrieve_or_abstain(query, store, source=source) if doc.metadata.get("source") == source]
    if not chunks:
        return ABSTAIN_MESSAGE
    prompt = build_prompt(query, chunks)
    try:
        response = get_llm().invoke(prompt)
        answer = extract_text(response.content).strip()
    except Exception:
        answer = ""
    if not answer:
        excerpts = "\n\n".join(
            f'{doc.page_content}\nSource: {doc.metadata["source"]}' for doc in chunks
        )
        return f"{SERVICE_BUSY_MESSAGE}\n\nPotentially relevant reference excerpt (may not answer your question):\n\n{excerpts}"
    if answer == ABSTAIN_MESSAGE:
        return answer
    sources = "\n".join(sorted({doc.metadata["source"] for doc in chunks}))
    return f"{answer}\n\nSources:\n{sources}"



