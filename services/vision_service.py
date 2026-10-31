import base64
import json

from langchain_core.messages import HumanMessage

from services.rag_service import extract_text, get_llm


def extract_prescription_candidates(image_bytes: bytes) -> list[dict[str, str]]:
    image_data = base64.b64encode(image_bytes).decode("ascii")
    prompt = (
        "Transcribe medicine details visible in this prescription image. "
        "Do not guess, infer, correct, or complete illegible text. "
        "Do not use diagnoses, symptoms, or medical knowledge to identify medicines. "
        "Return only a JSON array with one object per possible medicine. "
        'Each object must contain exactly the keys "name", "strength", and "form", '
        "with string values copied from the image. Use an empty string for any "
        "field that is unclear or missing. Return [] if no medicine is readable. "
        "Do not include patient details, instructions, or any other text."
    )
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
    ])
    response = get_llm().invoke([message], response_mime_type="application/json")
    candidates = json.loads(extract_text(response.content))
    if not isinstance(candidates, list):
        raise ValueError("Invalid vision response")

    fields = {"name", "strength", "form"}
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != fields:
            raise ValueError("Invalid vision response")
        if not all(isinstance(candidate[field], str) for field in fields):
            raise ValueError("Invalid vision response")
    return candidates


def extract_report_results(image_bytes: bytes) -> list[dict[str, str]]:
    image_data = base64.b64encode(image_bytes).decode("ascii")
    prompt = (
        "Transcribe test names, results, units, and reference ranges visible in this medical report image. "
        "Do not diagnose, interpret, guess, infer, calculate, correct, or complete illegible text. "
        "Copy the reference range exactly as printed for the same result; never supply a missing range. "
        "Return only a JSON array with one object per visible result. "
        'Each object must contain exactly the keys "name", "result", "unit", and "reference_range", '
        "with string values copied from the image. Use an empty string for any field that is unclear or missing. "
        "Return [] if no result is readable. Do not include patient names, dates, identifiers, or other patient details."
    )
    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
    ])
    response = get_llm().invoke([message], response_mime_type="application/json")
    results = json.loads(extract_text(response.content))
    if not isinstance(results, list):
        raise ValueError("Invalid report response")

    fields = {"name", "result", "unit", "reference_range"}
    for result in results:
        if not isinstance(result, dict) or set(result) != fields:
            raise ValueError("Invalid report response")
        if not all(isinstance(result[field], str) for field in fields):
            raise ValueError("Invalid report response")
    return results