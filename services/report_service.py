import re
from decimal import Decimal, InvalidOperation


NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"


def compare_printed_range(name: str, result: str, unit: str, reference_range: str) -> str:
    if not name.strip() or not unit.strip():
        return "Not compared"
    value_match = re.fullmatch(rf"\s*({NUMBER})\s*", result)
    range_match = re.fullmatch(rf"\s*({NUMBER})\s*[-\u2013]\s*({NUMBER})\s*", reference_range)
    if not value_match or not range_match:
        return "Not compared"
    try:
        value = Decimal(value_match.group(1))
        lower = Decimal(range_match.group(1))
        upper = Decimal(range_match.group(2))
    except InvalidOperation:
        return "Not compared"
    if lower > upper:
        return "Not compared"
    if value < lower:
        return "Low"
    if value > upper:
        return "High"
    return "OK (within range)"