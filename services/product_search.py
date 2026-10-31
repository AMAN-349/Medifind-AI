import json
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from urllib.error import URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen


def search_shopping(query: str, api_key: str) -> tuple[list[dict[str, str | Decimal | None]], str]:
    if not query.strip() or not api_key.strip():
        raise ValueError("A search query and API key are required.")

    parameters = urlencode({
        "engine": "google_shopping",
        "q": query,
        "gl": "in",
        "hl": "en",
        "no_cache": "true",
        "api_key": api_key,
    })
    try:
        with urlopen(f"https://serpapi.com/search.json?{parameters}", timeout=15) as response:
            payload = json.loads(response.read(1_000_001))
    except (URLError, TimeoutError, ValueError, OSError):
        raise RuntimeError("Shopping search is unavailable.") from None

    if not isinstance(payload, dict) or "error" in payload:
        raise RuntimeError("Shopping search is unavailable.")
    results = payload.get("shopping_results", [])
    if not isinstance(results, list):
        raise RuntimeError("Shopping search returned an invalid response.")

    listings = []
    for result in results:
        if not isinstance(result, dict):
            continue
        title, source, price = (result.get(field) for field in ("title", "source", "price"))
        if not all(isinstance(value, str) and value.strip() for value in (title, source, price)):
            continue
        link = result.get("product_link", "")
        if not isinstance(link, str):
            link = ""
        try:
            parsed = urlsplit(link)
            if (parsed.scheme != "https" or parsed.hostname != "www.google.com" or parsed.port not in (None, 443)
                    or parsed.username or parsed.password or parsed.path != "/search"):
                link = ""
        except ValueError:
            link = ""
        price_inr = None
        price_match = re.fullmatch(r"(?:₹|INR)\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)", price.strip(), re.IGNORECASE)
        extracted_price = result.get("extracted_price")
        if price_match and isinstance(extracted_price, (int, float)) and not isinstance(extracted_price, bool):
            try:
                displayed_price = Decimal(price_match.group(1).replace(",", ""))
                numeric_price = Decimal(str(extracted_price))
                if numeric_price.is_finite() and numeric_price >= 0 and numeric_price == displayed_price:
                    price_inr = numeric_price
            except InvalidOperation:
                pass
        listings.append({"title": title, "source": source, "price": price, "google_link": link, "price_inr": price_inr})

    return listings, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def sort_listings(listings: list[dict], order: str) -> list[dict]:
    if order == "Relevance":
        return list(listings)
    priced = [listing for listing in listings if listing.get("price_inr") is not None]
    unpriced = [listing for listing in listings if listing.get("price_inr") is None]
    return sorted(priced, key=lambda listing: listing["price_inr"], reverse=order == "Price: high to low") + unpriced