import json
import unittest
from decimal import Decimal
from io import BytesIO
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from services.product_search import search_shopping, sort_listings


class ProductSearchTests(unittest.TestCase):
    @patch("services.product_search.urlopen")
    def test_searches_india_and_keeps_only_complete_listings(self, mock_open):
        payload = {"shopping_results": [
            {"title": "Example product", "source": "Example shop", "price": "Rs 20",
             "product_link": "https://www.google.com/search?q=example"},
            {"title": "Missing price", "source": "Example shop"},
            {"title": "Untrusted link", "source": "Example shop", "price": "Rs 30",
             "product_link": "https://not-google.example/search"},
            {"title": "Wrong port", "source": "Example shop", "price": "Rs 40",
             "product_link": "https://www.google.com:444/search"},
        ]}
        mock_open.return_value.__enter__.return_value = BytesIO(json.dumps(payload).encode())
        listings, retrieved_at = search_shopping("example medicine 500mg tablet", "test-key")
        request_url = mock_open.call_args.args[0]
        parameters = parse_qs(urlsplit(request_url).query)
        self.assertEqual(parameters["gl"], ["in"])
        self.assertEqual(parameters["no_cache"], ["true"])
        self.assertEqual(len(listings), 3)
        self.assertEqual(listings[0]["google_link"], "https://www.google.com/search?q=example")
        self.assertEqual(listings[1]["google_link"], "")
        self.assertEqual(listings[2]["google_link"], "")
        self.assertTrue(retrieved_at.endswith(" UTC"))

    @patch("services.product_search.urlopen")
    def test_invalid_or_error_response_does_not_expose_details(self, mock_open):
        for payload in ({"error": "secret details"}, {"shopping_results": {}}, []):
            mock_open.return_value.__enter__.return_value = BytesIO(json.dumps(payload).encode())
            with self.subTest(payload=payload), self.assertRaises(RuntimeError) as error:
                search_shopping("example", "test-key")
            self.assertNotIn("secret", str(error.exception))

    @patch("services.product_search.urlopen")
    def test_empty_query_or_key_never_sends_request(self, mock_open):
        with self.assertRaises(ValueError):
            search_shopping("", "test-key")
        with self.assertRaises(ValueError):
            search_shopping("example", "")
        mock_open.assert_not_called()

    @patch("services.product_search.urlopen")
    def test_only_agreeing_single_inr_prices_can_be_sorted(self, mock_open):
        payload = {"shopping_results": [
            {"title": "Higher", "source": "Shop", "price": "₹100.00", "extracted_price": 100},
            {"title": "Ambiguous", "source": "Shop", "price": "₹20 - ₹50", "extracted_price": 20},
            {"title": "Lower", "source": "Shop", "price": "INR 25.50", "extracted_price": 25.5},
            {"title": "Mismatch", "source": "Shop", "price": "₹60", "extracted_price": 6},
        ]}
        mock_open.return_value.__enter__.return_value = BytesIO(json.dumps(payload).encode())
        listings, _ = search_shopping("example", "test-key")
        self.assertEqual([item["price_inr"] for item in listings], [Decimal("100"), None, Decimal("25.5"), None])
        self.assertEqual([item["title"] for item in sort_listings(listings, "Price: low to high")],
                         ["Lower", "Higher", "Ambiguous", "Mismatch"])
        self.assertEqual([item["title"] for item in sort_listings(listings, "Price: high to low")],
                         ["Higher", "Lower", "Ambiguous", "Mismatch"])
        self.assertEqual(sort_listings(listings, "Relevance"), listings)