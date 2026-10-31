import unittest
from unittest.mock import Mock, patch

from services.vision_service import extract_prescription_candidates, extract_report_results


class VisionServiceTests(unittest.TestCase):
    @patch("services.vision_service.get_llm")
    def test_returns_candidates_and_sends_image(self, get_llm):
        get_llm.return_value.invoke.return_value = Mock(
            content='[{"name":"Paracetamol","strength":"500 mg","form":"Tab"}]'
        )

        result = extract_prescription_candidates(b"synthetic-image")

        self.assertEqual(result, [{"name": "Paracetamol", "strength": "500 mg", "form": "Tab"}])
        message = get_llm.return_value.invoke.call_args.args[0][0]
        self.assertTrue(message.content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(get_llm.return_value.invoke.call_args.kwargs["response_mime_type"], "application/json")

    @patch("services.vision_service.get_llm")
    def test_rejects_malformed_and_unexpected_responses(self, get_llm):
        for content in ("not json", '{}', '[{"name": null, "strength": "", "form": ""}]',
                        '[{"name":"x","strength":"","form":"","patient":"name"}]'):
            with self.subTest(content=content):
                get_llm.return_value.invoke.return_value = Mock(content=content)
                with self.assertRaises(ValueError):
                    extract_prescription_candidates(b"synthetic-image")

    @patch("services.vision_service.get_llm")
    def test_extracts_only_report_fields(self, get_llm):
        get_llm.return_value.invoke.return_value = Mock(
            content='[{"name":"Example test","result":"5.0","unit":"mg/dL","reference_range":"4.0-6.0"}]'
        )
        results = extract_report_results(b"synthetic-image")
        self.assertEqual(results, [{"name": "Example test", "result": "5.0", "unit": "mg/dL", "reference_range": "4.0-6.0"}])
        message = get_llm.return_value.invoke.call_args.args[0][0]
        self.assertIn("never supply a missing range", message.content[0]["text"])
        self.assertEqual(get_llm.return_value.invoke.call_args.kwargs["response_mime_type"], "application/json")

    @patch("services.vision_service.get_llm")
    def test_report_rejects_malformed_or_extra_fields(self, get_llm):
        for content in ("not json", '{}', '[{"name":"x","result":null,"unit":"","reference_range":""}]',
                        '[{"name":"x","result":"1","unit":"","reference_range":"","patient":"name"}]'):
            with self.subTest(content=content):
                get_llm.return_value.invoke.return_value = Mock(content=content)
                with self.assertRaises(ValueError):
                    extract_report_results(b"synthetic-image")


if __name__ == "__main__":
    unittest.main()