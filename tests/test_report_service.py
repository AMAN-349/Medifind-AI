import unittest

from services.report_service import compare_printed_range


class ReportServiceTests(unittest.TestCase):
    def test_compares_only_against_printed_range(self):
        for result, expected in [("3.9", "Low"), ("4.0", "OK (within range)"),
                     ("6.0", "OK (within range)"), ("6.1", "High")]:
            with self.subTest(result=result):
                self.assertEqual(compare_printed_range("Example", result, "mg/dL", "4.0-6.0"), expected)

    def test_unclear_or_qualitative_results_are_not_compared(self):
        for name, result, unit, reference_range in [
            ("Example", "positive", "mg/dL", "4-6"),
            ("Ultrasound", "normal appearance", "", ""),
            ("Example", "<4", "mg/dL", "4-6"),
            ("Example", "5", "mg/dL", "<6"),
            ("Example", "5", "mg/dL", "6-4"),
            ("Example", "5", "", "4-6"),
            ("", "5", "mg/dL", "4-6"),
            ("Example", "5", "mg/dL", "4-6 mg/dL"),
        ]:
            with self.subTest(result=result, reference_range=reference_range):
                self.assertEqual(compare_printed_range(name, result, unit, reference_range), "Not compared")

    def test_accepts_signed_values_and_en_dash(self):
        self.assertEqual(compare_printed_range("Example", "-1.0", "units", "-2.0\u2013-0.5"), "OK (within range)")


if __name__ == "__main__":
    unittest.main()