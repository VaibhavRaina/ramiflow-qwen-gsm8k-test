import unittest

from benchmark_utils import extract_predicted_answer, extract_reference_answer


class BenchmarkAnswerTest(unittest.TestCase):
    def test_extracts_gsm8k_reference_answer(self) -> None:
        self.assertEqual(extract_reference_answer("Reasoning\n#### 1,234"), "1234")

    def test_uses_the_last_number_in_model_output(self) -> None:
        output = "First I considered 12, then corrected the final answer to 14."
        self.assertEqual(extract_predicted_answer(output), "14")

    def test_reports_missing_predictions(self) -> None:
        self.assertIsNone(extract_predicted_answer("I cannot determine the result."))


if __name__ == "__main__":
    unittest.main()
