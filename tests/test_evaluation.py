import unittest

from evaluation import score_predictions


class EvaluationTest(unittest.TestCase):
    def test_scores_predictions_and_keeps_example_details(self) -> None:
        rows = [
            {"question": "Two plus two?", "answer": "Work\n#### 4"},
            {"question": "Three plus five?", "answer": "Work\n#### 8"},
        ]

        result = score_predictions(rows, ["The answer is 4.", "The answer is 7."])

        self.assertEqual(result["metrics"], {"gsm8k_accuracy": 0.5})
        self.assertEqual(result["examples"], [
            {
                "index": 0,
                "question": "Two plus two?",
                "reference_answer": "4",
                "prediction": "4",
                "raw_output": "The answer is 4.",
                "correct": True,
            },
            {
                "index": 1,
                "question": "Three plus five?",
                "reference_answer": "8",
                "prediction": "7",
                "raw_output": "The answer is 7.",
                "correct": False,
            },
        ])

    def test_rejects_missing_prediction_rows(self) -> None:
        rows = [{"question": "Two plus two?", "answer": "Work\n#### 4"}]

        with self.assertRaisesRegex(ValueError, "prediction count"):
            score_predictions(rows, [])


if __name__ == "__main__":
    unittest.main()
