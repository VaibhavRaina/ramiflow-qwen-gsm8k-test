import json
import tempfile
import unittest
from pathlib import Path

from experiment_config import format_training_example, load_experiment_config


class ExperimentContractTest(unittest.TestCase):
    @staticmethod
    def valid_config() -> dict[str, object]:
        return {
            "mode": "lora",
            "seed": 1337,
            "train_examples": 512,
            "eval_examples": 64,
            "max_new_tokens": 128,
            "max_steps": 50,
            "learning_rate": 0.0002,
            "lora_rank": 16,
            "lora_alpha": 32,
        }

    def test_loads_a_valid_lora_experiment(self) -> None:
        config = self.valid_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            path.write_text(json.dumps(config), encoding="utf-8")

            loaded = load_experiment_config(path)

        self.assertEqual(loaded.mode, "lora")
        self.assertEqual(loaded.seed, 1337)
        self.assertEqual(loaded.train_examples, 512)

    def test_rejects_an_unknown_experiment_mode(self) -> None:
        config = self.valid_config()
        config["mode"] = "full_finetune"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            path.write_text(json.dumps(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "mode"):
                load_experiment_config(path)

    def test_rejects_non_positive_training_limits(self) -> None:
        config = self.valid_config()
        config["max_steps"] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            path.write_text(json.dumps(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "max_steps"):
                load_experiment_config(path)

    def test_rejects_unknown_configuration_fields(self) -> None:
        config = self.valid_config()
        config["untracked_override"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            path.write_text(json.dumps(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "untracked_override"):
                load_experiment_config(path)

    def test_formats_gsm8k_as_conversational_prompt_completion(self) -> None:
        example = format_training_example({
            "question": "What is 3 plus 4?",
            "answer": "Three plus four is seven.\n#### 7",
        })

        self.assertEqual(example, {
            "prompt": [{
                "role": "user",
                "content": "Solve the problem and end with the numeric answer. What is 3 plus 4?",
            }],
            "completion": [{
                "role": "assistant",
                "content": "Three plus four is seven.\n#### 7",
            }],
        })


if __name__ == "__main__":
    unittest.main()
