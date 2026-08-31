import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str
    seed: int
    train_examples: int
    eval_examples: int
    max_new_tokens: int
    max_steps: int
    learning_rate: float
    lora_rank: int
    lora_alpha: int


def load_experiment_config(path: Path) -> ExperimentConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("experiment configuration must be an object")
    expected = {field.name for field in fields(ExperimentConfig)}
    unexpected = sorted(set(raw) - expected)
    missing = sorted(expected - set(raw))
    if unexpected:
        raise ValueError(f"unknown experiment configuration field: {unexpected[0]}")
    if missing:
        raise ValueError(f"missing experiment configuration field: {missing[0]}")
    if raw["mode"] not in {"baseline", "lora"}:
        raise ValueError("mode must be baseline or lora")
    for name in (
        "train_examples", "eval_examples", "max_new_tokens", "max_steps",
        "lora_rank", "lora_alpha",
    ):
        if isinstance(raw[name], bool) or not isinstance(raw[name], int) or raw[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(raw["seed"], bool) or not isinstance(raw["seed"], int) or raw["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        isinstance(raw["learning_rate"], bool)
        or not isinstance(raw["learning_rate"], (int, float))
        or raw["learning_rate"] <= 0
    ):
        raise ValueError("learning_rate must be positive")
    return ExperimentConfig(**raw)


def format_training_example(row: dict[str, str]) -> dict[str, object]:
    return {
        "prompt": [{
            "role": "user",
            "content": "Solve the problem and end with the numeric answer. " + row["question"],
        }],
        "completion": [{"role": "assistant", "content": row["answer"]}],
    }
