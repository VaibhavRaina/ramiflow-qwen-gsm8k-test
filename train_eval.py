import os
from collections.abc import Callable
from pathlib import Path

from experiment_config import ExperimentConfig, format_training_example, load_experiment_config


def load_runtime():
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    model_path = os.environ["RAMIFLOW_MODEL_PATH"]
    train_path = os.environ["RAMIFLOW_TRAIN_DATASET_PATH"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", local_files_only=True,
    )
    return load_from_disk(train_path), tokenizer, model, set_seed


def train_lora(config, dataset, tokenizer, model, checkpoint: Path):
    from peft import LoraConfig, set_peft_model_state_dict
    from safetensors.torch import load_file
    from trl import SFTConfig, SFTTrainer

    records = dataset.select(range(config.train_examples)).map(
        format_training_example, remove_columns=dataset.column_names,
    )
    arguments = SFTConfig(
        output_dir="/ramiflow/output/trainer",
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        bf16=True,
        max_length=512,
        completion_only_loss=True,
        seed=config.seed,
    )
    adapter = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    trainer = SFTTrainer(
        model=model,
        args=arguments,
        train_dataset=records,
        processing_class=tokenizer,
        peft_config=adapter,
    )
    if checkpoint.exists() and checkpoint.stat().st_size > 0:
        set_peft_model_state_dict(trainer.model, load_file(str(checkpoint)))
    result = trainer.train()
    return trainer.model, result.metrics


def checkpoint_metadata(config: ExperimentConfig) -> dict[str, str]:
    return {
        "format": "ramiflow-peft-v1",
        "lora_alpha": str(config.lora_alpha),
        "lora_rank": str(config.lora_rank),
    }


def save_checkpoint(model, config: ExperimentConfig, checkpoint: Path) -> None:
    temporary = checkpoint.with_suffix(".tmp")
    if config.mode == "lora":
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file

        state = {
            key: value.detach().cpu().contiguous()
            for key, value in get_peft_model_state_dict(model).items()
        }
        save_file(state, str(temporary), metadata=checkpoint_metadata(config))
    else:
        temporary.write_text('{"format":"ramiflow-baseline-v1"}\n', encoding="utf-8")
    os.replace(temporary, checkpoint)


RuntimeLoader = Callable[[], tuple[object, object, object, Callable[[int], None]]]


def run_candidate(
    config_path: Path,
    checkpoint: Path,
    runtime_loader: RuntimeLoader = load_runtime,
) -> None:
    config = load_experiment_config(config_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    train_data, tokenizer, model, set_seed = runtime_loader()
    set_seed(config.seed)
    if config.mode == "lora":
        model, _training_metrics = train_lora(
            config, train_data, tokenizer, model, checkpoint,
        )
    save_checkpoint(model, config, checkpoint)


def main() -> None:
    run_candidate(
        Path("experiment.json"),
        Path(os.environ["RAMIFLOW_CHECKPOINT_PATH"]),
    )


if __name__ == "__main__":
    main()
