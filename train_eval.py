import json
import os
from dataclasses import asdict
from pathlib import Path

from evaluation import score_predictions
from experiment_config import format_training_example, load_experiment_config


def load_runtime():
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    model_path = os.environ["RAMIFLOW_MODEL_PATH"]
    train_path = os.environ["RAMIFLOW_TRAIN_DATASET_PATH"]
    eval_path = os.environ["RAMIFLOW_EVAL_DATASET_PATH"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", local_files_only=True,
    )
    return torch, load_from_disk(train_path), load_from_disk(eval_path), tokenizer, model, set_seed


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


def save_checkpoint(model, mode: str, checkpoint: Path) -> None:
    temporary = checkpoint.with_suffix(".tmp")
    if mode == "lora":
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file

        state = {
            key: value.detach().cpu().contiguous()
            for key, value in get_peft_model_state_dict(model).items()
        }
        save_file(state, str(temporary), metadata={"format": "ramiflow-peft-v1"})
    else:
        temporary.write_text('{"format":"ramiflow-baseline-v1"}\n', encoding="utf-8")
    os.replace(temporary, checkpoint)


def generate_predictions(torch, rows, tokenizer, model, max_new_tokens: int) -> list[str]:
    outputs: list[str] = []
    model.eval()
    for row in rows:
        messages = format_training_example(row)["prompt"]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                pad_token_id=tokenizer.eos_token_id,
            )
        suffix = generated[0][inputs["input_ids"].shape[-1]:]
        outputs.append(tokenizer.decode(suffix, skip_special_tokens=True).strip())
    return outputs


def main() -> None:
    config = load_experiment_config(Path("experiment.json"))
    output = Path("/ramiflow/output")
    checkpoint = Path(os.environ["RAMIFLOW_CHECKPOINT_PATH"])
    output.mkdir(parents=True, exist_ok=True)
    torch, train_data, eval_data, tokenizer, model, set_seed = load_runtime()
    set_seed(config.seed)
    training_metrics: dict[str, object] = {}
    if config.mode == "lora":
        model, training_metrics = train_lora(
            config, train_data, tokenizer, model, checkpoint,
        )
    save_checkpoint(model, config.mode, checkpoint)
    rows = [dict(row) for row in eval_data.select(range(config.eval_examples))]
    set_seed(config.seed)
    predictions = generate_predictions(
        torch, rows, tokenizer, model, config.max_new_tokens,
    )
    scored = score_predictions(rows, predictions)
    correct = sum(int(item["correct"]) for item in scored["examples"])
    metrics = {
        **scored["metrics"],
        "eval_examples": len(rows),
        "correct_examples": correct,
        "train_examples": config.train_examples if config.mode == "lora" else 0,
        "train_steps": config.max_steps if config.mode == "lora" else 0,
        "train_loss": float(training_metrics.get("train_loss", 0.0)),
    }
    details = {
        "config": asdict(config),
        "model": {"id": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca"},
        "dataset": {"id": "openai/gsm8k", "revision": "740312add88f781978c0658806c59bc2815b9866"},
        "training_metrics": training_metrics,
        "evaluation": scored,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    (output / "details.json").write_text(
        json.dumps(details, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    for example in scored["examples"]:
        print(json.dumps({"event": "evaluation_example", **example}, sort_keys=True))
    print(json.dumps({"event": "evaluation_summary", "metrics": metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
