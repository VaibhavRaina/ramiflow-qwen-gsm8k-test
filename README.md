# Ramiflow Qwen GSM8K validation

This public, secret-free repository validates Ramiflow's managed GPU lifecycle with a reproducible
Qwen3-0.6B GSM8K baseline and LoRA child experiments. The candidate container has no network access
and contains only the public training split. A separate backend-controlled evaluator image owns the
hidden evaluation split and returns only the aggregate required metric.

## Reproducibility

- Model: `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`
- Dataset: `openai/gsm8k@740312add88f781978c0658806c59bc2815b9866`
- Container base: `pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime@sha256:6acf597eeb8e376a96580dde4952f37cc017fef732bb40bfc73f28f25e3f64b4`
- Experiment parameters: `experiment.json`
- Required hosted metric: `gsm8k_accuracy`

Run unit tests with `python -m unittest discover -s tests -p 'test_*.py'`. Ramiflow runs
`python train_eval.py` inside the digest-pinned candidate image, persists the adapter checkpoint,
and evaluates that checkpoint in the isolated trusted evaluator. Hidden examples and evaluator
stdout are never returned to the candidate or synchronized into the experiment DAG.
