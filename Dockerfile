FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/ramiflow/huggingface \
    RAMIFLOW_MODEL_PATH=/opt/ramiflow/model \
    RAMIFLOW_TRAIN_DATASET_PATH=/opt/ramiflow/data/train \
    RAMIFLOW_EVAL_DATASET_PATH=/opt/ramiflow/data/eval

WORKDIR /opt/ramiflow/build
COPY requirements.txt ./requirements.txt
RUN python -m pip install --requirement requirements.txt

RUN python - <<'PY'
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-0.6B"
model_revision = "c1899de289a04d12100db370d81485cdf75e47ca"
dataset_id = "openai/gsm8k"
dataset_revision = "740312add88f781978c0658806c59bc2815b9866"

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision)
model = AutoModelForCausalLM.from_pretrained(model_id, revision=model_revision)
tokenizer.save_pretrained("/opt/ramiflow/model")
model.save_pretrained("/opt/ramiflow/model", safe_serialization=True)

train = load_dataset(dataset_id, "main", revision=dataset_revision, split="train")
test = load_dataset(dataset_id, "main", revision=dataset_revision, split="test")
train.select(range(512)).save_to_disk("/opt/ramiflow/data/train")
test.select(range(64)).save_to_disk("/opt/ramiflow/data/eval")
PY

WORKDIR /workspace
