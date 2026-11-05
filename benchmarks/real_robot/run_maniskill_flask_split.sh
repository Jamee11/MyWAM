#!/usr/bin/env bash
# Temporary capacity workaround. The original single-GPU launcher is unchanged.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${TOKEN_FILE:?Set TOKEN_FILE to your private API token file}"
export CUDA_VISIBLE_DEVICES=${GPU_IDS:-6,5}  # policy, T5
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/outputs/real_robot/http_deps:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PYTHON_BIN=${PYTHON_BIN:-$PWD/.venv/bin/python}
exec "$PYTHON_BIN" benchmarks/real_robot/serve_maniskill_flask.py \
  --checkpoint "${CKPT:-outputs/real_robot/maniskill20872_flask/deployment/checkpoint/checkpoint_step_20872.safetensors}" \
  --policy sidecar --host "${HOST:-0.0.0.0}" --port "${PORT:-5901}" --token-file "$TOKEN_FILE" \
  --manifest "${MANIFEST:-outputs/real_robot/maniskill20872_flask/server_manifest.json}" \
  --split-text-encoder --min-free-gib "${MIN_FREE_GIB:-15}" --max-memory-gib "${MAX_MEMORY_GIB:-15.25}"
