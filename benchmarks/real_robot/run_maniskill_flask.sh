#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
MODEL=${MODEL:-sidecar}
case "$MODEL" in
  sidecar) DEFAULT_CKPT=outputs/maniskill_sidecar/joint_20261019T222758Z/2026-10-19_22-33-43/checkpoint_step_33880.safetensors ;;
  baseline) DEFAULT_CKPT=outputs/maniskill_baseline/six_task_bs64_10epochs_20261017T050252Z/runs/2026-10-17_05-16-28/checkpoint_step_21180.safetensors ;;
  *) echo 'MODEL must be sidecar or baseline' >&2; exit 2 ;;
esac
: "${TOKEN_FILE:?Set TOKEN_FILE to your private API token file}"
export CUDA_VISIBLE_DEVICES=${GPU_ID:-7}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTHONPATH="$PWD/outputs/real_robot/http_deps:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PYTHON_BIN=${PYTHON_BIN:-/dev/shm/openwam-sidecar-runtime-20261016.3vRKd9/venv/bin/python}
exec "$PYTHON_BIN" benchmarks/real_robot/serve_maniskill_flask.py \
  --checkpoint "${CKPT:-$DEFAULT_CKPT}" --policy "$MODEL" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-5901}" --token-file "$TOKEN_FILE" \
  --manifest "${MANIFEST:-outputs/real_robot/${MODEL}_http_manifest.json}"
