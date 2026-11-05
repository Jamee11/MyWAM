#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
: "${GPU_ID:?Set GPU_ID to a GPU with at least 35GiB free}"
: "${TOKEN_FILE:?Set TOKEN_FILE to an existing private token file (16+ characters)}"
CKPT=${CKPT:-outputs/piper_sidecar/midtrain_patchwam_2epochs_20261102T101245Z/runs/2026-11-02_10-18-00/checkpoint_step_12000.safetensors}
PYTHON_BIN=${PYTHON_BIN:-/dev/shm/openwam-sidecar-runtime-20261016.3vRKd9/venv/bin/python}
export CUDA_VISIBLE_DEVICES="$GPU_ID"
exec "$PYTHON_BIN" benchmarks/real_robot/piper/serve_piper_flask.py \
  --checkpoint "$CKPT" --token-file "$TOKEN_FILE" --host 0.0.0.0 --port "${PORT:-5902}"
