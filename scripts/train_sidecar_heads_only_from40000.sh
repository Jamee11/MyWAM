#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${HEADS_ONLY_FORK_ROOT:?Set the prepared independent fork root}"
export HEADS_ONLY_FORK_RUN="${HEADS_ONLY_FORK_RUN:-$HEADS_ONLY_FORK_ROOT/runs/from40000}"
export EA_HANDOVER_DIR="$HEADS_ONLY_FORK_ROOT/handover"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export EA_SIDECAR_MANIFEST="${EA_SIDECAR_MANIFEST:-$PWD/outputs/sidecar/aloha_4350/manifest.ready.json}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_WARM_POOL=0 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
[[ -f "$HEADS_ONLY_FORK_ROOT/fork_ready.json" ]] || { echo "Fork copy/configuration not yet verified" >&2; exit 2; }
args=(--config-name=train_relational_sidecar_aloha50_heads_only_10epochs)
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  exec "$PYTHON_BIN" scripts/train_sidecar_heads_only_resume.py "${args[@]}" "$@" --cfg job --resolve
fi
exec "$PYTHON_BIN" -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
  --master_port="${MASTER_PORT:-29693}" scripts/train_sidecar_heads_only_resume.py "${args[@]}" "$@"
