#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
cluster_base="${OPENWAM_CLUSTER_BASE:-/home/mingfeihangroup/xiaoming}"
foundation="${OPENWAM_FOUNDATION:-$cluster_base/ckpts/OpenWAM-Alpha-Pretrain-Foundation-Model}"
export PATH="${OPENWAM_VENV:-$project_root/.venv}/bin:$PATH"
export EA_LITE_MANIFEST="${OPENWAM_MANIFEST:-$project_root/data/baseline/manifest.ready.json}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$project_root/.cache/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$project_root/.cache/triton}"
args=(--config-name=train_alpha_baseline_clean50
      "training.finetune_ckpt_path=$foundation"
      "training.output_path=${OPENWAM_OUTPUT:-$project_root/outputs/baseline_clean50}")
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  exec python scripts/train.py "${args[@]}" "$@" --cfg job --resolve
fi
[[ -f "$EA_LITE_MANIFEST" ]] || { echo 'Run prepare_baseline_paths.py once after unpacking.' >&2; exit 2; }
[[ -f "$foundation/checkpoint_step_154000.safetensors" ]] || { echo "Missing Foundation: $foundation" >&2; exit 2; }
# Slurm owns CUDA_VISIBLE_DEVICES; never replace its GPU allocation with physical indices.
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29500}"
exec bash scripts/train.sh "${args[@]}" "$@"
