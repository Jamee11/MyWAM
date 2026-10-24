#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
cluster_base="${OPENWAM_CLUSTER_BASE:-/home/mingfeihangroup/xiaoming}"
dataset="${OPENWAM_DATASET:-$cluster_base/datasets/robotwin2_official_clean_ea_rsig_v3_4350_wrist020_schema5_20261015}"
foundation="${OPENWAM_FOUNDATION:-$cluster_base/ckpts/OpenWAM-Alpha-Pretrain-Foundation-Model}"
export PATH="${OPENWAM_VENV:-$project_root/.venv}/bin:$PATH"
export EA_SIDECAR_MANIFEST="${OPENWAM_MANIFEST:-$dataset/openwam_sidecar/aloha50/manifest.ready.json}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TORCH_WARM_POOL=0 PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$project_root/.cache/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$project_root/.cache/triton}"
args=(--config-name="${OPENWAM_TRAIN_CONFIG:-train_relational_sidecar_aloha50_isolated_10epochs}"
      "training.finetune_ckpt_path=$foundation"
      "training.output_path=${OPENWAM_OUTPUT:-$project_root/outputs/sidecar/aloha50_isolated2k_10epochs}")
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  exec python scripts/train.py "${args[@]}" "$@" --cfg job --resolve
fi
[[ "${CHECK_ONLY:-0}" == 0 ]] || { echo 'CHECK_ONLY must be 0 or 1' >&2; exit 2; }
[[ -f "$EA_SIDECAR_MANIFEST" ]] || { echo 'Relocate the supplied manifest first; see README_TRANSFER.md.' >&2; exit 2; }
# No physical GPU indices or handover delay: Slurm already owns the allocation.
python scripts/preflight_ea_lite.py --manifest "$EA_SIDECAR_MANIFEST" --foundation "$foundation" --data-check fast
exec python -m torch.distributed.run --nnodes=1 --nproc_per_node="${NPROC_PER_NODE:-4}" \
  --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29500}" \
  scripts/cluster/train_sidecar_metrics.py "${args[@]}" "$@"
