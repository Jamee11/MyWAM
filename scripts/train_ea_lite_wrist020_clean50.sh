#!/usr/bin/env bash
# Default: inspect only. The user explicitly starts the new run with CHECK_ONLY=0.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export PATH="${OPENWAM_VENV:-$project_root/.venv}/bin:$PATH"
export EA_WRIST020_MANIFEST="${EA_WRIST020_MANIFEST:-$project_root/outputs/ea_lite/wrist020_clean50_20261015/data/manifest.ready.json}"
foundation="${EA_FOUNDATION_DIR:-$project_root/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model}"
output="${EA_OUTPUT_DIR:-$project_root/outputs/ea_lite/wrist020_clean50_C_joint_bs64_5epochs}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
args=(--config-name=train_ea_lite_wrist020_clean50
      "training.finetune_ckpt_path=$foundation" "training.output_path=$output")
if [[ "${CHECK_ONLY:-1}" == 1 ]]; then
  exec python scripts/train.py "${args[@]}" "$@" --cfg job --resolve
fi
[[ "${CHECK_ONLY:-1}" == 0 ]] || { echo 'CHECK_ONLY must be 0 or 1.' >&2; exit 2; }
: "${CUDA_VISIBLE_DEVICES:?Select the four GPUs after stopping the old run yourself}"
case "${EA_DATA_CHECK:-fast}" in
  fast|full)
    python scripts/preflight_ea_lite.py --manifest "$EA_WRIST020_MANIFEST" \
      --foundation "$foundation" --data-check "${EA_DATA_CHECK:-fast}" \
      --num-gpus 4 --gpu-profile 80 --gpu-name A100 ;;
  none)
    python -c 'from scripts.preflight_ea_lite import check_gpus; check_gpus(4, 80, "A100")' ;;
  *) echo 'EA_DATA_CHECK must be fast, full or none.' >&2; exit 2 ;;
esac
export NPROC_PER_NODE=4 NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29875}"
exec bash scripts/train.sh "${args[@]}" "$@"
