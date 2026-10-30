#!/usr/bin/env bash
# Defaults to config inspection. Actual four-GPU work requires CHECK_ONLY=0.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
experiment="${EA_EXPERIMENT:-C}"
stage="${EA_STAGE:-warmup}"
profile="${EA_GPU_PROFILE:-80}"
processes="${EA_NUM_GPUS:-4}"
if [[ ! "$processes" =~ ^[1-9][0-9]*$ ]]; then
  echo 'EA_NUM_GPUS must be a positive integer' >&2; exit 2
fi
foundation="${EA_FOUNDATION_DIR:-$PWD/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model}"
export EA_LITE_MANIFEST="${EA_LITE_MANIFEST:-$PWD/outputs/ea_lite/data/manifest.ready.json}"
args=(--config-name=train_ea_lite_4gpu "training.finetune_ckpt_path=$foundation")
case "$experiment" in
  A) args+=(model=dual_system dataloader.auxiliary=false); stage=joint ;;
  B) args+=(model.architecture.ea_lite.action_conditioning=false "model.architecture.ea_lite.stage=$stage") ;;
  C) args+=(model.architecture.ea_lite.action_conditioning=true "model.architecture.ea_lite.stage=$stage") ;;
  *) echo 'EA_EXPERIMENT must be A, B, or C' >&2; exit 2 ;;
esac
case "$stage" in warmup|joint|action_only) ;; *) echo 'Invalid EA_STAGE' >&2; exit 2 ;; esac
case "$profile" in
  80) args+=("training.offload_optimizer_device=${EA_OFFLOAD:-none}") ;;
  40) args+=(training.offload_optimizer_device=cpu) ;;
  *) echo 'EA_GPU_PROFILE must be 80 or 40' >&2; exit 2 ;;
esac
args+=("training.output_path=outputs/ea_lite/debug_${experiment}_${stage}" "$@")
if [[ "${CHECK_ONLY:-1}" == 1 ]]; then
  export CUDA_VISIBLE_DEVICES=''
  exec python scripts/train.py "${args[@]}" --cfg job --resolve
fi
if [[ "${CHECK_ONLY:-1}" != 0 ]]; then
  echo 'CHECK_ONLY must be 1 (inspect) or explicit 0 (launch)' >&2
  exit 2
fi
: "${CUDA_VISIBLE_DEVICES:?Set the physical GPU indices allocated to this job}"
python scripts/preflight_ea_lite.py --manifest "$EA_LITE_MANIFEST" --foundation "$foundation" --num-gpus "$processes" --gpu-profile "$profile" --gpu-name "${EA_GPU_NAME:-A100}" --data-check "${EA_DATA_CHECK:-fast}"
export NPROC_PER_NODE="$processes"
export NCCL_DEBUG=WARN OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec torchrun --nnodes=1 --nproc_per_node="$processes" --master_port="${MASTER_PORT:-29874}" scripts/debug_ea_lite.py "${args[@]}"
