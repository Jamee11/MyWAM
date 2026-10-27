#!/usr/bin/env bash
# A/B/C share data, FM objective and optimizer settings; config selects the architecture.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${OPENWAM_VENV:-$PWD/.venv}/bin:$PATH"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" NCCL_DEBUG=WARN
# Eager Sidecar debugging does not need Inductor's import-time worker pool.
export TORCH_WARM_POOL=0
foundation="${EA_FOUNDATION_DIR:-$PWD/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model}"
export EA_SIDECAR_MANIFEST="${EA_SIDECAR_MANIFEST:-$PWD/outputs/sidecar/aloha_4350/manifest.ready.json}"
stage="${EA_STAGE:-warmup}"
processes="${EA_NUM_GPUS:-4}"
[[ "$processes" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid EA_NUM_GPUS' >&2; exit 2; }
case "$stage" in warmup|action_only|joint) ;; *) echo 'Invalid EA_STAGE' >&2; exit 2 ;; esac
args=(--config-name=train_relational_sidecar_4gpu "training.finetune_ckpt_path=$foundation"
      "training.output_path=${EA_OUTPUT_DIR:-outputs/sidecar/${EA_EXPERIMENT:-C}_${stage}}")
case "${EA_EXPERIMENT:-C}" in
  A) args+=(model=dual_system dataloader.auxiliary=false) ;;
  B) args+=(model.architecture.relational_sidecar.action_conditioning=false "model.architecture.relational_sidecar.stage=$stage") ;;
  C) args+=("model.architecture.relational_sidecar.stage=$stage") ;;
  *) echo 'EA_EXPERIMENT must be A, B or C' >&2; exit 2 ;;
esac
args+=("$@")
if [[ "${CHECK_ONLY:-1}" == 1 ]]; then
  export CUDA_VISIBLE_DEVICES=''
  exec python scripts/train.py "${args[@]}" --cfg job --resolve
fi
[[ "${CHECK_ONLY:-1}" == 0 ]] || { echo 'CHECK_ONLY must be 0 or 1' >&2; exit 2; }
: "${CUDA_VISIBLE_DEVICES:?Select allocated GPUs}"
python scripts/preflight_ea_lite.py --manifest "$EA_SIDECAR_MANIFEST" --foundation "$foundation" \
  --data-check "${EA_DATA_CHECK:-fast}" --num-gpus "$processes" --gpu-profile "${EA_GPU_PROFILE:-80}" --gpu-name "${EA_GPU_NAME:-A100}"
export NPROC_PER_NODE="$processes"
exec python -m torch.distributed.run --nnodes=1 --nproc_per_node="$processes" --master_port="${MASTER_PORT:-29876}" scripts/debug_relational_sidecar.py "${args[@]}"
