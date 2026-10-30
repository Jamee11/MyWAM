#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# This run explicitly uses the original shared-disk environment.
export PATH="$PWD/.venv/bin:$PATH"
unset PYTHONHOME PYTHONPATH EA_STOP_AFTER_OPT_STEPS
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TORCH_WARM_POOL=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EA_SIDECAR_MANIFEST="$PWD/outputs/sidecar/aloha_4350/manifest.ready.json"
foundation="$PWD/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model"
: "${EA_HANDOVER_DIR:?Set a fresh handover directory}"
args=(--config-name="${EA_TRAIN_CONFIG:-train_relational_sidecar_aloha50_10epochs}"
      "training.finetune_ckpt_path=$foundation"
      "training.output_path=${EA_OUTPUT_DIR:-outputs/sidecar/aloha50_C_joint_bs32acc2_10epochs}")
if [[ "${CHECK_ONLY:-1}" == 1 ]]; then
  exec python scripts/train.py "${args[@]}" "$@" --cfg job --resolve
fi
[[ "${CHECK_ONLY:-1}" == 0 ]] || { echo 'CHECK_ONLY must be 0 or 1' >&2; exit 2; }
# Availability is checked after CPU preparation, before DeepSpeed's large GPU move.
python scripts/preflight_ea_lite.py --manifest "$EA_SIDECAR_MANIFEST" --foundation "$foundation" --data-check fast
exec python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port="${MASTER_PORT:-29880}" \
  scripts/train_relational_sidecar_formal.py "${args[@]}" "$@"
