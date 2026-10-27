#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${EA_OUTPUT_DIR:?Set an independent experiment output directory}"
: "${EA_HANDOVER_DIR:?Set a fresh GPU handover directory}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export EA_SIDECAR_MANIFEST="${EA_SIDECAR_MANIFEST:-$PWD/outputs/sidecar/aloha_4350/manifest.ready.json}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled PYTHONNOUSERSITE=1
export TORCH_WARM_POOL=0 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PYTHONHOME PYTHONPATH EA_STOP_AFTER_OPT_STEPS
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
foundation="$PWD/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model"
args=(--config-name=train_relational_sidecar_aloha50_ea_grad_anneal_10epochs
      "training.finetune_ckpt_path=$foundation" "training.output_path=$EA_OUTPUT_DIR")
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  exec "$PYTHON_BIN" scripts/train.py "${args[@]}" "$@" --cfg job --resolve
fi
"$PYTHON_BIN" scripts/preflight_ea_lite.py --manifest "$EA_SIDECAR_MANIFEST" --foundation "$foundation" --data-check fast
exec "$PYTHON_BIN" -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
  --master_port="${MASTER_PORT:-29793}" scripts/train_relational_sidecar_formal.py "${args[@]}" "$@"
