#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
unset PYTHONHOME PYTHONPATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TORCH_WARM_POOL=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON_BIN="${PYTHON_BIN:-${OPENWAM_VENV:-$PWD/.venv}/bin/python}"
args=(--config-name=train_maniskill_archive_baseline_2epochs)
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  exec "$PYTHON_BIN" scripts/train_maniskill_archive_baseline.py "${args[@]}" "$@" --cfg job --resolve
fi
: "${EA_HANDOVER_DIR:?Set a fresh handover directory}"
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/train_maniskill_archive_baseline.py "${args[@]}" "$@"
