#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_WARM_POOL=0
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false

PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
BASE_MODEL="${BASE_MODEL:-$PWD/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
export MIDTRAIN_GLOBAL_BATCH="$((NPROC_PER_NODE * BATCH_SIZE * GRAD_ACCUM))"
export MIDTRAIN_MANIFEST="${MIDTRAIN_MANIFEST:-/remote-home/jinminghao/datasets/openwam_sidecar_multisim_midtrain_v1_20261024/manifest.ready.json}"
export MIDTRAIN_OUTPUT_DIR="${MIDTRAIN_OUTPUT_DIR:-$PWD/outputs/midtrain/multisim_$(date -u +%Y%m%dT%H%M%SZ)}"

"$PYTHON_BIN" scripts/check_multisim_sidecar_data.py --manifest "$MIDTRAIN_MANIFEST"
if [[ "${CHECK_ONLY:-1}" == 1 ]]; then
  exit 0
fi
exec "$PYTHON_BIN" -m torch.distributed.run --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="${MASTER_PORT:-29684}" scripts/train_multisim_sidecar.py \
  --config-name=train_multisim_sidecar_midtrain \
  "training.finetune_ckpt_path=$BASE_MODEL" "training.batch_size=$BATCH_SIZE" \
  "training.gradient_accumulation_steps=$GRAD_ACCUM" "$@"
