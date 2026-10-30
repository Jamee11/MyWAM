#!/usr/bin/env bash
# One mixture epoch; 4 x 32 x 4 = 512 effective batch by default.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MIDTRAIN_EPOCHS="${MIDTRAIN_EPOCHS:-1}"
if [[ ! "$MIDTRAIN_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MIDTRAIN_EPOCHS must be a positive integer" >&2
  exit 2
fi
export NPROC_PER_NODE=4
export BATCH_SIZE="${BATCH_SIZE:-32}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export MIDTRAIN_MANIFEST="${MIDTRAIN_MANIFEST:-$PWD/datasets/multisim_sidecar_archive_v2_20261025/manifest.ready.json}"
exec bash scripts/train_multisim_sidecar_midtrain.sh \
  training.max_steps=null "training.num_epochs=$MIDTRAIN_EPOCHS" "$@"
