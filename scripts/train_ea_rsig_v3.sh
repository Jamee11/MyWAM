#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${ALLOW_LEGACY_EA_V3:-0}" != 1 ]]; then
  echo 'Legacy EA-v3 uses 14D qpos/40 steps/H18. The current design uses scripts/train_ea_lite_4gpu.sh (default CHECK_ONLY=1). Set ALLOW_LEGACY_EA_V3=1 only to reproduce the archived experiment.' >&2
  exit 2
fi
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH="$PWD/.venv/bin:$PATH"
: "${EA_DATASET_DIRS:?Set a JSON list of audited schema-5 dataset directories}"
foundation="${EA_FOUNDATION_DIR:-$PWD/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model}"
if [[ ! -f "$foundation/download_verification.json" ]]; then
  echo 'Foundation download verification is missing.' >&2
  exit 1
fi
python - "$foundation" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
proof = json.loads((root / 'download_verification.json').read_text())
name = 'checkpoint_step_154000.safetensors'
record = proof['files'].get(name)
if not record or not record['verified_upstream'] or not (root / name).is_file():
    raise SystemExit('Foundation download is not complete and verified.')
if (root / name).stat().st_size != record['size']:
    raise SystemExit('Foundation file size changed after verification.')
PY
# Explicit GPU selection avoids taking GPUs occupied by unrelated work.
: "${CUDA_VISIBLE_DEVICES:?Set available GPU indices explicitly}"
exec bash scripts/train.sh model=dual_system_ea_rsig_v3 dataloader=ea_rsig_v3 \
  "training.finetune_ckpt_path=$foundation" training.batch_size=1 \
  training.adam_foreach=false training.adam_fused=true \
  training.use_gradient_checkpointing=true training.dataset_num_workers=0 \
  "$@"
