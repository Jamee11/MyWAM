#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"
unset PYTHONHOME PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TORCH_WARM_POOL=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MANISKILL_MANIFEST="${MANISKILL_MANIFEST:-$PWD/outputs/maniskill_baseline/data_absolute_eef_20261017/manifest.ready.json}"
test -f "$MANISKILL_MANIFEST"
python - <<'PY'
import subprocess
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
free={int(r.split(',')[0]):int(r.split(',')[1]) for r in rows.splitlines()}
if any(free[g]<70000 for g in range(4)):
    raise SystemExit(f'GPUs 0–3 require >=70000 MiB free each; current free MiB: {free}')
print('GPU 0–3 memory availability passed',flush=True)
PY
exec python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port="${MASTER_PORT:-29885}" \
  scripts/train_maniskill_baseline.py --config-name=train_maniskill_baseline_10epochs \
  "training.output_path=${MANISKILL_OUTPUT_DIR:-outputs/maniskill_baseline/six_task_bs64_10epochs}" "$@"
