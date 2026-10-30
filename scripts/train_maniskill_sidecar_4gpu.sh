#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${OPENWAM_VENV:-$PWD/.venv}/bin:$PATH"
unset PYTHONHOME PYTHONPATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TORCH_WARM_POOL=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MANISKILL_SIDECAR_MANIFEST="${MANISKILL_SIDECAR_MANIFEST:-$PWD/outputs/maniskill_sidecar/data_absolute_eef_20261019/manifest.ready.json}"
args=(--config-name=train_maniskill_sidecar_10epochs
      "training.output_path=${MANISKILL_SIDECAR_OUTPUT_DIR:-outputs/maniskill_sidecar/six_task_joint_bs32acc2_10epochs}")
if [[ "${CHECK_ONLY:-1}" == 1 ]]; then
  exec python scripts/train_maniskill_sidecar.py "${args[@]}" "$@" --cfg job --resolve
fi
[[ "${CHECK_ONLY:-1}" == 0 ]] || { echo 'CHECK_ONLY must be 0 or 1' >&2; exit 2; }
test -f "$MANISKILL_SIDECAR_MANIFEST"
python - <<'PY'
import os
import subprocess
selected = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
if len(selected) != 4 or len(set(selected)) != 4 or not all(s.isdigit() for s in selected):
    raise SystemExit('Select four distinct physical GPU indices')
rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free',
                                '--format=csv,noheader,nounits'], text=True)
devices = {r[0]: r for line in rows.splitlines() if (r := [s.strip() for s in line.split(',')])}
for index in selected:
    row = devices[index]
    if 'A100' not in row[1] or int(row[2]) < 75000 or int(row[3]) < 70000:
        raise SystemExit(f'GPU{index} needs A100-80GB and >=70000 MiB free; observed {row[1:]}')
print('Four-GPU availability passed; no jobs will be stopped automatically', flush=True)
PY
exec python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port="${MASTER_PORT:-29886}" \
  scripts/train_maniskill_sidecar.py "${args[@]}" "$@"
