#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
project_root="$PWD"
python_bin="${PYTHON_BIN:-${OPENWAM_VENV:-$project_root/.venv}/bin/python}"
dataset="${OPENWAM_DATASET:-$project_root/datasets/robotwin2_official_clean_ea_rsig_v3_4350_wrist020_schema5_20261015}"
checkpoint="${OPENWAM_INIT_CKPT:-$project_root/assets/openwam_ckpt/openwam_alpha/Sidecar-PatchWAM-Warmup2K}"
export EA_SIDECAR_MANIFEST="${OPENWAM_MANIFEST:-$dataset/openwam_sidecar/aloha50/manifest.ready.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#devices[@]} == 8 ]] || { echo 'Allocate exactly eight visible GPUs.' >&2; exit 2; }
[[ -x "$python_bin" && -f "$EA_SIDECAR_MANIFEST" && -f "$checkpoint/checkpoint_step_4000.safetensors" ]] || {
  echo 'Check Python, the relocated manifest and the packaged 2K checkpoint directory.' >&2; exit 2;
}
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$project_root/.cache/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$project_root/.cache/triton}"
unset PYTHONHOME PYTHONPATH EA_STOP_AFTER_OPT_STEPS
args=(--config-name=train_sidecar_patchwam_warmstart_8gpu_10epochs
      "training.finetune_ckpt_path=$checkpoint"
      "training.output_path=${OPENWAM_OUTPUT:-$project_root/outputs/sidecar/patchwam_warmstart8_$(date -u +%Y%m%dT%H%M%SZ)}")
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  exec "$python_bin" scripts/train.py "${args[@]}" "$@" --cfg job --resolve
fi
[[ "${CHECK_ONLY:-0}" == 0 ]] || { echo 'CHECK_ONLY must be 0 or 1.' >&2; exit 2; }
exec "$python_bin" -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29502}" \
  scripts/cluster/train_sidecar_metrics.py "${args[@]}" "$@"
