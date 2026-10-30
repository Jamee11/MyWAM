#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
root="$PWD"
python_bin="${PYTHON_BIN:-${OPENWAM_VENV:-$root/.venv}/bin/python}"
checkpoint="${OPENWAM_INIT_CKPT:-$root/assets/openwam_ckpt/openwam_alpha/Sidecar-PatchWAM-Warmup2K}"
dataset="${OPENWAM_DATASET:-$root/datasets/robotwin2_official_clean_ea_rsig_v3_4350_wrist020_schema5_20261015}"
export EA_SIDECAR_MANIFEST="${OPENWAM_MANIFEST:-$dataset/openwam_sidecar/aloha50/manifest.ready.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#devices[@]} == 8 && -x "$python_bin" && -f "$EA_SIDECAR_MANIFEST" && -f "$checkpoint/checkpoint_step_4000.safetensors" ]] || {
  echo 'Need eight GPUs, runtime, relocated manifest and ORIGINAL transferred checkpoint_step_4000.' >&2; exit 2;
}
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_EXTENSIONS_DIR="$root/.cache/torch_extensions" TRITON_CACHE_DIR="$root/.cache/triton"
export HDF5_USE_FILE_LOCKING=FALSE
export TRITON_LIBCUDA_PATH="$root/.local/lib"
export LIBRARY_PATH="$root/.local/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$root/.local/lib:${LD_LIBRARY_PATH:-}"
unset PYTHONHOME PYTHONPATH EA_STOP_AFTER_OPT_STEPS
args=(--config-name=train_sidecar_xa_warmup2k_8gpu_10epochs
      "training.finetune_ckpt_path=$checkpoint/checkpoint_step_4000.safetensors"
      "training.output_path=${OPENWAM_OUTPUT:-$root/outputs/sidecar/xa_warmup2k_$(date -u +%Y%m%dT%H%M%SZ)}")
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  exec "$python_bin" scripts/train.py "${args[@]}" "$@" --cfg job --resolve
fi
exec "$python_bin" -m torch.distributed.run --standalone --local-addr=127.0.0.1 \
  --nnodes=1 --nproc_per_node=8 \
  scripts/cluster/train_sidecar_metrics.py "${args[@]}" "$@"
