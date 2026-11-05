#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
cluster_base="${OPENWAM_CLUSTER_BASE:-$(dirname "$(dirname "$project_root")")}"
dataset="${OPENWAM_DATASET:-$cluster_base/datasets/robotwin2_official_clean_ea_rsig_v3_4350_wrist020_schema5_20261015}"
foundation="${OPENWAM_FOUNDATION:-$cluster_base/ckpts/OpenWAM-Alpha-Pretrain-Foundation-Model}"
python_bin="${PYTHON_BIN:-${OPENWAM_VENV:-$project_root/.venv}/bin/python}"
output="${OPENWAM_OUTPUT:-$project_root/outputs/sidecar/grounded_read_job${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}}"
export EA_SIDECAR_MANIFEST="${OPENWAM_MANIFEST:-$dataset/openwam_sidecar/aloha50/manifest.ready.json}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" NCCL_DEBUG=WARN HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$project_root/.cache/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$project_root/.cache/triton}"
unset PYTHONHOME PYTHONPATH EA_STOP_AFTER_OPT_STEPS
configs=(train_sidecar_grounded_warmup2k_4gpu train_sidecar_grounded_joint10epoch_4gpu)
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  for phase in 0 1; do
    source="$foundation"
    [[ $phase == 0 ]] || source="$output/warmup/runs/AFTER_WARMUP/checkpoint_step_2000.safetensors"
    "$python_bin" scripts/train.py "--config-name=${configs[$phase]}" \
      "training.finetune_ckpt_path=$source" "training.output_path=$output/phase$phase/runs" --cfg job --resolve
  done
  exit 0
fi
[[ "${CHECK_ONLY:-0}" == 0 ]] || { echo 'CHECK_ONLY must be 0 or 1' >&2; exit 2; }
[[ -x "$python_bin" && -f "$foundation/config.yaml" && -f "$EA_SIDECAR_MANIFEST" ]] || {
  echo 'Check Python, the Foundation checkpoint directory and relocated EA manifest.' >&2; exit 2;
}
[[ ! -e "$output/warmup" && ! -e "$output/joint" ]] || {
  echo 'Use a new OPENWAM_OUTPUT; do not mix this fresh two-stage run with an existing run.' >&2; exit 2;
}
mkdir -p "$output/warmup" "$output/joint"
run_stage() {
  local name="$1" config="$2" source="$3"
  echo "Starting $name from $source; batch32 x acc2 x four Slurm GPUs = 256"
  "$python_bin" -m torch.distributed.run --standalone --local-addr=127.0.0.1 \
    --nnodes=1 --nproc_per_node=4 scripts/cluster/train_sidecar_metrics.py \
    "--config-name=$config" "training.finetune_ckpt_path=$source" \
    "training.output_path=$output/$name/runs" 2>&1 | tee "$output/$name/launch.log"
}
run_stage warmup "${configs[0]}" "$foundation"
warmup_checkpoint="$("$python_bin" scripts/cluster/sidecar_stage_checkpoint.py "$output/warmup" 2000)"
echo "Warmup saved: $warmup_checkpoint; reloading with fresh optimizer, scheduler and counters"
run_stage joint "${configs[1]}" "$warmup_checkpoint"
"$python_bin" scripts/cluster/sidecar_stage_checkpoint.py "$output/joint" 21480 | tee "$output/final_checkpoint.txt"
