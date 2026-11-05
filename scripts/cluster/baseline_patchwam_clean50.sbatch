#!/usr/bin/env bash
#SBATCH --job-name=openwam-patchwam-baseline10ep
#SBATCH --partition=mhan0035
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
project_root="${OPENWAM_PROJECT:-${SLURM_SUBMIT_DIR:?Submit from the OpenWAM project root}}"
cd "$project_root"
if [[ -n "${OPENWAM_CLUSTER_SETUP:-}" ]]; then
  source "$OPENWAM_CLUSTER_SETUP"
fi
cluster_base="${OPENWAM_CLUSTER_BASE:-$(dirname "$(dirname "$project_root")")}"
foundation="${OPENWAM_FOUNDATION:-$cluster_base/ckpts/OpenWAM-Alpha-Pretrain-Foundation-Model}"
dataset="${OPENWAM_DATASET:-$cluster_base/datasets/robotwin2_official_clean_ea_rsig_v3_4350_wrist020_schema5_20261015}"
python_bin="${OPENWAM_VENV:-$project_root/.venv}/bin/python"
output="${OPENWAM_OUTPUT:-$project_root/outputs/baseline_patchwam_job${SLURM_JOB_ID}}"
export EA_SIDECAR_MANIFEST="${OPENWAM_MANIFEST:-$dataset/openwam_sidecar/aloha50/manifest.ready.json}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" NCCL_DEBUG=WARN HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset PYTHONHOME PYTHONPATH EA_STOP_AFTER_OPT_STEPS
cd "${OPENWAM_CODE:-$project_root}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/.cache/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/.cache/triton}"
echo "Job=$SLURM_JOB_ID node=$(hostname) code=$PWD output=$output; original WAM, no Sidecar, global batch256"
mkdir -p "$output"
exec srun --ntasks=1 --kill-on-bad-exit=1 "$python_bin" -m torch.distributed.run \
  --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc_per_node=4 \
  scripts/cluster/train_sidecar_metrics.py --config-name=train_alpha_patchwam_baseline_10epoch_4gpu \
  "training.finetune_ckpt_path=$foundation" "training.output_path=$output/runs"
