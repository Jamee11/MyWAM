#!/usr/bin/env bash
#SBATCH --job-name=openwam-sidecar-frozen2k
#SBATCH --partition=GPU-8A100
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
project_root="${OPENWAM_PROJECT:-/home/mingfeihangroup/xiaoming/projects/OpenWAM}"
cd "$project_root"
if [[ -n "${OPENWAM_CLUSTER_SETUP:-}" ]]; then
  source "$OPENWAM_CLUSTER_SETUP"
fi
export OPENWAM_TRAIN_CONFIG=train_relational_sidecar_aloha50_frozen_backbone_10epochs
export OPENWAM_OUTPUT="${OPENWAM_OUTPUT:-$project_root/outputs/sidecar/aloha50_frozen_backbone2k_job${SLURM_JOB_ID}}"
export NPROC_PER_NODE=4
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
echo "Job=$SLURM_JOB_ID partition=GPU-8A100 node=$(hostname) project=$project_root"
nvidia-smi --query-gpu=name,memory.total --format=csv
exec srun --ntasks=1 --kill-on-bad-exit=1 bash scripts/cluster/train_sidecar_isolated.sh "$@"
