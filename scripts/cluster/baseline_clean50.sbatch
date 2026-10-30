#!/usr/bin/env bash
#SBATCH --job-name=openwam-alpha-baseline
#SBATCH --partition=mhan0035
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# No --time: use the partition default. Override resources/account with sbatch arguments.
set -euo pipefail
project_root="${OPENWAM_PROJECT:-/home/mingfeihangroup/xiaoming/projects/OpenWAM}"
cd "$project_root"
# If the cluster requires modules, put module loads in OPENWAM_CLUSTER_SETUP.
if [[ -n "${OPENWAM_CLUSTER_SETUP:-}" ]]; then
  source "$OPENWAM_CLUSTER_SETUP"
fi
export NPROC_PER_NODE=4
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
echo "Job=$SLURM_JOB_ID node=$(hostname) project=$project_root"
nvidia-smi --query-gpu=name,memory.total --format=csv
# One Slurm task launches four torchrun workers; do not use four srun tasks.
exec srun --ntasks=1 --kill-on-bad-exit=1 bash scripts/cluster/train_baseline.sh "$@"
