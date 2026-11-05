#!/usr/bin/env bash
#SBATCH --job-name=openwam-grounded-2k-10ep
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
export OPENWAM_CLUSTER_BASE="${OPENWAM_CLUSTER_BASE:-$(dirname "$(dirname "$project_root")")}"
export OPENWAM_VENV="${OPENWAM_VENV:-$project_root/.venv}"
export OPENWAM_OUTPUT="${OPENWAM_OUTPUT:-$project_root/outputs/sidecar/grounded_read_job${SLURM_JOB_ID}}"
cd "${OPENWAM_CODE:-$project_root}"
echo "Job=$SLURM_JOB_ID partition=mhan0035 node=$(hostname) project=$project_root"
exec srun --ntasks=1 --kill-on-bad-exit=1 bash scripts/cluster/train_sidecar_grounded_two_stage.sh
