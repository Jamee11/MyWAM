#!/usr/bin/env bash
#SBATCH --job-name=openwam-multisim
#SBATCH --partition=GPU-8A100
#SBATCH --qos=gpu_8a100
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
cd "${OPENWAM_PROJECT:-${SLURM_SUBMIT_DIR:?Submit from the OpenWAM project root}}"
if [[ -n "${OPENWAM_CLUSTER_SETUP:-}" ]]; then
  source "$OPENWAM_CLUSTER_SETUP"
fi
export PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
export MIDTRAIN_OUTPUT_DIR="${MIDTRAIN_OUTPUT_DIR:-$PWD/outputs/midtrain/job${SLURM_JOB_ID}}"
export CHECK_ONLY="${CHECK_ONLY:-0}"
exec srun --ntasks=1 --kill-on-bad-exit=1 bash scripts/train_multisim_sidecar_4a100.sh "$@"
