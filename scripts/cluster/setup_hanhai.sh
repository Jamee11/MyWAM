#!/usr/bin/env bash
# Source through OPENWAM_CLUSTER_SETUP from baseline_clean50.sbatch.
# Hanhai's /etc/profile module initialization does not support nounset.
set +u
source /etc/profile
module load cuda/12.8
set -u
export PYTHONNOUSERSITE=1
# train_baseline.sh selects project/.venv; do not activate the old cosmos env.
