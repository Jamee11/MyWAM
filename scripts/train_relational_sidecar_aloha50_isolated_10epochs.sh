#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export EA_TRAIN_CONFIG=train_relational_sidecar_aloha50_isolated_10epochs
export EA_OUTPUT_DIR="${EA_OUTPUT_DIR:-outputs/sidecar/aloha50_isolated2k_10epochs}"
exec bash scripts/train_relational_sidecar_aloha50_10epochs.sh "$@"
