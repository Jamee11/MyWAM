#!/usr/bin/env bash
# Single-node only. Shares the four-GPU recipe and defaults to config inspection.
set -euo pipefail
: "${EA_NUM_GPUS:?Set EA_NUM_GPUS explicitly, for example 4 or 8}"
exec bash "$(dirname "$0")/train_ea_lite_4gpu.sh" "$@"
