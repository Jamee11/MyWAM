#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OPENWAM_OUTPUT="${OPENWAM_OUTPUT:-$PWD/outputs/sidecar/allstage_warmup2k_$(date -u +%Y%m%dT%H%M%SZ)}"
# Reuse the working standalone launcher and runtime paths; the config keeps XA disabled.
exec bash scripts/train_sidecar_xa_warmup2k_8gpu.sh \
  --config-name=train_sidecar_allstage_warmup2k_8gpu_10epochs "$@"
