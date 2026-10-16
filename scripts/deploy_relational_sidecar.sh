#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${OPENWAM_VENV:-$PWD/.venv}/bin:$PATH"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
checkpoint="${1:?Pass a NEW Sidecar checkpoint directory}"
shift
exec bash scripts/deploy.sh "$checkpoint" --config configs/deploy_relational_sidecar.yaml "$@"
