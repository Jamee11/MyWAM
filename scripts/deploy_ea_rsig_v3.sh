#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH="$PWD/.venv/bin:$PATH"
checkpoint="${1:?Pass an EA checkpoint directory}"
shift
exec bash scripts/deploy.sh "$checkpoint" --compile-enabled false "$@" optimization.dit_cache.enabled=false inference.cfg_scale=1
