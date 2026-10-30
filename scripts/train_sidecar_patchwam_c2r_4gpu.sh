#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${CUDA_VISIBLE_DEVICES:?Set the four GPUs allocated to this separate experiment}"
export OPENWAM_TRAIN_CONFIG=train_relational_sidecar_aloha50_patchwam_c2r_10epochs
export OPENWAM_MANIFEST="${OPENWAM_MANIFEST:-$PWD/outputs/sidecar/aloha_4350/manifest.ready.json}"
export OPENWAM_FOUNDATION="${OPENWAM_FOUNDATION:-$PWD/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model}"
export OPENWAM_OUTPUT="${OPENWAM_OUTPUT:-$PWD/outputs/sidecar/aloha50_patchwam_c2r_$(date -u +%Y%m%dT%H%M%SZ)}"
export NPROC_PER_NODE=4 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
unset PYTHONHOME PYTHONPATH EA_STOP_AFTER_OPT_STEPS
exec bash scripts/cluster/train_sidecar_isolated.sh "$@"
