#!/usr/bin/env bash
# Dedicated Piper queue; defaults and files used by Aloha are not changed.
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT/.venv/bin/python}"
SIM_PY="${SIM_PY:-$PROJECT/.conda/robotwin-official/bin/python}"
ROBOTWIN_PATH="${ROBOTWIN_PATH:-$PROJECT/.external/robotwin-official-20261018/RoboTwin}"
# Same working 4090/L20 simulation environment as the existing Aloha native runs.
export ROBOTWIN_PATH CUDA_HOME="${CUDA_HOME:-/remote-home/jinminghao/miniconda3/envs/nvcu121_cuda-12.4}"
export PATH="$(dirname "$SIM_PY"):$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$ROBOTWIN_PATH:$PROJECT/benchmarks/robotwin:$ROBOTWIN_PATH/envs/curobo/src:$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-$PROJECT/.conda/robotwin-official/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PROJECT/.cache/robotwin/4090_trend30_extensions}"
export PYTHONNOUSERSITE=1 HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SAPIEN_DISABLE_RAY_TRACING=0 ROBOTWIN_ENABLE_PLANNER_FALLBACK=0
OUTPUT="${OUTPUT:-$PROJECT/outputs/robotwin_eval/piper_$(date -u +%Y%m%dT%H%M%SZ)}"
args=(--output "$OUTPUT" --sim-python "$SIM_PY" --sim-gpus "${SIM_GPUS:-0}" --ports "${PORTS:-43400}" --host "${MODEL_HOST:-127.0.0.1}" --episodes "${EPISODES:-100}" --tasks "${TASKS:-all}")
if [[ -n "${CHECKPOINT:-}" ]]; then args+=(--checkpoint "$CHECKPOINT"); fi
if [[ -n "${PIPER_MANIFEST:-}" ]]; then args+=(--manifest "$PIPER_MANIFEST"); fi
args+=(--robotwin "$ROBOTWIN_PATH")
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then args+=(--prepare-only); fi
exec "$PYTHON_BIN" -m benchmarks.robotwin.piper_eval "${args[@]}" "$@"
