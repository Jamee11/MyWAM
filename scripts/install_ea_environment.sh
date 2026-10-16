#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${PYTHON_BIN:-python3.10}"
if [[ ! -x .venv/bin/python ]]; then
  "$python_bin" -m venv .venv
fi
installer="${UV_BIN:-uv}"
torch_requirements=(torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128)
torch_index=(--default-index https://download.pytorch.org/whl/cu128)
wheel_links=()
if [[ -n "${OPENWAM_TORCH_WHEEL_DIR:-}" ]]; then
  wheel_dir="$(cd -- "$OPENWAM_TORCH_WHEEL_DIR" && pwd)"
  torch_requirements=()
  for spec in torch-2.7.1 torchvision-0.22.1 torchaudio-2.7.1; do
    wheel="$wheel_dir/${spec}+cu128-cp310-cp310-manylinux_2_28_x86_64.whl"
    [[ -f "$wheel" ]] || { echo "Missing locked wheel: $wheel" >&2; exit 2; }
    torch_requirements+=("$wheel")
  done
  # Only the Torch trio is local. Other dependencies use the reachable PyPI mirror.
  torch_index=(--default-index "${UV_DEFAULT_INDEX:-${UV_INDEX_URL:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}}")
  wheel_links=(--find-links "$wheel_dir")
fi
if [[ -n "${OPENWAM_CUDA_WHEEL_DIR:-}" ]]; then
  cuda_wheel_dir="$(cd -- "$OPENWAM_CUDA_WHEEL_DIR" && pwd)"
  runtime_requirements="$cuda_wheel_dir/../requirements-cuda-runtime.txt"
  [[ -f "$runtime_requirements" ]] || { echo "Missing runtime requirements: $runtime_requirements" >&2; exit 2; }
  # Install the complete CUDA runtime set offline before the project dependency pass.
  "$installer" pip install --python .venv/bin/python --no-index \
    --find-links "$cuda_wheel_dir" -r "$runtime_requirements"
  wheel_links+=(--find-links "$cuda_wheel_dir")
fi
"$installer" pip install --python .venv/bin/python \
  "${torch_requirements[@]}" "${torch_index[@]}" "${wheel_links[@]}"
# Fail before installing the remaining lock if the requested CUDA builds differ.
.venv/bin/python - <<'PY'
from importlib.metadata import version
for package, expected in (("torch", "2.7.1+cu128"), ("torchvision", "0.22.1+cu128"),
                          ("torchaudio", "2.7.1+cu128")):
    actual = version(package)
    if actual != expected:
        raise SystemExit(f"Wrong {package} build: {actual}; expected {expected}")
PY
mkdir -p outputs/setup
cat > outputs/setup/torch_constraints.txt <<'EOF'
torch==2.7.1+cu128
torchvision==0.22.1+cu128
torchaudio==2.7.1+cu128
EOF
# Freeze snapshots contain this checkout's absolute editable path; use the new checkout.
sed '/^-e file:/d' requirements-ea-lock.txt > outputs/setup/requirements-ea-relocatable.txt
DS_BUILD_OPS=0 UV_LINK_MODE=copy "$installer" pip install --python .venv/bin/python \
  -r outputs/setup/requirements-ea-relocatable.txt -e . --constraint outputs/setup/torch_constraints.txt \
  "${wheel_links[@]}"
.venv/bin/python -m pip check
