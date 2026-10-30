#!/usr/bin/env bash
# Run once on a host allowed to install packages, not inside every batch job.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
python_bin="${PYTHON_BIN:-python3.10}"
installer="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "$installer" ]]; then
  echo 'uv is required. Install the small bootstrap tool in your active environment:' >&2
  echo '  python -m pip install uv' >&2
  echo 'Then rerun this script. Training dependencies will still go into project/.venv.' >&2
  exit 2
fi
if [[ ! -x .venv/bin/python || ! -f .venv/bin/activate ]]; then
  command -v "$python_bin" >/dev/null || { echo "Python interpreter not found: $python_bin" >&2; exit 2; }
  if [[ -e .venv || -L .venv ]]; then
    backup=".venv.incomplete.$(date +%Y%m%dT%H%M%S).$$"
    mv .venv "$backup"
    echo "Preserved incomplete environment at $project_root/$backup"
  fi
  # uv creates and seeds the environment without the system ensurepip module.
  "$installer" venv --python "$python_bin" --no-python-downloads --seed .venv
elif ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  "$installer" pip install --python .venv/bin/python pip
fi
export PYTHON_BIN="$python_bin" UV_BIN="$installer"
exec bash scripts/install_ea_environment.sh
