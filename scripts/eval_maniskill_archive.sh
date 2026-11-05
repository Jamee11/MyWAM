#!/usr/bin/env bash
# Portable entry: regenerate local paths, then run the unchanged 8-task held-out protocol.
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
MODEL_PY="${MODEL_PY:-$PROJECT/.venv/bin/python}"
if [[ -z "${SIM_PY:-}" ]]; then
  SIM_PY="$PROJECT/.venv-maniskill/bin/python"
  if [[ ! -x "$SIM_PY" ]]; then
    SIM_PY="/remote-home/jinminghao/miniconda3/envs/maniskill-eort-v1/bin/python"
  fi
fi
DATA_ROOT="${DATA_ROOT:-$PROJECT/datasets/multisim_sidecar_archive_v2_20261025}"
: "${CHECKPOINT:?Export CHECKPOINT pointing to the ManiSkill safetensors file}"
MODEL_GPU="${MODEL_GPU:-0}"
SIM_GPU="${SIM_GPU:-1}"
if [[ "$MODEL_GPU" == "$SIM_GPU" ]]; then
  WORKERS="${WORKERS:-1}"
else
  WORKERS="${WORKERS:-4}"
fi
EPISODES_PER_TASK="${EPISODES_PER_TASK:-100}"
PORT="${PORT:-22400}"
CHECK_ONLY="${CHECK_ONLY:-0}"
REVALIDATE="${REVALIDATE:-0}"
VALIDATED_RUN="${VALIDATED_RUN:-}"
if [[ -z "$VALIDATED_RUN" && -f "$PROJECT/outputs/maniskill_eval/validated_run.txt" ]]; then
  VALIDATED_RUN="$(cat "$PROJECT/outputs/maniskill_eval/validated_run.txt")"
fi
[[ "$CHECK_ONLY" == 0 || "$CHECK_ONLY" == 1 ]] || { echo 'CHECK_ONLY must be 0 or 1' >&2; exit 2; }
for value in "$MODEL_GPU" "$SIM_GPU" "$WORKERS" "$EPISODES_PER_TASK" "$PORT"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo 'GPU indices and budgets must be nonnegative integers' >&2; exit 2; }
done
(( WORKERS > 0 && EPISODES_PER_TASK > 0 && PORT > 0 && PORT+WORKERS-1 <= 65535 ))
[[ -x "$MODEL_PY" && -x "$SIM_PY" ]] || { echo 'Set MODEL_PY and SIM_PY to existing Python executables' >&2; exit 2; }
[[ -f "$CHECKPOINT" && -f "$DATA_ROOT/manifest.ready.json" ]]
CKPT_DIR="$(dirname "$CHECKPOINT")"
for name in config.yaml normalization_stats.npy; do
  [[ -f "$CKPT_DIR/$name" ]] || { echo "Missing checkpoint asset: $CKPT_DIR/$name" >&2; exit 2; }
done
[[ -d "$CKPT_DIR/tokenizer" && -d "$DATA_ROOT/replay/ManiSkill" && -d "$DATA_ROOT/replay/DiT4DiT/examples/ManiSkill3" ]]
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 TORCH_WARM_POOL=0
unset PYTHONHOME PYTHONPATH
EVAL_ROOT="${EVAL_ROOT:-$PROJECT/outputs/maniskill_eval/archive_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ ! -e "$EVAL_ROOT" ]] || { echo "Use a new EVAL_ROOT; refusing to overwrite: $EVAL_ROOT" >&2; exit 2; }
mkdir -p "$EVAL_ROOT"
plan_args=(--manifest "$DATA_ROOT/manifest.ready.json" --count "$EPISODES_PER_TASK" --output "$EVAL_ROOT/plan.json")
if [[ "$REVALIDATE" != 1 && -n "$VALIDATED_RUN" ]]; then
  plan_args+=(--reuse-plan "$VALIDATED_RUN/plan.json")
fi
"$MODEL_PY" -m benchmarks.maniskill.archive_protocol "${plan_args[@]}" > "$EVAL_ROOT/plan_summary.log"
args=("$MODEL_PY" -u benchmarks/maniskill/run_archive_eval.py
      --plan "$EVAL_ROOT/plan.json" --checkpoint "$CHECKPOINT" --output "$EVAL_ROOT/run"
      --model-python "$MODEL_PY" --sim-python "$SIM_PY"
      --model-gpu "$MODEL_GPU" --sim-gpu "$SIM_GPU" --workers "$WORKERS" --port "$PORT")
if [[ "$REVALIDATE" != 1 && -n "$VALIDATED_RUN" ]]; then
  args+=(--validated-run "$VALIDATED_RUN")
fi
echo "Evaluation root: $EVAL_ROOT"
echo "Episodes per task/condition: $EPISODES_PER_TASK; model GPU=$MODEL_GPU; simulator GPU=$SIM_GPU; workers=$WORKERS"
if [[ "$CHECK_ONLY" == 1 ]]; then
  printf 'Command: '; printf '%q ' "${args[@]}"; printf '\n'
  echo 'Path/plan preparation complete; no GPU evaluation started.'
  exit 0
fi
exec "${args[@]}" > "$EVAL_ROOT/run.log" 2>&1
