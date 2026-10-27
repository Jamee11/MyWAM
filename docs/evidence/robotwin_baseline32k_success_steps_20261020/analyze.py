"""Read-only comparison of complete baseline retests and Sidecar32k conditions."""
import csv
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
EVAL = ROOT / "outputs/robotwin_eval"
BASE = EVAL / "baseline_official_manifest_deadline_20261018"
NATIVE = EVAL / "rebuild_official_20261018/official_diagnostic100"
SIDE = EVAL / "sidecar32000_officialenv_l20_20261019"
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
EVENT = re.compile(r"step:\s*(\d+)\s*/\s*(\d+)|Success rate:\s*(\d+)/(\d+)[^\n]*?current seed:\s*(\d+)|\[RESULT\] episode=(\d+) seed=(\d+) success=(True|False)")


def parse(path, kind):
    text = ANSI.sub("", path.read_text(errors="replace"))
    rows = []
    last_step = None
    budget = None
    previous_successes = 0
    for m in EVENT.finditer(text):
        if m[1]:
            step, budget = int(m[1]), int(m[2])
            assert step == (1 if last_step is None else last_step + 1), (path, step, last_step)
            last_step = step
        elif m[3] and kind == "native":
            successes, completed, seed = map(int, (m[3], m[4], m[5]))
            assert completed == len(rows) + 1
            assert successes - previous_successes in (0, 1)
            rows.append(dict(episode=completed - 1, seed=seed, success=successes > previous_successes, steps=last_step, budget=budget, source=str(path)))
            previous_successes = successes
            last_step = None
        elif m[6] and kind == "manifest":
            rows.append(dict(episode=int(m[6]), seed=int(m[7]), success=m[8] == "True", steps=last_step, budget=budget, source=str(path)))
            last_step = None
    assert rows and all(r["steps"] is not None and 0 < r["steps"] <= r["budget"] for r in rows), path
    return rows


def summarize(rows):
    success = [r["steps"] for r in rows if r["success"]]
    failed = [r["steps"] for r in rows if not r["success"]]
    return dict(episodes=len(rows), successes=len(success), success_rate=100 * len(success) / len(rows),
                success_mean_steps=statistics.mean(success), success_median_steps=statistics.median(success),
                all_mean_steps=statistics.mean(r["steps"] for r in rows),
                failure_mean_steps=statistics.mean(failed) if failed else None,
                failure_at_budget=sum(r["steps"] == r["budget"] for r in rows if not r["success"]))


baseline_run = json.loads((BASE / "run.json").read_text())
assert baseline_run["status"] == "complete" and not baseline_run["errors"]
side_state = json.loads((SIDE / "global_remaining_20261020/parallel.json").read_text())
side_jobs = {(j["task"], j["mode"]): j for j in side_state["jobs"]}
comparisons = list(csv.DictReader((EVAL / "baseline_official_comparison_20261019/per_condition_comparison.csv").open()))
report = []
raw_rows = {}
for condition in comparisons:
    if condition["retested"] != "True":
        continue
    task, mode = condition["task"], condition["mode"]
    job = side_jobs[task, mode]
    if job["status"] != "complete":
        continue
    assert job["result"]["complete"] and job["result"]["completed_episodes"] == 100 and not job["result"]["errors"]
    if condition["protocol"] == "traditional_native":
        baseline_rows = parse(NATIVE / mode / task / "client.log", "native")
    else:
        baseline_rows = []
        for shard in sorted((BASE / "shards" / mode / task).iterdir()):
            result = json.loads((shard / "result.json").read_text())
            actual = parse(shard / "client.log", "manifest")
            assert [(r["episode"], r["seed"], r["success"]) for r in actual] == [(r["episode"], r["seed"], r["success"]) for r in result["episodes"]]
            baseline_rows.extend(actual)
    assert sorted(r["episode"] for r in baseline_rows) == list(range(100))
    assert len({r["seed"] for r in baseline_rows}) == 100
    side_path = Path(job["output"]) / mode / task / "client.log"
    side_rows = parse(side_path, "native")
    assert len(side_rows) == 100 and sum(r["success"] for r in side_rows) == job["result"]["successes"]
    assert [r["seed"] for r in side_rows] == job["result"]["accepted_seeds"]
    b, s = summarize(baseline_rows), summarize(side_rows)
    assert b["successes"] == int(condition["new_retest"])
    b_seeds, s_seeds = ({r["seed"]: r for r in rows} for rows in (baseline_rows, side_rows))
    shared = sorted(b_seeds.keys() & s_seeds.keys())
    both_success = [seed for seed in shared if b_seeds[seed]["success"] and s_seeds[seed]["success"]]
    paired = dict(shared_accepted_seeds=len(shared), both_success=len(both_success))
    if both_success:
        paired.update(baseline_mean_steps=statistics.mean(b_seeds[k]["steps"] for k in both_success),
                      sidecar_mean_steps=statistics.mean(s_seeds[k]["steps"] for k in both_success),
                      mean_difference_steps=statistics.mean(s_seeds[k]["steps"] - b_seeds[k]["steps"] for k in both_success))
    row = dict(task=task, mode=mode, old_environment_baseline_success_rate=int(condition["old"]),
               baseline_protocol=condition["protocol"], sidecar_protocol="traditional_native", baseline=b, sidecar32k=s,
               success_difference_pp=s["success_rate"] - b["success_rate"],
               success_step_difference=s["success_mean_steps"] - b["success_mean_steps"],
               success_step_difference_pct=100 * (s["success_mean_steps"] / b["success_mean_steps"] - 1),
               shared_seed_both_success_descriptive_only=paired)
    report.append(row)
    raw_rows[f"{task}/{mode}"] = dict(baseline=baseline_rows, sidecar32k=side_rows)
    print(json.dumps(row), flush=True)

summary = {}
for mode in ("demo_clean", "demo_randomized", "all"):
    selected = [r for r in report if mode == "all" or r["mode"] == mode]
    summary[mode] = dict(conditions=len(selected), baseline_macro_success_rate=statistics.mean(r["baseline"]["success_rate"] for r in selected),
                        sidecar_macro_success_rate=statistics.mean(r["sidecar32k"]["success_rate"] for r in selected),
                        baseline_macro_success_mean_steps=statistics.mean(r["baseline"]["success_mean_steps"] for r in selected),
                        sidecar_macro_success_mean_steps=statistics.mean(r["sidecar32k"]["success_mean_steps"] for r in selected))
result = dict(utc=datetime.now(timezone.utc).isoformat(), comparisons=report, summary=summary,
              limitations="Success-only populations differ. Baseline 9 conditions use sealed manifest shards; Sidecar uses native seed scanning. Shared seed is not a strict matched scene/instruction/RNG experiment. Steps are native take_action counts, not physical dt, denoising steps or wall-clock time.")
(OUT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
(OUT / "episodes.json").write_text(json.dumps(raw_rows, indent=2) + "\n")
print(json.dumps(summary, indent=2))
