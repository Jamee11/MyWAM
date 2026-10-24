"""Exercise queue control flow without a model, GPU or simulator."""
import json
import sys
import threading

import pytest

from benchmarks.robotwin import eval_remaining24 as queue


@pytest.mark.parametrize("explicit", [False, True])
def test_queue_runs_all_cases_and_keeps_failures_out_of_denominator(tmp_path, monkeypatch, capsys, explicit):
    plan = queue.load_plan(queue.PLAN)
    if explicit:
        plan = {**plan, "expected_conditions": 3, "conditions": plan["conditions"][:3]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    model_root = tmp_path / "models"
    for model in [5, 6]:
        directory = model_root / f"model{model}"
        directory.mkdir(parents=True)
        for slot in range(3):
            (directory / f"session_slot{slot}.json").write_text(json.dumps(
                dict(slot=slot, checkpoint_sha256=plan["checkpoint_sha256"])))
    fake = tmp_path / "fake_python"
    fake.write_text(f"#!{sys.executable}\n" + '''import json,sys
from pathlib import Path
def arg(k): return sys.argv[sys.argv.index(k)+1]
p=Path(arg('--output')); p.mkdir()
if arg('--task')=='open_microwave': raise SystemExit(3)
(p/'run.json').write_text(json.dumps(dict(status='complete', result=dict(
    complete=True, completed_episodes=100, successes=50, success_rate=.5), wall_seconds=.01)))
''')
    fake.chmod(0o755)
    output = tmp_path / "result"
    argv = ["queue", "--model-run", str(model_root), "--output", str(output),
            "--plan", str(plan_path), "--sim-python", str(fake)]
    if explicit:
        argv += ["--sessions", str(model_root / "model6/session_slot0.json"),
                 "--sim-gpus", "4", "--ports", "22730", "--wait-for-capacity"]
    gpu_queries = []

    def gpu_output(command, **_):
        if "-i" in command:
            gpu_queries.append(command)
            return "1000" if len(gpu_queries) == 1 else "24000"
        return "\n".join(f"{i}, NVIDIA GeForce RTX 4090, 24000" for i in range(8))

    original_event = threading.Event

    class FastEvent(original_event):
        def wait(self, timeout=None):
            return super().wait(.001 if timeout == 30 else timeout)

    monkeypatch.setattr(queue.threading, "Event", FastEvent)
    monkeypatch.setattr(queue.subprocess, "check_output", gpu_output)
    monkeypatch.setattr(queue, "probe", lambda *_: {})
    monkeypatch.setattr(queue.signal, "signal", lambda *_: None)
    monkeypatch.setattr(sys, "argv", argv)
    assert queue.main() == 1
    summary = json.loads((output / "summary.json").read_text())
    expected = len(plan["conditions"])
    failures = sum(r["task"] == "open_microwave" for r in plan["conditions"])
    assert summary["planned_conditions"] == len(summary["results"]) == expected
    assert len({(r["task"], r["mode"]) for r in summary["results"]}) == expected
    assert summary["episodes"] == (expected - failures) * 100
    assert summary["complete_conditions"] == expected - failures
    if explicit:
        assert len(gpu_queries) >= 4 and "WAIT_CAPACITY" in capsys.readouterr().out
        assert all(r["port"] == 22730 for r in summary["results"])
    with pytest.raises(FileExistsError):
        queue.main()
