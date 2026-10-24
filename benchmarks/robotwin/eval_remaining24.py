"""Six remote policy slots consuming 24 untouched Sidecar42960 conditions."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.robotwin.remote_eval import digest, probe, save

PLAN = ROOT / "benchmarks/robotwin/plans/sidecar42960_remaining24_20261022.json"


def load_plan(path):
    plan = json.loads(path.read_text())
    old = ROOT / plan["previous_final_record"]
    if digest(old) != plan["previous_final_record_sha256"]:
        raise ValueError("Previous sealed result changed; re-audit selection")
    record = json.loads(old.read_text())
    completed = {(r["task"], r["mode"]) for r in record["completed"]}
    partial = {(r["task"], r["mode"]) for r in record["partial"]}
    keys = [(r["task"], r["mode"]) for r in plan["conditions"]]
    expected = plan.get("expected_conditions", 24)
    if expected < 1 or len(keys) != expected or len(set(keys)) != expected or set(keys) & (completed | partial):
        raise ValueError("Expected distinct previously unstarted conditions matching the plan count")
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-run", type=Path, required=True, help="contains model5/model6 session files")
    p.add_argument("--output", type=Path, required=True, help="new batch directory")
    p.add_argument("--sim-gpus", default="0,1,2,4,5,6")
    p.add_argument("--ports", default="22700,22701,22702,22710,22711,22712")
    p.add_argument("--sessions", type=Path, nargs="+", help="Explicit sessions for a different number of independent slots")
    p.add_argument("--wait-for-capacity", action="store_true", help="Wait for free memory before every condition instead of abandoning the queue")
    p.add_argument("--sim-python", type=Path, default=ROOT / ".conda/robotwin-official/bin/python")
    p.add_argument("--robotwin-path", type=Path, default=ROOT / ".external/robotwin-official-20261018/RoboTwin")
    p.add_argument("--plan", type=Path, default=PLAN)
    p.add_argument("--episodes", type=int, choices=[1, 100], default=100)
    p.add_argument("--min-free-mib", type=int, default=8000)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    plan = load_plan(args.plan)
    gpus, ports = args.sim_gpus.split(","), [int(s) for s in args.ports.split(",")]
    count = len(args.sessions) if args.sessions else 6
    if len(gpus) != count or len(ports) != count or len(set(ports)) != count or any(not 1 <= n <= 65535 for n in ports):
        p.error("GPU, port and session counts must match, with distinct valid ports")
    run, out = args.model_run.absolute(), args.output.absolute()
    workers = [dict(index=i, gpu=gpus[i], port=ports[i],
        session=str(args.sessions[i].absolute() if args.sessions else
            run / f"model{5 if i < 3 else 6}" / f"session_slot{i % 3}.json")) for i in range(count)]
    if args.dry_run:
        print(json.dumps(dict(conditions=plan["conditions"], workers=workers,
            episodes=args.episodes, total_episodes=len(plan["conditions"]) * args.episodes,
            old_completed_overlap=0, old_partial_overlap=0, video=False), indent=2))
        return 0
    # Run this dispatcher on the simulator machine, not on the model GPU host.
    gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.free",
        "--format=csv,noheader,nounits"], text=True)
    free = {}
    for row in gpu_rows.splitlines():
        gpu, name, mib = [s.strip() for s in row.split(",")]
        free[gpu] = (name, int(mib))
    for gpu in set(gpus):
        if gpu not in free or "4090" not in free[gpu][0]:
            raise RuntimeError(f"Expected a 4090 simulator GPU: {gpu}")
        if not args.wait_for_capacity and free[gpu][1] < args.min_free_mib * gpus.count(gpu):
            raise RuntimeError(f"GPU {gpu}: insufficient free memory for assigned simulators")
    sessions = []
    for w in workers:
        session = json.loads(Path(w["session"]).read_text())
        if session["checkpoint_sha256"] != plan["checkpoint_sha256"]:
            raise RuntimeError("Session is not the selected Sidecar42960 checkpoint")
        expected_slot = w["index"] % 3
        if not args.sessions and session["slot"] != expected_slot:
            raise RuntimeError("Session file does not match its assigned slot")
        sessions.append({**w, "health": probe(session, w["port"])})
    # Save an immutable copy; old runs/results are never modified or appended.
    out.mkdir(parents=True, exist_ok=False)
    save(out / "selection.json", plan)
    state = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
        episodes_per_condition=args.episodes, scope="standard100" if args.episodes == 100 else "smoke_only",
        gpu_snapshot=gpu_rows, workers=sessions, results=[], active={}, errors=[],
        selection_sha256=digest(args.plan), source_sha256=digest(Path(__file__)))
    pending = queue.Queue()
    for case in plan["conditions"]:
        pending.put(case)
    lock, stopping = threading.RLock(), threading.Event()
    active = {}

    def persist():
        save(out / "state.json", state)

    def terminate(*_):
        stopping.set()
        with lock:
            for child in list(active.values()):
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    def worker(w):
        while not stopping.is_set():
            if pending.empty():
                return
            if args.wait_for_capacity:
                while not stopping.is_set():
                    mib = int(subprocess.check_output(["nvidia-smi", "-i", w["gpu"],
                        "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
                    with lock:
                        waiting = state.setdefault("waiting_capacity", {})
                        if mib >= args.min_free_mib * gpus.count(w["gpu"]):
                            waiting.pop(str(w["index"]), None)
                            persist()
                            break
                        waiting[str(w["index"])] = dict(gpu=w["gpu"], free_mib=mib)
                        persist()
                    print(f"WAIT_CAPACITY slot={w['index']} gpu={w['gpu']} free_mib={mib}", flush=True)
                    stopping.wait(30)
                if stopping.is_set():
                    return
            try:
                case = pending.get_nowait()
            except queue.Empty:
                return
            task, mode = case["task"], case["mode"]
            directory = out / mode / task
            directory.parent.mkdir(parents=True, exist_ok=True)
            command = [str(args.sim_python.absolute()), "-u", str(ROOT / "benchmarks/robotwin/remote_eval.py"),
                "sim", "--session", w["session"], "--port", str(w["port"]), "--gpu", w["gpu"],
                "--robotwin-path", str(args.robotwin_path.absolute()), "--task", task, "--mode", mode,
                "--episodes", str(args.episodes), "--output", str(directory)]
            row = dict(task=task, mode=mode, worker=w["index"], gpu=w["gpu"], port=w["port"],
                output=str(directory), complete=False)
            try:
                with (directory.parent / f"{task}_launcher.log").open("w") as log:
                    with lock:
                        if stopping.is_set():
                            return
                        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
                        active[child.pid] = child
                        state["active"][str(w["index"])] = {**row, "pid": child.pid}
                        persist()
                    print(f"START slot={w['index']} {mode}/{task} pid={child.pid}", flush=True)
                    code = child.wait()
                    with lock:
                        active.pop(child.pid, None)
                row["returncode"] = code
                native = json.loads((directory / "run.json").read_text())
                result = native.get("result", {})
                if code != 0 or native.get("status") != "complete" or not result.get("complete"):
                    raise RuntimeError(f"Incomplete condition (exit {code}); inspect launcher/client logs")
                if result["completed_episodes"] != args.episodes:
                    raise RuntimeError("Actual episode denominator mismatch")
                row.update(complete=True, successes=result["successes"], episodes=args.episodes,
                    success_rate=result["success_rate"], wall_seconds=native["wall_seconds"])
            except Exception as exc:
                row["error"] = repr(exc)
            with lock:
                state["active"].pop(str(w["index"]), None)
                state["results"].append(row)
                if not row["complete"]:
                    state["errors"].append(row)
                persist()
            print(f"DONE {mode}/{task} complete={row['complete']}", flush=True)
            pending.task_done()

    persist()
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        futures = [executor.submit(worker, w) for w in workers]
        for f in futures:
            f.result()
    rows = [r for r in state["results"] if r["complete"]]
    state["status"] = "interrupted" if stopping.is_set() else (
        "complete" if len(rows) == len(plan["conditions"]) else "finished_with_errors")
    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    persist()
    successes, episodes = sum(r["successes"] for r in rows), sum(r["episodes"] for r in rows)
    summary = dict(status=state["status"], complete_conditions=len(rows), planned_conditions=len(plan["conditions"]),
        successes=successes, episodes=episodes, success_rate=successes / episodes if episodes else None,
        scope=state["scope"], results=state["results"], note="only complete conditions; old A100 results remain separate")
    save(out / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
