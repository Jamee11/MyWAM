"""Dynamically distribute the 50×2 native RoboTwin evaluations over isolated endpoints."""
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.robotwin.standard_eval import (  # noqa: E402 -- source-tree imports follow sys.path bootstrap
    save,
    task_names,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pool-manifest", type=Path, required=True)
    p.add_argument("--pool-pid", type=int, required=True)
    p.add_argument("--sim-gpus", default="0,1,1,1,2,2,3,3,4,4,5,5,6,6,7,7")
    p.add_argument("--legacy-server-pid", type=int)
    p.add_argument("--legacy-port", type=int, default=18848)
    p.add_argument("--legacy-server-gpu", default="7")
    p.add_argument("--legacy-sim-gpu", default="1")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--cuda-home", type=Path, required=True)
    p.add_argument("--cuda-arch", default="8.9")
    args = p.parse_args()
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "parallel.json").exists():
        p.error("output already contains a queue; do not duplicate or overwrite it")
    (output / "logs").mkdir(exist_ok=True)
    state = {"status": "running", "episodes": args.episodes, "jobs": [], "workers": {},
             "pool_pid": args.pool_pid, "pool_manifest": str(args.pool_manifest.absolute())}
    jobs = queue.Queue()
    for task in task_names():
        for mode in ("demo_clean", "demo_randomized"):
            row = {"task": task, "mode": mode, "status": "queued"}
            state["jobs"].append(row)
            jobs.put(row)
    lock = threading.Lock()
    stopping = threading.Event()
    active = {}

    def snapshot():
        save(output / "parallel.json", state)

    def process_alive(pid):
        try:
            return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
        except FileNotFoundError:
            return False

    def wait_pool(slot):
        from benchmarks.utils.transport import WSPolicyClient
        while not stopping.is_set():
            if not process_alive(args.pool_pid):
                raise RuntimeError("shared model process exited during startup/evaluation")
            if args.pool_manifest.is_file():
                pool = json.loads(args.pool_manifest.read_text())
                if pool["pid"] != args.pool_pid:
                    raise RuntimeError("pool manifest PID mismatch")
                if slot >= len(pool["ports"]):
                    raise ValueError("more simulator slots than shared policy endpoints")
                try:
                    with WSPolicyClient(f"ws://127.0.0.1:{pool['ports'][slot]}", timeout=5, open_timeout=5) as ws:
                        pong = ws.ping()
                    if pong.get("readiness_token") != pool["readiness_token"] or pong.get("pool_slot") != slot:
                        raise RuntimeError("pool readiness identity mismatch")
                    return pool
                except (OSError, TimeoutError):
                    pass
            stopping.wait(2)
        return None

    def worker(index, sim_gpu, legacy=False):
        try:
            with lock:
                state["workers"][str(index)] = {"sim_gpu": sim_gpu, "status": "waiting_model"}
                snapshot()
            pool = None if legacy else wait_pool(index)
            if stopping.is_set():
                return
            server_pid = args.legacy_server_pid if legacy else pool["pid"]
            server_gpu = args.legacy_server_gpu if legacy else pool["cuda_visible_devices"]
            port = args.legacy_port if legacy else pool["ports"][index]
            while not stopping.is_set():
                try:
                    row = jobs.get_nowait()
                except queue.Empty:
                    break
                job_out = output / "jobs" / row["mode"] / row["task"]
                command = [sys.executable, "-u", str(Path(__file__).with_name("standard_eval.py")),
                           "--checkpoint", str(args.checkpoint.absolute()), "--output", str(job_out),
                           "--tasks", row["task"], "--modes", row["mode"], "--episodes", str(args.episodes),
                           "--server-gpu", server_gpu, "--sim-gpu", sim_gpu, "--port", str(port),
                           "--reuse-server-pid", str(server_pid), "--cuda-home", str(args.cuda_home.absolute()),
                           "--cuda-arch", args.cuda_arch]
                log_path = output / "logs" / f"{row['mode']}_{row['task']}.log"
                with log_path.open("w") as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    with lock:
                        active[index] = process
                        row.update(status="running", worker=index, sim_gpu=sim_gpu, server_port=port,
                                   pid=process.pid, output=str(job_out), log=str(log_path))
                        state["workers"][str(index)].update(status="running", task=row["task"], mode=row["mode"])
                        snapshot()
                    print(f"START worker={index} gpu={sim_gpu} {row['mode']}/{row['task']} pid={process.pid}", flush=True)
                    code = process.wait()
                with lock:
                    active.pop(index, None)
                    run_file = job_out / "run.json"
                    report = json.loads(run_file.read_text()) if run_file.is_file() else {}
                    valid = code == 0 and report.get("status") == "complete" and len(report.get("results", [])) == 1
                    row.update(status="complete" if valid else "failed", returncode=code)
                    if valid:
                        result = report["results"][0]
                        valid = result["complete"] and result["completed_episodes"] == args.episodes
                        row.update(status="complete" if valid else "failed", result=result)
                    snapshot()
                print(f"END worker={index} {row['mode']}/{row['task']} status={row['status']}", flush=True)
                jobs.task_done()
            with lock:
                state["workers"][str(index)]["status"] = "finished"
                snapshot()
        except BaseException as exc:
            with lock:
                state["workers"].setdefault(str(index), {}).update(status="failed", error=f"{type(exc).__name__}: {exc}")
                snapshot()
            print(f"WORKER_FAILED {index}: {type(exc).__name__}: {exc}", flush=True)

    def stop(*_):
        stopping.set()
        with lock:
            for process in active.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    snapshot()
    sim_gpus = args.sim_gpus.split(",")
    with ThreadPoolExecutor(max_workers=len(sim_gpus) + bool(args.legacy_server_pid)) as executor:
        futures = [executor.submit(worker, i, gpu) for i, gpu in enumerate(sim_gpus)]
        if args.legacy_server_pid:
            futures.append(executor.submit(worker, len(sim_gpus), args.legacy_sim_gpu, True))
        for future in futures:
            future.result()
    completed = [row for row in state["jobs"] if row["status"] == "complete"]
    state["status"] = "complete" if len(completed) == 100 else "incomplete"
    state["completed_splits"] = len(completed)
    state["full_50_task_benchmark"] = len(completed) == 100 and args.episodes == 100
    state["macro_success_rate"] = {
        mode: sum(row["result"]["success_rate"] for row in completed if row["mode"] == mode) / 50
        for mode in ("demo_clean", "demo_randomized")
    } if len(completed) == 100 else None
    snapshot()
    print(json.dumps({key: state[key] for key in ("status", "completed_splits", "full_50_task_benchmark", "macro_success_rate")}), flush=True)
    return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
