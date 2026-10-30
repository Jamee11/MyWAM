"""Schedule unmodified official manifest/sharded evaluation for the deadline audit.

The original native single_eval run remains separate and untouched. This run
uses upstream's deterministic manifest protocol, not its traditional loop.
"""
from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.robotwin.standard_eval import save  # noqa: E402


def main():
    out = ROOT / "outputs/robotwin_eval/baseline_official_manifest_deadline_20261018"
    out.mkdir(exist_ok=False)
    source = ROOT / ".external/robotwin-official-20261018"
    wrapper = source / "OpenWAM/benchmarks/robotwin/eval_policy_wrapper.py"
    policy = source / "OpenWAM/benchmarks/robotwin/policy_config.yml"
    python = ROOT / ".conda/robotwin-official/bin/python"
    pool_file = out / "pool.json"
    base = dict(os.environ, HF_ENDPOINT="https://hf-mirror.com", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                ROBOTWIN_PATH=str(source / "RoboTwin"), ROBOTWIN_PYTHON=str(python),
                CUDA_HOME="/remote-home/jinminghao/miniconda3/envs/nvcu121_cuda-12.4", TORCH_CUDA_ARCH_LIST="8.9",
                TORCH_EXTENSIONS_DIR=str(ROOT / ".cache/robotwin/torch_extensions_official"), MAX_JOBS="4",
                PYTHONPATH=str(source / "RoboTwin") + os.pathsep + str(wrapper.parent),
                SAPIEN_DISABLE_RAY_TRACING="0", ROBOTWIN_ENABLE_PLANNER_FALLBACK="0",
                __EGL_VENDOR_LIBRARY_FILENAMES=str(python.parent.parent / "lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"))
    bin_dir = out / "bin"
    bin_dir.mkdir()
    (bin_dir / "ffmpeg").symlink_to("/usr/bin/ffmpeg")
    base["PATH"] = os.pathsep.join([str(bin_dir), str(python.parent), base["CUDA_HOME"] + "/bin", base["PATH"]])
    cases = [("place_dual_shoes", "demo_randomized")]
    cases += [(t, m) for m in ["demo_randomized", "demo_clean"]
              for t in ["handover_block", "scan_object", "place_shoe", "click_bell"]]
    pending = queue.Queue()
    for case in cases:
        pending.put(case)
    jobs = queue.Queue()
    done = threading.Event()
    stop = threading.Event()
    lock = threading.RLock()
    state = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
             "target_deadline_utc": "2026-10-19T01:43:00+00:00", "workers": 8, "manifest_builders": 3,
             "protocol": "unmodified upstream build_manifest/run_eval; fixed contiguous seed scan, sealed instructions and logical episode shards",
             "traditional_run": "rebuild_official_20261018/official_diagnostic100 (retained separately; do not mix protocols)",
             "completed_native_clean": {"task": "place_dual_shoes", "successes": 92, "episodes": 100},
             "cases": [dict(task=t, mode=m) for t, m in cases], "processes": [], "errors": [], "results": []}
    active = {}

    def run(command, directory, env, label):
        directory.mkdir(parents=True, exist_ok=True)
        env = dict(env, ROBOTWIN_RUNTIME_ROOT=str(directory / "runtime"))
        for key in [k for k in env if k.startswith("ROBOTWIN_LABTASKER") or k == "ROBOTWIN_TEST_NUM"]:
            env.pop(key)
        with (directory / "client.log").open("w") as f:
            p = subprocess.Popen(command, cwd=source / "OpenWAM", env=env, stdout=f, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            with lock:
                active[p.pid] = p
                state["processes"].append({"pid": p.pid, "label": label, "command": command, "path": str(directory)})
            code = p.wait()
            with lock:
                active.pop(p.pid, None)
        if code:
            raise RuntimeError(f"{label} exited {code}; see {directory / 'client.log'}")

    def command(task, mode, operation, directory):
        return [str(python), "-u", str(wrapper), "labtasker", "--task", task, "--mode", mode,
                "--policy-config", str(policy), "--operation", operation, "--total-episodes", "100",
                "--result-file", str(directory / "result.json"), "--progress-file", str(directory / "progress.json")]

    def producer(gpu):
        while not stop.is_set():
            try:
                task, mode = pending.get_nowait()
            except queue.Empty:
                return
            directory = out / "manifests" / mode / task
            try:
                run(command(task, mode, "build_manifest", directory), directory,
                    dict(base, CUDA_VISIBLE_DEVICES=gpu), f"manifest:{mode}/{task}")
                manifest = json.loads((directory / "result.json").read_text())
                assert len(manifest["entries"]) == 100
                for start in range(0, 100, 10):
                    jobs.put((task, mode, start, directory / "result.json", manifest["manifest_hash"]))
            except Exception as exc:
                with lock:
                    state["errors"].append(str(exc))

    def worker(slot, gpu):
        while not stop.is_set() and not pool_file.exists():
            time.sleep(2)
        if stop.is_set():
            return
        pool = json.loads(pool_file.read_text())
        from benchmarks.utils.transport import WSPolicyClient

        port = pool["ports"][slot]
        while not stop.is_set():
            try:
                with WSPolicyClient(f"ws://127.0.0.1:{port}", timeout=5, open_timeout=5) as ws:
                    ws.ping()
                break
            except (OSError, TimeoutError):
                time.sleep(2)
        while not stop.is_set():
            try:
                task, mode, start, manifest_file, manifest_hash = jobs.get(timeout=2)
            except queue.Empty:
                if done.is_set():
                    return
                continue
            directory = out / "shards" / mode / task / f"{start:03d}"
            cmd = command(task, mode, "run_eval", directory)
            cmd += ["--host", "127.0.0.1", "--port", str(port), "--episode-start", str(start),
                    "--num-episodes", "10", "--manifest-file", str(manifest_file), "--manifest-hash", manifest_hash]
            try:
                run(cmd, directory, dict(base, CUDA_VISIBLE_DEVICES=gpu), f"eval:{mode}/{task}/{start}")
                result = json.loads((directory / "result.json").read_text())
                assert result["manifest_hash"] == manifest_hash and result["num_episodes"] == 10
                assert [r["episode"] for r in result["episodes"]] == list(range(start, start + 10))
                with lock:
                    state["results"].append(result)
            except Exception as exc:
                with lock:
                    state["errors"].append(str(exc))

    def stopping(*_):
        stop.set()
        with lock:
            for p in active.values():
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    with ThreadPoolExecutor(max_workers=11) as executor:
        producers = [executor.submit(producer, g) for g in ["3", "5", "6"]]
        workers = [executor.submit(worker, i, g) for i, g in enumerate(["3", "3", "3", "5", "5", "5", "6", "6"])]
        while not all(f.done() for f in producers + workers):
            if all(f.done() for f in producers):
                done.set()
            with lock:
                state["active_pids"] = list(active)
                save(out / "run.json", state)
            time.sleep(5)
        for future in producers + workers:
            future.result()
    by_case = {}
    for result in state["results"]:
        key = result["mode"] + "/" + result["task"]
        table = by_case.setdefault(key, {})
        for episode in result["episodes"]:
            assert episode["episode"] not in table
            table[episode["episode"]] = episode
    state["summary"] = {k: {"completed": len(v), "successes": sum(x["success"] for x in v.values()),
                              "complete": set(v) == set(range(100))} for k, v in by_case.items()}
    state["status"] = "complete" if len(by_case) == 9 and all(v["complete"] for v in state["summary"].values()) else "incomplete"
    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save(out / "run.json", state)


if __name__ == "__main__":
    main()
