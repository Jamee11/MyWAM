"""Launch the authorized new-environment queue after isolated acceleration gates."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.robotwin.standard_eval import save, task_names  # noqa: E402


def main():
    out = ROOT / "outputs/robotwin_eval/sidecar20000_officialenv_20261018"
    state_file = out / "launch_state.json"
    if state_file.exists():
        raise FileExistsError("Launch state exists; inspect it before any repeat")
    state = {"status": "waiting_validation", "processes": []}
    save(state_file, state)
    deadline = time.monotonic() + 1800
    result_file = out / "fast_validation/result.json"
    validation_record = json.loads((out / "fast_validation_process.json").read_text())
    validation_pid = validation_record["pid"]
    while not result_file.exists() and Path(f"/proc/{validation_pid}").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Acceleration validation not finished; no unverified fast queue launched")
        time.sleep(5)
    result = json.loads(result_file.read_text()) if result_file.exists() else {}
    tests = (out / "fast_runtime_tests.log").read_text()
    if not re.search(r"\d+ passed", tests) or "FAILED" in tests:
        raise RuntimeError("CPU Sidecar input/computation gates did not pass")
    fast = bool(result.get("passed"))
    if not fast:
        raise RuntimeError("GPU acceleration gate failed; preserved logs require inspection")
    while not validation_record.get("becomes_server") and Path(f"/proc/{validation_pid}").exists():
        time.sleep(1)
    save(out / "validation_gpu_released.json", {"validation_passed": True, "validator_pid": validation_pid})
    # Finite, numerically checked compilation is required above. Caching remains
    # approximate; results are explicitly labeled and never merged with eager runs.
    state.update(status="releasing_smoke_services", acceleration="compile_and_joint_velocity_cache",
                 approximate_cache=True, validation=str(result_file))
    save(state_file, state)
    for record in json.loads((out / "eager_smoke_processes.json").read_text()):
        pid = record["pid"]
        proc = Path(f"/proc/{pid}/cmdline")
        if proc.exists():
            assert b"standard_eval.py" in proc.read_bytes() and b"eager_smoke" in proc.read_bytes()
            os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        if not any(Path(f"/proc/{v['pid']}").exists() for v in json.loads((out / "eager_smoke_processes.json").read_text())):
            break
        time.sleep(1)
    for pid, port in [(2389543, b"19200"), (2390211, b"19300")]:
        p = Path(f"/proc/{pid}/cmdline")
        if p.exists():
            cmd = p.read_bytes()
            assert b"serve_parallel.py" in cmd and port in cmd
            os.kill(pid, signal.SIGTERM)
    time.sleep(5)
    env = dict(os.environ, HF_ENDPOINT="https://hf-mirror.com", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               TORCHINDUCTOR_COMPILE_THREADS="4", TORCH_WARM_POOL="0",
               TORCHINDUCTOR_CACHE_DIR=str(ROOT / ".cache/robotwin/inductor_sidecar_fast"))
    checkpoint = ROOT / "outputs/robotwin_eval/sidecar20000_currentenv_20261018/checkpoint_step_20000"
    config = out / "deploy_selected.yaml"
    priority = ["adjust_bottle", "place_dual_shoes", "beat_block_hammer", "place_shoe", "scan_object",
                "handover_block", "place_container_plate", "place_cans_plasticbox"]
    tasks = priority + [t for t in task_names() if t not in priority]
    partitions = {str(g): tasks[i::3] for i, g in enumerate([0, 1, 7])}
    assert len({t for ts in partitions.values() for t in ts}) == 50
    save(out / "task_partitions.json", partitions)

    def spawn(command, process_env, log):
        with log.open("w") as f:
            p = subprocess.Popen(command, cwd=ROOT, env=process_env, stdout=f, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        state["processes"].append({"pid": p.pid, "command": command, "log": str(log)})
        save(state_file, state)
        return p.pid

    pools = []
    for i, gpu in enumerate([0, 1, 7]):
        manifest = out / f"pool{gpu}.json"
        if (out / f"pool{gpu}_process.json").exists() and gpu == 0:
            record = json.loads((out / f"pool{gpu}_process.json").read_text())
            pid = record["pid"]
            assert Path(f"/proc/{pid}").exists()
            state["processes"].append(record)
            save(state_file, state)
        else:
            port = {0: 19600, 1: 19700, 7: 19800}[gpu]
            command = [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "benchmarks/robotwin/sidecar_fast_runtime.py"),
                       "--ckpt-dir", str(checkpoint), "--config", str(config), "--port", str(port), "--slots", "3",
                       "--manifest", str(manifest), "--readiness-token", f"sidecar20k-official-fast-gpu{gpu}"]
            pid = spawn(command, dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), out / f"pool{gpu}.log")
        pools.append((i, gpu, pid, manifest))

    def warm(pool):
        from benchmarks.utils.transport import WSPolicyClient

        _, gpu, pid, manifest = pool
        until = time.monotonic() + 1800
        while not manifest.exists():
            if not Path(f"/proc/{pid}").exists() or time.monotonic() > until:
                raise RuntimeError(f"GPU{gpu} model did not become ready")
            time.sleep(2)
        port = json.loads(manifest.read_text())["ports"][0]
        payloads = json.loads((out / "fast_validation/inputs.json").read_text())
        while True:
            try:
                with WSPolicyClient(f"ws://127.0.0.1:{port}", timeout=1200, open_timeout=5) as ws:
                    ws.ping()
                    for payload in payloads:
                        ws.reset()
                        response = ws.predict(payload)
                        if response.get("action") is None:
                            raise RuntimeError("Warmup did not produce an action")
                    ws.reset()
                break
            except (OSError, TimeoutError):
                if time.monotonic() > until:
                    raise
                time.sleep(2)
        print(f"WARMED gpu={gpu}", flush=True)
        return gpu

    state["status"] = "warming_models_before_native_clients"
    save(state_file, state)
    state["warmed_gpus"] = []
    # The checked GPU0 server can start native rollouts while other independent
    # replicas load, with no requests to a server during its own warmup.
    for i, gpu, pid, manifest in sorted(pools, key=lambda p: {0: 0, 1: 1, 7: 2}[p[1]]):
        warm((i, gpu, pid, manifest))
        state["warmed_gpus"].append(gpu)
        save(state_file, state)
        command = [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "benchmarks/robotwin/parallel_eval.py"),
                   "--output", str(out / f"queue{gpu}"), "--checkpoint", str(checkpoint),
                   "--pool-manifest", str(manifest), "--pool-pid", str(pid),
                   "--sim-gpus", {0: "2,5,0", 1: "6,2,1", 7: "6,7"}[gpu],
                   "--cuda-home", "/remote-home/jinminghao/miniconda3/envs/nvcu121_cuda-12.4",
                   "--deploy-config", str(config), "--policy-config", str(ROOT / "benchmarks/robotwin/policy_config_relational_sidecar.yml"),
                   "--robotwin-path", str(ROOT / ".external/robotwin-official-20261018/RoboTwin"),
                   "--sim-python", str(ROOT / ".conda/robotwin-official/bin/python"),
                   "--tasks", ",".join(partitions[str(gpu)])]
        spawn(command, dict(env, PYTHONNOUSERSITE="1"), out / f"queue{gpu}.log")
    state.update(status="launched", native_workers=8, total_splits=100, episodes_per_split=100,
                 official_baseline_preserved=True)
    save(state_file, state)
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
