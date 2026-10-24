"""Split-host RoboTwin deployment; reuse the existing server and native evaluator."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def probe(session, port, host="127.0.0.1", timeout=15):
    from benchmarks.utils.transport import WSPolicyClient
    start = time.monotonic()
    with WSPolicyClient(f"ws://{host}:{port}", timeout=timeout, open_timeout=timeout) as ws:
        pong = ws.ping()  # Never reset or generate actions in a liveness probe.
    if pong.get("readiness_token") != session["readiness_token"]:
        raise RuntimeError("Endpoint belongs to a different model session")
    if pong.get("pool_slot") != session["slot"]:
        raise RuntimeError("Endpoint slot mismatch; do not share a port between simulators")
    return {"pong": pong, "ping_seconds": time.monotonic() - start}


def model(args):
    import yaml
    out = args.output.absolute()
    out.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(args.deploy_config.read_text())
    if args.kind == "sidecar":
        cfg["optimization"]["compile"].update(enabled=True, sidecar_strategy="ffn")
    deploy = out / "deploy.yaml"
    deploy.write_text(yaml.safe_dump(cfg))
    weights = sorted(args.checkpoint.glob("checkpoint_step_*.safetensors"),
                     key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not weights:
        raise ValueError("Checkpoint directory must contain checkpoint_step_*.safetensors and its artifacts")
    free = subprocess.check_output(["nvidia-smi", "-i", args.gpu,
        "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    if int(free.strip()) < args.min_free_mib:
        raise RuntimeError("Insufficient free GPU memory; existing jobs are left running")
    token = uuid.uuid4().hex
    entry = "serve_remote_sidecar.py" if args.kind == "sidecar" else "serve_parallel.py"
    command = [sys.executable, "-u", str(HERE / entry), "--ckpt-dir", str(args.checkpoint.absolute()),
        "--port", str(args.port), "--slots", str(args.slots), "--manifest", str(out / "pool.json"),
        "--readiness-token", token, "--config", str(deploy)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", TORCHINDUCTOR_CACHE_DIR=str(out / "inductor"))
    # Keep architecture-specific compiled kernels separate between A100 and L20.
    metadata = {"checkpoint": str(weights[-1].absolute()), "checkpoint_sha256": digest(weights[-1]),
        "deploy_config_sha256": digest(deploy), "kind": args.kind, "gpu": args.gpu,
        "readiness_token": token, "ports": list(range(args.port, args.port + args.slots)),
        "protocol": "loopback WebSocket via SSH; one independent policy state per slot"}
    metadata["checkpoint_artifact_sha256"] = {name: digest(args.checkpoint / name)
        for name in ["config.yaml", "normalization_stats.npy"]}
    save(out / "model.json", metadata)
    with (out / "server.log").open("w") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    metadata["pid"] = child.pid
    save(out / "model.json", metadata)
    try:
        deadline = time.monotonic() + args.wait
        while not (out / "pool.json").exists():
            if child.poll() is not None:
                raise RuntimeError("Model initialization failed; inspect server.log")
            if time.monotonic() > deadline:
                raise TimeoutError("Model initialization timed out")
            time.sleep(2)
        for slot, port in enumerate(metadata["ports"]):
            session = {**metadata, "slot": slot, "model_port": port}
            # Manifest creation precedes listener threads; permit their startup.
            while True:
                try:
                    health = probe(session, port)
                    break
                except (OSError, TimeoutError):
                    if child.poll() is not None or time.monotonic() > deadline:
                        raise
                    time.sleep(2)
            save(out / f"session_slot{slot}.json", {**session, "health": health})
        print(f"READY: {out}/session_slot*.json", flush=True)
        return child.wait()
    finally:
        if child.poll() is None:
            child.terminate()  # Only this launcher's own child.
            child.wait(timeout=30)


def sim(args):
    import fcntl
    import yaml
    from benchmarks.robotwin.standard_eval import validate_result
    session = json.loads(args.session.read_text())
    out = args.output.absolute()
    out.mkdir(parents=True, exist_ok=False)
    # Local coordination supplements the explicit one-slot-per-simulator assignment.
    lock_path = Path(os.environ.get("TMPDIR", "/tmp")) / (
        "openwam-remote-" + hashlib.sha256(
            f"{session['readiness_token']}:{session['slot']}".encode()).hexdigest()[:24] + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        health = probe(session, args.port, args.host)
        if yaml.safe_load((HERE / "step_limits.yml").read_text()):
            raise RuntimeError("Standard eval requires empty step-limit overrides")
        cfg = yaml.safe_load(args.policy_config.read_text())
        cfg["eval_video_log"] = False
        policy = out / "policy.yml"
        policy.write_text(yaml.safe_dump(cfg))
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("ROBOTWIN_") and k != "POLICY_CONFIG_PATH"}
        env.update(ROBOTWIN_PATH=str(args.robotwin_path.absolute()),
            ROBOTWIN_PYTHON=str(Path(sys.executable).absolute()),
            ROBOTWIN_RUNTIME_ROOT=str(out / "runtime"), ROBOTWIN_TEST_NUM=str(args.episodes),
            POLICY_CONFIG_PATH=str(policy), SAPIEN_DISABLE_RAY_TRACING="0",
            ROBOTWIN_ENABLE_PLANNER_FALLBACK="0", PYTHONNOUSERSITE="1",
            OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            TORCH_EXTENSIONS_DIR=str(ROOT / ".cache/robotwin/remote_sim_extensions" / socket.gethostname()))
        env["PYTHONPATH"] = str(args.robotwin_path.absolute() / "envs/curobo/src")
        command = ["bash", str(HERE / "single_eval.sh"), args.task, args.mode,
            out.name, args.gpu, str(args.port), args.host]
        paths = [HERE / n for n in ["eval_policy_wrapper.py", "openwam2robotwin_interface.py", "step_limits.yml"]]
        paths += [args.robotwin_path / n for n in ["script/eval_policy.py", "envs/camera/camera.py",
            "envs/_base_task.py", "task_config/_camera_config.yml"]]
        record = {"status": "running", "session": session, "health": health, "command": command,
            "scope": "standard100" if args.episodes == 100 else "smoke_only",
            "source_sha256": {str(p): digest(p) for p in paths},
            "gpu_snapshot": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.free",
                "--format=csv"], text=True), "video": False,
            "near": "unchanged official camera source; this is not wrist-near A/B"}
        save(out / "run.json", record)
        started = time.monotonic()
        with (out / "client.log").open("w") as log:
            code = subprocess.call(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        result = validate_result((out / "client.log").read_text(), out / "runtime", args.episodes, code)
        record.update(status="complete" if result["complete"] else "failed", result=result,
                      wall_seconds=time.monotonic() - started)
        save(out / "run.json", record)
        print(json.dumps(record["result"], indent=2))
        return 0 if result["complete"] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="operation", required=True)
    m = sub.add_parser("model")
    m.add_argument("--checkpoint", type=Path, required=True)
    m.add_argument("--kind", choices=["sidecar", "baseline"], required=True)
    m.add_argument("--gpu", required=True)
    m.add_argument("--port", type=int, default=22700)
    m.add_argument("--slots", type=int, default=2)
    m.add_argument("--output", type=Path, required=True)
    m.add_argument("--deploy-config", type=Path, default=ROOT / "configs/deploy.yaml")
    m.add_argument("--min-free-mib", type=int, default=33000)
    m.add_argument("--wait", type=int, default=3600)
    for name in ["probe", "sim"]:
        s = sub.add_parser(name)
        s.add_argument("--session", type=Path, required=True)
        s.add_argument("--host", default="127.0.0.1")
        s.add_argument("--port", type=int, default=22700)
        if name == "sim":
            s.add_argument("--robotwin-path", type=Path, required=True)
            s.add_argument("--gpu", default="0")
            s.add_argument("--task", default="adjust_bottle")
            s.add_argument("--mode", choices=["demo_clean", "demo_randomized"], default="demo_clean")
            s.add_argument("--episodes", type=int, default=1)
            s.add_argument("--output", type=Path, required=True)
            s.add_argument("--policy-config", type=Path, default=HERE / "policy_config_relational_sidecar.yml")
    args = p.parse_args()
    if not 1 <= args.port <= 65535:
        p.error("Invalid port")
    if args.operation == "model" and (args.slots < 1 or args.port + args.slots > 65536):
        p.error("Invalid slot range")
    if args.operation == "sim" and args.episodes < 1:
        p.error("Episodes must be positive")
    if args.operation == "probe":
        print(json.dumps(probe(json.loads(args.session.read_text()), args.port, args.host), indent=2))
        return 0
    return model(args) if args.operation == "model" else sim(args)


if __name__ == "__main__":
    def terminate(*_):
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, terminate)
    raise SystemExit(main())
