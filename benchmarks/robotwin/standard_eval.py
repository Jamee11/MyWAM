"""Run and audit native RoboTwin evaluation using an isolated local checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_names():
    # Keep the established traditional launcher's task list authoritative.
    source = (HERE / "multi_eval.sh").read_text()
    return re.search(r"ROBOTWIN_ALL_TASKS=\((.*?)\)", source, re.S).group(1).split()


def validate_result(log: str, runtime: Path, expected: int, returncode: int) -> dict:
    plain = re.sub(r"\x1b\[[0-9;]*m", "", log)
    rows = [(int(a), int(b), int(c)) for a, b, c in re.findall(
        r"Success rate:\s*(\d+)/(\d+).*?current seed:\s*(\d+)", plain)]
    errors = []
    if returncode != 0:
        errors.append(f"client exited {returncode}")
    if [row[1] for row in rows] != list(range(1, expected + 1)):
        errors.append(f"expected {expected} episode records, got {[row[1] for row in rows]}")
    previous = 0
    for success, count, _ in rows:
        if not 0 <= success <= count or success - previous not in (0, 1):
            errors.append("invalid cumulative successes")
        previous = success
    if any(b[2] <= a[2] for a, b in zip(rows, rows[1:])):
        errors.append("accepted seeds did not increase")
    if any(row[2] < 100000 for row in rows):
        errors.append("accepted seed is outside seed-group-0 evaluation range")
    files = re.findall(r"Data has been saved to (.+)", plain)
    result_path, native_rate = None, None
    if len(files) != 1:
        errors.append("missing or ambiguous native result completion marker")
    else:
        result_path = (runtime / files[0].strip()).resolve()
        if not result_path.is_relative_to(runtime.resolve()) or not result_path.is_file():
            errors.append("native result missing or outside this run's runtime")
        else:
            try:
                result_text = result_path.read_text()
                if "Instruction Type: unseen" not in result_text:
                    errors.append("native result does not confirm unseen instructions")
                native_rate = float(result_text.splitlines()[-1])
                if not math.isfinite(native_rate) or not rows or abs(native_rate - rows[-1][0] / expected) > 1e-10:
                    errors.append("native result differs from actual episode denominator")
            except (ValueError, IndexError):
                errors.append("native result is not a scalar success rate")
    return {"complete": not errors, "errors": errors, "requested_episodes": expected,
            "completed_episodes": len(rows), "successes": rows[-1][0] if rows else 0,
            "success_rate": native_rate, "accepted_seeds": [r[2] for r in rows],
            "native_result": str(result_path) if result_path else None, "returncode": returncode}


def inspect_server(pid: int, checkpoint: Path, port: int) -> dict:
    """Verify a reused local listener belongs to the declared checkpoint process."""
    proc = Path(f"/proc/{pid}")
    command = [v.decode() for v in (proc / "cmdline").read_bytes().split(b"\0") if v]
    cwd = (proc / "cwd").resolve()
    if "--ckpt-dir" not in command or "--port" not in command:
        raise ValueError("Reused server must explicitly declare --ckpt-dir and --port")
    declared = (cwd / command[command.index("--ckpt-dir") + 1]).resolve()
    base_port = int(command[command.index("--port") + 1])
    pool = None
    if "--slots" in command and "--manifest" in command:
        pool_file = (cwd / command[command.index("--manifest") + 1]).resolve()
        pool = json.loads(pool_file.read_text())
        if (pool["pid"] != pid or pool["checkpoint_dir"] != str(checkpoint.resolve())
                or pool["deploy_config_sha256"] != sha256(ROOT / "configs/deploy.yaml")
                or port not in pool["ports"]):
            raise ValueError("Shared server manifest/checkpoint/deploy configuration mismatch")
    if declared != checkpoint.resolve() or (pool is None and base_port != port):
        raise ValueError("Reused server checkpoint/port mismatch")
    if "--ckpt-name" in command:
        raise ValueError("For reused servers, use the default latest checkpoint selection")
    allowed = {"--ckpt-dir", "--host", "--port", "--device", "--readiness-token"}
    if pool is not None:
        allowed |= {"--slots", "--manifest"}
    if any(item.startswith("--") and item not in allowed for item in command):
        raise ValueError("Reused server has deploy overrides; launch a fresh standard server")
    listeners = set()
    for line in Path("/proc/net/tcp").read_text().splitlines()[1:]:
        cols = line.split()
        if cols[3] == "0A" and int(cols[1].split(":")[1], 16) == port:
            listeners.add(cols[9])
    sockets = set()
    for path in (proc / "fd").iterdir():
        try:
            sockets.add(os.readlink(path))
        except FileNotFoundError:
            # Concurrent endpoint health checks can close their socket mid-scan.
            pass
    if not any(f"socket:[{inode}]" in sockets for inode in listeners):
        raise ValueError("Declared server PID does not own the listening socket")
    environment = dict(v.split("=", 1) for v in (proc / "environ").read_bytes().decode().split("\0") if "=" in v)
    return {"pid": pid, "command": command, "cwd": str(cwd), "pool": pool,
            "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
            "ownership": "reused; never terminated by runner"}


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def select_ffmpeg(explicit: Path | None = None) -> tuple[Path, str]:
    candidates = [explicit] if explicit else [Path("/usr/bin/ffmpeg"), Path("/bin/ffmpeg")]
    if not explicit and shutil.which("ffmpeg"):
        candidates.append(Path(shutil.which("ffmpeg")))
    errors = []
    for path in candidates:
        if path is None or not path.is_file():
            continue
        version = subprocess.run([str(path), "-version"], capture_output=True, text=True, timeout=15)
        encoders = subprocess.run([str(path), "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=15)
        if version.returncode == encoders.returncode == 0 and "libx264 " in encoders.stdout:
            return path.resolve(), version.stdout.splitlines()[0]
        errors.append(str(path))
    raise RuntimeError(f"No working ffmpeg with libx264; checked {errors}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--robotwin-path", type=Path, default=ROOT / ".external/RoboTwin")
    p.add_argument("--sim-python", type=Path, default=ROOT / ".venv-robotwin/bin/python")
    p.add_argument("--cuda-home", type=Path)
    p.add_argument("--cuda-arch", default=os.environ.get("TORCH_CUDA_ARCH_LIST"),
                   help="CUDA extension target, e.g. 8.9 for L20; default is PyTorch auto-detection")
    p.add_argument("--ffmpeg", type=Path, help="working ffmpeg binary; system binary preferred by default")
    p.add_argument("--server-gpu", required=True)
    p.add_argument("--sim-gpu", required=True)
    p.add_argument("--port", type=int, default=18848)
    p.add_argument("--tasks", default="all", help="all or comma-separated task names")
    p.add_argument("--modes", nargs="+", choices=["demo_clean", "demo_randomized"], default=["demo_clean", "demo_randomized"])
    p.add_argument("--episodes", type=int, default=100, help="100 is standard; other counts are smoke only")
    p.add_argument("--reuse-server-pid", type=int)
    p.add_argument("--server-wait", type=int, default=1800)
    args = p.parse_args()
    if args.episodes < 1 or not 1 <= args.port <= 65535:
        p.error("episodes must be positive and port must be 1..65535")
    tasks = task_names() if args.tasks == "all" else list(dict.fromkeys(item.strip() for item in args.tasks.split(",")))
    args.modes = list(dict.fromkeys(args.modes))
    if not tasks or set(tasks) - set(task_names()):
        p.error("unknown task names")
    for key in ("checkpoint", "output", "robotwin_path", "sim_python"):
        # Do not resolve the interpreter symlink: conda/venv prefix comes from its entry path.
        setattr(args, key, getattr(args, key).absolute())
    if not (args.robotwin_path / "OPENWAM_COPY_MANIFEST.json").is_file():
        p.error("use scripts/copy_robotwin_runtime.py to create an independent local checkout first")
    if args.robotwin_path.is_symlink() or not args.sim_python.is_file():
        p.error("invalid independent checkout or simulator interpreter")
    if yaml.safe_load((HERE / "step_limits.yml").read_text()):
        p.error("standard evaluation requires no task step-limit overrides")
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoints = sorted(args.checkpoint.glob("checkpoint_step_*.safetensors"),
                         key=lambda f: int(f.stem.rsplit("_", 1)[1]))
    if not checkpoints:
        p.error("checkpoint weights not found")
    weight = checkpoints[-1]
    print(f"Verifying checkpoint SHA256: {weight}", flush=True)
    record = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "preparing",
              "protocol": "native expert-filtered RoboTwin; unseen; seed group 0; RT; original step limits",
              "seed_scope": {"simulator_seed_group": 0,
                             "policy_rng": "upstream server defaults; launcher does not reseed per episode"},
              "scope": "standard100" if args.episodes == 100 else "smoke_only",
              "tasks": tasks, "modes": args.modes, "episodes": args.episodes,
              "checkpoint": str(weight), "checkpoint_size": weight.stat().st_size,
              "checkpoint_sha256": sha256(weight), "results": []}
    for label, source in {"training_config": args.checkpoint / "config.yaml",
                          "normalization_stats": args.checkpoint / "normalization_stats.npy",
                          "deploy_config": ROOT / "configs/deploy.yaml",
                          "policy_config": HERE / "policy_config.yml",
                          "step_limits": HERE / "step_limits.yml",
                          "wrapper": HERE / "eval_policy_wrapper.py",
                          "adapter": HERE / "openwam2robotwin_interface.py",
                          "robotwin_eval": args.robotwin_path / "script/eval_policy.py"}.items():
        record[label] = {"path": str(source), "sha256": sha256(source)}
        if source.suffix in {".yaml", ".yml"}:
            record[label]["contents"] = yaml.safe_load(source.read_text())
    record["robotwin_commit"] = subprocess.check_output(["git", "-C", str(args.robotwin_path), "rev-parse", "HEAD"], text=True).strip()
    manifest_snapshot = args.output / "robotwin_copy_manifest.json"
    manifest_snapshot.write_bytes((args.robotwin_path / "OPENWAM_COPY_MANIFEST.json").read_bytes())
    record["robotwin_copy_manifest_sha256"] = sha256(manifest_snapshot)
    extension_manifest = args.robotwin_path / "OPENWAM_EXTENSION_INSTALL.json"
    if extension_manifest.is_file():
        extension_snapshot = args.output / "curobo_extensions.json"
        extension_snapshot.write_bytes(extension_manifest.read_bytes())
        installed = json.loads(extension_snapshot.read_text())
        record["curobo_extensions"] = {"manifest_sha256": sha256(extension_snapshot), "binaries": {}}
        for entry in installed["modules"]:
            binary = args.robotwin_path / "envs/curobo/src/curobo/curobolib" / Path(entry["target"]).name
            actual = sha256(binary)
            if actual != entry["sha256"]:
                raise RuntimeError(f"Installed cuRobo binary differs from its build record: {binary}")
            record["curobo_extensions"]["binaries"][entry["module"]] = actual
    record["openwam_commit"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    (args.output / "openwam.patch").write_bytes(subprocess.check_output(
        ["git", "-C", str(ROOT), "diff", "--binary", "--", "openwam", "configs", "benchmarks", "scripts"]))
    sources = ["envs/_base_task.py", "envs/robot/robot.py", "envs/robot/planner.py",
               "task_config/_camera_config.yml", "task_config/_embodiment_config.yml",
               "task_config/_eval_step_limit.yml", "task_config/demo_clean.yml", "task_config/demo_randomized.yml",
               "assets/embodiments/aloha-agilex/config.yml"] + [f"envs/{task}.py" for task in tasks]
    embodiment = args.robotwin_path / "assets/embodiments/aloha-agilex"
    urdf = embodiment / yaml.safe_load((embodiment / "config.yml").read_text())["urdf_path"]
    sources.append(str(urdf.relative_to(args.robotwin_path)))
    record["robotwin_source_sha256"] = {name: sha256(args.robotwin_path / name) for name in sources}
    version_code = (
        "import sys,json,importlib.metadata as m; "
        "print(json.dumps({'python':sys.version,'prefix':sys.prefix,'packages':"
        "{name:m.version(name) for name in ['torch','sapien','warp-lang','mplib','websockets']}}))"
    )
    print(f"Inspecting simulator environment: {args.sim_python}", flush=True)
    record["sim_environment"] = json.loads(subprocess.check_output([str(args.sim_python), "-c", version_code], text=True))
    (args.output / "robotwin.patch").write_bytes(subprocess.check_output(["git", "-C", str(args.robotwin_path), "diff", "--binary"]))
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("ROBOTWIN_") or key == "POLICY_CONFIG_PATH":
            env.pop(key)
    env.update(HF_ENDPOINT=env.get("HF_ENDPOINT", "https://hf-mirror.com"), HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               TORCHINDUCTOR_CACHE_DIR=str(ROOT / ".cache/robotwin/inductor"))
    server, client = None, None
    server_log = None
    token = uuid.uuid4().hex
    policy_snapshot = args.output / "policy_config.yml"
    policy_snapshot.write_bytes((HERE / "policy_config.yml").read_bytes())
    deploy_snapshot = args.output / "deploy.yaml"
    deploy_snapshot.write_bytes((ROOT / "configs/deploy.yaml").read_bytes())
    try:
        ffmpeg, ffmpeg_version = select_ffmpeg(args.ffmpeg)
        private_bin = args.output / "bin"
        private_bin.mkdir()
        (private_bin / "ffmpeg").symlink_to(ffmpeg)
        record["ffmpeg"] = {"binary": str(ffmpeg), "version": ffmpeg_version}
        if args.reuse_server_pid:
            record["server"] = inspect_server(args.reuse_server_pid, args.checkpoint, args.port)
            if record["server"]["cuda_visible_devices"] != args.server_gpu:
                raise ValueError("Reused server GPU does not match --server-gpu")
        else:
            server_env = dict(env, CUDA_VISIBLE_DEVICES=args.server_gpu)
            command = [sys.executable, "-u", str(ROOT / "scripts/deploy.py"), "--ckpt-dir", str(args.checkpoint),
                       "--config", str(deploy_snapshot), "--host", "127.0.0.1", "--port", str(args.port),
                       "--readiness-token", token]
            server_log = (args.output / "server.log").open("w")
            server = subprocess.Popen(command, cwd=ROOT, env=server_env, stdout=server_log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            record["server"] = {"pid": server.pid, "command": command, "gpu": args.server_gpu}
        save(args.output / "run.json", record)
        from benchmarks.utils.transport import WSPolicyClient
        deadline = time.monotonic() + args.server_wait
        while True:
            if server is not None and server.poll() is not None:
                raise RuntimeError("policy server exited before READY; see server.log")
            try:
                with WSPolicyClient(f"ws://127.0.0.1:{args.port}", timeout=5, open_timeout=5) as ws:
                    pong = ws.ping()
                if server is not None and pong.get("readiness_token") != token:
                    raise RuntimeError("port belongs to a different server")
                pool = record["server"].get("pool")
                if pool and (pong.get("readiness_token") != pool["readiness_token"]
                             or pong.get("pool_slot") != pool["ports"].index(args.port)):
                    raise RuntimeError("shared server session identity mismatch")
                record["server_pong"] = pong
                break
            except (OSError, TimeoutError):
                if time.monotonic() > deadline:
                    raise TimeoutError("policy server did not become ready")
                time.sleep(2)
        sim_env = dict(env, ROBOTWIN_PATH=str(args.robotwin_path), ROBOTWIN_PYTHON=str(args.sim_python),
                       ROBOTWIN_TEST_NUM=str(args.episodes), SAPIEN_DISABLE_RAY_TRACING="0",
                       ROBOTWIN_ENABLE_PLANNER_FALLBACK="0", MAX_JOBS=env.get("MAX_JOBS", "4"),
                       TORCH_EXTENSIONS_DIR=str(ROOT / ".cache/robotwin/torch_extensions"),
                       POLICY_CONFIG_PATH=str(policy_snapshot),
                       PYTHONPATH=str(args.robotwin_path / "envs/curobo/src"))
        sim_env["PATH"] = str(args.sim_python.parent) + os.pathsep + env["PATH"]
        if args.cuda_home:
            sim_env["CUDA_HOME"] = str(args.cuda_home.absolute())
            sim_env["PATH"] = str(args.cuda_home.absolute() / "bin") + os.pathsep + sim_env["PATH"]
        if args.cuda_arch:
            sim_env["TORCH_CUDA_ARCH_LIST"] = args.cuda_arch
        sim_env["PATH"] = str(private_bin) + os.pathsep + sim_env["PATH"]
        record["simulator"] = {key: sim_env[key] for key in (
            "ROBOTWIN_PATH", "ROBOTWIN_PYTHON", "TORCH_EXTENSIONS_DIR",
            "SAPIEN_DISABLE_RAY_TRACING", "ROBOTWIN_ENABLE_PLANNER_FALLBACK")}
        record["simulator"]["TORCH_CUDA_ARCH_LIST"] = sim_env.get("TORCH_CUDA_ARCH_LIST", "auto")
        record["simulator"]["gpu"] = args.sim_gpu
        record["status"] = "running"
        for mode in args.modes:
            for task in tasks:
                job = args.output / mode / task
                if sha256(HERE / "step_limits.yml") != record["step_limits"]["sha256"]:
                    raise RuntimeError("Step-limit overrides changed during evaluation")
                job.mkdir(parents=True)
                runtime = job / "runtime"
                sim_env["ROBOTWIN_RUNTIME_ROOT"] = str(runtime)
                record["current"] = {"task": task, "mode": mode}
                save(args.output / "run.json", record)
                command = ["bash", str(HERE / "single_eval.sh"), task, mode, args.output.name,
                           args.sim_gpu, str(args.port), "127.0.0.1"]
                print(f"Running {mode}/{task}, episodes={args.episodes}; log={job / 'client.log'}", flush=True)
                with (job / "client.log").open("w") as log:
                    client = subprocess.Popen(command, cwd=ROOT, env=sim_env, stdout=log,
                                              stderr=subprocess.STDOUT, start_new_session=True)
                    code = client.wait()
                client = None
                result = validate_result((job / "client.log").read_text(), runtime, args.episodes, code)
                result.update(task=task, mode=mode)
                save(job / "result.json", result)
                record["results"].append(result)
                save(args.output / "run.json", record)
                print(f"{mode}/{task}: {result['successes']}/{result['completed_episodes']}; "
                      f"complete={result['complete']}; details={job / 'result.json'}", flush=True)
                if not result["complete"]:
                    raise RuntimeError(f"Incomplete evaluation: {mode}/{task}; see client.log")
        record["status"] = "complete"
        record["full_50_task_benchmark"] = len(tasks) == 50 and set(args.modes) == {"demo_clean", "demo_randomized"} and args.episodes == 100
        record["macro_success_rate"] = {
            mode: sum(row["success_rate"] for row in record["results"] if row["mode"] == mode) / len(tasks)
            for mode in args.modes}
        print(json.dumps({key: record[key] for key in ("status", "full_50_task_benchmark", "macro_success_rate")}), flush=True)
        return 0
    except BaseException as exc:
        record.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for owned in (client, server):
            if owned is not None and owned.poll() is None:
                os.killpg(owned.pid, signal.SIGTERM)
                try:
                    owned.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(owned.pid, signal.SIGKILL)
                    owned.wait()
        if server_log is not None:
            server_log.close()
        save(args.output / "run.json", record)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    raise SystemExit(main())
