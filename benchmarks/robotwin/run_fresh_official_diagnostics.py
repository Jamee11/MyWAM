"""Run unmodified official launchers on a fresh, verified RoboTwin environment."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.robotwin.standard_eval import save, validate_result  # noqa: E402


def main():
    work = ROOT / "outputs/robotwin_eval/rebuild_official_20261018"
    runtime_source = ROOT / ".external/robotwin-official-20261018/RoboTwin"
    evaluation_source = runtime_source.parent / "OpenWAM"
    python = ROOT / ".conda/robotwin-official/bin/python"
    required = [work / "pip_freeze.txt", work / "runtime_smoke.json"] + [work / "assets" / (n + ".manifest.json") for n in
                ["embodiments.zip", "objects.zip", "background_texture.zip"]]
    parity_result = work / "action_parity/result.json"
    deadline = time.monotonic() + 4 * 3600
    while not all(p.is_file() for p in required):
        if time.monotonic() > deadline:
            raise TimeoutError("Official environment/assets did not finish; inspect setup logs")
        time.sleep(5)
    # The action comparison borrows this single-session policy endpoint first.
    # Wait for its process to close/reset before starting a native client.
    while True:
        comparing = False
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                command = (proc / "cmdline").read_bytes().split(b"\0")
                comparing |= len(command) > 2 and command[2].endswith(b"compare_endpoints.py")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
        if parity_result.is_file() and not comparing:
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Action parity did not finish; do not share a single-session endpoint")
        time.sleep(5)
    commit = subprocess.check_output(["git", "-C", str(runtime_source), "rev-parse", "HEAD"], text=True).strip()
    if commit != "0aeea2d669c0f8516f4d5785f0aa33ba812c14b4":
        raise ValueError("Official RoboTwin commit mismatch")
    with (work / "configure_asset_paths_from_runner.log").open("w") as log:
        subprocess.run([str(python), "script/update_embodiment_config_path.py"], cwd=runtime_source,
                       env={**os.environ, "PYTHONNOUSERSITE": "1"}, stdout=log, stderr=subprocess.STDOUT, check=True)
    for side in ("left", "right"):
        config = runtime_source / f"assets/embodiments/aloha-agilex/curobo_{side}.yml"
        if not config.is_file() or str(runtime_source) not in config.read_text():
            raise ValueError("Official asset path generation did not bind the fresh checkout")
    assets = [json.loads(p.read_text()) for p in required[2:]]
    manifest = {"source": "fresh upstream git clone", "commit": commit,
                "asset_revision": assets[0]["revision"],
                "asset_archives": [{k: v[k] for k in ["archive", "size", "sha256"]} for v in assets],
                "environment": json.loads((work / "environment_installed.json").read_text()),
                "rt_smoke": json.loads((work / "runtime_smoke.json").read_text()),
                "pytorch3d": "compiled and installed; native RGB configs have pointcloud=false"}
    save(runtime_source / "OPENWAM_OFFICIAL_MANIFEST.json", manifest)
    out = work / "official_diagnostic100"
    out.mkdir(exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith("ROBOTWIN_") and k != "POLICY_CONFIG_PATH"}
    env.update(ROBOTWIN_PATH=str(runtime_source), ROBOTWIN_PYTHON=str(python),
               HF_ENDPOINT="https://hf-mirror.com", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", CUDA_HOME="/remote-home/jinminghao/miniconda3/envs/nvcu121_cuda-12.4",
               TORCH_CUDA_ARCH_LIST="8.9", MAX_JOBS="4", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               TORCH_EXTENSIONS_DIR=str(ROOT / ".cache/robotwin/torch_extensions_official"))
    bin_dir = out / "bin"
    bin_dir.mkdir()
    (bin_dir / "ffmpeg").symlink_to("/usr/bin/ffmpeg")
    env["PATH"] = os.pathsep.join([str(bin_dir), str(python.parent), env["CUDA_HOME"] + "/bin", env["PATH"]])
    env["PYTHONPATH"] = str(runtime_source / "envs/curobo/src")
    record = {"status": "running", "protocol": "unmodified official single_eval.sh/wrapper and upstream native100",
              "sources": {"robotwin": commit, "openwam": subprocess.check_output(
                  ["git", "-C", str(evaluation_source), "rev-parse", "HEAD"], text=True).strip()},
              "setup": manifest, "results": []}
    save(out / "run.json", record)
    from benchmarks.utils.transport import WSPolicyClient
    while True:
        try:
            with WSPolicyClient("ws://127.0.0.1:19400", timeout=10, open_timeout=5) as ws:
                ws.ping()
            break
        except (OSError, TimeoutError):
            if time.monotonic() > deadline:
                raise
            time.sleep(2)
    # This single official PolicyServer has one action deque: one native client
    # at a time. Do not multiplex concurrent tasks onto this endpoint.
    for task in ["place_dual_shoes", "place_shoe", "scan_object", "handover_block", "click_bell"]:
        for mode in ["demo_clean", "demo_randomized"]:
            job = out / mode / task
            job.mkdir(parents=True)
            runtime = job / "runtime"
            env["ROBOTWIN_RUNTIME_ROOT"] = str(runtime)
            command = ["bash", str(evaluation_source / "benchmarks/robotwin/single_eval.sh"),
                       task, mode, "fresh_official", "3", "19400", "127.0.0.1"]
            record["current"] = {"task": task, "mode": mode}
            save(out / "run.json", record)
            print(f"START native100 {mode}/{task}", flush=True)
            with (job / "client.log").open("w") as log:
                proc = subprocess.Popen(command, cwd=evaluation_source, env=env, stdout=log,
                                        stderr=subprocess.STDOUT)
                record["native_client_pid"] = proc.pid
                save(out / "run.json", record)
                code = proc.wait()
            result = validate_result((job / "client.log").read_text(), runtime, 100, code)
            result.update(task=task, mode=mode)
            save(job / "result.json", result)
            record["results"].append(result)
            save(out / "run.json", record)
            if not result["complete"]:
                record["status"] = "incomplete"
                save(out / "run.json", record)
                raise RuntimeError(f"Native diagnostic failed: {mode}/{task}; inspect client.log")
            print(f"END {mode}/{task} {result['successes']}/100", flush=True)
    record["status"] = "complete"
    save(out / "run.json", record)


if __name__ == "__main__":
    main()
