"""Preload an independent full run, then hand over GPUs after verified normal completion."""

import argparse
import json
import math
import os
import signal
import socket
import struct
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def latest_metrics(run, rank):
    path = run / f"metrics_rank{rank}.jsonl"
    if not path.exists():
        return None
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 8000))
        for line in reversed(stream.read().decode(errors="replace").splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    return None


def complete_weights(path):
    if not path.exists():
        return False
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(length))
    ends = [v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"]
    return bool(ends) and path.stat().st_size == 8 + length + max(ends)


def final_evidence(run):
    rows = [latest_metrics(run, rank) for rank in range(4)]
    weights = run / "checkpoint_step_42960.safetensors"
    complete = all(
        row and (row["micro_step"], row["optimizer_step"]) == (42960, 21480)
        and all(math.isfinite(v) for v in row["metrics"].values()) for row in rows
    ) and complete_weights(weights)
    return dict(complete=complete, final_weights=str(weights), rank_metrics=rows)


def predecessor_complete(source):
    launch = json.loads((source / "launch.json").read_text())
    stat = Path(f"/proc/{launch['pid']}/stat")
    if stat.exists() and stat.read_text().split()[21] == launch["start_ticks"]:
        return None
    status = json.loads((source / "job_status.json").read_text())
    if "returncode" not in status:
        return None  # Allow the existing monitor to reap its child and flush the final status.
    if status["returncode"] != 0:
        raise RuntimeError(f"Predecessor exited with {status['returncode']}; no automatic replacement")
    evidence = final_evidence(Path(launch["output_run"]))
    if not evidence["complete"]:
        raise RuntimeError("Predecessor exited zero but final four-rank counters/weights are incomplete")
    return dict(status="complete", utc=datetime.now(timezone.utc).isoformat(), **evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    root = parser.parse_args().root.resolve()
    plan = json.loads((root / "plan.json").read_text())
    if not (root / "engineering/acceptance.json").exists() or (root / "launch.json").exists():
        raise RuntimeError("Require passing engineering acceptance and a fresh launch directory")
    acceptance = json.loads((root / "engineering/acceptance.json").read_text())
    if acceptance.get("status") != "pass" or root != Path(plan["root"]):
        raise RuntimeError("Engineering acceptance or output identity mismatch")
    marker = root / "predecessor_complete.json"
    if marker.exists():
        raise RuntimeError("Use a fresh predecessor marker")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, PYTHON_BIN=plan["python"], EA_OUTPUT_DIR=str(root / "runs"),
               EA_HANDOVER_DIR=str(root / "handover"), EA_PREDECESSOR_DONE_FILE=str(marker),
               EA_SIDECAR_MANIFEST=plan["manifest"], MASTER_PORT=str(port),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", CHECK_ONLY="0")
    command = ["bash", "scripts/train_sidecar_ea_grad_anneal_4gpu.sh"]
    with (root / "launch.log").open("xb") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        write_json(root / "launch.json", dict(pid=child.pid, start_ticks=Path(f"/proc/{child.pid}/stat").read_text().split()[21],
                   utc=datetime.now(timezone.utc).isoformat(), command=command, python=plan["python"], master_port=port))
        try:
            while child.poll() is None:
                if not marker.exists():
                    evidence = predecessor_complete(Path(plan["predecessor_root"]))
                    if evidence:
                        write_json(marker, evidence)
                runs = list((root / "runs").glob("*/config.yaml"))
                run = runs[0].parent if len(runs) == 1 else None
                rows = [latest_metrics(run, rank) for rank in range(4)] if run else []
                for row in rows:
                    if not row:
                        continue
                    metrics = row["metrics"]
                    u = (row["micro_step"] - 1) // 2
                    rho = 1.0 if u < 2000 else 1.0 - 0.99 * min((u - 2000) / 500, 1)
                    if not all(math.isfinite(v) for v in metrics.values()):
                        raise RuntimeError("Nonfinite metrics")
                    if abs(metrics.get("ea_shared_gradient_scale", -1) - rho) > 1e-6:
                        raise RuntimeError("Actual EA gradient schedule differs from the approved optimizer-step recipe")
                status = "initializing_cpu_model"
                if (root / "handover/cpu_ready.json").exists():
                    contract = json.loads((root / "handover/cpu_contract.json").read_text())
                    expected = dict(dataset_windows=549775, world_size=4, batch_per_gpu=32, accumulation=2,
                                    backbone_warmup_updates=2000, aux_shared_gradient_scale=0.01,
                                    aux_gradient_anneal_updates=500)
                    if contract != expected:
                        raise RuntimeError(f"Actual CPU model/dataset contract differs: {contract}")
                    status = "cpu_ready_waiting_for_predecessor_and_gpu0_3"
                if (root / "handover/gpu_release.json").exists():
                    status = "preparing_deepspeed_on_gpu0_3"
                if rows and all(row and row["micro_step"] >= 2 for row in rows):
                    status = "training"
                    if not (root / "startup_acceptance.json").exists():
                        write_json(root / "startup_acceptance.json", dict(status="pass_first_real_update", rank_metrics=rows,
                                   output_run=str(run), utc=datetime.now(timezone.utc).isoformat(),
                                   real_2k_boundary_observed=False, closed_loop_performance_tested=False))
                write_json(root / "job_status.json", dict(utc=datetime.now(timezone.utc).isoformat(), status=status,
                           pid=child.pid, output_run=str(run) if run else None, latest=rows[0] if rows else None))
                time.sleep(15)
            runs = list((root / "runs").glob("*/config.yaml"))
            final = final_evidence(runs[0].parent) if len(runs) == 1 else dict(complete=False)
            write_json(root / "job_status.json", dict(utc=datetime.now(timezone.utc).isoformat(),
                       status="complete_training_only" if child.returncode == 0 and final["complete"] else "failed_or_incomplete",
                       returncode=child.returncode, **final, closed_loop_performance_tested=False))
        except BaseException as error:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)  # Only this supervisor's newly created child group.
                child.wait(timeout=60)
            write_json(root / "job_status.json", dict(status="failed_supervision", error=str(error), returncode=child.returncode,
                       utc=datetime.now(timezone.utc).isoformat()))
            raise


if __name__ == "__main__":
    main()
