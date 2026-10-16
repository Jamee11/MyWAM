#!/usr/bin/env python3
"""Read-only readiness gates. No torch import, model allocation, or GPU training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess


def check_gpus(num_gpus=4, gpu_profile=80, gpu_name="A100"):
    if num_gpus < 1 or not gpu_name or gpu_profile not in (40, 80):
        raise ValueError("Need positive GPU count, GPU name and profile 40/80")
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    count = "four" if num_gpus == 4 else str(num_gpus)
    if len(selected) != num_gpus or len(set(selected)) != num_gpus or not all(i.isdigit() for i in selected):
        raise ValueError(f"Select {count} distinct physical GPU indices")
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
                                      "--format=csv,noheader,nounits"], text=True)
    devices = {row[0]: row for line in output.splitlines() if (row := [x.strip() for x in line.split(",")])}
    minimum_total, minimum_free = (75000, 70000) if gpu_profile == 80 else (39000, 35000)
    for index in selected:
        if index not in devices:
            raise ValueError(f"Selected GPU index {index} is absent from nvidia-smi")
        row = devices[index]
        if gpu_name not in row[1] or int(row[2]) < minimum_total or int(row[3]) < minimum_free:
            raise ValueError(f"GPU {index} does not meet {gpu_name}/profile-{gpu_profile} free-memory gate: {row}")
    return selected


def check_four_gpus(gpu_profile=80):
    return check_gpus(4, gpu_profile)


def check(manifest, foundation, require_four_gpus=False, gpu_profile=80, num_gpus=None, gpu_name="A100", data_check="full"):
    if data_check not in ("fast", "full"):
        raise ValueError("data_check must be fast or full")
    if require_four_gpus and num_gpus not in (None, 4):
        raise ValueError("--require-four-gpus conflicts with --num-gpus")
    required = 4 if require_four_gpus else num_gpus
    if required is not None:
        check_gpus(required, gpu_profile, gpu_name)
    manifest, foundation = Path(manifest).resolve(), Path(foundation).resolve()
    data = json.loads(manifest.read_text())
    if (data.get("contract"), data.get("split"), data.get("plan_steps"), data.get("video_anchors")) != (
            "openwam_alpha_eef32_ea_lite_v1", "Clean", [8, 16, 32], list(range(0, 33, 4))):
        raise ValueError("Wrong Lite dataset contract")
    stats = Path(data["normalization_stats_path"])
    if hashlib.sha256(stats.read_bytes()).hexdigest() != data["normalization_stats_sha256"]:
        raise ValueError("Normalization artifact changed")
    if data_check == "fast" and any(task.get("startup_metadata", {}).get("version") != 1 for task in data["tasks"]):
        raise ValueError("Run scripts/cache_ea_lite_startup.py offline before prepared startup")
    total = 0
    for task in data["tasks"]:
        for episode in task["episodes"]:
            if data_check == "full":
                path = Path(episode["path"])
                stat = path.stat()
                if (stat.st_size, stat.st_mtime_ns) != (episode["size"], episode["mtime_ns"]):
                    raise ValueError(f"Source changed: {path}")
                labels = Path(episode["labels"])
                if hashlib.sha256(labels.read_bytes()).hexdigest() != episode["labels_sha256"]:
                    raise ValueError(f"Sidecar changed: {labels}")
            total += 1
    if not total:
        raise ValueError("Empty dataset")
    proof = json.loads((foundation / "download_verification.json").read_text())
    name = "checkpoint_step_154000.safetensors"
    record = proof["files"][name]
    weight = foundation / name
    if not record["verified_upstream"] or weight.stat().st_size != record["size"]:
        raise ValueError("Unverified or incomplete foundation")
    with weight.open("rb") as stream:
        header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
    for key, axis in (("action_backbone.action_encoder.weight", 1),
                      ("action_backbone.action_decoder.weight", 0), ("proprio_encoder.weight", 1)):
        if header[key]["shape"][axis] != 80:
            raise ValueError("Wrong Foundation head width")
    report = dict(status="cpu_preflight_passed", tasks=len(data["tasks"]), episodes=total,
                  frames=data["total_frames"], foundation_head_width=80,
                  gpu_training="not_run", gpu_memory_and_speed="unverified")
    report["data_check"] = data_check
    report["full_dataset_integrity_audited"] = data_check == "full"
    if required is not None:
        # Recheck after the data audit; this is an availability gate, not a reservation.
        check_gpus(required, gpu_profile, gpu_name)
        report["gpu_availability_gate"] = dict(count=required, name=gpu_name,
                                               status="passed; capacity still requires actual updates")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--foundation", required=True)
    parser.add_argument("--require-four-gpus", action="store_true")
    parser.add_argument("--gpu-profile", type=int, choices=(40, 80), default=80)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--gpu-name", default="A100")
    parser.add_argument("--data-check", choices=("fast", "full"), default="full")
    args = parser.parse_args()
    print(json.dumps(check(**vars(args)), indent=2))
