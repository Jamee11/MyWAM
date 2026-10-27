#!/usr/bin/env python3
"""Package the current Sidecar code and its exact paired Aloha data; reuse the tar writer."""
import argparse
import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from package_training_transfer import ROOT, write_archive

DATASET_NAME = "robotwin2_official_clean_ea_rsig_v3_4350_wrist020_schema5_20261015"


def collect_data(manifest_path):
    manifest_path = Path(manifest_path)
    data = json.loads(manifest_path.read_text())
    raw_root = Path(data["source_root"])
    metadata = manifest_path.parent
    if data.get("contract") != "openwam_alpha_eef32_ea_lite_v1" or (
            len(data["tasks"]), data["total_episodes"], data["plan_steps"], data["video_anchors"]) != (
            50, 2500, [8, 16, 32], list(range(0, 33, 4))):
        raise ValueError("Expected the audited Aloha50 Sidecar dataset")
    files, expected = [], {}
    for task in data["tasks"]:
        if task.get("startup_metadata", {}).get("version") != 1:
            raise ValueError("Prepared startup metadata is required")
        for episode in task["episodes"]:
            source = Path(episode["path"])
            stat = source.stat()
            if (stat.st_size, stat.st_mtime_ns) != (episode["size"], episode["mtime_ns"]):
                raise ValueError(f"Changed source: {source}")
            name = str(Path(DATASET_NAME) / "raw/aloha" / source.relative_to(raw_root))
            files.append((source, name))
            label = Path(episode["labels"])
            name = str(Path(DATASET_NAME) / "openwam_sidecar/aloha50" / label.relative_to(metadata))
            files.append((label, name))
            expected[name] = episode["labels_sha256"]
        directory = Path(task["data_root"]).parent
        for source in sorted((directory / "instructions").glob("*.json")):
            files.append((source, str(Path(DATASET_NAME) / "raw/aloha" / source.relative_to(raw_root))))
        for source in (directory / "scene_info.json", directory / "data/scene_info.json"):
            if source.is_file():
                files.append((source, str(Path(DATASET_NAME) / "raw/aloha" / source.relative_to(raw_root))))
    stats_name = f"{DATASET_NAME}/openwam_sidecar/aloha50/normalization_stats.npy"
    files.extend([(Path(data["normalization_stats_path"]), stats_name),
                  (manifest_path, f"{DATASET_NAME}/openwam_sidecar/aloha50/manifest.source.json")])
    expected[stats_name] = data["normalization_stats_sha256"]
    names = [name for _, name in files]
    if len(names) != len(set(names)) or sum(len(t["episodes"]) for t in data["tasks"]) != 2500:
        raise ValueError("Duplicate paths or inconsistent episode count")
    return data, files, expected


def package(output, manifest):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data, data_files, expected = collect_data(manifest)
    required_bytes = sum(source.stat().st_size for source, _ in data_files)
    if shutil.disk_usage(output).free < required_bytes + (1 << 30):
        raise OSError("Need enough archive space for the uncompressed data plus 1GiB")
    code_files = []
    for directory in ("openwam", "configs", "scripts", "assets/ea_robotwin", "tests"):
        for source in sorted((ROOT / directory).rglob("*")):
            if not source.is_file() or "__pycache__" in source.parts or source.suffix in {".pyc", ".log"}:
                continue
            if source.stat().st_size > 5_000_000:
                raise ValueError(f"Unexpected large code file: {source}")
            code_files.append((source, str(Path("OpenWAM") / source.relative_to(ROOT))))
    for name in ("AGENTS.md", "LICENSE", "EA_RSIG_DIT4DIT_LICENSE", "EA_RSIG_README.md", "README.md",
                 "pyproject.toml", "requirements-ea-lock.txt", "requirements-ci.txt", "MODIFICATION_LOG.md",
                 "docs/思考与隐患.md", "docs/BASELINE_CLUSTER_TRANSFER_20261015.md",
                 "docs/ROBOTWIN_SIDECAR_ISOLATED_WARMUP_20261021.md",
                 "docs/SIDECAR_ISOLATED_CLUSTER_TRANSFER_20261021.md"):
        code_files.append((ROOT / name, f"OpenWAM/{name}"))
    names = ["openwam_sidecar_isolated_code.tar.gz", "openwam_sidecar_aloha50_data.tar.gz"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        code = pool.submit(write_archive, output / names[0], code_files)
        dataset = pool.submit(write_archive, output / names[1], data_files, expected=expected)
        reports = [code.result(), dataset.result()]
    (output / "SHA256SUMS").write_text("".join(f"{r['sha256']}  {r['archive']}\n" for r in reports))
    shutil.copyfile(ROOT / "docs/SIDECAR_ISOLATED_CLUSTER_TRANSFER_20261021.md", output / "README_TRANSFER.md")
    info = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                source_manifest=str(Path(manifest).resolve()),
                source_manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                tasks=50, episodes=2500, frames=data["total_frames"],
                raw_bytes=sum(e["size"] for t in data["tasks"] for e in t["episodes"]),
                dataset_name=DATASET_NAME, project_destination="/home/mingfeihangroup/xiaoming/projects/OpenWAM",
                data_destination=f"/home/mingfeihangroup/xiaoming/datasets/{DATASET_NAME}",
                foundation="Reuse existing ckpts/OpenWAM-Alpha-Pretrain-Foundation-Model; not packaged",
                venv="Reuse target project/.venv; not packaged", archives=[{k:r[k] for k in (
                    "archive", "size_bytes", "file_count", "sha256")} for r in reports])
    (output / "TRANSFER_INFO.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/sidecar/aloha_4350/manifest.ready.json")
    package(**vars(parser.parse_args()))
