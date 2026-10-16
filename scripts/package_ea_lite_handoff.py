#!/usr/bin/env python3
"""Package the current checkout, Lite sidecars and small evidence; no weights/raw data/venv."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def package(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT)
    paths = {ROOT / p.decode() for p in names.split(b"\0") if p}
    for directory in ("outputs/ea_lite/data", "outputs/ea_lite/readiness", "outputs/batch_benchmark"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and path.suffix in (".json", ".log", ".yaml", ".err", ".png", ".npz", ".npy", ".sh"):
                paths.add(path)
    paths.update((ROOT / "outputs/setup").glob("*.json"))
    # Gitlinks are directories. This bundle targets Alpha/Wan, not Cosmos submodule reproduction.
    paths = sorted(p for p in paths if p.is_file() and p != output)
    required = ["AGENTS.md", "scripts/train_ea_lite_multigpu.sh", "scripts/train_ea_lite_4gpu.sh",
                "outputs/ea_lite/data/manifest.json", "outputs/ea_lite/data/manifest.ready.json",
                "scripts/cache_ea_lite_startup.py", "assets/ea_robotwin/role_registry_lite.json"]
    if any(ROOT / name not in paths for name in required):
        raise ValueError("Missing required handoff code/data assets")
    inventory = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "x:gz", format=tarfile.PAX_FORMAT, compresslevel=1) as archive:
        for path in paths:
            archive.add(path, arcname=str(Path("OpenWAM") / path.relative_to(ROOT)), recursive=False)
    digest = hashlib.sha256()
    with output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    report = dict(archive=str(output), size_bytes=output.stat().st_size, sha256=digest.hexdigest(),
                  scope="Alpha checkout + Lite labels/stats + small evidence; excludes weights, raw HDF5, venv, .git and Cosmos gitlinks",
                  files=inventory)
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    package(parser.parse_args().output)
