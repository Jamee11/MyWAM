"""Materialize a private RoboTwin checkout and its linked assets, without touching the source."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def copy_runtime(source: Path, destination: Path) -> dict:
    source = source.resolve(strict=True)
    destination = destination.absolute()
    if destination.exists():
        raise FileExistsError(f"Refusing to merge into existing checkout: {destination}")
    if source == destination or source in destination.parents:
        raise ValueError("Destination must be outside source checkout")
    if not (source / "script/eval_policy.py").is_file():
        raise ValueError("Source has no script/eval_policy.py")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    tracked = subprocess.check_output(["git", "-C", str(source), "ls-files", "-z"]).decode().split("\0")
    tracked_directories = {str(parent) for name in tracked if name for parent in Path(name).parents}

    def ignore(directory, names):
        excluded = set()
        cache = str((Path(directory) / "__pycache__").relative_to(source))
        if cache not in tracked_directories:
            excluded.add("__pycache__")
        if Path(directory) == source:
            excluded |= {"data", "eval_result", "logs", ".cache"}
        return set(names) & excluded

    def copy_file(src, dst):
        shutil.copy2(src, dst)
        before, after = os.stat(src), os.stat(dst)
        if before.st_size != after.st_size or (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino):
            raise RuntimeError(f"Copy is incomplete or shares source inode: {dst}")
        relative = str(Path(dst).relative_to(destination))
        row = {"path": relative, "size": after.st_size}
        # Hash executable/config sources; bulk assets retain explicit size provenance.
        if Path(dst).suffix in {".py", ".yaml", ".yml", ".urdf", ".srdf", ".json", ".sh"}:
            digest = hashlib.sha256(Path(src).read_bytes()).hexdigest()
            if hashlib.sha256(Path(dst).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Copy hash mismatch: {dst}")
            row["sha256"] = digest
        manifest.append(row)
        if len(manifest) % 500 == 0:
            print(f"copied {len(manifest)} files, {sum(r['size'] for r in manifest):,} bytes", flush=True)
        return dst

    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "-C", str(source), "diff", "--binary"])
    shutil.copytree(source, destination, symlinks=False, ignore=ignore, copy_function=copy_file)
    # Some upstream commits track helpers under data/ and even .pyc files.
    # Preserve all existing tracked files, while excluding untracked run data.
    for name in tracked:
        if name and (source / name).is_file() and not (destination / name).exists():
            (destination / name).parent.mkdir(parents=True, exist_ok=True)
            copy_file(source / name, destination / name)
    links = [str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_symlink()]
    if links:
        raise RuntimeError(f"Unexpected remaining links: {links[:10]}")
    (destination / "OPENWAM_SOURCE_DIFF.patch").write_bytes(diff)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source), "destination": str(destination), "source_commit": head,
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "copy_method": "independent files; all symbolic links materialized",
        "excluded_root_directories": ["data", "eval_result", "logs", ".cache"],
        "file_count": len(manifest), "size_bytes": sum(row["size"] for row in manifest),
        "verification": "all file sizes and distinct inodes; source/config SHA256; bulk asset contents not fully hashed",
        "files": manifest,
    }
    (destination / "OPENWAM_COPY_MANIFEST.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"complete: {len(manifest)} independent files at {destination}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path(".external/RoboTwin"))
    args = parser.parse_args()
    copy_runtime(args.source, args.destination)
