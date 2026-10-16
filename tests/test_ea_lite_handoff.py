"""Portable paths and launcher routing; no torch imports, network or GPU execution."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.relocate_ea_lite_manifest import relocate

ROOT = Path(__file__).resolve().parents[1]


def copied_data(tmp_path):
    raw, data = tmp_path / "new_raw", tmp_path / "new_data"
    (raw / "task/data").mkdir(parents=True)
    (data / "labels/task").mkdir(parents=True)
    episode = raw / "task/data/episode0.hdf5"
    episode.write_bytes(b"copied source")
    labels = data / "labels/task/episode0.npz"
    labels.write_bytes(b"copied labels")
    (data / "normalization_stats.npy").write_bytes(b"copied stats")
    source = dict(contract="openwam_alpha_eef32_ea_lite_v1", source_root="/old/raw",
                  normalization_stats_path="/old/data/normalization_stats.npy",
                  tasks=[dict(name="task", data_root="/old/raw/task/data", episodes=[dict(
                      path="/old/raw/task/data/episode0.hdf5", size=episode.stat().st_size,
                      mtime_ns=episode.stat().st_mtime_ns, labels="/old/data/labels/task/episode0.npz",
                      labels_sha256=hashlib.sha256(labels.read_bytes()).hexdigest())])])
    manifest = data / "manifest.json"
    manifest.write_text(json.dumps(source))
    return manifest, raw, data, source


def test_relocate_copied_tree_without_rewriting_content_bindings(tmp_path):
    manifest, raw, data, source = copied_data(tmp_path)
    original = manifest.read_bytes()
    output = data / "manifest.relocated.json"
    result = relocate(manifest, raw, data, output)
    moved = json.loads(output.read_text())
    assert result["episodes"] == 1 and manifest.read_bytes() == original
    assert moved["source_root"] == str(raw)
    assert moved["tasks"][0]["data_root"] == str(raw / "task/data")
    episode = moved["tasks"][0]["episodes"][0]
    assert episode["path"] == str(raw / "task/data/episode0.hdf5")
    assert episode["labels"] == str(data / "labels/task/episode0.npz")
    assert episode["labels_sha256"] == source["tasks"][0]["episodes"][0]["labels_sha256"]
    with pytest.raises(FileExistsError):
        relocate(manifest, raw, data, output)


def test_relocate_rejects_changed_source_and_missing_labels(tmp_path):
    manifest, raw, data, _ = copied_data(tmp_path)
    episode = raw / "task/data/episode0.hdf5"
    stamp = episode.stat()
    os.utime(episode, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 10))
    with pytest.raises(ValueError, match="size/mtime"):
        relocate(manifest, raw, data, data / "bad.json")
    assert not (data / "bad.json").exists()
    os.utime(episode, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    (data / "labels/task/episode0.npz").unlink()
    with pytest.raises(FileNotFoundError):
        relocate(manifest, raw, data, data / "bad.json")


@pytest.mark.parametrize("mode,count,profile", [("1", "4", "80"), ("0", "8", "80"), ("0", "2", "40")])
def test_launchers_route_without_loading_model_or_allocating_gpu(tmp_path, mode, count, profile):
    scripts, binaries = tmp_path / "scripts", tmp_path / ".venv/bin"
    scripts.mkdir(); binaries.mkdir(parents=True)
    for name in ("train_ea_lite_4gpu.sh", "train_ea_lite_multigpu.sh"):
        shutil.copy2(ROOT / "scripts" / name, scripts / name)
    for program in ("python", "torchrun"):
        stub = binaries / program
        stub.write_text(f'#!/bin/sh\nprintf "%s\\n" "{program}: $*" "CUDA=$CUDA_VISIBLE_DEVICES NPROC=$NPROC_PER_NODE"\n')
        stub.chmod(0o755)
    env = dict(os.environ, CHECK_ONLY=mode, EA_NUM_GPUS=count, EA_GPU_PROFILE=profile,
               EA_GPU_NAME="H200", CUDA_VISIBLE_DEVICES=",".join(map(str, range(int(count)))))
    result = subprocess.run(["bash", str(scripts / "train_ea_lite_multigpu.sh")], env=env,
                            text=True, capture_output=True, check=True)
    if mode == "1":
        assert "--cfg job --resolve" in result.stdout and "torchrun:" not in result.stdout
        assert "CUDA= NPROC=" in result.stdout
    else:
        assert f"--num-gpus {count}" in result.stdout and "--gpu-name H200" in result.stdout
        assert f"--nproc_per_node={count}" in result.stdout and "--nnodes=1" in result.stdout
        assert "scripts/debug_ea_lite.py" in result.stdout
        if profile == "40":
            assert "training.offload_optimizer_device=cpu" in result.stdout


def test_install_script_removes_old_editable_path_using_explicit_tools(tmp_path):
    scripts, binaries = tmp_path / "scripts", tmp_path / ".venv/bin"
    scripts.mkdir(); binaries.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/install_ea_environment.sh", scripts)
    (tmp_path / "requirements-ea-lock.txt").write_text("torch==2.7.1+cu128\n-e file:///old/OpenWAM\npytest==9.0.3\n")
    python = binaries / "python"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "python: $*"\n'); python.chmod(0o755)
    uv = tmp_path / "uv"
    uv.write_text('#!/bin/sh\nprintf "%s\\n" "uv: $*"\n'); uv.chmod(0o755)
    result = subprocess.run(["bash", str(scripts / "install_ea_environment.sh")],
                            env=dict(os.environ, UV_BIN=str(uv)), text=True, capture_output=True, check=True)
    assert "-e ." in result.stdout and "-r outputs/setup/requirements-ea-relocatable.txt" in result.stdout
    assert "-e file:" not in (tmp_path / "outputs/setup/requirements-ea-relocatable.txt").read_text()
