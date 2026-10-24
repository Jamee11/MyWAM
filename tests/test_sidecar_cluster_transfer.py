"""Cluster allocation, metrics entry and actual archive preservation without GPUs."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("profile", [None, "train_relational_sidecar_aloha50_frozen_backbone_10epochs"])
def test_cluster_launcher_preserves_slurm_allocation(tmp_path, profile):
    venv = tmp_path / "venv/bin"
    venv.mkdir(parents=True)
    python = venv / "python"
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$CUDA_VISIBLE_DEVICES" "$@"\n')
    python.chmod(0o755)
    env = dict(os.environ, OPENWAM_VENV=str(venv.parent), CUDA_VISIBLE_DEVICES="GPU-slurm-a,GPU-slurm-b",
               CHECK_ONLY="1", OPENWAM_FOUNDATION="/existing/foundation")
    env.pop("OPENWAM_TRAIN_CONFIG", None)
    if profile:
        env["OPENWAM_TRAIN_CONFIG"] = profile
    output = subprocess.check_output(["bash", str(ROOT / "scripts/cluster/train_sidecar_isolated.sh")],
                                     env=env, text=True)
    assert output.splitlines()[0] == "GPU-slurm-a,GPU-slurm-b"
    assert f"--config-name={profile or 'train_relational_sidecar_aloha50_isolated_10epochs'}" in output
    assert "training.finetune_ckpt_path=/existing/foundation" in output
    assert "--cfg\njob\n--resolve" in output


def test_frozen_backbone_gpu8a100_submission_preserves_allocation(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    srun = binaries / "srun"
    srun.write_text('#!/bin/bash\nprintf "%s\\n" "$CUDA_VISIBLE_DEVICES" "$OPENWAM_TRAIN_CONFIG" "$OPENWAM_OUTPUT" "$NPROC_PER_NODE" "$@"\n')
    srun.chmod(0o755)
    smi = binaries / "nvidia-smi"
    smi.write_text('#!/bin/bash\nexit 0\n')
    smi.chmod(0o755)
    script = ROOT / "scripts/cluster/sidecar_frozen_backbone_gpu8a100.sbatch"
    text = script.read_text()
    assert "#SBATCH --partition=GPU-8A100" in text and "#SBATCH --gres=gpu:4" in text
    assert "#SBATCH --time=10-00:00:00" in text and "--nodelist" not in text
    env = dict(os.environ, PATH=f'{binaries}:{os.environ["PATH"]}', OPENWAM_PROJECT=str(ROOT),
               SLURM_JOB_ID="321", CUDA_VISIBLE_DEVICES="GPU-slurm-a,GPU-slurm-b,GPU-slurm-c,GPU-slurm-d")
    for key in ("OPENWAM_OUTPUT", "OPENWAM_CLUSTER_SETUP"):
        env.pop(key, None)
    output = subprocess.check_output(["bash", str(script), "training.max_steps=8"], env=env, text=True)
    assert env["CUDA_VISIBLE_DEVICES"] in output
    assert "train_relational_sidecar_aloha50_frozen_backbone_10epochs" in output
    assert str(ROOT / "outputs/sidecar/aloha50_frozen_backbone2k_job321") in output
    assert "--ntasks=1\n--kill-on-bad-exit=1" in output and "training.max_steps=8" in output


@pytest.mark.parametrize("finite", [True, False])
def test_cluster_metrics_entry(monkeypatch, tmp_path, finite):
    import torch
    from openwam.train.openwam_trainer import OpenWAMTrainer
    from scripts.cluster import train_sidecar_metrics as entry

    original_calls = []
    monkeypatch.setattr(OpenWAMTrainer, "log_step", lambda self, **kwargs: original_calls.append(kwargs))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 123)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 456)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    trainer = SimpleNamespace(_rank=0, accelerator=SimpleNamespace(sync_gradients=True))

    def launch():
        OpenWAMTrainer.log_step(trainer, metrics=dict(ea_isolation_warmup=1, loss_total=1 if finite else float("nan")),
                               global_step=4, opt_step=2, epoch=0, steps_per_sec=2, lr=1e-4, output_path=tmp_path)

    monkeypatch.setitem(sys.modules, "train", SimpleNamespace(main=launch))
    if finite:
        entry.main()
        row = json.loads((tmp_path / "metrics_rank0.jsonl").read_text())
        assert row["optimizer_step"] == 2 and row["metrics"]["ea_isolation_warmup"] == 1
        assert row["peak_allocated_bytes"] == 123 and len(original_calls) == 1
    else:
        with pytest.raises(RuntimeError, match="Nonfinite"):
            entry.main()
        assert not (tmp_path / "metrics_rank0.jsonl").exists()


def test_archive_dereferences_preserves_nanoseconds_and_hashes(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/cluster"))
    package = importlib.import_module("package_sidecar_transfer")
    source = tmp_path / "source.hdf5"
    source.write_bytes(b"audited hdf5 fixture")
    stamp = 1792067570112390011
    os.utime(source, ns=(stamp, stamp))
    link = tmp_path / "link.hdf5"
    link.symlink_to(source)
    destination = tmp_path / "data.tar.gz"
    name = "dataset/raw/aloha/task/data/episode0.hdf5"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    report = package.write_archive(destination, [(link, name)], expected={name: digest})
    assert report["sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    with tarfile.open(destination) as archive:
        member = archive.getmembers()[0]
        assert member.isfile() and not member.issym() and not member.islnk()
        assert member.pax_headers["mtime"] == "1792067570.112390011"
        assert archive.extractfile(member).read() == source.read_bytes()
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    subprocess.run(["tar", "-xzf", str(destination), "-C", str(extracted)], check=True)
    assert (extracted / name).stat().st_mtime_ns == stamp


def test_collect_rejects_wrong_data_contract(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/cluster"))
    package = importlib.import_module("package_sidecar_transfer")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(source_root="/old", tasks=[], total_episodes=0,
                                        plan_steps=[8, 16, 32], video_anchors=list(range(0, 33, 4)))))
    with pytest.raises(ValueError, match="Aloha50"):
        package.collect_data(manifest)
