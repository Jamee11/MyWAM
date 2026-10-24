"""Prepared reader parity and lazy validation on real tiny HDF5s; CPU only."""

import hashlib
import io
import json
import random
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image

from openwam.dataloader.ea_rsig_lite import EARSIGLiteDataset
from scripts.cache_ea_lite_startup import prepare
from scripts.preflight_ea_lite import check


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "raw/task/demo_clean"
    (root / "data").mkdir(parents=True)
    (root / "instructions").mkdir()
    (root / "instructions/episode0.json").write_text(json.dumps({"seen": ["pick object", "move object"]}))
    (root / "scene_info.json").write_text(json.dumps({"episode_0": {"active_arm": "left"}}))
    episode = root / "data/episode0.hdf5"
    n = 40
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), "red").save(buffer, format="JPEG")
    jpeg = np.frombuffer(buffer.getvalue(), np.uint8)
    with h5py.File(episode, "w") as file:
        for camera in ("head_camera", "left_camera", "right_camera"):
            images = file.create_dataset(f"observation/{camera}/rgb", (n,), dtype=h5py.vlen_dtype(np.uint8))
            for row in range(n):
                images[row] = jpeg
        for arm in ("left", "right"):
            pose = np.zeros((n, 7))
            pose[:, 0] = np.arange(n) / 100
            pose[:, 3] = 1
            file[f"endpose/{arm}_endpose"] = pose
            file[f"endpose/{arm}_gripper"] = np.ones(n) * 0.5
    labels = tmp_path / "labels.npz"
    np.savez(
        labels,
        targets=np.zeros((n, 130), np.float32),
        effectors=np.zeros((n, 2, 10), np.float32),
        arm_present=np.ones((n, 2), np.float32),
    )
    stats = tmp_path / "stats.npy"
    np.save(stats, dict(eef=dict(mean=np.zeros(20), std=np.ones(20), min=-np.ones(20), max=np.ones(20))))
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    stat = episode.stat()
    manifest = dict(
        contract="openwam_alpha_eef32_ea_lite_v1",
        split="Clean",
        source_quaternion_order="wxyz",
        ea_context="observed_endpose_plus_local_x_0.12m_TCP",
        plan_steps=[8, 16, 32],
        video_anchors=list(range(0, 33, 4)),
        normalization_stats_path=str(stats),
        normalization_stats_sha256=sha(stats),
        total_frames=n,
        tasks=[
            dict(
                name="task",
                data_root=str(root / "data"),
                episodes=[
                    dict(
                        path=str(episode),
                        frames=n,
                        size=stat.st_size,
                        mtime_ns=stat.st_mtime_ns,
                        labels=str(labels),
                        labels_sha256=sha(labels),
                    )
                ],
            )
        ],
    )
    source, ready = tmp_path / "manifest.json", tmp_path / "ready.json"
    source.write_text(json.dumps(manifest))
    prepare(source, ready)
    foundation = tmp_path / "foundation"
    foundation.mkdir()
    header = {
        k: {"shape": shape}
        for k, shape in (
            ("action_backbone.action_encoder.weight", [32, 80]),
            ("action_backbone.action_decoder.weight", [80, 32]),
            ("proprio_encoder.weight", [32, 80]),
        )
    }
    data = json.dumps(header).encode()
    weight = foundation / "checkpoint_step_154000.safetensors"
    weight.write_bytes(len(data).to_bytes(8, "little") + data)
    (foundation / "download_verification.json").write_text(
        json.dumps(dict(files={weight.name: dict(verified_upstream=True, size=weight.stat().st_size)}))
    )
    return source, ready, episode, labels, foundation


def test_prepared_constructor_opens_no_hdf5_and_does_not_discover_files(prepared, monkeypatch):
    _, ready, _, _, _ = prepared

    def forbidden(*args, **kwargs):
        raise AssertionError("startup must not scan raw HDF5s")

    import openwam.dataloader.robotwin as official

    monkeypatch.setattr(official.h5py, "File", forbidden)
    monkeypatch.setattr(official.glob, "glob", forbidden)
    monkeypatch.setattr(official.os, "listdir", forbidden)
    dataset = EARSIGLiteDataset(ready, startup_mode="prepared")
    assert len(dataset) == 39 and dataset.readers[0]._obs_image_size == (24, 32)
    assert dataset.readers[0]._instructions["episode0.json"]["seen"] == ["pick object", "move object"]


def test_prepared_and_scan_paths_have_identical_real_samples(prepared):
    _, ready, _, _, _ = prepared
    scanned = EARSIGLiteDataset(ready, startup_mode="scan")
    cached = EARSIGLiteDataset(ready, startup_mode="prepared")
    assert cached.readers[0]._window_index == scanned.readers[0]._window_index
    for index in (0, 20, 38):
        random.seed(123)
        expected = scanned[index]
        random.seed(123)
        actual = cached[index]
        assert expected.keys() == actual.keys()
        for key in expected:
            if key in ("video", "first_frame_image"):
                assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(expected[key], actual[key]))
            elif hasattr(expected[key], "numpy"):
                np.testing.assert_array_equal(expected[key].numpy(), actual[key].numpy())
            else:
                assert expected[key] == actual[key]
    assert actual["active_arm"] == "left" and int(actual["action_mask"].any(-1).sum()) == 1


def test_source_check_is_lazy_and_rejects_changed_episode_on_read(prepared):
    _, ready, episode, _, _ = prepared
    dataset = EARSIGLiteDataset(ready, startup_mode="prepared")
    with episode.open("ab") as stream:
        stream.write(b"changed")
    # Startup itself did not scan the source; first use still rejects a stale binding.
    with pytest.raises(ValueError, match="Source episode changed"):
        dataset[0]


def test_sidecar_hash_and_stats_check_remain_active(prepared):
    _, ready, _, labels, _ = prepared
    dataset = EARSIGLiteDataset(ready, startup_mode="prepared")
    labels.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Corrupt Lite sidecar"):
        dataset[0]
    stats = Path(json.loads(ready.read_text())["normalization_stats_path"])
    stats.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Normalization artifact changed"):
        EARSIGLiteDataset(ready, startup_mode="prepared")


def test_missing_prepared_metadata_fails_without_building(prepared):
    source, _, _, _, _ = prepared
    with pytest.raises(ValueError, match="cache_ea_lite_startup"):
        EARSIGLiteDataset(source, startup_mode="prepared")


def test_full_audit_detects_damage_that_fast_preflight_intentionally_defers(prepared):
    _, ready, episode, _, foundation = prepared
    with episode.open("ab") as stream:
        stream.write(b"changed")
    fast = check(ready, foundation, data_check="fast")
    assert not fast["full_dataset_integrity_audited"]
    with pytest.raises(ValueError, match="Source changed"):
        check(ready, foundation, data_check="full")
