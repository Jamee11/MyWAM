#!/usr/bin/env python3
"""Audit all label hashes plus actual RGB/action/segmentation windows and local plan directions."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check(manifest, output):
    import h5py
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation

    from openwam.dataloader.maniskill_absolute import ManiSkillAbsoluteDataset
    from openwam.dataloader.maniskill_sidecar import ManiSkillSidecarDataset

    started = time.monotonic()
    metadata = json.loads(Path(manifest).read_text())
    baseline_path = Path(metadata["baseline_manifest"])
    if hashlib.sha256(baseline_path.read_bytes()).hexdigest() != metadata["baseline_manifest_sha256"]:
        raise ValueError("Baseline manifest changed")
    baseline = ManiSkillAbsoluteDataset(baseline_path)
    dataset = ManiSkillSidecarDataset(manifest)
    if len(dataset) != len(baseline) or len(dataset.records) != len(baseline.records):
        raise ValueError("Sidecar and baseline inventory differs")
    for index, record in enumerate(dataset.records):
        old = baseline.records[index]
        assert all(record[k] == v for k, v in old.items())
        target = dataset._labels(index)
        assert (target[:, 0] == 1).all() and (target[:, 1] == 0).all()
        assert not target[:, 64:70].reshape(-1, 3, 2)[:, :, 1].any()
        assert not target[:, 118:130].reshape(-1, 4, 3)[:, :, 2].any()
        assert not target[:, 118:130].reshape(-1, 4, 3)[:, 2:].any()
        if hashlib.sha256(Path(record["derived_source"]).read_bytes()).hexdigest() != record["derived_sha256"]:
            raise ValueError("Derived oracle changed")
    print(f"All {len(dataset.records)} label/source hashes and baseline identities passed", flush=True)
    windows = []
    for task in sorted({r["task"] for r in dataset.records}):
        episodes = [i for i, r in enumerate(dataset.records) if r["task"] == task]
        for episode in (episodes[0], episodes[len(episodes) // 2], episodes[-1]):
            record = dataset.records[episode]
            length = record["steps"]
            first = dataset.ends[episode - 1] if episode else 0
            with h5py.File(record["source_h5"], "r") as handle:
                tr = handle[record["trajectory"]]
                for start in (0, length // 2, length - 1):
                    sample, ref = dataset[first + start], baseline[first + start]
                    assert set(sample) - set(ref) == {"ea_targets", "ea_plan_steps", "ea_video_anchors"}
                    for key in ("action", "proprio", "action_mask", "proprio_mask", "video_mask"):
                        torch.testing.assert_close(sample[key], ref[key], rtol=0, atol=0)
                    assert sample["prompt"] == ref["prompt"]
                    for image, other in zip(sample["video"], ref["video"]):
                        np.testing.assert_array_equal(np.asarray(image), np.asarray(other))
                    target = sample["ea_targets"].numpy()[0]
                    boxes = target[70:118].reshape(4, 3, 4)
                    visible = target[118:130].reshape(4, 3)
                    for view, camera in enumerate(("front_camera", "hand_camera")):
                        segmentation = np.asarray(tr[f"obs/sensor_data/{camera}/segmentation"][start])[..., 0]
                        for role, field in enumerate(("obj_segmentation_id", "goal_segmentation_id")):
                            actor = int(np.asarray(tr[f"obs/extra/{field}"][start]).item())
                            y, x = np.nonzero(segmentation == actor) if actor > 0 else ([], [])
                            assert bool(len(x)) == bool(visible[role, view])
                            expected = [min(x), min(y), max(x) + 1, max(y) + 1] if len(x) else [0] * 4
                            np.testing.assert_allclose(boxes[role, view] * 256, expected, atol=1e-5)
                    for hidx, horizon in enumerate((8, 16, 32)):
                        valid = start + horizon <= length
                        assert bool(target[64 + hidx * 2]) == valid
                        if valid:
                            for role, field in enumerate(("tcp_pose", "obj_pose")):
                                current = np.asarray(tr[f"obs/extra/{field}"][start])
                                future = np.asarray(tr[f"obs/extra/{field}"][start + horizon])
                                local = Rotation.from_quat(current[[4, 5, 6, 3]]).inv().apply(future[:3] - current[:3])
                                moving = np.linalg.norm(local) > .001
                                assert bool(target[52 + hidx * 4 + role]) == moving
                                expected = local / np.linalg.norm(local) if moving else np.zeros(3)
                                actual = target[16:52].reshape(3, 2, 2, 3)[hidx, 0, role]
                                np.testing.assert_allclose(actual, expected, atol=2e-5)
                    windows.append(dict(task=task, episode=episode, start=start))
        print(f"{task}: first/middle/last episode x first/middle/last window passed", flush=True)
    # Actual list collation used by trainer; no oracle context is added to policy inputs.
    batch = [dataset[i] for i in range(32)]
    assert torch.stack([s["ea_targets"] for s in batch]).shape == (32, 1, 130)
    assert torch.stack([s["action"] for s in batch]).shape == (32, 32, 80)
    report = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  manifest=str(Path(manifest).resolve()), episodes=len(dataset.records), windows=len(dataset),
                  checked_windows=windows, batch_size=32, elapsed_seconds=time.monotonic() - started,
                  gpu_training_tested=False, closed_loop_tested=False)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "checked_windows"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    check(**vars(parser.parse_args()))
