#!/usr/bin/env python3
"""Audit Clean HDF5s, fit official EEF statistics, and regenerate Lite labels.

Writes only output-dir; original RGB/actions/oracle files are never modified.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(raw_root, output_dir, tasks=None):
    import h5py
    import numpy as np

    from openwam.dataloader.ea_rsig_lite import CAMERAS, LiteRoboTwinSource, targets_from_oracle
    from openwam.dataloader.robotwin import ROBOTWIN_ALL_TASKS, RoboTwinDataset
    from openwam.dataloader.utils.stats_computation.robotwin_stats_computation import (
        _ModeAccumulator,
        _pin_eef_rot6d_identity,
    )
    from openwam.ea_lite_contract import tcp_context_from_eef20
    raw_root, output_dir = Path(raw_root).resolve(), Path(output_dir).resolve()
    if output_dir == raw_root or raw_root in output_dir.parents:
        raise ValueError("Output must be outside source data")
    task_names = sorted(tasks if tasks is not None else ROBOTWIN_ALL_TASKS)
    if len(set(task_names)) != len(task_names) or set(task_names) - set(ROBOTWIN_ALL_TASKS):
        raise ValueError("Need unique official RoboTwin tasks")
    output_dir.mkdir(parents=True, exist_ok=True)
    accumulator = _ModeAccumulator(20)
    entries = []
    for task in task_names:
        source = raw_root / task / "demo_clean"
        collection = json.loads((source / "ea_rsig_v3_collection_manifest.json").read_text())
        if collection.get("task") != task:
            raise ValueError(f"Wrong role/task manifest: {task}")
        if collection.get("oracle_contract") != "ea_rsig_v3_dual_track_articulation_safe_goal_valid_v6":
            raise ValueError(f"Need audited V6 role/oracle contract: {task}")
        files = sorted((source / "data").glob("episode*.hdf5"))
        if not files:
            raise FileNotFoundError(source / "data")
        episodes = []
        label_dir = output_dir / "labels" / task
        label_dir.mkdir(parents=True, exist_ok=True)
        for path in files:
            with h5py.File(path, "r") as file:
                length = len(file["endpose/left_endpose"])
                eef = LiteRoboTwinSource._read_eef_actions(None, file, 0, length)
                if eef.ndim != 2 or eef.shape[1] != 20 or not np.isfinite(eef).all():
                    raise ValueError(f"Invalid official EEF in {path}")
                if np.any((eef[:, [9, 19]] < -1e-5) | (eef[:, [9, 19]] > 1 + 1e-5)):
                    raise ValueError(f"Unexpected raw gripper convention {path}")
                length = len(eef)
                sizes = []
                for camera in CAMERAS:
                    images = file[f"observation/{camera}/rgb"]
                    if len(images) != length:
                        raise ValueError(f"Camera/EEF timestamps mismatch {path}")
                    sizes.append(RoboTwinDataset._decode_jpeg(None, images[0]).size)
                oracle = {k: np.asarray(v) for k, v in file["ea_rsig_oracle_v3"].items()}
                labels = targets_from_oracle(oracle, sizes, task)
                if len(labels) != length:
                    raise ValueError(f"Oracle/EEF length mismatch {path}")
                # Official HDF5 endpose and oracle poses must describe one shared frame.
                oracle_eef = oracle["eef_pose_world_wxyz"]
                effectors = tcp_context_from_eef20(eef)
                if not np.allclose(effectors[..., :3], oracle_eef[..., :3], atol=2e-5):
                    raise ValueError(f"Observed endpose-to-TCP geometry mismatch {path}")
                # Also reject a quaternion-order mismatch hidden by unit quaternion norms.
                from scipy.spatial.transform import Rotation
                rotation = Rotation.from_quat(oracle_eef[..., [4, 5, 6, 3]].reshape(-1, 4)).as_matrix()
                expected_rot6d = np.concatenate((rotation[:, :, 0], rotation[:, :, 1]), axis=-1).reshape(length, 2, 6)
                if not np.allclose(effectors[..., 3:9], expected_rot6d, atol=2e-5):
                    raise ValueError(f"Observed endpose/oracle orientation mismatch {path}")
                accumulator.update(eef)
                dest = label_dir / (path.stem + ".npz")
                temp = dest.with_suffix(".tmp.npz")
                if not np.all(oracle["arm_present"] == 1):
                    raise ValueError("This bimanual Aloha contract needs both physical arms")
                np.savez_compressed(temp, targets=labels, effectors=effectors,
                                    arm_present=np.ones((length, 2), np.float32))
                temp.replace(dest)
            stat = path.stat()
            episodes.append(dict(path=str(path), frames=length, size=stat.st_size,
                                 mtime_ns=stat.st_mtime_ns, labels=str(dest), labels_sha256=sha256(dest)))
        entries.append(dict(name=task, data_root=str(source / "data"), episodes=episodes))
        print(f"Lite labels: {task}: {len(episodes)} episodes", flush=True)
    stats_path = output_dir / "normalization_stats.npy"
    temp = output_dir / "normalization_stats.tmp.npy"
    np.save(temp, {"eef": _pin_eef_rot6d_identity(accumulator.finalize())})
    temp.replace(stats_path)
    manifest = dict(contract="openwam_alpha_eef32_ea_lite_v1", split="Clean",
                    source_root=str(raw_root), action_frame="official_robotwin_endpose_reference",
                    source_quaternion_order="wxyz", ea_context="observed_endpose_plus_local_x_0.12m_TCP",
                    action_semantics="absolute_eef_t_plus_1_to_t_plus_32",
                    plan_frame="each_role_current_local_rotation", plan_steps=[8, 16, 32],
                    video_anchors=list(range(0, 33, 4)), gripper="raw01_to_official_minmax_minus1_plus1",
                    normalization_stats_path=str(stats_path), normalization_stats_sha256=sha256(stats_path),
                    tasks=entries, total_episodes=sum(len(e["episodes"]) for e in entries),
                    total_frames=sum(ep["frames"] for e in entries for ep in e["episodes"]))
    target = output_dir / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {target}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+")
    args = parser.parse_args()
    prepare(args.raw_root, args.output_dir, args.tasks)
