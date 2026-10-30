#!/usr/bin/env python3
"""Reuse audited baseline identities/actions/stats; regenerate 130D auxiliary labels offline."""

import argparse
import hashlib
import json
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(baseline_manifest, output_dir):
    import h5py
    import numpy as np

    from openwam.maniskill_eef_contract import achieved_states
    from openwam.maniskill_sidecar_contract import CONTRACT, build_targets

    baseline_manifest, output_dir = Path(baseline_manifest).resolve(), Path(output_dir).resolve()
    baseline = json.loads(baseline_manifest.read_text())
    if baseline["contract"] != "maniskill_panda_absolute_command_eef_v1":
        raise ValueError("Need the audited absolute-command baseline manifest")
    if output_dir.exists():
        raise FileExistsError("Use a fresh output directory; never overwrite baseline or old labels")
    if sha(baseline["normalization_stats_path"]) != baseline["normalization_stats_sha256"]:
        raise ValueError("Baseline normalization changed")
    mappings = {}
    for source in baseline["source_manifests"]:
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("Source manifest changed")
        manifest_path = Path(source["path"])
        for line in manifest_path.read_text().splitlines():
            row = json.loads(line)
            key = (source["task"], str(Path(row["source_h5"]).resolve()), row["source_trajectory"])
            if key in mappings:
                raise ValueError("Duplicate derived source mapping")
            mappings[key] = (manifest_path.parent / row["output"], row)
    if len(mappings) != len(baseline["episodes"]):
        raise ValueError("Derived and baseline trajectory inventories differ")
    output_dir.mkdir(parents=True)
    episodes, coverage = [], {}
    with ExitStack() as stack:
        handles = {}
        for index, record in enumerate(baseline["episodes"]):
            source = record["source_h5"]
            if source not in handles:
                st, expected = Path(source).stat(), baseline["sources"][source]
                if (st.st_size, st.st_mtime_ns) != (expected["size"], expected["mtime_ns"]):
                    raise ValueError("Raw HDF5 changed")
                handles[source] = stack.enter_context(h5py.File(source, "r"))
            derived_path, row = mappings[(record["task"], source, record["trajectory"])]
            if row["num_steps"] != record["steps"] or row["task_instruction"] != record["prompt"]:
                raise ValueError("Trajectory identity/length/instruction mismatch")
            if sha(record["numeric"]) != record["numeric_sha256"]:
                raise ValueError("Baseline numeric artifact changed")
            tr = handles[source][record["trajectory"]]
            extra = tr["obs/extra"]
            tcp, obj, goal, base = (np.asarray(extra[k]) for k in
                                    ("tcp_pose", "obj_pose", "goal_pos", "robot_base_pose"))
            length = record["steps"]
            if len(tcp) != length + 1:
                raise ValueError("Raw observations must include terminal frame")
            with np.load(record["numeric"], allow_pickle=False) as numeric:
                np.testing.assert_allclose(achieved_states(tcp, base, np.asarray(tr["obs/agent/qpos"])),
                                           numeric["state"], atol=2e-6, rtol=1e-6)
            force = np.asarray(extra["robot_obj_contact_force_norm"]) if "robot_obj_contact_force_norm" in extra else None
            # Read only required arrays; never reuse the legacy future_horizons/125D vector.
            derived_hash = sha(derived_path)
            with np.load(derived_path, allow_pickle=False) as derived:
                for key, raw in (("tcp_pose_wxyz", tcp), ("object_pose_wxyz", obj), ("goal_pos", goal)):
                    np.testing.assert_allclose(derived[key], raw[:-1], atol=2e-6, rtol=1e-6)
                boxes = np.zeros((length, 2, 2, 4), np.float32)
                visible = np.zeros((length, 2, 2), np.float32)
                for role, name in enumerate(("object", "goal")):
                    for view, camera in enumerate(("front_camera", "hand_camera")):
                        boxes[:, role, view] = derived[f"{camera}_{name}_bbox_xyxy"]
                        visible[:, role, view] = derived[f"{camera}_{name}_visible"].reshape(length)
                if force is not None:
                    np.testing.assert_array_equal(derived["physical_contact"].reshape(length), force[:-1].reshape(length) > 1e-6)
            if sha(derived_path) != derived_hash:
                raise ValueError("Derived source changed while reading")
            targets = build_targets(tcp, obj, goal, np.asarray(extra["obj_segmentation_id"]),
                                    np.asarray(extra["goal_segmentation_id"]), force, boxes, visible)
            path = output_dir / "labels" / record["task"] / f"episode{record['episode']:04d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, targets=targets)
            episodes.append(dict(record, labels=str(path), labels_sha256=sha(path),
                                 derived_source=str(derived_path), derived_sha256=derived_hash))
            counts = coverage.setdefault(record["task"], dict(episodes=0, windows=0, contact_positive=0,
                                                           ground_visible=[0, 0, 0, 0], future_valid=[0, 0, 0]))
            counts["episodes"] += 1
            counts["windows"] += length
            counts["contact_positive"] += int(targets[:, 8].sum())
            counts["ground_visible"] = (np.asarray(counts["ground_visible"]) + visible.sum(0).reshape(4)).astype(int).tolist()
            counts["future_valid"] = (np.asarray(counts["future_valid"]) + targets[:, 64:70:2].sum(0)).astype(int).tolist()
            if (index + 1) % 250 == 0:
                print(f"Prepared {index + 1}/{len(baseline['episodes'])} episodes", flush=True)
    for source, expected in baseline["sources"].items():
        st = Path(source).stat()
        if (st.st_size, st.st_mtime_ns) != (expected["size"], expected["mtime_ns"]):
            raise ValueError("Raw source changed during preparation")
    manifest = dict(baseline, episodes=episodes, sidecar_contract=CONTRACT, auxiliary_labels_used=True,
                    baseline_manifest=str(baseline_manifest), baseline_manifest_sha256=sha(baseline_manifest),
                    prepared_utc=datetime.now(timezone.utc).isoformat(), plan_steps=[8, 16, 32],
                    plan_frame="each_role_current_local_rotation_world_pose_differences",
                    contact_threshold_n=1e-6, near_threshold_m=0.1, goal_distance_scale_m=0.5,
                    terminal_observation_used_for_plan=True, coverage=coverage)
    path = output_dir / "manifest.ready.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(manifest=str(path), episodes=len(episodes), windows=manifest["total_windows"],
                          coverage=coverage), indent=2), flush=True)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    prepare(**vars(parser.parse_args()))
