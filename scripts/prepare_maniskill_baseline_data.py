#!/usr/bin/env python3
"""Build numeric absolute-EEF sidecars; read original RGB HDF5 in training.

Only train manifests from the requested six-task LeRobot link collection are
used, including aggregate replacement episodes. Never rewrite source files.
"""

import argparse
import hashlib
import json
import sys
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(dataset_root, output_dir):
    import h5py
    import numpy as np

    from openwam.dataloader.utils.normalization import pin_rot6d_identity
    from openwam.dataloader.utils.stats_computation.robotwin_stats_computation import _ModeAccumulator
    from openwam.maniskill_eef_contract import absolute_command_targets, achieved_states

    dataset_root, output_dir = Path(dataset_root).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Use a fresh output directory; no source or prepared data is overwritten")
    links = sorted((dataset_root / "lerobot/train/ea_rsig_v3").iterdir())
    if len(links) != 6:
        raise ValueError("Expected six requested task buckets")
    output_dir.mkdir(parents=True)
    accumulator = _ModeAccumulator(10)
    episodes, manifests, source_stats, controller_metadata = [], [], {}, {}
    seen = set()
    errors, success_counts = {}, {}
    with ExitStack() as stack:
        handles = {}
        for link in links:
            bucket = link.resolve()
            info = json.loads((bucket / "meta/info.json").read_text())
            task = info["robot_type"].removeprefix("maniskill_eort_").removesuffix("_ea_rsig_v3")
            manifest = bucket.parents[3] / task / "train/color_distractors/derived_objectcentric_v4/manifest.jsonl"
            rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
            if len(rows) != 1000 or info["total_episodes"] != 1000:
                raise ValueError(f"Incomplete task {task}")
            manifests.append(dict(task=task, path=str(manifest), sha256=sha(manifest), lerobot_bucket=str(bucket)))
            task_dir = output_dir / "numeric" / task
            task_dir.mkdir(parents=True)
            errors[task], success_counts[task] = [], 0
            for index, row in enumerate(rows):
                if row.get("source_control_mode") != "pd_ee_delta_pose" or row.get("quaternion_convention") != "wxyz":
                    raise ValueError(f"Unsupported source controller/quaternion: {manifest}")
                source, trajectory = Path(row["source_h5"]).resolve(), row["source_trajectory"]
                if (str(source), trajectory) in seen:
                    raise ValueError("Duplicate training trajectory")
                seen.add((str(source), trajectory))
                if str(source) not in handles:
                    handles[str(source)] = stack.enter_context(h5py.File(source, "r"))
                    st = source.stat()
                    source_stats[str(source)] = dict(size=st.st_size, mtime_ns=st.st_mtime_ns)
                    meta = json.loads(source.with_suffix(".json").read_text())
                    env = meta["env_info"]
                    if (
                        env["env_kwargs"]["control_mode"] != "pd_ee_delta_pose"
                        or env["env_kwargs"]["sim_backend"] != "physx_cpu"
                    ):
                        raise ValueError("Unverified collection controller/backend")
                    controller_metadata[str(source)] = env
                tr = handles[str(source)][trajectory]
                action = np.asarray(tr["actions"], np.float32)
                length = len(action)
                if action.shape != (length, 7) or row["num_steps"] != length or length < 1:
                    raise ValueError(f"Bad trajectory length: {source}:{trajectory}")
                world = np.asarray(tr["obs/extra/tcp_pose"], np.float64)
                base = np.asarray(tr["obs/extra/robot_base_pose"], np.float64)
                qpos = np.asarray(tr["obs/agent/qpos"], np.float32)
                if world.shape != (length + 1, 7) or base.shape != world.shape or qpos.shape != (length + 1, 9):
                    raise ValueError("Need synchronized T actions and T+1 actual observations")
                for camera in ("front_camera", "hand_camera"):
                    images = tr[f"obs/sensor_data/{camera}/rgb"]
                    if images.shape != (length + 1, 256, 256, 3) or images.dtype != np.uint8:
                        raise ValueError("Unexpected RGB shape/codec")
                state = achieved_states(world, base, qpos)
                targets = absolute_command_targets(world[:-1], base[:-1], action)
                if not np.isfinite(state).all() or not np.isfinite(targets).all():
                    raise ValueError("Nonfinite EEF output")
                # Source NPZ maps source episodes, not guessed LeRobot numeric IDs.
                derived = manifest.parent / row["output"]
                with np.load(derived) as old:
                    if not np.allclose(old["tcp_pose_base_wxyz"][:, :3], state[:-1, :3], atol=2e-6):
                        raise ValueError("Raw and derived trajectory mapping disagree")
                accumulator.update(state)
                accumulator.update(targets)
                distance = np.linalg.norm(targets[:, :3] - state[1:, :3], axis=-1)
                errors[task].append(distance)
                if "success" in tr:
                    success_counts[task] += int(np.asarray(tr["success"]).any())
                path = task_dir / f"episode{index:04d}.npz"
                np.savez_compressed(path, state=state, action=targets)
                prompt = row["task_instruction"]
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("Missing instruction")
                episodes.append(
                    dict(
                        task=task,
                        episode=index,
                        source_h5=str(source),
                        trajectory=trajectory,
                        steps=length,
                        numeric=str(path),
                        numeric_sha256=sha(path),
                        prompt=prompt,
                    )
                )
            print(
                f"Prepared {task}: 1000 episodes, {sum(e['steps'] for e in episodes if e['task'] == task)} actions",
                flush=True,
            )
    # Stats fit train states and commanded targets only; no held-out leakage.
    stats = accumulator.finalize()
    pin_rot6d_identity(stats, list(range(3, 9)))
    stats["min"][9], stats["max"][9] = 0.0, 1.0
    stats_path = output_dir / "normalization_stats.npy"
    np.save(stats_path, {"eef": stats})
    for source, old in source_stats.items():
        st = Path(source).stat()
        if (st.st_size, st.st_mtime_ns) != (old["size"], old["mtime_ns"]):
            raise ValueError("Source changed while preparing")
    report = dict(
        contract="maniskill_panda_absolute_command_eef_v1",
        source_dataset=str(dataset_root),
        action_semantics="absolute_controller_target_at_t_for_transition_t_to_t_plus_1",
        state_semantics="achieved_base_tcp_at_t",
        action_frame="robot_base",
        eef_reference="panda_hand_tcp",
        source_quaternion="wxyz",
        rotation="first_two_matrix_columns",
        tool_offset_m=0.0,
        action_slots=list(range(10)),
        num_frames=33,
        video_stride=4,
        action_horizon=32,
        camera_layout=["front_camera", "hand_camera", None],
        fps=20,
        normalization_stats_path=str(stats_path),
        normalization_stats_sha256=sha(stats_path),
        source_manifests=manifests,
        sources=source_stats,
        controller_metadata=controller_metadata,
        episodes=episodes,
        total_episodes=len(episodes),
        total_windows=sum(e["steps"] for e in episodes),
        raw_rgb_copied=False,
        auxiliary_labels_used=False,
        recorded_success_episodes=success_counts,
        target_vs_next_achieved_position_m={
            k: dict(mean=float(np.concatenate(v).mean()), maximum=float(np.concatenate(v).max()))
            for k, v in errors.items()
        },
    )
    target = output_dir / "manifest.ready.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps({k: report[k] for k in ["total_episodes", "total_windows", "recorded_success_episodes"]}), flush=True
    )
    print(f"READY_MANIFEST={target}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    prepare(**vars(parser.parse_args()))
