#!/usr/bin/env python3
"""Small CPU-physics expert comparison, using recorded initial states.

Uses native absolute pose control (XYZ Euler at execution only); the saved
training contract remains XYZ + rotation6D + normalized gripper.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import gymnasium as gym
    import h5py
    import mani_skill.envs  # noqa: F401 - register native environments.
    import numpy as np
    from mani_skill.trajectory.utils import index_dict
    from scipy.spatial.transform import Rotation

    from openwam.maniskill_eef_contract import absolute_command_targets

    root = Path(__file__).resolve().parents[1]
    output = args.output or root / "outputs/maniskill_baseline/expert_replay.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    audit = json.loads((root / "docs/evidence/franka_maniskill_20261016/absolute_eef_source_audit.json").read_text())
    results = []
    for task in audit["tasks"]:
        source = Path(task["source_h5"][0])
        metadata = json.loads(source.with_suffix(".json").read_text())
        with h5py.File(source, "r") as f:
            tr = f["traj_0"]
            a = np.asarray(tr["actions"])
            world = np.asarray(tr["obs/extra/tcp_pose"])
            base = np.asarray(tr["obs/extra/robot_base_pose"])
            initial = index_dict(tr["env_states"], 0)
        targets = absolute_command_targets(world[:-1], base[:-1], a)
        rmat = np.stack([targets[:, 3:6], targets[:, 6:9], np.cross(targets[:, 3:6], targets[:, 6:9])], axis=-1)
        native_abs = np.concatenate(
            [targets[:, :3], Rotation.from_matrix(rmat).as_euler("XYZ"), 2 * targets[:, 9:10] - 1], axis=-1
        ).astype(np.float32)
        traces = {}
        success = {}
        for mode, commands in [("pd_ee_delta_pose", a), ("pd_ee_pose", native_abs)]:
            started = time.monotonic()
            kwargs = dict(metadata["env_info"]["env_kwargs"])
            kwargs.update(obs_mode="none", render_mode=None, render_backend="none", control_mode=mode)
            env = gym.make(metadata["env_info"]["env_id"], **kwargs).unwrapped
            env.reset(**metadata["episodes"][0]["reset_kwargs"])
            env.set_state_dict(initial)
            trace = []
            ok = False
            for command in commands:
                _, _, _, _, info = env.step(command)
                ok |= bool(info["success"].item())
                trace.append(env.agent.tcp.pose.raw_pose.cpu().numpy()[0].copy())
            traces[mode] = np.asarray(trace)
            success[mode] = ok
            env.close()
            print(
                json.dumps(
                    dict(
                        task=task["task"],
                        control_mode=mode,
                        success=ok,
                        steps=len(commands),
                        seconds=time.monotonic() - started,
                    )
                ),
                flush=True,
            )
        dp = float(np.linalg.norm(traces["pd_ee_delta_pose"][:, :3] - traces["pd_ee_pose"][:, :3], axis=-1).max())
        row = dict(task=task["task"], trajectory="traj_0", success=success, maximum_tcp_position_difference_m=dp)
        results.append(row)
        output.write_text(json.dumps(results, indent=2) + "\n")
        if not all(success.values()) or dp > 0.02:
            raise RuntimeError(f"Expert replay failed: {row}")


if __name__ == "__main__":
    main()
