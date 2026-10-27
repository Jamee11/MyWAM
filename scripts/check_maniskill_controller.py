#!/usr/bin/env python3
"""Compare new targets with actual local ManiSkill controller methods on CPU.

Run with the existing simulator environment and its local ManiSkill PYTHONPATH.
This is a same-state controller comparison, not a physics success-rate replay.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import h5py
    import numpy as np
    import torch
    from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseController
    from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
    from mani_skill.utils.structs.pose import Pose
    from scipy.spatial.transform import Rotation

    from openwam.maniskill_eef_contract import absolute_command_targets, pose_in_base

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.manifest).read_text())
    # EORT environments default to PandaWristCam (inherits Panda controllers).
    panda = PandaWristCam.__new__(PandaWristCam)
    cfg = panda._controller_configs["pd_ee_delta_pose"]["arm"]
    controller = PDEEPoseController.__new__(PDEEPoseController)
    controller.config = cfg
    controller.action_space_low = torch.tensor([cfg.pos_lower] * 3 + [cfg.rot_lower] * 3)
    controller.action_space_high = torch.tensor([cfg.pos_upper] * 3 + [cfg.rot_upper] * 3)
    assert cfg.use_delta and not cfg.use_target
    assert cfg.frame == "root_translation:root_aligned_body_rotation"
    assert cfg.ee_link == "panda_hand_tcp"
    rows = []
    for task in sorted({r["task"] for r in data["episodes"]}):
        records = [r for r in data["episodes"] if r["task"] == task]
        for record in (records[0], records[500], records[-1]):
            with h5py.File(record["source_h5"], "r") as f:
                tr = f[record["trajectory"]]
                actions = np.asarray(tr["actions"])
                world = np.asarray(tr["obs/extra/tcp_pose"])[:-1]
                base = np.asarray(tr["obs/extra/robot_base_pose"])[:-1]
            p, r = pose_in_base(world, base)
            q = r.as_quat()[:, [3, 0, 1, 2]]
            current = Pose.create_from_pq(torch.tensor(p, dtype=torch.float32), torch.tensor(q, dtype=torch.float32))
            scaled = controller._clip_and_scale_action(torch.tensor(actions[:, :6]))
            target = controller.compute_target_pose(current, scaled)
            expected = absolute_command_targets(world, base, actions)
            actual_r = Rotation.from_quat(target.q.numpy()[:, [1, 2, 3, 0]])
            expected_r = np.stack(
                [expected[:, 3:6], expected[:, 6:9], np.cross(expected[:, 3:6], expected[:, 6:9])], axis=-1
            )
            pe = float(np.abs(target.p.numpy() - expected[:, :3]).max())
            re = float((actual_r.inv() * Rotation.from_matrix(expected_r)).magnitude().max())
            assert pe < 2e-6 and re < 2e-6, (task, pe, re)
            rows.append(
                dict(
                    task=task,
                    trajectory=record["trajectory"],
                    steps=len(actions),
                    position_max_abs_m=pe,
                    rotation_max_rad=re,
                )
            )
    import inspect

    source = Path(inspect.getfile(PDEEPoseController))
    result = dict(
        robot_uid=panda.uid,
        controller_config={
            k: getattr(cfg, k)
            for k in [
                "pos_lower",
                "pos_upper",
                "rot_lower",
                "rot_upper",
                "frame",
                "use_delta",
                "use_target",
                "ee_link",
                "urdf_path",
            ]
        },
        controller_source=str(source),
        controller_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        comparisons=rows,
        physics_replay=False,
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
