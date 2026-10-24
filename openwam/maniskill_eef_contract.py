"""ManiSkill Panda recorded delta commands -> absolute robot-base TCP targets.

Matches the audited local PDEEPoseController: normalized +/-0.1 translation,
norm-clipped XYZ Euler rotation scaled by rot_lower=-0.1, use_target=False.
No RoboTwin tool offset is applied. Numerical arrays are never simulator GT inputs.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def pose_in_base(tcp_world, base_world):
    tcp_world, base_world = np.asarray(tcp_world), np.asarray(base_world)
    for poses in (tcp_world, base_world):
        if poses.shape[-1] != 7 or not np.isfinite(poses).all():
            raise ValueError("Need finite xyz+wxyz poses")
        if not np.allclose(np.linalg.norm(poses[..., 3:], axis=-1), 1, atol=1e-3):
            raise ValueError("Non-unit source quaternion")
    tcp = Rotation.from_quat(tcp_world[..., [4, 5, 6, 3]])
    base = Rotation.from_quat(base_world[..., [4, 5, 6, 3]])
    position = base.inv().apply(tcp_world[..., :3] - base_world[..., :3])
    rotation = base.inv() * tcp
    return position, rotation


def eef10(position, rotation, gripper):
    matrix = rotation.as_matrix()
    return np.concatenate((position, matrix[..., :, 0], matrix[..., :, 1],
                           np.asarray(gripper).reshape(-1, 1)), axis=-1).astype(np.float32)


def absolute_command_targets(tcp_world, base_world, recorded_actions):
    action = np.asarray(recorded_actions, np.float64)
    if action.ndim != 2 or action.shape[1] != 7 or not np.isfinite(action).all():
        raise ValueError("Need T x 7 finite native Panda commands")
    position, rotation = pose_in_base(tcp_world, base_world)
    rot = action[:, 3:6] / np.maximum(1, np.linalg.norm(action[:, 3:6], axis=-1, keepdims=True))
    target_position = position + .1 * np.clip(action[:, :3], -1, 1)
    target_rotation = Rotation.from_euler("XYZ", -.1 * rot) * rotation
    grip = (np.clip(action[:, 6], -1, 1) + 1) / 2
    return eef10(target_position, target_rotation, grip)


def achieved_states(tcp_world, base_world, qpos):
    p, r = pose_in_base(tcp_world, base_world)
    qpos = np.asarray(qpos)
    if not np.isfinite(qpos).all() or qpos.shape[-1] != 9:
        raise ValueError("Expected Panda qpos7 + two finger positions")
    return eef10(p, r, np.clip(qpos[:, -2:].mean(-1) / .04, 0, 1))
