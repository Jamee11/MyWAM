"""Portable, ROS/model-free absolute EEF geometry and fail-closed motion checks."""
import numpy as np

CONTRACT = dict(schema='openwam.maniskill.absolute_eef10.v1', action_dim=10, horizon=32,
                frame='model_robot_base', position_unit='metre', rotation='first_two_columns',
                gripper='0_closed_1_open')


def rigid(value):
    t = np.asarray(value, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError('Need a finite 4x4 rigid transform')
    if (not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5)):
        raise ValueError('Invalid rigid transform: no scale/reflection/projective transforms')
    return t


def eef_to_matrix(action):
    a = np.asarray(action, dtype=float)
    if a.shape != (10,) or not np.isfinite(a).all():
        raise ValueError('Expected finite EEF10: xyz + column6D + gripper')
    x = a[3:6]
    if np.linalg.norm(x) < 1e-6:
        raise ValueError('Degenerate first rotation column')
    x = x / np.linalg.norm(x)
    y = a[6:9] - x * np.dot(x, a[6:9])
    if np.linalg.norm(y) < 1e-6:
        raise ValueError('Degenerate second rotation column')
    y /= np.linalg.norm(y)
    t = np.eye(4)
    t[:3, :3] = np.stack((x, y, np.cross(x, y)), axis=1)
    t[:3, 3] = a[:3]
    return t


def matrix_to_eef(transform, gripper):
    t = rigid(transform)
    if not np.isfinite(gripper) or not 0 <= gripper <= 1:
        raise ValueError('Proprio gripper must be an open fraction in [0,1]')
    return np.r_[t[:3, 3], t[:3, 0], t[:3, 1], gripper].astype(np.float32)


def action_chunk(value):
    actions = np.asarray(value, dtype=float)
    if actions.shape != (32, 10) or not np.isfinite(actions).all():
        raise ValueError('Expected a finite physical 32x10 absolute EEF chunk')
    if np.any((actions[:, 9] < -.25) | (actions[:, 9] > 1.25)):
        raise ValueError('Grossly invalid gripper prediction')
    for a in actions:
        eef_to_matrix(a)
    return actions


def model_state(real_link, model_from_real, link_from_tcp, gripper):
    return matrix_to_eef(rigid(model_from_real) @ rigid(real_link) @ rigid(link_from_tcp), gripper)


def real_target(action, model_from_real, link_from_tcp):
    tcp = np.linalg.inv(rigid(model_from_real)) @ eef_to_matrix(action)
    # Absolute target, NOT current_pose + action and NOT camera-to-base.
    link = tcp @ np.linalg.inv(rigid(link_from_tcp))
    return rigid(link), rigid(tcp)


def pose_guard(current_tcp, target_tcp, cfg):
    current, target = rigid(current_tcp), rigid(target_tcp)
    bounds = np.asarray(cfg['workspace_xyz'], dtype=float)
    if (bounds.shape != (2, 3) or not np.isfinite(bounds).all()
            or np.any(bounds[0] >= bounds[1])):
        raise ValueError('Configure measured real-base TCP workspace bounds, [[min],[max]]')
    if np.any(target[:3, 3] < bounds[0]) or np.any(target[:3, 3] > bounds[1]):
        raise ValueError('Target TCP outside approved real workspace')
    translation = float(np.linalg.norm(target[:3, 3] - current[:3, 3]))
    angle = float(np.arccos(np.clip((np.trace(current[:3, :3].T @ target[:3, :3])-1)/2, -1, 1)))
    if translation > cfg['max_translation_m'] or angle > cfg['max_rotation_rad']:
        raise ValueError(f'Pose jump rejected: translation={translation:.4f}m, rotation={angle:.4f}rad')


def joint_guard(previous, target, limits, cfg):
    q, before, bounds = np.asarray(target), np.asarray(previous), np.asarray(limits)
    if (q.shape != (7,) or before.shape != (7,) or bounds.shape != (7, 2)
            or not np.isfinite([q, before]).all() or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1])):
        raise ValueError('Need finite named seven-joint positions and URDF limits')
    if np.any(q < bounds[:, 0]) or np.any(q > bounds[:, 1]):
        raise ValueError('IK result outside URDF joint limits')
    delta = np.abs(q - before)
    if np.any(delta > cfg['max_joint_step_rad']) or np.any(delta / cfg['step_seconds'] > cfg['max_joint_speed_rad_s']):
        raise ValueError('IK joint jump/velocity rejected')
