"""Portable, ROS/model-free absolute EEF geometry and fail-closed motion checks."""
import numpy as np

SCHEMA = 'openwam.piper.absolute_eef20.v1'


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


def state20(value):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (20,) or not np.isfinite(value).all():
        raise ValueError('Expected finite physical EEF20, left10 then right10')
    for offset in (0, 10):
        eef_to_matrix(value[offset:offset + 10])
        if not 0 <= value[offset + 9] <= 1:
            raise ValueError('Observed gripper must be 0=closed, 1=open')
    return value


def action_chunk(value):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (32, 20) or not np.isfinite(value).all():
        raise ValueError('Expected finite physical 32x20 absolute EEF actions')
    for row in value:
        for offset in (0, 10):
            eef_to_matrix(row[offset:offset + 10])
    return value


class Frames:
    """T_model_tool = T_model_base @ T_base_feedback @ T_feedback_tool."""
    def __init__(self, config):
        self.config = config
        self.base = [rigid(config[side]['model_from_base']) for side in ('left', 'right')]
        self.tool = [rigid(config[side]['feedback_from_tool']) for side in ('left', 'right')]
        self.open = np.asarray([config[s]['gripper_open_m'] for s in ('left', 'right')], float)
        if not np.isfinite(self.open).all() or np.any(self.open <= 0):
            raise ValueError('Configure measured gripper_open_m for both arms')

    def state(self, feedback, grippers):
        values = [matrix_to_eef(self.base[i] @ rigid(feedback[i]) @ self.tool[i],
                               np.clip(grippers[i] / self.open[i], 0, 1)) for i in range(2)]
        return state20(np.concatenate(values))

    def targets(self, action):
        from scipy.spatial.transform import Rotation
        value = np.asarray(action, float)
        if value.shape != (20,) or not np.isfinite(value).all():
            raise ValueError('Expected finite absolute EEF20 target')
        result = []
        for i in range(2):
            model = eef_to_matrix(value[10*i:10*i+10])
            real = np.linalg.inv(self.base[i]) @ model @ np.linalg.inv(self.tool[i])
            result.append(np.r_[real[:3, 3], Rotation.from_matrix(real[:3, :3]).as_euler('xyz'),
                                np.clip(value[10*i+9], 0, 1) * self.open[i]])
        return np.asarray(result)


def guard_targets(previous, targets, config):
    from scipy.spatial.transform import Rotation
    for i, side in enumerate(('left', 'right')):
        target = targets[i]
        bounds = np.asarray(config[side]['workspace_xyz'], float)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
            raise ValueError('Configure measured real-base workspace_xyz for both arms')
        distance = np.linalg.norm(target[:3] - previous[i, :3])
        angle = (Rotation.from_euler('xyz', previous[i, 3:6]).inv() *
                 Rotation.from_euler('xyz', target[3:6])).magnitude()
        if (np.any(target[:3] < bounds[0]) or np.any(target[:3] > bounds[1]) or
                distance > config['max_translation_m'] or angle > config['max_rotation_rad']):
            raise ValueError(f'{side} EEF limit rejected: delta={distance:.4f}m, rotation={angle:.4f}rad')


