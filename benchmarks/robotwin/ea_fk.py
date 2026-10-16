"""Aloha FK snapshot from DiT4DiT (MIT); no imports from the old project."""
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ARM_LINK_NAMES = ("fl_link6", "fr_link6")

def _transform(xyz, rotation):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3], result[:3, 3] = rotation, xyz
    return result


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
         [-sp, cp * sr, cp * cr]],
        dtype=np.float64,
    )


class AlohaBaseKinematics:
    """Minimal URDF FK for the two RoboTwin link6 frames, in robot base."""

    def __init__(self, urdf: Path) -> None:
        children: dict[str, list[tuple[str, str, np.ndarray, str, np.ndarray]]] = {}
        self.joints: dict[str, tuple[str, str, np.ndarray, str, np.ndarray]] = {}
        for joint in ET.parse(urdf).getroot().findall("joint"):
            name = joint.attrib["name"]
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            origin = joint.find("origin")
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            axis_node = joint.find("axis")
            axis = np.fromstring(axis_node.attrib.get("xyz", "1 0 0"), sep=" ") if axis_node is not None else np.array([1.0, 0.0, 0.0])
            record = (parent, child, _transform(xyz, _rpy_matrix(rpy)), joint.attrib["type"], axis)
            self.joints[name] = record
            children.setdefault(parent, []).append((child, name, *record[2:]))
        self.children = children

    def poses(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.asarray(qpos).shape != (14,):
            raise ValueError(f"Expected 14D source qpos, got {np.asarray(qpos).shape}")
        values = {f"fl_joint{i + 1}": float(qpos[i]) for i in range(6)}
        values.update({f"fr_joint{i + 1}": float(qpos[7 + i]) for i in range(6)})
        link_poses = {"footprint": np.eye(4, dtype=np.float64)}
        pending = ["footprint"]
        while pending:
            parent = pending.pop()
            for child, name, origin, kind, axis in self.children.get(parent, []):
                joint_pose = link_poses[parent] @ origin
                motion = np.eye(4, dtype=np.float64)
                if kind in {"revolute", "continuous"}:
                    motion[:3, :3] = Rotation.from_rotvec(axis * values.get(name, 0.0)).as_matrix()
                elif kind == "prismatic":
                    motion[:3, 3] = axis * values.get(name, 0.0)
                link_poses[child] = joint_pose @ motion
                pending.append(child)
        try:
            return tuple(link_poses[name] for name in ARM_LINK_NAMES)  # type: ignore[return-value]
        except KeyError as error:
            raise ValueError(f"URDF lacks expected end link: {error}") from error


class AlohaEEF:
    """Same unnormalized 20D FK features as the schema-5 converter."""
    def __init__(self, urdf, config, conversion_summary):
        urdf = Path(urdf)
        summary = json.loads(Path(conversion_summary).read_text())
        if "drive-target" not in summary.get("deployable_effector_contract", ""):
            raise ValueError("This client requires drive-target EEF labels; actual-joint FK needs its own sensor adapter")
        expected = summary["base_delta_ee_backup"]
        if expected.get("kinematics_version") != "urdf_child_link6_v2":
            raise ValueError("Unsupported FK convention; do not silently substitute Piper")
        if hashlib.sha256(urdf.read_bytes()).hexdigest() != expected["urdf_sha256"]:
            raise ValueError("URDF hash differs from training data")
        self.fk = AlohaBaseKinematics(urdf)
        self.bias = float(yaml.safe_load(Path(config).read_text())["gripper_bias"])

    def __call__(self, qpos):
        qpos = np.asarray(qpos, dtype=np.float32)
        if qpos.shape != (14,) or not np.isfinite(qpos).all():
            raise ValueError("Expected finite native Aloha qpos [left6,grip,right6,grip]")
        result = np.zeros((2, 10), dtype=np.float32)
        for arm, pose in enumerate(self.fk.poses(qpos)):
            r = pose[:3, :3]
            result[arm, :3] = pose[:3, 3] + r @ np.array([self.bias, 0, 0])
            result[arm, 3:9] = r[:2].reshape(6)
            result[arm, 9] = qpos[6 if arm == 0 else 13]
        return result
