"""Official RoboTwin EEF reader plus independently regenerated 8/16/32 labels.

Source HDF5s are read-only. No legacy 10/20/40 target is interpolated or reused.
"""
import hashlib
import json
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.robotwin import RoboTwinDataset
from openwam.dataloader.transforms.rotation import quat_xyzw_to_rotation_6d

PLAN_STEPS = (8, 16, 32)
VIDEO_ANCHORS = tuple(range(0, 33, 4))
CAMERAS = ("head_camera", "left_camera", "right_camera")
MASK_RANDOM_GOALS = frozenset(("blocks_ranking_rgb", "blocks_ranking_size"))


class LiteRoboTwinSource(RoboTwinDataset):
    """Declare this raw collection's wxyz encoding at the reader boundary."""
    def __init__(self, *args, patchwam_augmentation=None, retain_clean_video=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._retain_clean_video = bool(retain_clean_video)
        self._patchwam_augmentation = None
        if patchwam_augmentation and patchwam_augmentation.get('enabled', True) and self.split == 'train':
            if self._color_jitter is not None or not self.multiview:
                raise ValueError('PatchWAM requires multiview=true and the old color_jitter disabled')
            from openwam.dataloader.transforms.patchwam_video import PatchWAMVideoAugment

            self._patchwam_augmentation = PatchWAMVideoAugment(patchwam_augmentation)

    def _read_multiview_frames(self, file, cameras, start, end):
        if self._patchwam_augmentation is not None:
            # Do not infer camera validity from black pixels inside a real view.
            self._patchwam_camera_present = [f'observation/{camera}/rgb' in file for camera in self.camera_layout]
        return super()._read_multiview_frames(file, cameras, start, end)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if self._retain_clean_video:
            sample['xa_clean_video'] = [frame.copy() for frame in sample['video']]
        if self._patchwam_augmentation is not None:
            sample = self._patchwam_augmentation(sample, self._patchwam_camera_present)
        return sample

    def _read_eef_actions(self, file, start, end):
        arms = []
        for arm in ("left", "right"):
            pose = np.asarray(file[f"endpose/{arm}_endpose"][start:end], np.float32)
            grip = np.asarray(file[f"endpose/{arm}_gripper"][start:end], np.float32).reshape(-1, 1)
            quaternion = pose[:, [4, 5, 6, 3]]
            if not np.allclose(np.linalg.norm(quaternion, axis=-1), 1, atol=.01):
                raise ValueError("Raw endpose requires unit wxyz quaternion")
            arms.append(np.concatenate((pose[:, :3], quat_xyzw_to_rotation_6d(quaternion), grip), axis=-1))
        return np.concatenate(arms, axis=-1).astype(np.float32)


def future_local_motion(position, rotation, role_id, valid, horizons=PLAN_STEPS):
    length = len(position)
    directions = np.zeros((length, 3, 3), np.float32)
    moving = np.zeros((length, 3), np.float32)
    future_valid = np.zeros((length, 3), np.float32)
    for index, horizon in enumerate(horizons):
        count = length - horizon
        if count <= 0:
            continue
        delta = np.einsum("tji,tj->ti", rotation[:count], position[horizon:] - position[:count])
        norm = np.linalg.norm(delta, axis=-1)
        pair = (valid[:count] > .5) & (valid[horizon:] > .5)
        pair &= (role_id[:count] > 0) & (role_id[:count] == role_id[horizon:])
        active = (norm > .001) & pair
        directions[:count, index][active] = delta[active] / norm[active, None]
        moving[:count, index] = active
        future_valid[:count, index] = pair
    return directions, moving, future_valid


def targets_from_oracle(oracle, image_sizes, task):
    """Preserve V3 role/near/contact/local-direction semantics; change horizons only."""
    valid = np.asarray(oracle["track_valid"], np.float32)
    length = len(valid)
    expected = {"eef_pose_world_wxyz": (length, 2, 7),
                "track_object_pose_world_wxyz": (length, 2, 7),
                "track_goal_pose_world_wxyz": (length, 2, 7),
                "track_role_id": (length, 2), "track_contact": (length, 2),
                "goal_valid": (length, 2), "arm_present": (length, 2),
                "ground_bbox_pixels": (length, 4, 3, 4),
                "ground_bbox_valid": (length, 4, 3)}
    for key, shape in expected.items():
        value = np.asarray(oracle[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid raw oracle {key}; expected {shape}")
    eef = np.asarray(oracle["eef_pose_world_wxyz"], np.float32)
    obj = np.asarray(oracle["track_object_pose_world_wxyz"], np.float32).copy()
    goal = np.asarray(oracle["track_goal_pose_world_wxyz"], np.float32).copy()
    obj[valid <= .5] = (0, 0, 0, 1, 0, 0, 0)
    goal[valid <= .5] = (0, 0, 0, 1, 0, 0, 0)
    def matrices(poses):
        quat = poses[..., [4, 5, 6, 3]].reshape(-1, 4)
        if not np.allclose(np.linalg.norm(quat, axis=-1), 1, atol=.01):
            raise ValueError("Oracle rotations must be unit wxyz quaternions")
        return Rotation.from_quat(quat).as_matrix().reshape(length, 2, 3, 3)
    er, ob = matrices(eef), matrices(obj)
    relation = np.einsum("taij,tai->taj", er, obj[..., :3] - eef[..., :3])
    proximity = np.clip(1 - np.linalg.norm(goal[..., :3] - obj[..., :3], axis=-1) / .5, 0, 1)
    goal_valid = np.asarray(oracle["goal_valid"], np.float32) * valid
    if task in MASK_RANDOM_GOALS:
        goal_valid[:] = 0
    direction = np.zeros((length, 3, 2, 2, 3), np.float32)
    moving = np.zeros((length, 3, 2, 2), np.float32)
    future_valid = np.zeros((length, 3, 2), np.float32)
    for arm in range(2):
        results = []
        for role, (pose, rot) in enumerate(((eef, er), (obj, ob))):
            d, m, v = future_local_motion(pose[:, arm, :3], rot[:, arm],
                                          oracle["track_role_id"][:, arm], valid[:, arm])
            direction[:, :, arm, role], moving[:, :, arm, role] = d, m
            results.append(v)
        future_valid[:, :, arm] = results[0] * results[1] * valid[:, arm, None]
    boxes = np.asarray(oracle["ground_bbox_pixels"], np.float32).copy()
    ground_valid = np.asarray(oracle["ground_bbox_valid"], np.float32).copy()
    for camera, (width, height) in enumerate(image_sizes):
        boxes[:, :, camera] /= np.array([width, height, width, height], np.float32)
    visible = ground_valid > .5
    if np.any((boxes[visible] < 0) | (boxes[visible] > 1)):
        raise ValueError("Visible bbox is outside its source view")
    boxes[~visible] = 0
    target = np.concatenate((valid, relation.reshape(length, 6), oracle["track_contact"], valid,
                             proximity, goal_valid, direction.reshape(length, 36),
                             moving.reshape(length, 12), future_valid.reshape(length, 6),
                             boxes.reshape(length, 48), ground_valid.reshape(length, 12)), axis=1)
    if target.shape != (length, 130) or not np.isfinite(target).all():
        raise ValueError("Invalid Lite auxiliary labels")
    return target.astype(np.float32)


class EARSIGLiteDataset(BaseDataset):
    """Wrap the official reader; A/B/C share the same samples/actions/stats."""
    action_dim = 80

    def __init__(self, manifest_path, *, tasks=None, auxiliary=True, color_jitter=None, startup_mode="auto",
                 patchwam_augmentation=None, retain_clean_video=False):
        self.manifest_path = str(Path(manifest_path).resolve())
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest.get("contract") != "openwam_alpha_eef32_ea_lite_v1" or manifest.get("split") != "Clean":
            raise ValueError("Need audited Clean-only Lite manifest")
        if manifest.get("source_quaternion_order") != "wxyz" or manifest.get("ea_context") != "observed_endpose_plus_local_x_0.12m_TCP":
            raise ValueError("Need explicit source rotation and EE/TCP contract")
        self.normalization_stats_path = manifest["normalization_stats_path"]
        if hashlib.sha256(Path(self.normalization_stats_path).read_bytes()).hexdigest() != manifest["normalization_stats_sha256"]:
            raise ValueError("Normalization artifact changed")
        self.normalization_stats = np.load(self.normalization_stats_path, allow_pickle=True).item()
        stats = self.normalization_stats["eef"]
        rotation_dims = list(range(3, 9)) + list(range(13, 19))
        if not (np.all(np.asarray(stats["min"])[rotation_dims] == -1) and np.all(np.asarray(stats["max"])[rotation_dims] == 1)):
            raise ValueError("Official Rot6D normalization must be identity")
        self.auxiliary = bool(auxiliary)
        self.episode_records = {}
        self.readers, self.ends, self.task_names = [], [], []
        entries = manifest["tasks"]
        if tasks is not None:
            missing = set(tasks) - {entry["name"] for entry in entries}
            if missing:
                raise ValueError(f"Missing Clean tasks {missing}")
            entries = [entry for entry in entries if entry["name"] in tasks]
        if not entries:
            raise ValueError("Empty training set")
        if startup_mode not in ("auto", "prepared", "scan"):
            raise ValueError("startup_mode must be auto, prepared or scan")
        prepared = all(entry.get("startup_metadata", {}).get("version") == 1 for entry in entries)
        if startup_mode == "prepared" and not prepared:
            raise ValueError("Run scripts/cache_ea_lite_startup.py offline; training never builds missing metadata")
        self._prepared_startup = prepared and startup_mode != "scan"
        total = 0
        for entry in entries:
            metadata = None
            if self._prepared_startup:
                metadata = dict(entry["startup_metadata"],
                                episode_files=[ep["path"] for ep in entry["episodes"]],
                                episode_lengths=[ep["frames"] for ep in entry["episodes"]])
            reader = LiteRoboTwinSource(entry["data_root"], num_frames=33, video_stride=4,
                height=384, width=320, split="train", task_name=entry["name"],
                normalization_stats_path=self.normalization_stats_path, normalize_mode="min-max",
                multiview=True, camera_layout=list(CAMERAS), action_mode="eef", unify_action=True,
                unify_action_map=["0-9", "34-43"], color_jitter=color_jitter, prepared_metadata=metadata,
                patchwam_augmentation=patchwam_augmentation, retain_clean_video=retain_clean_video)
            if reader._normalizer is None:
                raise ValueError("Official EEF normalization is mandatory")
            recorded = {ep["path"]: ep for ep in entry["episodes"]}
            if set(recorded) != set(reader._episode_files):
                raise ValueError("Raw episode inventory changed after preparation")
            if not self._prepared_startup:
                for path, length in zip(reader._episode_files, reader._episode_lengths):
                    stat, record = Path(path).stat(), recorded[path]
                    if (stat.st_size, stat.st_mtime_ns, length) != (record["size"], record["mtime_ns"], record["frames"]):
                        raise ValueError("Source episode changed; regenerate Lite sidecars")
            self.episode_records.update(recorded)
            self.readers.append(reader)
            self.task_names.append(entry["name"])
            total += len(reader)
            self.ends.append(total)

    @classmethod
    def from_config(cls, cfg, split="train"):
        if split != "train":
            raise ValueError("Explicit held-out data required; no implicit training-set validation")
        expected = dict(num_frames=33, video_stride=4, action_mode="eef", unify_action=True,
                        normalize_mode="min-max", variant="clean_50", height=384, width=320)
        if any(cfg.get(k, v) != v for k, v in expected.items()):
            raise ValueError("Lite must preserve the official Alpha EEF/temporal contract")
        return cls(cfg["manifest_path"], tasks=cfg.get("tasks"), auxiliary=cfg.get("auxiliary", True),
                   color_jitter=cfg.get("color_jitter"), startup_mode=cfg.get("startup_mode", "prepared"),
                   patchwam_augmentation=cfg.get("patchwam_augmentation"),
                   retain_clean_video=cfg.get('retain_clean_video',False))

    def __len__(self):
        return self.ends[-1]

    @lru_cache(maxsize=None)
    def _check_source_on_first_read(self, path):
        record = self.episode_records[path]
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (record["size"], record["mtime_ns"]):
            raise ValueError(f"Source episode changed; regenerate Lite sidecars: {path}")

    @lru_cache(maxsize=8)
    def _labels(self, reader_index, path):
        from hashlib import sha256
        record = self.episode_records[path]
        if sha256(Path(record["labels"]).read_bytes()).hexdigest() != record["labels_sha256"]:
            raise ValueError("Corrupt Lite sidecar")
        with np.load(record["labels"]) as data:
            return data["targets"], data["arm_present"], data["effectors"]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ridx = bisect_right(self.ends, index)
        reader = self.readers[ridx]
        local = index - (self.ends[ridx - 1] if ridx else 0)
        ep, start = reader._window_index[local]
        path = reader._episode_files[ep]
        if self._prepared_startup:
            self._check_source_on_first_read(path)
        sample = reader[local]
        if not self.auxiliary:
            return sample
        labels, arms, current = self._labels(ridx, path)
        effectors = current[start]
        sample.update(ea_effectors=torch.from_numpy(effectors.copy()),
                      ea_arm_present=torch.from_numpy(arms[start].copy()),
                      ea_targets=torch.from_numpy(labels[start:start + 1].copy()),
                      ea_plan_steps=list(PLAN_STEPS), ea_video_anchors=list(VIDEO_ANCHORS))
        return sample
