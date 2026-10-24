"""Common numeric and image contract for the independently prepared sim mixture."""

import cv2
import numpy as np
from PIL import Image

CONTRACT = "openwam_multisim_absolute_eef20_sidecar130_v1"
PLAN_STEPS = (8, 16, 32)
VIDEO_ANCHORS = tuple(range(0, 33, 4))
ACTION_SLOTS = tuple(range(10)) + tuple(range(34, 44))
SOURCE_GROUPS = ("rlbench_franka", "maniskill_franka", "robotwin_aloha", "robotwin_piper")


def validate_numeric(state, action, targets, arms):
    state, action, targets = (np.asarray(x, np.float32) for x in (state, action, targets))
    arms = np.asarray(arms, bool)
    length = len(action)
    if length < 1 or state.shape != (length + 1, 20) or action.shape != (length, 20):
        raise ValueError("Need T absolute commands and T+1 observed EEF20 states")
    if targets.shape != (length, 130) or arms.shape != (2,) or not arms.any():
        raise ValueError("Need T x130 targets and explicit observed-arm mask")
    if not all(np.isfinite(x).all() for x in (state, action, targets)):
        raise ValueError("Nonfinite prepared numeric data")
    for i, present in enumerate(arms):
        section = slice(10 * i, 10 * (i + 1))
        if not present:
            if np.any(state[:, section]) or np.any(action[:, section]) or np.any(targets[:, i]):
                raise ValueError("Absent-arm numeric values and track validity must be zero")
            continue
        for values in (state[:, section], action[:, section]):
            a, b = values[:, 3:6], values[:, 6:9]
            if not (np.allclose(np.linalg.norm(a, axis=1), 1, atol=.015)
                    and np.allclose(np.linalg.norm(b, axis=1), 1, atol=.015)
                    and np.allclose((a * b).sum(1), 0, atol=.015)):
                raise ValueError("EEF rotation must be two orthonormal matrix columns")
            if (values[:, 9] < -.001).any() or (values[:, 9] > 1.001).any():
                raise ValueError("Gripper must use 0=closed, 1=open")
    return state, action, targets, arms


def pad_single_arm(values):
    values = np.asarray(values, np.float32)
    if values.ndim != 2 or values.shape[1] != 10:
        raise ValueError("Expected native single-arm EEF10")
    return np.pad(values, ((0, 0), (0, 10)))


def decode_rgb(value, codec):
    if codec == "uint8_rgb":
        pixels = np.asarray(value)
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3:
            raise ValueError("Expected HxWx3 uint8 RGB")
    else:
        content = bytes(value)
        marked = b"XPL-RGB1" in content
        if codec not in ("robotwin_legacy_rgb_jpeg", "standard_rgb_jpeg"):
            raise ValueError(f"Unsupported image codec: {codec}")
        if marked and codec != "standard_rgb_jpeg":
            raise ValueError("Corrected RGB JPEG cannot use the legacy channel decoder")
        pixels = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if pixels is None:
            raise ValueError("Corrupt JPEG")
        if codec == "standard_rgb_jpeg":
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
    return Image.fromarray(pixels)


def segmentation_boxes(segmentation, object_ids, goal_ids):
    """Same-frame, exclusive-xyxy boxes; invisible and virtual actors stay invalid."""
    seg = np.asarray(segmentation)
    if seg.ndim != 4 or seg.shape[-1] != 1:
        raise ValueError("Need T+1 x H x W x1 actor segmentation")
    ids = [np.asarray(x).reshape(len(seg)) for x in (object_ids, goal_ids)]
    boxes = np.zeros((len(seg), 2, 4), np.float32)
    visible = np.zeros((len(seg), 2), np.float32)
    for role, actors in enumerate(ids):
        for t, actor in enumerate(actors):
            if actor <= 0:
                continue
            y, x = np.nonzero(seg[t, ..., 0] == actor)
            if len(x):
                boxes[t, role] = (x.min(), y.min(), x.max() + 1, y.max() + 1)
                visible[t, role] = 1
    return boxes, visible


def single_arm_targets(tcp, objects, goals, object_ids, contact, boxes, visible, goal_valid):
    from openwam.dataloader.ea_rsig_lite import targets_from_oracle

    tcp, objects, goals = (np.asarray(x, np.float32) for x in (tcp, objects, goals))
    length = len(tcp)
    if tcp.shape != (length, 7) or objects.shape != tcp.shape or goals.shape != (length, 3):
        raise ValueError("Need synchronized observed world poses, including terminal observation")
    pose = np.zeros((length, 2, 7), np.float32)
    pose[..., 3] = 1
    eef, obj, goal = pose.copy(), pose.copy(), pose.copy()
    eef[:, 0], obj[:, 0], goal[:, 0, :3] = tcp, objects, goals
    valid = np.zeros((length, 2), np.float32)
    valid[:, 0] = 1
    goal_mask = valid.copy()
    goal_mask[:, 0] *= np.asarray(goal_valid).reshape(length)
    ids = np.zeros((length, 2), np.int64)
    ids[:, 0] = np.asarray(object_ids).reshape(length)
    forces = np.zeros((length, 2), np.float32)
    if contact is not None:
        forces[:, 0] = np.asarray(contact).reshape(length)
    bbox = np.zeros((length, 4, 3, 4), np.float32)
    visibility = np.zeros((length, 4, 3), np.float32)
    bbox[:, :2, :2], visibility[:, :2, :2] = boxes, visible
    visibility[:, 1] *= goal_mask[:, 0, None]
    oracle = dict(track_valid=valid, eef_pose_world_wxyz=eef, track_object_pose_world_wxyz=obj,
                  track_goal_pose_world_wxyz=goal, track_role_id=ids, track_contact=forces,
                  goal_valid=goal_mask, arm_present=valid.copy(),
                  ground_bbox_pixels=bbox, ground_bbox_valid=visibility)
    result = targets_from_oracle(oracle, [(256, 256)] * 3, "maniskill")[:-1]
    if contact is None:
        result[:, 10] = 0
    return result
