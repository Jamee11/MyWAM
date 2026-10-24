"""Single-arm ManiSkill oracle -> existing Sidecar 130D supervision, never policy inputs."""

import numpy as np

from openwam.dataloader.ea_rsig_lite import targets_from_oracle

CONTRACT = "maniskill_panda_sidecar130_v1"


def build_targets(tcp, objects, goals, object_ids, goal_ids, contact_force, boxes, visible):
    """T+1 observed poses; T segmentation boxes. Keep terminal observations for plan anchors.

    Boxes are source-view xyxy pixels, front then hand; the missing third view
    and track1 are invalid. Goal geometry can be valid while the goal is hidden.
    """
    tcp, objects, goals = (np.asarray(x, np.float32) for x in (tcp, objects, goals))
    length = len(tcp)
    if length < 2 or tcp.shape != (length, 7) or objects.shape != tcp.shape or goals.shape != (length, 3):
        raise ValueError("Need T+1 synchronized TCP/object/goal observations")
    if not all(np.isfinite(x).all() for x in (tcp, objects, goals)):
        raise ValueError("Nonfinite observed pose")
    ids = np.asarray(object_ids).reshape(length)
    gids = np.asarray(goal_ids).reshape(length)
    if not (ids > 0).all() or not (ids == ids[0]).all():
        raise ValueError("Expected one persistent target actor; role switches require explicit handling")
    if not (gids == gids[0]).all():
        raise ValueError("Goal actor changed within an episode")
    boxes, visible = np.asarray(boxes, np.float32), np.asarray(visible, np.float32)
    if boxes.shape != (length - 1, 2, 2, 4) or visible.shape != (length - 1, 2, 2):
        raise ValueError("Need T x object/goal x front/hand boxes and visibility")
    if not np.isfinite(boxes).all() or not np.isin(visible, [0, 1]).all():
        raise ValueError("Invalid segmentation supervision")
    selected = boxes[visible > 0]
    if np.any((selected < 0) | (selected > 256)) or np.any(selected[:, 2:] <= selected[:, :2]):
        raise ValueError("Visible exclusive-xyxy bbox is outside 256x256 source image")
    poses = np.zeros((length, 2, 7), np.float32)
    poses[..., 3] = 1
    eef, obj, goal = poses.copy(), poses.copy(), poses.copy()
    eef[:, 0], obj[:, 0], goal[:, 0, :3] = tcp, objects, goals
    valid = np.zeros((length, 2), np.float32)
    valid[:, 0] = 1
    bbox = np.zeros((length, 4, 3, 4), np.float32)
    ground_valid = np.zeros((length, 4, 3), np.float32)
    bbox[:-1, :2, :2], ground_valid[:-1, :2, :2] = boxes, visible
    # No visible-goal target for virtual actors, but retain their known goal geometry.
    ground_valid[:, 1] *= (gids > 0)[:, None]
    contact = np.zeros((length, 2), np.float32)
    if contact_force is not None:
        force = np.asarray(contact_force).reshape(length)
        if not np.isfinite(force).all() or (force < 0).any():
            raise ValueError("Invalid physical contact force")
        contact[:, 0] = force > 1e-6
    oracle = dict(
        track_valid=valid, eef_pose_world_wxyz=eef, track_object_pose_world_wxyz=obj,
        track_goal_pose_world_wxyz=goal, track_role_id=np.stack((ids, np.zeros(length)), -1),
        track_contact=contact, goal_valid=valid.copy(), arm_present=valid.copy(),
        ground_bbox_pixels=bbox, ground_bbox_valid=ground_valid,
    )
    targets = targets_from_oracle(oracle, [(256, 256)] * 3, "maniskill")[:-1]
    if contact_force is None:
        targets[:, 10] = 0  # Unknown is not a known negative contact.
    return targets
