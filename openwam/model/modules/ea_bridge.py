"""Explicit current-frame EA-RSIG conditioning, with no oracle or mutable hooks."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .ea_rsig_v3 import EARSIGV3Module, validate_ea_rsig_v3_video_plan_alignment


class EARSIGBridge(nn.Module):
    def __init__(self, video_dim, action_dim, *, layer_index, action_horizon,
                 video_anchors, layout="robotwin", num_views=3, num_tracks=2,
                 num_arms=2, grid_height=8, view_grid_width=10, slot_dim=512,
                 num_heads=8, action_conditioning=True, video_residual=False,
                 loss_scale=0.1, ground_weight=0.25, relation_weight=1.0,
                 plan_weight=1.0, action_space="qpos"):
        super().__init__()
        self.layer_index = int(layer_index)
        self.action_horizon = int(action_horizon)
        self.video_anchors = tuple(int(x) for x in video_anchors)
        self.plan_steps = validate_ea_rsig_v3_video_plan_alignment(action_horizon, video_anchors)
        if layout not in ("robotwin", "horizontal") or (layout == "robotwin" and num_views != 3):
            raise ValueError("Invalid camera layout/views")
        if self.layer_index < 0 or grid_height < 1 or view_grid_width < 1:
            raise ValueError("Invalid layer/grid")
        self.layout = layout
        if action_space not in ("qpos", "robot_base_delta_ee"):
            raise ValueError("Declare the action space explicitly")
        self.action_space = action_space
        self.slot_dim = int(slot_dim)
        self.action_conditioning = bool(action_conditioning)
        self.video_residual = bool(video_residual)
        self.loss_scale = float(loss_scale)
        self.loss_weights = dict(ground=ground_weight, relation=relation_weight, plan=plan_weight)
        self.core = EARSIGV3Module(hidden_dim=video_dim, slot_dim=slot_dim,
                                  num_views=num_views, num_arms=num_arms,
                                  num_tracks=num_tracks, grid_height=grid_height,
                                  view_grid_width=view_grid_width, num_heads=num_heads)
        self.action_projection = nn.Linear(video_dim, action_dim, bias=False)

    def contract(self):
        contract = dict(version=1, layer_index=self.layer_index, action_horizon=self.action_horizon,
                    action_space=self.action_space, slot_dim=self.slot_dim,
                    video_anchors=list(self.video_anchors), plan_steps=list(self.plan_steps),
                    layout=self.layout, num_views=self.core.num_views,
                    num_tracks=self.core.num_tracks, num_arms=self.core.num_arms,
                    target_dim=self.core.target_dim, grid_height=self.core.grid_height,
                    view_grid_width=self.core.view_grid_width,
                    action_conditioning=self.action_conditioning,
                    video_residual=self.video_residual)
        # Preserve existing first-frame checkpoint contracts exactly.
        if getattr(self, "joint_attention", False):
            contract["action_video_attention"] = "full_video"
        return contract

    def observe(self, hidden, grid, effectors, arm_present):
        frames, height, width = map(int, grid)
        if hidden.shape[1] != frames * height * width:
            raise ValueError("Video grid/token count mismatch")
        current = hidden[:, :height * width].reshape(hidden.shape[0], height, width, -1)
        current = current.permute(0, 3, 1, 2)
        if self.layout == "robotwin":
            if height % 3 or width % 2:
                raise ValueError("RoboTwin grid must preserve the 2:1 head/wrist layout")
            split = 2 * height // 3
            views = [current[:, :, :split], current[:, :, split:, :width//2],
                     current[:, :, split:, width//2:]]
        else:
            if width % self.core.num_views:
                raise ValueError("Horizontal grid must divide into equal camera views")
            views = current.chunk(self.core.num_views, dim=-1)
        views = [F.interpolate(v, size=(self.core.grid_height, self.core.view_grid_width),
                               mode="bilinear", align_corners=False) for v in views]
        # Core expects row-major horizontal VHW, with boxes normalized per view.
        packed = torch.cat(views, dim=-1).flatten(2).transpose(1, 2)
        output = self.core(packed, effectors.to(hidden), arm_present.to(hidden))
        output["condition_tokens"] = self.action_projection(output["action_tokens"])
        return output

    @staticmethod
    def append_context(payload, tokens):
        if payload is None or payload.get("context") is None:
            raise ValueError("EA action conditioning requires the native action context")
        context = payload["context"]
        mask = payload.get("mask")
        if mask is not None:
            added = torch.ones((*mask.shape[:-1], tokens.shape[1]), device=mask.device, dtype=mask.dtype)
            mask = torch.cat((mask, added), dim=-1)
        return {**payload, "context": torch.cat((context, tokens.to(context)), dim=1), "mask": mask}

    def inject_future(self, video_tokens, grid, output):
        """No re-prediction, no retained graph, no change to the current-frame prefix."""
        prefix = int(grid[1]) * int(grid[2])
        if not self.video_residual or video_tokens.shape[1] <= prefix:
            return video_tokens
        gate = output["track_gate"]
        any_track = 1 - (1 - gate).prod(dim=1, keepdim=True)
        eo, og = output["eef_object_relations"], output["object_goal_relation"]
        if self.core.num_tracks == 2:
            relation = torch.cat(((eo * gate[..., None]).flatten(1),
                                  (og * gate[..., None]).flatten(1),
                                  output["progress"] * any_track), dim=-1)
        else:
            relation = torch.cat((eo[:, 0] * gate, og[:, 0] * gate,
                                  output["progress"] * any_track), dim=-1)
        future = video_tokens[:, prefix:]
        condition = self.core.video_context(relation).to(future.dtype)
        delta = self.core.video_out(F.silu(self.core.video_query(future) + condition[:, None]))
        return torch.cat((video_tokens[:, :prefix], future + self.core.video_alpha.to(future.dtype) * delta), dim=1)

    def validate_sample(self, sample):
        spaces = sample.get("ea_action_space")
        if isinstance(spaces, str):
            spaces = [spaces]
        if not spaces or any(s != self.action_space for s in spaces):
            raise ValueError("Sample action space disagrees with EA contract")
        if sample["action"].shape[1] != self.action_horizon:
            raise ValueError("EA action horizon does not match the model contract")
        if sample["video"].shape[2] != len(self.video_anchors):
            raise ValueError("EA video frame count does not match anchors")
        # Metadata stays on CPU. No GPU .item() in the per-microbatch path.
        for key, expected in (("ea_plan_steps", self.plan_steps), ("ea_video_anchors", self.video_anchors)):
            value = sample.get(key)
            if value is None:
                raise ValueError(f"Missing explicit temporal metadata: {key}")
            if isinstance(value, str) or (isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value)):
                values = [value] if isinstance(value, str) else value
                if any(v != ",".join(map(str, expected)) for v in values):
                    raise ValueError(f"{key} disagrees with EA temporal contract")
                continue
            value = torch.as_tensor(value)
            if value.device.type != "cpu":
                raise ValueError(f"Keep {key} on CPU for validation")
            if value.ndim == 1:
                value = value.unsqueeze(0)
            target = torch.tensor(expected).expand(value.shape[0], -1)
            if value.shape != target.shape or not torch.equal(value, target):
                raise ValueError(f"{key} disagrees with EA temporal contract")
        targets = sample.get("ea_targets")
        if targets is None or targets.shape != (sample["action"].shape[0], 1, self.core.target_dim):
            raise ValueError("Missing or invalid EA supervision shape")

    def losses(self, output, targets):
        parts = self.core.losses(output, targets)
        loss = self.loss_scale * sum(self.loss_weights[k] * v for k, v in parts.items())
        return loss, {"loss_ea": loss.detach(), **{f"loss_ea_{k}": v.detach() for k, v in parts.items()}}
