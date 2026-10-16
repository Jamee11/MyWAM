"""Causal patch-grid EA and a zero-initialized Action-stream residual.

No video block is run here, and no tokens are added to joint attention.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .ea_rsig_v3 import EARSIGV3Module


class EARSIGLiteBridge(nn.Module):
    def __init__(self, video_dim, action_dim, text_dim, *, slot_dim=512,
                 num_heads=8, grid_height=14, view_grid_width=14,
                 action_conditioning=True):
        super().__init__()
        self.action_conditioning = bool(action_conditioning)
        self.patch_projection = nn.Sequential(nn.LayerNorm(video_dim), nn.Linear(video_dim, slot_dim))
        self.text_projection = nn.Linear(text_dim, slot_dim)
        self.core = EARSIGV3Module(hidden_dim=slot_dim, slot_dim=slot_dim,
                                  num_views=3, num_arms=2, num_tracks=2,
                                  grid_height=grid_height, view_grid_width=view_grid_width,
                                  num_heads=num_heads)
        # Retain the reusable V3 core without enabling its legacy video branch.
        for name, parameter in self.core.named_parameters():
            if name.startswith("video_"):
                parameter.requires_grad_(False)
        self.action_query = nn.Linear(action_dim, slot_dim)
        self.action_attention = nn.MultiheadAttention(slot_dim, num_heads, dropout=0, batch_first=True)
        self.action_out = nn.Linear(slot_dim, action_dim, bias=False)
        nn.init.zeros_(self.action_out.weight)
        if not self.action_conditioning:
            unused = ("action_query.", "action_attention.", "action_out.",
                      "core.ground_to_hidden.", "core.relation_to_hidden.", "core.progress_to_hidden.",
                      "core.eef_plan_to_hidden.", "core.object_plan_to_hidden.",
                      "core.ground_gate", "core.relation_gate", "core.progress_gate")
            for name, parameter in self.named_parameters():
                if name.startswith(unused):
                    parameter.requires_grad_(False)

    def observe(self, patches, context, context_mask, effectors, arms):
        """patches: clean CURRENT-frame patchify output [B,C,1,H,W]."""
        if patches.ndim != 5 or patches.shape[2] != 1:
            raise ValueError("Lite requires exactly one clean observation frame")
        b, _, _, h, w = patches.shape
        if h % 3 or w % 2:
            raise ValueError("Patch grid must retain RoboTwin head/left/right layout")
        if effectors.shape != (b, 2, 10) or arms.shape != (b, 2):
            raise ValueError("Expected current EEF [B,2,10] and arm mask [B,2]")
        if not torch.isfinite(effectors).all() or not torch.isfinite(arms).all():
            raise ValueError("Nonfinite current EEF context")
        if not torch.all((arms == 0) | (arms == 1)) or not torch.all(arms.any(dim=-1)):
            raise ValueError("Invalid current arm mask")
        current = patches[:, :, 0]
        split = 2 * h // 3
        views = (current[:, :, :split], current[:, :, split:, :w // 2], current[:, :, split:, w // 2:])
        resized = [F.interpolate(v, size=(self.core.grid_height, self.core.view_grid_width),
                                 mode="bilinear", align_corners=False) for v in views]
        packed = torch.cat(resized, dim=-1).flatten(2).transpose(1, 2)
        features = self.patch_projection(packed)
        mask = (torch.ones(context.shape[:2], device=context.device, dtype=torch.bool)
                if context_mask is None else context_mask.bool())
        text = (context * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        # Instruction-dependent channel scaling preserves the spatial grid.
        features = features * (1 + 0.1 * self.text_projection(text).tanh()[:, None])
        return self.core(features, effectors.to(features), arms.to(features))

    def condition_action(self, hidden, tokens):
        if not self.action_conditioning:
            return hidden
        if tokens.shape[1:] != (15, self.core.slot_dim):
            raise ValueError("Lite requires 4 ground + 4 relation + 1 progress + 6 plan tokens")
        query = self.action_query(hidden)
        condition, _ = self.action_attention(query, tokens, tokens, need_weights=False)
        return hidden + self.action_out(condition)
