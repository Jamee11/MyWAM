"""Persistent observation-only relation tokens, outside the checkpointed MoT blocks."""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .ea_rsig_v3 import EARSIGV3Module


def canvas_geometry(height, width, device):
    """Cell centres in the actual L-shaped head/left/right canvas (no resampling)."""
    y, x = torch.meshgrid((torch.arange(height, device=device) + .5) / height,
                          (torch.arange(width, device=device) + .5) / width, indexing="ij")
    view = torch.where(y < 2 / 3, 0, torch.where(x < .5, 1, 2))
    return x, y, view


def canvas_bbox_targets(boxes, valid, height, width):
    """View-normalized xyxy -> full runtime canvas Gaussian, masked within each view."""
    boxes, valid = boxes.float(), valid.float()
    x, y, view = canvas_geometry(height, width, boxes.device)
    offset = boxes.new_tensor([[0, 0], [0, 2 / 3], [.5, 2 / 3]])
    scale = boxes.new_tensor([[1, 2 / 3], [.5, 1 / 3], [.5, 1 / 3]])
    lo = boxes[..., :2] * scale + offset
    hi = boxes[..., 2:] * scale + offset
    centre = (lo + hi) / 2
    sigma = ((hi - lo) / 2).clamp_min(boxes.new_tensor([1 / width, 1 / height]))
    distance = ((x - centre[..., 0, None, None]) / sigma[..., 0, None, None]).square()
    distance = distance + ((y - centre[..., 1, None, None]) / sigma[..., 1, None, None]).square()
    region = view[None, None] == torch.arange(3, device=boxes.device)[None, None, :, None, None]
    density = (torch.exp(-.5 * distance) * region * valid[..., None, None]).sum(2, keepdim=True)
    return density / density.sum((-3, -2, -1), keepdim=True).clamp_min(1e-8)


class ActionWriteAdapter(nn.Module):
    def __init__(self, action_dim, dim, heads, gate_init):
        super().__init__()
        self.query = nn.Sequential(nn.LayerNorm(action_dim), nn.Linear(action_dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0)
        self.projection = nn.Linear(dim, action_dim, bias=False)
        # Only the gate is zero: double-zero initialization would prevent it learning.
        self.beta = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, actions, tokens, valid):
        kv = self.norm(tokens)
        residual = self.attention(self.query(actions), kv, kv, key_padding_mask=~valid,
                                  need_weights=False)[0]
        return actions + self.beta.to(actions.dtype) * self.projection(residual)


class SidecarStage(nn.Module):
    def __init__(self, video_dim, action_dim, dim, heads, read_gate_init, write_gate_init):
        super().__init__()
        self.projector = nn.Sequential(nn.LayerNorm(video_dim), nn.Linear(video_dim, dim))
        self.position = nn.Linear(5, dim)
        self.read_norm = nn.LayerNorm(dim)
        self.read = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0)
        self.alpha = nn.Parameter(torch.tensor(float(read_gate_init)))
        self.refine_norm = nn.LayerNorm(dim)
        self.refine = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.SiLU(),
                                 nn.Linear(2 * dim, dim))
        self.context = nn.Linear(dim, dim)
        self.write = ActionWriteAdapter(action_dim, dim, heads, write_gate_init)

    def forward(self, tokens, current, context, valid, height, width):
        x, y, view = canvas_geometry(height, width, current.device)
        position = torch.cat((torch.stack((x, y), -1), F.one_hot(view, 3)), -1)
        features = self.projector(current) + self.position(position.flatten(0, 1).to(current.dtype))[None]
        read = self.read(self.read_norm(tokens), features, features, need_weights=False)[0]
        tokens = tokens + self.alpha.to(tokens.dtype) * read
        q = self.refine_norm(tokens + self.context(context)[:, None])
        tokens = tokens + self.refine(q, q, q, key_padding_mask=~valid, need_weights=False)[0]
        tokens = (tokens + self.ffn(tokens)) * valid[..., None]
        return tokens, features


class EAHeads(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ground_query = nn.Linear(dim, dim)
        self.ground_key = nn.Linear(dim, dim)
        self.presence = nn.Linear(dim, 1)
        self.track = nn.Linear(2 * dim, 1)
        self.near = nn.Linear(dim, 1)
        self.contact = nn.Linear(dim, 1)
        self.goal = nn.Linear(dim, 1)
        self.plan = nn.Linear(dim, 8)

    def forward(self, tokens, features, arms, height, width):
        b, _, dim = tokens.shape
        logits = torch.einsum("brd,bnd->brn", self.ground_query(tokens[:, :4]),
                               self.ground_key(features)) * dim ** -.5
        logits = logits.reshape(b, 4, 1, height, width)
        relations = tokens[:, 4:8].reshape(b, 2, 2, dim)
        raw = self.plan(tokens[:, 9:]).reshape(b, 3, 2, 2, 4)
        return dict(tokens=tokens, first_frame_features=features, grid=(height, width),
                    grounding=dict(logits=logits, attention=logits.flatten(2).softmax(-1),
                                   presence_logits=self.presence(tokens[:, :4]).squeeze(-1)),
                    track_valid_logits=self.track(tokens[:, :4].reshape(b, 2, 2 * dim)).squeeze(-1),
                    arm_present=arms, near_logits=self.near(relations[:, :, 0]).squeeze(-1),
                    contact_logits=self.contact(relations[:, :, 0]).squeeze(-1),
                    goal_proximity=self.goal(relations[:, :, 1]).squeeze(-1).sigmoid(),
                    plan_directions=F.normalize(raw[..., :3], dim=-1), plan_moving_logits=raw[..., 3])

    @staticmethod
    def losses(output, targets):
        if targets.ndim != 3 or targets.shape[1:] != (1, 130) or not torch.isfinite(targets).all():
            raise ValueError("Sidecar needs finite, separate [B,1,130] auxiliary targets")
        targets = targets.clone()
        arms = output["arm_present"].to(targets.dtype)
        targets[:, 0, :2] *= arms
        targets[:, 0, 118:130] *= arms.repeat_interleave(2, 1).repeat_interleave(3, 1)
        targets[:, 0, 121:124] *= targets[:, 0, 14:15]
        targets[:, 0, 127:130] *= targets[:, 0, 15:16]
        # Reuse the established V3 masked relation/plan definitions; only spatial geometry differs.
        proxy = SimpleNamespace(num_views=3, near_threshold=.10, near_loss_weight=.5,
                                goal_loss_weight=.5, bbox_gaussian_targets=lambda boxes, valid:
                                canvas_bbox_targets(boxes, valid, *output["grid"]))
        return EARSIGV3Module._dual_track_losses(proxy, output, targets)


class RelationalSidecar(nn.Module):
    def __init__(self, video_dim, action_dim, text_dim, *, latent_dim=512, heads=8,
                 num_ea_tokens=15, read_gate_init=.01, write_gate_init=0., action_conditioning=True):
        super().__init__()
        if num_ea_tokens != 15 or latent_dim % heads:
            raise ValueError("Sidecar requires 15 tokens and latent_dim divisible by heads")
        self.action_conditioning = bool(action_conditioning)
        self.role_tokens = nn.Parameter(torch.randn(15, latent_dim) * .02)
        self.text = nn.Linear(text_dim, latent_dim)
        self.proprio = nn.Sequential(nn.Linear(160, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim))
        self.stages = nn.ModuleList([SidecarStage(video_dim, action_dim, latent_dim, heads,
                                                read_gate_init, write_gate_init) for _ in range(3)])
        self.heads = EAHeads(latent_dim)
        if not self.action_conditioning:
            for stage in self.stages:
                stage.write.requires_grad_(False)

    def init_tokens(self, text, text_mask, proprio, proprio_mask):
        b = text.shape[0]
        proprio = proprio.reshape(b, 80)
        mask = proprio_mask.reshape(b, 80).bool()
        if not torch.isfinite(proprio).all() or not torch.isfinite(text).all():
            raise ValueError("Nonfinite current observation")
        arms = torch.stack((mask[:, :10].any(-1), mask[:, 34:44].any(-1)), -1)
        if not arms.any(-1).all():
            raise ValueError("At least one observed arm must be present")
        valid = torch.cat((arms.repeat_interleave(2, 1), arms.repeat_interleave(2, 1),
                           arms.any(-1, keepdim=True), arms.repeat(1, 3)), -1)
        if text_mask is None:
            text_mask = torch.ones(text.shape[:2], device=text.device, dtype=torch.bool)
        weights = text_mask.to(text.dtype)
        summary = (text * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1)
        context = self.text(summary) + self.proprio(torch.cat((proprio * mask, mask.to(proprio.dtype)), -1))
        tokens = (self.role_tokens[None].to(context.dtype) + context[:, None]) * valid[..., None]
        return tokens, context, valid, arms

    def read_refine(self, index, tokens, current, context, valid, arms, height, width):
        if current.shape[1] != height * width:
            raise ValueError("Sidecar must read exactly the runtime first-frame grid")
        tokens, features = self.stages[index](tokens, current, context, valid, height, width)
        return tokens, self.heads(tokens, features, arms, height, width)

    def write_action(self, index, actions, tokens, valid):
        return self.stages[index].write(actions, tokens, valid) if self.action_conditioning else actions
