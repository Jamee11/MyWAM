"""Optimizer group helpers for scale-oriented OpenWAM training."""

from __future__ import annotations

import math


def build_trainable_parameters(model, *, action_lr=None, video_lr=None, sidecar_lr=None):
    """Bucket every trainable top-level param by owner, then build optimizer groups:

    - ``video_backbone`` → optional ``video_lr``
    - ``action_backbone`` → optional ``action_lr``
    - ``sidecar`` → optional ``sidecar_lr`` (otherwise the optimizer base LR)
    - every other trainable top-level module (tri_system's ``vlm_backbone`` /
      ``understanding_expert`` / …) → the optimizer base ``learning_rate``, never a
      per-group override.

    Source of truth: ``BaseWAMArchitecture.get_trainable_modules()`` (top-level
    children that hold at least one ``requires_grad`` param). Only ``requires_grad``
    params are collected, so the model yaml ``freeze:`` list — which flips
    ``requires_grad`` — is the single gate for trainability; frozen params never
    reach the optimizer. With no per-group LR override, returns a flat param list;
    otherwise returns groups.

    Temporary WAM warmup keeps future-trainable parameters registered in separate
    groups; sidecar_warmup skips their optimizer updates after normal backward/clipping.
    """
    if sidecar_lr is not None and (not math.isfinite(sidecar_lr) or sidecar_lr <= 0):
        raise ValueError("sidecar_lr must be finite and positive")
    backbone_warmup = bool(getattr(model.architecture, "backbone_warmup_updates", 0))
    action_params, video_params, sidecar_params, other_params = [], [], [], []
    for mod_name, mod in model.architecture.get_trainable_modules().items():
        if mod_name == "video_backbone":
            bucket = video_params
        elif mod_name == "action_backbone":
            bucket = action_params
        elif mod_name == "sidecar" and (sidecar_lr is not None or backbone_warmup):
            bucket = sidecar_params
        else:
            bucket = other_params
        bucket.extend(param for param in mod.parameters() if param.requires_grad)

    if action_lr is None and video_lr is None and sidecar_lr is None and not backbone_warmup:
        return action_params + video_params + other_params

    # Unspecified modules retain the optimizer base LR.
    groups = []
    for params, lr in [(action_params, action_lr), (video_params, video_lr),
                       (sidecar_params, sidecar_lr), (other_params, None)]:
        if not params:
            continue
        group = {"params": params}
        if lr is not None:
            group["lr"] = lr
        if backbone_warmup:
            group["ea_backbone_group"] = params is not sidecar_params
        groups.append(group)
    return groups


__all__ = ["build_trainable_parameters"]
