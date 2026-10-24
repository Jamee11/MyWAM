"""Freeze WAM updates without rebuilding ZeRO's static parameter partitions."""

import torch


def install_backbone_warmup(optimizer, architecture):
    """Keep activation gradients, zero WAM leaf gradients, and skip its AdamW step.

    ZeRO-2 calls the underlying AdamW once per flattened parameter group. Group
    metadata survives that flattening; clearing its gradient skips both decay
    and momentum/step updates, unlike LR=0 or a zero gradient alone.
    """
    if not architecture.backbone_warmup_updates:
        return
    if not any(g.get("ea_backbone_group") is True for g in optimizer.param_groups) or not any(
        g.get("ea_backbone_group") is False for g in optimizer.param_groups
    ):
        raise ValueError("Backbone warmup needs separate WAM and Sidecar optimizer groups")

    def parameter_gradient(gradient):
        # Keep the distributed graph static and exclude WAM from global clipping.
        return torch.zeros_like(gradient) if architecture.backbone_warmup_active else gradient

    for name, parameter in architecture.named_parameters():
        if parameter.requires_grad and not name.startswith("sidecar."):
            parameter.register_hook(parameter_gradient)

    def before_step(opt, args, kwargs):
        if architecture.backbone_warmup_active:
            for group in opt.param_groups:
                if group["ea_backbone_group"]:
                    for parameter in group["params"]:
                        parameter.grad = None

    optimizer.register_step_pre_hook(before_step)
