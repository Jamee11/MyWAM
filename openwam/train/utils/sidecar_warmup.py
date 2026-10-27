"""Freeze WAM updates without rebuilding ZeRO's static parameter partitions."""


def install_backbone_warmup(optimizer, architecture):
    """Keep full backward and gradient clipping; skip only WAM's AdamW step.

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

    def before_step(opt, args, kwargs):
        if architecture.backbone_warmup_active:
            for group in opt.param_groups:
                if group["ea_backbone_group"]:
                    for parameter in group["params"]:
                        parameter.grad = None

    optimizer.register_step_pre_hook(before_step)
