"""Opt-in LR interventions retaining the original ZeRO groups and Adam history."""

import re

import torch
from torch.optim.adamw import adamw


class SidecarLRScreen:
    def __init__(self, optimizer, architecture, mode, ramp_updates=300):
        if mode not in ("control", "wam_ramp", "write2"):
            raise ValueError("Unknown Sidecar LR screening mode")
        if not isinstance(optimizer, torch.optim.AdamW) or ramp_updates < 1:
            raise ValueError("LR screening requires AdamW and a positive ramp")
        self.architecture, self.mode, self.ramp_updates = architecture, mode, ramp_updates
        self.write = [p for n, p in architecture.named_parameters()
                      if re.fullmatch(r"sidecar\.stages\.\d+\.write\..+", n) and p.requires_grad]
        if mode == "write2" and not self.write:
            raise ValueError("No actual stages.i.write parameters matched")
        self.pending, self.restore_lrs, self.trace = [], [], {}
        optimizer.register_step_pre_hook(self.before)
        optimizer.register_step_post_hook(self.after)

    def factor(self):
        u = self.architecture.completed_optimizer_updates
        start = self.architecture.backbone_warmup_updates
        return .1 + .9 * min(max((u - start) / self.ramp_updates, 0), 1)

    def fragments(self, group):
        for parameter in self.write:
            mapping = getattr(parameter, "_hp_mapping", None)
            if mapping is None:
                for master in group["params"]:
                    if master is parameter:
                        yield master, 0, master.numel()
                continue
            fragment = mapping.get_hp_fragment()
            for master in group["params"]:
                if fragment.untyped_storage().data_ptr() == master.untyped_storage().data_ptr():
                    address = mapping.get_hp_fragment_address()
                    yield master, address.start, address.numel

    @torch.no_grad()
    def before(self, optimizer, args, kwargs):
        if self.pending or self.restore_lrs:
            raise RuntimeError("Previous LR intervention did not finish")
        for group in optimizer.param_groups:
            if "ea_backbone_group" not in group:
                raise ValueError("LR screening needs the original WAM/Sidecar group metadata")
            if not any(p.grad is not None for p in group["params"]):
                continue
            backbone = group["ea_backbone_group"]
            base = float(group["lr"])
            factor = self.factor() if self.mode == "wam_ramp" and backbone else 1.0
            if factor != 1:
                self.restore_lrs.append((group, base))
                group["lr"] = base * factor
            self.trace["wam" if backbone else "sidecar"] = dict(scheduled=base, effective=base * factor)
            if self.mode != "write2" or backbone:
                continue
            if self.architecture.completed_optimizer_updates < self.architecture.backbone_warmup_updates:
                continue
            params, grads, first, second, steps, destinations = [], [], [], [], [], []
            for master, offset, length in self.fragments(group):
                if master.grad is None:
                    continue
                state = optimizer.state[master]
                if not state or group.get("amsgrad") or group.get("differentiable"):
                    raise ValueError("Write2 needs restored ordinary AdamW moments")
                view = master.view(-1).narrow(0, offset, length)
                destinations.append(view)
                params.append(view.clone())
                grads.append(master.grad.view(-1).narrow(0, offset, length))
                first.append(state["exp_avg"].view(-1).narrow(0, offset, length).clone())
                second.append(state["exp_avg_sq"].view(-1).narrow(0, offset, length).clone())
                steps.append(state["step"].clone())
            if params:
                # Re-evaluate only Write slices at 2x LR, with cloned moments/counters.
                # The ordinary step still updates the original moments exactly once.
                adamw(params, grads, first, second, [], steps,
                      foreach=group.get("foreach"), capturable=group.get("capturable", False),
                      differentiable=False, fused=group.get("fused"), amsgrad=False,
                      beta1=group["betas"][0], beta2=group["betas"][1], lr=2 * base,
                      weight_decay=group["weight_decay"], eps=group["eps"], maximize=group["maximize"])
                self.pending = list(zip(destinations, params))
            self.trace["write"] = dict(scheduled=base, effective=2 * base,
                                       local_elements=sum(x.numel() for x in params))

    @torch.no_grad()
    def after(self, optimizer, args, kwargs):
        # ZeRO copies/all-gathers the fp32 master only AFTER this AdamW post-hook.
        for destination, value in self.pending:
            destination.copy_(value)
        self.pending.clear()
        for group, value in self.restore_lrs:
            group["lr"] = value  # Keep the original recursive cosine schedule intact.
        self.restore_lrs.clear()
