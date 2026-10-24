"""Separate live loss gradients without an optimizer or a parameter update."""

import json
from pathlib import Path

import torch


def grouped_gradients(parameters, gradients):
    """Plan is rows 9:15 of role_tokens; zero/unused gradients have no cosine."""
    groups = {"plan_tokens": [("role_tokens", slice(9, 15))]}
    for index in range(3):
        prefix = f"stages.{index}."
        for label, prefixes in {
            "read_projector": ("projector.",),
            "read_attention": ("read.", "read_norm.", "alpha", "position."),
            "relational_refinement": ("refine.", "refine_norm.", "ffn.", "context."),
            "write_projection": ("write.projection.",),
            "beta": ("write.beta",),
        }.items():
            groups[f"tap{(7, 15, 23)[index]}/{label}"] = [
                (n, slice(None)) for n in parameters if any(n.startswith(prefix + p) for p in prefixes)
            ]
    report = {}
    for group, selected in groups.items():
        values = {
            loss: [g[n] if g[n] is None or g[n].ndim == 0 else g[n][s] for n, s in selected]
            for loss, g in gradients.items()
        }
        norms = {loss: sum(float(v.square().sum()) for v in vs if v is not None) ** 0.5 for loss, vs in values.items()}
        cosines = {}
        for a, b in (("action", "ea"), ("action", "video"), ("video", "ea")):
            pairs = [
                (x, y)
                for x, y in zip(values[a], values[b])
                if x is not None and y is not None and x.norm() > 0 and y.norm() > 0
            ]
            aa = sum(float(x.square().sum()) for x, y in pairs)
            bb = sum(float(y.square().sum()) for x, y in pairs)
            cosines[a + "_" + b] = dict(
                cosine=sum(float((x * y).sum()) for x, y in pairs) / (aa * bb) ** 0.5 if aa * bb > 0 else None,
                shared_nonzero_tensors=len(pairs),
            )
        report[group] = dict(
            norms=norms,
            cosines=cosines,
            unused={loss: sum(v is None for v in vs) for loss, vs in values.items()},
            parameter_tensors=len(selected),
        )
    return report


def measure(architecture, prepared):
    arch = architecture
    flags = {name: p.requires_grad for name, p in arch.named_parameters()}
    parameters = dict(arch.sidecar.named_parameters())
    before = {name: p.detach().cpu().clone() for name, p in parameters.items()}
    originals = {name: getattr(arch, name) for name in ("_compute_action_loss", "_compute_video_loss")}
    original_heads = arch.sidecar.heads.losses
    live = {}
    head_losses = []
    time_ids = {}

    def capture(name):
        def wrapped(*args, **kwargs):
            value = originals[name](*args, **kwargs)
            live[name] = value
            time_ids[name] = args[2].detach().cpu().tolist()
            return value

        return wrapped

    def heads(*args, **kwargs):
        value = original_heads(*args, **kwargs)
        head_losses.append(value)
        return value

    try:
        for name, p in arch.named_parameters():
            p.requires_grad_(name.startswith("sidecar."))
        for name in originals:
            setattr(arch, name, capture(name))
        arch.sidecar.heads.losses = heads
        with torch.enable_grad():
            result = arch.compute_loss(**dict(prepared))
            assert len(head_losses) == 3
            ea = arch.aux_loss_scale * (
                sum(arch.loss_weights[k] * v for k, v in head_losses[0].items())
                + arch.deep_supervision
                * (
                    arch.loss_weights["ground"] * head_losses[1]["ground"]
                    + arch.loss_weights["relation"] * head_losses[2]["relation"]
                )
            )
            terms = dict(
                action=prepared.get("lambda_action", 1.0) * live["_compute_action_loss"],
                video=prepared.get("lambda_video", 1.0) * live["_compute_video_loss"],
                ea=ea,
            )
            torch.testing.assert_close(sum(terms.values()), result["loss"])
            gradients = {}
            for label, loss in terms.items():
                values = torch.autograd.grad(loss, tuple(parameters.values()), retain_graph=True, allow_unused=True)
                gradients[label] = {
                    name: None if value is None else value.detach().float().cpu()
                    for name, value in zip(parameters, values)
                }
        report = dict(
            losses={key: float(value.detach()) for key, value in terms.items()},
            timestep_ids=time_ids,
            parameters={},
            gates={},
        )
        for name, param in parameters.items():
            report["parameters"][name] = dict(
                parameter_norm=float(before[name].float().norm()),
                gradients={
                    label: dict(
                        is_none=values[name] is None, norm=None if values[name] is None else float(values[name].norm())
                    )
                    for label, values in gradients.items()
                },
            )
            if name.endswith(".write.beta"):
                report["gates"][name] = dict(
                    beta=float(before[name]),
                    gradients={
                        label: None if values[name] is None else float(values[name])
                        for label, values in gradients.items()
                    },
                )
        report["d_loss_d_global_write_scale"] = {
            label: sum(
                float(before[name]) * float(values[name])
                for name in parameters
                if name.endswith(".write.beta") and values[name] is not None
            )
            for label, values in gradients.items()
        }
        report["shared_parameter_cosines"] = {}
        for left, right in (("action", "ea"), ("action", "video"), ("video", "ea")):
            selected = [
                name
                for name in parameters
                if gradients[left][name] is not None
                and gradients[right][name] is not None
                and gradients[left][name].norm() > 0
                and gradients[right][name].norm() > 0
            ]
            if selected:
                dot = sum(float((gradients[left][name] * gradients[right][name]).sum()) for name in selected)
                a = sum(float(gradients[left][name].square().sum()) for name in selected)
                b = sum(float(gradients[right][name].square().sum()) for name in selected)
                report["shared_parameter_cosines"][f"{left}_{right}"] = dict(
                    cosine=dot / (a * b) ** 0.5, parameter_tensors=len(selected)
                )
            else:
                report["shared_parameter_cosines"][f"{left}_{right}"] = None
        report["groups"] = grouped_gradients(parameters, gradients)
        for name, p in parameters.items():
            torch.testing.assert_close(p.detach().cpu(), before[name], rtol=0, atol=0)
        report["parameters_unchanged"] = True
        return report
    finally:
        for name, value in originals.items():
            setattr(arch, name, value)
        arch.sidecar.heads.losses = original_heads
        for name, p in arch.named_parameters():
            p.requires_grad_(flags[name])


def run_smoke(server, out, rows):
    """One real GT batch checks feasibility; it is not a multi-batch gradient conclusion."""
    from openwam.dataloader.ea_rsig_lite import EARSIGLiteDataset
    from openwam.model.video_backbone.wan import encode

    arch = server.engine.architecture
    out = Path(out)
    source = next(row for row in rows if row["domain"] == "Clean_seen_repeat")
    sample = EARSIGLiteDataset(
        "outputs/sidecar/aloha_4350/manifest.ready.json", tasks=["adjust_bottle"], startup_mode="prepared"
    )[0]
    sample["prompt"] = json.loads(Path(source["payload"]).read_text())["prompt"]
    original_encode = encode.encode_text

    def cached(prompts, **kwargs):
        contexts = [server.engine._prompt_embed_cache[p] for p in prompts]
        return tuple(torch.cat([pair[i].clone() for pair in contexts], dim=0) for i in (0, 1))

    try:
        encode.encode_text = cached
        arch.init_training_schedulers()
        arch.set_training_runtime(
            use_gradient_checkpointing=True, min_timestep_boundary=0.5, max_timestep_boundary=0.501
        )
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        prepared = arch.prepare_inputs([sample])
        # The training accelerator moves sample masks; this standalone reader does not.
        prepared = {
            key: value.to(arch.device) if isinstance(value, torch.Tensor) else value for key, value in prepared.items()
        }
        (out / "gradient_prepared_devices.json").write_text(
            json.dumps(
                {key: str(value.device) for key, value in prepared.items() if isinstance(value, torch.Tensor)}, indent=2
            )
        )
        report = measure(arch, prepared)
        report.update(
            scope="Single training-seen adjust_bottle batch, B=1; video t-index500, action sampled with seed42; no optimizer update",
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        (out / "gradient_smoke.json").write_text(json.dumps(report, indent=2))
    finally:
        encode.encode_text = original_encode
