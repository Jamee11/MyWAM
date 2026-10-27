"""Instance-local inference interventions; never mutate checkpoint tensors."""

MODES = ("full", "off", "first_write", "last_write")


def configure(model, mode):
    """Apply once, before first generation, in a dedicated eval process."""
    if mode not in MODES:
        raise ValueError(f"Unknown Sidecar ablation: {mode}")
    if model.training or getattr(model, "_evaluation_ablation", None) is not None:
        raise ValueError("Use a fresh eval model for each ablation")
    if not model.sidecar_enabled or not model.sidecar.action_conditioning:
        raise ValueError("Ablation requires the complete action-conditioned Sidecar checkpoint")
    writes = {"full": (0, 1, 2), "off": (), "first_write": (0,), "last_write": (2,)}[mode]
    if mode == "off":
        model.sidecar_enabled = False
    elif mode != "full":
        original = model.sidecar.write_action

        def selected_write(index, actions, tokens, valid):
            return original(index, actions, tokens, valid) if index in writes else actions

        model.sidecar.write_action = selected_write
        # The numerical validator reapplies compile settings between variants.
        if hasattr(model, "_evaluation_eager_write_action"):
            model._evaluation_eager_write_action = selected_write
    model._evaluation_ablation = mode
    return {"mode": mode, "active_write_blocks": [model.tap_blocks[i] for i in writes],
            "sidecar_computed": mode != "off", "weights_modified": False,
            "scope": "runtime branch dependence; not a separately trained no-EA baseline"}
