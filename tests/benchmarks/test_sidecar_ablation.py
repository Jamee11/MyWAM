"""Exercise interventions on the actual tiny MoT + Sidecar architecture."""
from copy import deepcopy

import pytest
import torch

from benchmarks.robotwin.sidecar_ablation import MODES, configure
from tests.benchmarks.test_sidecar_fast_runtime import scoped_install
from tests.test_relational_sidecar import inputs, make  # noqa: F401


@pytest.mark.parametrize("mode", MODES)
def test_ablation_uses_selected_writes_without_changing_weights(make, mode):  # noqa: F811
    model = make().eval()
    for stage in model.sidecar.stages:
        stage.write.beta.data.fill_(0.3)
    reference = deepcopy(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    report = configure(model, mode)
    calls, reads, hooks = [], [], []
    for i, stage in enumerate(model.sidecar.stages):
        hooks.append(stage.write.register_forward_hook(lambda module, args, out, i=i: calls.append(i)))
        hooks.append(stage.register_forward_hook(lambda module, args, out, i=i: reads.append(i)))
    if mode == "off":
        reference.sidecar.action_conditioning = False
    elif mode != "full":
        for i, stage in enumerate(reference.sidecar.stages):
            if reference.tap_blocks[i] not in report["active_write_blocks"]:
                stage.write.beta.data.zero_()
    actions, x = torch.randn(1, 32, 80), inputs()
    with torch.inference_mode():
        actual = model(actions, torch.ones(1), **x)
        expected = reference(actions, torch.ones(1), **x)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert calls == [model.tap_blocks.index(i) for i in report["active_write_blocks"]]
    assert reads == ([] if mode == "off" else [0, 1, 2])
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    for hook in hooks:
        hook.remove()
    if mode == "off":
        assert actual[1].shape == (1, 32, 80)
        assert actual[0].shape == expected[0].shape


def test_reject_training_unknown_and_switching(make):  # noqa: F811
    model = make()
    with pytest.raises(ValueError, match="fresh eval"):
        configure(model, "full")
    model.eval()
    with pytest.raises(ValueError, match="Unknown"):
        configure(model, "typo")
    configure(model, "full")
    with pytest.raises(ValueError, match="fresh eval"):
        configure(model, "off")


def test_compile_reset_cannot_restore_masked_writes(make, monkeypatch):  # noqa: F811
    scoped_install(monkeypatch)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    model = make().eval()
    model.apply_compile_optimizations({"enabled": True, "sidecar_strategy": "ffn"})
    configure(model, "last_write")
    selected = model.sidecar.write_action
    for enabled in [False, True]:
        model.apply_compile_optimizations({"enabled": enabled, "sidecar_strategy": "ffn"})
        assert model.sidecar.write_action is selected
