"""Check opt-in acceleration preserves Sidecar computation and input guards."""
import pytest
import torch

from benchmarks.robotwin.sidecar_fast_runtime import install
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.architectures.dual_system.relational_sidecar import RelationalSidecarArchitecture
from tests.test_relational_sidecar import inputs, make  # noqa: F401 -- shared actual tiny-model fixture


def scoped_install(monkeypatch):
    cls = RelationalSidecarArchitecture
    monkeypatch.setattr(cls, "generate", cls.generate)
    monkeypatch.setattr(cls, "apply_compile_optimizations", cls.apply_compile_optimizations)
    monkeypatch.setattr(cls, "_evaluation_fast_runtime_installed", False, raising=False)
    install()


@pytest.mark.parametrize("strategy", ["conservative", "sidecar", "ffn"])
def test_compiles_complete_sidecar_and_preserves_output(make, monkeypatch, strategy):  # noqa: F811 -- imported pytest fixture
    scoped_install(monkeypatch)
    model = make().eval()
    # Nonzero writes ensure a foundation-only loop cannot pass this check.
    for stage in model.sidecar.stages:
        stage.write.beta.data.fill_(0.3)
    actions, x = torch.randn(1, 32, 80), inputs()
    with torch.inference_mode():
        expected = model(actions, torch.ones(1), **x)
    captured = []

    def capture_compile(fn, **kwargs):
        captured.append(fn)
        assert kwargs["options"]["emulate_precision_casts"] is True
        assert kwargs["options"]["force_same_precision"] is True
        return fn

    monkeypatch.setattr(torch, "compile", capture_compile)
    model.apply_compile_optimizations({"enabled": True, "sidecar_strategy": strategy})
    if strategy == "conservative":
        assert captured[0].__func__ is RelationalSidecarArchitecture.forward
    elif strategy == "sidecar":
        assert captured[0].__func__.__name__ == "read_refine"
        assert captured[1].__func__.__name__ == "write_action"
    else:
        assert len(captured) == 3
    assert model._compiled_mot_run_joint_loop is None
    with torch.inference_mode():
        actual = model(actions, torch.ones(1), **x)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_cache_forwarded_without_losing_state_mask_or_guards(make, monkeypatch):  # noqa: F811 -- imported pytest fixture
    scoped_install(monkeypatch)
    model = make()
    cache, received = object(), {}

    def base_generate(self, **kwargs):
        received.update(kwargs)
        return "generated"

    monkeypatch.setattr(DualSystemSelfAttnArchitecture, "generate", base_generate)
    mask = torch.zeros(80, dtype=torch.bool)
    mask[:10] = True
    assert model.generate(proprio=torch.zeros(1, 80), active_action_mask=mask, dit_cache=cache) == "generated"
    assert received["dit_cache"] is cache
    assert received["_proprio_sample_mask"].shape == (1, 1, 80)
    with pytest.raises(ValueError, match="Privileged"):
        model.generate(ea_targets=torch.zeros(1))
    with pytest.raises(ValueError, match="CFG"):
        model.generate(cfg_scale=2)
