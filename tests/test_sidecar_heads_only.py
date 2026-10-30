"""The opt-in route changes gradients, while preserving all forward values."""

import json

import pytest
import torch

from tests.test_relational_sidecar import inputs, target
from tests.test_sidecar_backbone_warmup import loss, optimizer_for

pytest_plugins = ["tests.test_relational_sidecar"]


def auxiliary(model, outputs):
    parts = model.sidecar.heads.losses(outputs[-1], target())
    deep = (
        model.loss_weights["ground"] * model.sidecar.heads.losses(outputs[0], target())["ground"]
        + model.loss_weights["relation"] * model.sidecar.heads.losses(outputs[1], target())["relation"]
    )
    return model.aux_loss_scale * (
        sum(model.loss_weights[key] * value for key, value in parts.items()) + model.deep_supervision * deep
    )


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward_identity_and_all_tap_gradient_routes(make, checkpointed, dtype):
    original = make(dtype=dtype, backbone_updates=500)
    routed = make(dtype=dtype, backbone_updates=500, heads_only=True)
    for stage in original.sidecar.stages:
        stage.write.beta.data.fill_(0.1)
    routed.load_state_dict(original.state_dict(), strict=True)
    original.set_training_progress(20000)
    routed.set_training_progress(20000)
    x, actions = inputs(dtype), torch.randn(1, 32, 80, dtype=dtype)
    expected = original.forward_with_aux(actions, torch.ones(1, dtype=dtype), use_gradient_checkpointing=checkpointed, **x)
    actual = routed.forward_with_aux(actions, torch.ones(1, dtype=dtype), use_gradient_checkpointing=checkpointed, **x)
    for a, b in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(actual[2], expected[2]):
        for key in ("plan_directions", "near_logits", "contact_logits", "tokens", "first_frame_features"):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    aux = auxiliary(routed, actual[2])
    torch.testing.assert_close(aux, auxiliary(original, expected[2]), rtol=0, atol=0)
    aux.backward(retain_graph=True)
    assert any(p.grad is not None and p.grad.norm() > 0 for p in routed.sidecar.heads.parameters())
    assert all(p.grad is None for name, p in routed.named_parameters() if not name.startswith("sidecar.heads."))
    routed.zero_grad(set_to_none=True)
    actual[1].float().square().mean().backward(retain_graph=True)
    for stage in routed.sidecar.stages:
        assert stage.projector[1].weight.grad.norm() > 0
        assert stage.refine.in_proj_weight.grad.norm() > 0
        assert stage.write.projection.weight.grad.norm() > 0
        assert stage.write.beta.grad.abs() > 0
    assert routed.sidecar.role_tokens.grad.norm() > 0
    assert any(p.grad is not None and p.grad.norm() > 0 for p in routed.video_backbone.parameters())
    assert all(p.grad is None for p in routed.sidecar.heads.parameters())
    routed.zero_grad(set_to_none=True)
    actual[0].float().square().mean().backward()
    assert all(stage.write.projection.weight.grad.norm() > 0 for stage in routed.sidecar.stages)


@pytest.mark.parametrize("warmup", ["backbone_updates", "isolation_updates"])
def test_500_update_boundary_and_eval_keep_original_auxiliary_learning(make, warmup):
    model = make(heads_only=True, **{warmup: 500})
    model.set_training_progress(499)
    assert not model.aux_heads_only_active
    outputs = model.forward_with_aux(torch.randn(1, 32, 80), torch.ones(1), **inputs())[2]
    auxiliary(model, outputs).backward()
    assert model.sidecar.stages[0].projector[1].weight.grad.norm() > 0
    model.zero_grad(set_to_none=True)
    model.set_training_progress(500)
    assert model.aux_heads_only_active
    outputs = model.forward_with_aux(torch.randn(1, 32, 80), torch.ones(1), **inputs())[2]
    auxiliary(model, outputs).backward()
    assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith("sidecar.heads."))
    model.eval()
    assert not model.aux_heads_only_active


def test_full_loss_identity_and_restored_optimizer_can_switch_route(make):
    original = make(backbone_updates=2)
    optimizer = optimizer_for(original)
    for step in range(3):
        original.set_training_progress(step)
        torch.manual_seed(step)
        loss(original).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    routed = make(backbone_updates=2, heads_only=True)
    restored_optimizer = optimizer_for(routed)
    routed.load_state_dict(original.state_dict(), strict=True)
    restored_optimizer.load_state_dict(optimizer.state_dict())
    routed.set_training_progress(3)
    original.set_training_progress(3)
    torch.manual_seed(101)
    expected = loss(original)
    torch.manual_seed(101)
    actual = loss(routed)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    before = routed.sidecar.heads.plan.weight.detach().clone()
    restored_optimizer.step()
    assert not torch.equal(before, routed.sidecar.heads.plan.weight)
    assert routed.aux_heads_only_active


def test_resume_output_cannot_accidentally_target_original_run(tmp_path):
    from scripts.train_sidecar_heads_only_resume import validate_resume_identity

    run = tmp_path / "independent_run"
    state = run / "accel_state_step_40000"
    state.mkdir(parents=True)
    (state / "trainer_state.json").write_text(json.dumps(dict(global_step=40000, opt_step=20000, epoch=9)))
    assert validate_resume_identity(run, run, state)["opt_step"] == 20000
    with pytest.raises(ValueError, match="own exact"):
        validate_resume_identity(run, tmp_path / "original", state)
    (state / "trainer_state.json").write_text(json.dumps(dict(global_step=36000, opt_step=18000, epoch=8)))
    with pytest.raises(ValueError, match="micro40000"):
        validate_resume_identity(run, run, state)
