"""Actual tiny WAM: scale only shared EA gradients, preserving FM and head learning."""

import pytest
import torch

from tests.test_relational_sidecar import inputs
from tests.test_sidecar_backbone_warmup import loss, optimizer_for
from tests.test_sidecar_heads_only import auxiliary

pytest_plugins = ["tests.test_relational_sidecar"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_all_taps_auxiliary_and_fm_gradient_routes(make, dtype, checkpointed):
    original = make(dtype=dtype, backbone_updates=2000)
    routed = make(dtype=dtype, backbone_updates=2000, shared_gradient_scale=0.01, gradient_anneal_updates=500)
    for stage in original.sidecar.stages:
        stage.write.beta.data.fill_(0.1)
    routed.load_state_dict(original.state_dict(), strict=True)
    original.set_training_progress(2500)
    routed.set_training_progress(2500)
    x, actions = inputs(dtype), torch.randn(1, 32, 80, dtype=dtype)
    kwargs = dict(use_gradient_checkpointing=checkpointed, **x)
    expected = original.forward_with_aux(actions, torch.ones(1, dtype=dtype), **kwargs)
    actual = routed.forward_with_aux(actions, torch.ones(1, dtype=dtype), **kwargs)
    for a, b in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(actual[2], expected[2]):
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    for model, outputs in ((original, expected), (routed, actual)):
        auxiliary(model, outputs[2]).backward(retain_graph=True)
    for stage in routed.sidecar.stages:
        assert stage.projector[1].weight.grad.norm() > 0
        assert stage.refine.in_proj_weight.grad.norm() > 0
    assert routed.sidecar.role_tokens.grad.norm() > 0
    shared_expected, shared_actual = [], []
    for name, parameter in original.named_parameters():
        a, b = routed.get_parameter(name).grad, parameter.grad
        if b is None:
            assert a is None
        elif name.startswith("sidecar.heads."):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
        else:
            shared_expected.append(b.float().flatten() * 0.01)
            shared_actual.append(a.float().flatten())
    reference, scaled = torch.cat(shared_expected), torch.cat(shared_actual)
    assert reference.norm() > 0
    assert (scaled - reference).norm() / reference.norm() < (0.035 if dtype == torch.bfloat16 else 2e-5)
    for output_index in (0, 1):
        for model, outputs in ((original, expected), (routed, actual)):
            model.zero_grad(set_to_none=True)
            outputs[output_index].float().square().mean().backward(retain_graph=True)
        for name, parameter in original.named_parameters():
            a, b = routed.get_parameter(name).grad, parameter.grad
            if b is None:
                assert a is None
            else:
                torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)


def test_optimizer_counter_schedule_and_eval(make):
    model = make(backbone_updates=2000, shared_gradient_scale=0.01, gradient_anneal_updates=500)
    for step, expected in ((0, 1), (1999, 1), (2000, 1), (2250, 0.505), (2500, 0.01), (21480, 0.01)):
        model.set_training_progress(step)
        assert model.aux_head_input_gradient_scale == pytest.approx(expected)
    model.eval()
    assert model.aux_head_input_gradient_scale == 1
    model.train()
    assert model.aux_head_input_gradient_scale == pytest.approx(0.01)


@pytest.mark.parametrize("saved_step", [1, 2, 3, 4])
def test_optimizer_resume_during_warmup_and_anneal(make, tmp_path, saved_step):
    settings = dict(backbone_updates=2, shared_gradient_scale=0.01, gradient_anneal_updates=2)
    model = make(**settings)
    optimizer = optimizer_for(model)

    def update(net, opt, step):
        net.set_training_progress(step)
        opt.zero_grad(set_to_none=True)
        for micro in range(2):
            torch.manual_seed(100 + step * 2 + micro)
            (loss(net, checkpointed=True) / 2).backward()
        opt.step()

    for step in range(saved_step):
        update(model, optimizer, step)
    path = tmp_path / "state.pt"
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), completed_updates=saved_step), path)
    restored = make(**settings)
    restored_optimizer = optimizer_for(restored)
    state = torch.load(path, weights_only=True)
    restored.load_state_dict(state["model"], strict=True)
    restored_optimizer.load_state_dict(state["optimizer"])
    update(model, optimizer, saved_step)
    update(restored, restored_optimizer, state["completed_updates"])
    assert restored.aux_head_input_gradient_scale == model.aux_head_input_gradient_scale
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, restored.get_parameter(name), rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("scale,ramp,hard", [(float("nan"), 500, False), (-0.1, 0, False),
                                            (1.1, 0, False), (0.01, -1, False),
                                            (0.01, 0.5, False), (0.01, 500, True)])
def test_invalid_or_conflicting_route_rejected(make, scale, ramp, hard):
    with pytest.raises(ValueError):
        make(shared_gradient_scale=scale, gradient_anneal_updates=ramp, heads_only=hard)
