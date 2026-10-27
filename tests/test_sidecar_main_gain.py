"""A Write-token gradient gain preserves forward values and the auxiliary path."""
import pytest
import torch

from tests.test_relational_sidecar import inputs
from tests.test_sidecar_heads_only import auxiliary

pytest_plugins = ['tests.test_relational_sidecar']


@pytest.mark.parametrize('gain', [0, 5, float('nan'), float('inf')])
def test_gain_rejects_values_outside_the_authorized_range(make, gain):
    with pytest.raises(ValueError, match='Main-task token gradient gain'):
        make(main_task_gain=gain)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('checkpointed', [False, True])
def test_gain_scales_shared_main_gradient_not_write_parameters_or_ea(make, dtype, checkpointed):
    normal = make(dtype=dtype, backbone_updates=2000)
    gained = make(dtype=dtype, backbone_updates=2000, main_task_gain=2)
    for stage in normal.sidecar.stages:
        stage.write.beta.data.fill_(.1)
    gained.load_state_dict(normal.state_dict(), strict=True)
    for model in (normal, gained):
        model.set_training_progress(2000)
    x, action = inputs(dtype), torch.randn(1, 32, 80, dtype=dtype)
    outputs = [m.forward_with_aux(action, torch.ones(1, dtype=dtype),
               use_gradient_checkpointing=checkpointed, **x) for m in (normal, gained)]
    for a, b in zip(outputs[0][:2], outputs[1][:2]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for index in (0, 1, 2):
        for model, output in zip((normal, gained), outputs):
            model.zero_grad(set_to_none=True)
            value = auxiliary(model, output[2]) if index == 2 else output[index].float().square().mean()
            value.backward(retain_graph=True)
        for name, p in normal.named_parameters():
            actual = gained.get_parameter(name).grad
            if p.grad is None:
                assert actual is None
                continue
            if index == 2 or name.startswith('sidecar.'):
                factor = 2 if index != 2 and name.startswith('sidecar.') and '.write.' not in name else 1
                torch.testing.assert_close(actual.float(), p.grad.float() * factor,
                                           rtol=.035 if dtype == torch.bfloat16 else 1e-5, atol=2e-6, msg=name)


def test_gain_has_no_effect_during_wam_warmup_or_eval(make):
    a, b = make(backbone_updates=2000), make(backbone_updates=2000, main_task_gain=2)
    for stage in a.sidecar.stages:
        stage.write.beta.data.fill_(.1)
    b.load_state_dict(a.state_dict())
    x, action = inputs(), torch.randn(1, 32, 80)
    for model in (a, b):
        model.set_training_progress(1999)
        model.forward_with_aux(action, torch.ones(1), **x)[1].square().mean().backward()
    for name, p in a.named_parameters():
        q = b.get_parameter(name)
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    for model in (a, b):
        model.set_training_progress(2000)
        model.eval()
        model.zero_grad(set_to_none=True)
        model.forward_with_aux(action, torch.ones(1), **x)[1].square().mean().backward()
    for name, p in a.named_parameters():
        q = b.get_parameter(name)
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
