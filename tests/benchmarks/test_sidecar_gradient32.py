"""The standalone diagnostic must preserve FM RNG and loss/parameter grouping."""
import pytest
import torch

from benchmarks.robotwin.sidecar_gradient32 import fixed_times, signed_stats
from benchmarks.robotwin.sidecar_gradient_probe import grouped_gradients
from tests.test_relational_sidecar import make


def test_fixed_fm_draws_preserve_rng_and_restore_on_error():
    original=torch.randint
    torch.manual_seed(7)
    original(500,501,(1,));original(0,1000,(1,));expected=torch.randn(3)
    torch.manual_seed(7)
    with fixed_times(500,900,(500,501),1000):
        assert torch.randint(500,501,(1,)).item()==500
        assert torch.randint(0,1000,(1,)).item()==900
    torch.testing.assert_close(torch.randn(3),expected)
    assert torch.randint is original
    with pytest.raises(AssertionError),fixed_times(500,900,(500,501),1000):
        torch.randint(0,20,(1,))
    assert torch.randint is original


def test_plan_rows_and_structural_zero_have_correct_metrics():
    gradients={k:{'role_tokens':torch.zeros(15,2)} for k in ('action','ea','video')}
    gradients['action']['role_tokens'][9:]=2
    gradients['ea']['role_tokens'][:9]=100
    gradients['ea']['role_tokens'][9:]=3
    report=grouped_gradients({'role_tokens':None},gradients)
    plan=report['plan_tokens']
    assert plan['norms']['action']==pytest.approx((12*4)**.5)
    assert plan['norms']['ea']==pytest.approx((12*9)**.5)
    assert plan['cosines']['action_ea']['cosine']==pytest.approx(1)
    assert plan['cosines']['action_video']['cosine'] is None
    assert report['tap7/beta']['norms']['ea']==0


def test_gate_cancellation_is_signed_not_mean_norm():
    values=signed_stats([1.,-1.,None,0.])
    assert values['n']==3 and values['mean']==0 and values['cancellation_ratio']==0
    assert values['positive_fraction']==pytest.approx(1/3)
    assert values['negative_fraction']==pytest.approx(1/3)
    assert signed_stats([0.])['cancellation_ratio'] is None


def test_fixed_times_with_real_three_loss_backward(make):
    from tests.test_relational_sidecar import inputs, target
    from benchmarks.robotwin.sidecar_gradient_probe import measure
    model=make();model.eval()
    for stage in model.sidecar.stages:stage.write.beta.data.fill_(.1)
    x=inputs()
    with fixed_times(5,9,(0,10),10):
        report=measure(model,dict(actions=torch.randn(1,32,80),input_latents=x['latents'],
            first_frame_latents=x['first_frame_latents'],context=x['context'],context_mask=x['context_mask'],
            proprio=x['proprio'],proprio_mask=x['_proprio_sample_mask'],
            fuse_vae_embedding_in_latents=True,ea_targets=target()))
    assert report['timestep_ids']=={'_compute_video_loss':[5],'_compute_action_loss':[9]}
    assert report['parameters_unchanged']
