"""Write-only intervention preserves all tensors, read calls and physical-unit reporting."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.robotwin.sidecar_write_diagnostics import WriteProbe, action_metrics, distribution
from tests.test_relational_sidecar import make, inputs


def test_diagnostic_devices_preserve_defaults_and_allow_one_gpu():
    from benchmarks.robotwin.sidecar_gt_diagnostics import diagnostic_devices
    assert tuple(map(str, diagnostic_devices({}))) == ('cuda:0', 'cuda:1')
    assert tuple(map(str, diagnostic_devices(dict(encoder_device='cuda:0', model_device='cuda:0')))) == ('cuda:0', 'cuda:0')
    with pytest.raises(ValueError):
        diagnostic_devices(dict(model_device='cpu'))


def test_bf16_nonzero_gated_residual_can_round_away_completely():
    projection=torch.nn.Linear(2,2,bias=False,dtype=torch.bfloat16)
    with torch.no_grad(): projection.weight.copy_(torch.eye(2,dtype=torch.bfloat16))
    write=SimpleNamespace(beta=torch.nn.Parameter(torch.tensor(1e-4,dtype=torch.bfloat16)),projection=projection)
    sidecar=SimpleNamespace(action_conditioning=True,stages=[SimpleNamespace(write=write)])
    sidecar.write_action=lambda i,actions,tokens,valid: actions+write.beta.to(actions.dtype)*projection(tokens)
    model=SimpleNamespace(training=False,sidecar_enabled=True,sidecar=sidecar,tap_blocks=[7],forward=lambda *a,**k: None)
    probe=WriteProbe(model);probe.time=(0.,0.)
    values=torch.ones(1,2,2,dtype=torch.bfloat16)
    with torch.inference_mode(),probe.use('full',collect=True):probe.write(0,values,values,None)
    row=probe.rows[0]
    assert row['gated_native_l2']>0 and row['native_pre_addition_ratio']==pytest.approx(1e-4,rel=.01)
    assert row['ratio']==0 and row['effective_delta_l2']==0 and row['changed_fraction']==0


def test_gt_summary_accepts_full_off_without_extra_strength_scan(tmp_path):
    import json
    from pathlib import Path
    from benchmarks.robotwin.sidecar_gt_diagnostics import summarize
    source=Path('outputs/robotwin_eval/sidecar42960_gt_diagnostic_20261021T135950Z/heldout_eval/results.json')
    if not source.exists():
        pytest.skip('Archived GT diagnostic fixture unavailable')
    raw=json.loads(source.read_text())
    raw['results']=[r for r in raw['results'] if r['scale'] in (0,1)]
    (tmp_path/'results.json').write_text(json.dumps(raw))
    result=summarize(tmp_path)
    assert {r['scale'] for r in result['action_sweep']}=={0,1}
    assert len(result['representations'])==9
    assert all(r['windows'] in (1,12) for r in result['action_sweep'])


def test_write_only_ablation_and_residuals(make):
    model = make()
    model.eval()
    for stage in model.sidecar.stages:
        stage.write.beta.data.fill_(.1)
    saved = {key:value.clone() for key,value in model.state_dict().items()}
    reads = []
    for stage in model.sidecar.stages:
        stage.register_forward_hook(lambda *args: reads.append(1))
    probe = WriteProbe(model)
    x, actions = inputs(), torch.randn(1,32,80)
    with torch.inference_mode(), probe.use('full',collect=True):
        full = model(actions,torch.ones(1),**x)[1]
    assert len(probe.rows)==3 and any(row['ratio']>0 for row in probe.rows)
    for row in probe.rows:
        assert row['hidden_l2']>0 and row['gated_native_l2']>0
        assert row['ratio']==pytest.approx(row['effective_delta_l2']/(row['hidden_l2']+1e-8))
        assert row['native_pre_addition_ratio']==pytest.approx(row['gated_native_l2']/(row['hidden_l2']+1e-8))
        assert 0<=row['changed_fraction']<=1
    probe.rows=[]
    with torch.inference_mode(), probe.use('write_off',collect=True):
        off = model(actions,torch.ones(1),**x)[1]
    assert all(row['ratio']==0 for row in probe.rows)
    assert all(row['gated_native_l2']==0 and row['changed_fraction']==0 for row in probe.rows)
    assert len(reads)==6 and model.sidecar_enabled and model.sidecar.action_conditioning
    assert not torch.equal(full,off)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,saved[key],rtol=0,atol=0)


def test_gate_restored_on_exception(make):
    model=make();model.eval()
    probe=WriteProbe(model)
    model.sidecar.stages[0].write.beta.data.fill_(.3)
    before=model.sidecar.stages[0].write.beta.detach().clone()
    def fail(*args):
        assert model.sidecar.stages[0].write.beta==0
        raise RuntimeError('fixture')
    probe.original_write=fail
    with pytest.raises(RuntimeError,match='fixture'),probe.use('write_off'):
        probe.write(0,None,None,None)
    torch.testing.assert_close(model.sidecar.stages[0].write.beta,before,rtol=0,atol=0)


def test_physical_target_differences_and_distribution():
    from benchmarks.utils.action_conversion import quat_xyzw_to_rot6d
    off=np.zeros((32,20),dtype=np.float32)
    off[:,3:9]=off[:,13:19]=quat_xyzw_to_rot6d(np.array([0,0,0,1]))
    full=off.copy();full[:,0]+=.001
    report=action_metrics(full,off,SimpleNamespace(normalize=lambda x:x))
    assert report['position_difference_mm']['max']==pytest.approx(1)
    assert report['position_difference_mm']['mean']==pytest.approx(.5)
    assert report['rotation_difference_degrees']['max']==pytest.approx(0)
    assert distribution([0,1,2])['p50']==1
    with pytest.raises(ValueError):distribution([float('nan')])


def test_scaled_write_restores_gate_and_matches_off(make):
    model=make();model.eval()
    for stage in model.sidecar.stages:stage.write.beta.data.fill_(.1)
    saved=[stage.write.beta.detach().clone() for stage in model.sidecar.stages]
    probe=WriteProbe(model);x=inputs();actions=torch.randn(1,32,80)
    with torch.inference_mode(),probe.use('full',scale=0):
        zero=model(actions,torch.ones(1),**x)[1]
    with torch.inference_mode(),probe.use('write_off'):
        off=model(actions,torch.ones(1),**x)[1]
    torch.testing.assert_close(zero,off,rtol=0,atol=0)
    with torch.inference_mode(),probe.use('full',scale=2,collect=True):
        model(actions,torch.ones(1),**x)
    assert any(row['ratio']>0 for row in probe.rows)
    for stage,beta in zip(model.sidecar.stages,saved):
        torch.testing.assert_close(stage.write.beta,beta,rtol=0,atol=0)
    with pytest.raises(ValueError),probe.use('full',scale=float('nan')):pass


def test_gt_representation_metrics_masks_and_local_directions():
    from benchmarks.robotwin.sidecar_gt_diagnostics import representation_metrics,binary_report
    target=np.zeros(130);target[:2]=1;target[8]=1;target[10:12]=1
    target[16:52]=np.tile([1,0,0],12);target[52:70]=1
    target[70:74]=[0,0,1,1];target[118]=1
    prediction=dict(grid=[3,2],attention=np.full((4,6),1/6),presence_logits=[10,-10,-10,-10],
        track_valid_logits=[10,10],contact_logits=[10,-10],near_logits=[10,10],goal_proximity=[0,0],
        plan_moving_logits=np.full((3,2,2),10),plan_directions=np.tile([1,0,0],(3,2,2,1)))
    metrics=representation_metrics(prediction,target,[1,1])
    assert metrics['ground_mass']==pytest.approx([2/3])
    assert metrics['ground_hits']==[True]
    assert metrics['direction_cosine']==pytest.approx([1]*12)
    assert binary_report(metrics['contact'])['f1']==1
    assert binary_report(metrics['presence'])['f1']==1
    target[64:70]=0
    masked=representation_metrics(prediction,target,[1,1])
    assert masked['direction_cosine']==[] and binary_report(masked['moving'])['n']==0


def test_three_live_loss_gradients_without_updates(make):
    from benchmarks.robotwin.sidecar_gradient_probe import measure
    from tests.test_relational_sidecar import target
    model=make();model.eval()
    for stage in model.sidecar.stages:stage.write.beta.data.fill_(.1)
    x=inputs();flags={k:p.requires_grad for k,p in model.named_parameters()}
    report=measure(model,dict(actions=torch.randn(1,32,80),input_latents=x['latents'],
        first_frame_latents=x['first_frame_latents'],context=x['context'],context_mask=x['context_mask'],
        proprio=x['proprio'],proprio_mask=x['_proprio_sample_mask'],fuse_vae_embedding_in_latents=True,ea_targets=target()))
    assert report['parameters_unchanged']
    assert {k:p.requires_grad for k,p in model.named_parameters()}==flags
    assert all(p.grad is None for p in model.parameters())
    for name,values in report['parameters'].items():
        if '.write.' in name:
            assert values['gradients']['ea']['is_none'] or values['gradients']['ea']['norm']==0
        if name.startswith('heads.'):assert values['gradients']['action']['is_none']
    assert report['parameters']['stages.0.write.projection.weight']['gradients']['action']['norm']>0
