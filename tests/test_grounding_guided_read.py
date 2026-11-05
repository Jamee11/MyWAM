"""Cross-stage routing, baseline compatibility and two-stage training contracts."""
import json
import os
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest
import torch
from safetensors.torch import save_file

from openwam.model.modules.relational_sidecar import grounding_read_bias
from scripts.cluster.sidecar_stage_checkpoint import completed_checkpoint
from tests.test_cross_appearance import loss_inputs
from tests.test_relational_sidecar import make, inputs


def prediction():
    maps = torch.eye(4)[None].requires_grad_()
    return dict(grounding=dict(attention=maps, presence_logits=torch.full((1, 4), 20.0)),
                track_valid_logits=torch.full((1, 2), 20.0), arm_present=torch.ones(1, 2))


def test_role_pairing_global_fallback_and_gradient():
    output = prediction()
    bias = grounding_read_bias(output, 2, torch.float32, .5, .5)
    assert bias.shape == (2, 15, 4) and torch.equal(bias[0], bias[1])
    assert torch.count_nonzero(bias[:, :4]) == torch.count_nonzero(bias[:, 8]) == 0
    assert bias[0, 4].argmax() == 0 and bias[0, 6].argmax() == 2
    for q in (5, 9, 11, 13):
        assert bias[0, q, 0] == bias[0, q, 1] > bias[0, q, 2]
    for q in (7, 10, 12, 14):
        assert bias[0, q, 2] == bias[0, q, 3] > bias[0, q, 0]
    assert bias.min() >= .5 * torch.log(torch.tensor(.5))
    bias[0, 4, 0].backward()
    assert output['grounding']['attention'].grad.abs().sum() > 0
    output['arm_present'][:, 1] = 0
    absent = grounding_read_bias(output, 2, torch.float32, .5, .5)
    assert torch.count_nonzero(absent[:, [6, 7, 10, 12, 14]]) == 0
    assert torch.count_nonzero(grounding_read_bias(output, 2, torch.float32, .5, 1)) == 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_zero_strength_preserves_forward_rng_loss_and_gradients(make, dtype):
    old = make(dtype=dtype)
    new = make(dtype=dtype, grounding_guided_read=dict(enabled=True))
    new.load_state_dict(old.state_dict(), strict=True)
    assert old.state_dict().keys() == new.state_dict().keys()
    batch = loss_inputs()
    batch.pop('xa_clean_inputs')
    batch = {k: v.to(dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
             for k, v in batch.items()}
    torch.manual_seed(27)
    a = old.compute_loss(**batch)
    rng = torch.get_rng_state()
    torch.manual_seed(27)
    b = new.compute_loss(**batch)
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(a['loss'], b['loss'], rtol=0, atol=0)
    a['loss'].backward()
    b['loss'].backward()
    for (name, p), (_, q) in zip(old.named_parameters(), new.named_parameters()):
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('checkpointed', [False, True])
def test_active_read_routes_action_gradients_to_grounding_head(make, dtype, checkpointed):
    model = make(dtype=dtype, grounding_guided_read=dict(enabled=True))
    model.set_training_progress(500)
    assert model.grounding_guidance_strength == .25
    model.set_training_progress(1000)
    assert model.grounding_guidance_strength == .5
    for stage in model.sidecar.stages:
        stage.write.beta.data.fill_(.2)
    masks = []
    handles = [s.read.register_forward_pre_hook(lambda m, a, kw: masks.append(kw.get('attn_mask')),
                                               with_kwargs=True) for s in model.sidecar.stages]
    x = inputs(dtype)
    _, action = model(torch.randn(1, 32, 80, dtype=dtype), torch.ones(1, dtype=dtype),
                      use_gradient_checkpointing=checkpointed, **x)
    assert masks[0] is None and all(mask is not None for mask in masks[1:])
    action.float().square().mean().backward()
    assert torch.isfinite(action).all()
    assert model.sidecar.heads.ground_query.weight.grad.abs().sum() > 0
    assert model.sidecar.heads.ground_key.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.video_backbone.parameters())
    for h in handles:
        h.remove()
    model.eval()
    model.set_training_progress(0)
    assert model.grounding_guidance_strength == .5


@pytest.mark.parametrize('cfg', [dict(strength=float('nan')), dict(strength=2), dict(uniform_mix=0),
                                 dict(uniform_mix=2), dict(ramp_updates=-1)])
def test_invalid_guidance_config(make, cfg):
    with pytest.raises(ValueError, match='grounding-guided'):
        make(grounding_guided_read=dict(enabled=True, **cfg))


def test_two_stage_configs_and_original_default():
    with initialize_config_dir(config_dir=str(Path('configs').resolve()), version_base=None):
        old = compose(config_name='train_sidecar_patchwam_warmstart_8gpu_10epochs')
        warm = compose(config_name='train_sidecar_grounded_warmup2k_4gpu')
        joint = compose(config_name='train_sidecar_grounded_joint10epoch_4gpu')
    assert 'grounding_guided_read' not in old.model.architecture.relational_sidecar
    assert 'grounding_guided_read' not in warm.model.architecture.relational_sidecar
    assert joint.model.architecture.relational_sidecar.grounding_guided_read.enabled
    assert warm.training.stop_after_optimizer_updates == 2000
    assert warm.model.architecture.relational_sidecar.backbone_warmup_updates == 2000
    assert joint.model.architecture.relational_sidecar.backbone_warmup_updates == 0
    for cfg in (warm, joint):
        assert cfg.dataloader == old.dataloader
        assert cfg.training.num_epochs == 10 and cfg.training.max_steps is None
        assert cfg.training.batch_size == 32 and cfg.training.gradient_accumulation_steps == 2
        assert cfg.training.learning_rate == 1e-4 and cfg.training.sidecar_lr == 5e-5
        assert cfg.training.warmup_ratio == .05 and cfg.training.sidecar_lr_screen is None
        assert cfg.training.resume_ckpt_path is None and not cfg.training.save_full_states_for_resume
    assert joint.training.save_optimizer_steps == 5000


def test_stage_handover_requires_four_completed_ranks_and_full_weights(tmp_path):
    run = tmp_path/'runs/test'
    run.mkdir(parents=True)
    checkpoint = run/'checkpoint_step_2000.safetensors'
    save_file({'weight': torch.ones(4)}, str(checkpoint))
    marker = checkpoint.with_suffix('.complete.json')
    marker.write_text(json.dumps(dict(step=2000, bytes=checkpoint.stat().st_size)))
    for rank in range(4):
        (run/f'metrics_rank{rank}.jsonl').write_text(json.dumps(dict(optimizer_step=2000, metrics={'loss': .1}))+'\n')
    assert completed_checkpoint(tmp_path, 2000) == checkpoint
    (run/'metrics_rank3.jsonl').write_text(json.dumps(dict(optimizer_step=1999, metrics={'loss': .1}))+'\n')
    with pytest.raises(RuntimeError, match='rank 3'):
        completed_checkpoint(tmp_path, 2000)
    (run/'metrics_rank3.jsonl').write_text(json.dumps(dict(optimizer_step=2000, metrics={'loss': .1}))+'\n')
    checkpoint.write_bytes(checkpoint.read_bytes()[:-1])
    with pytest.raises(RuntimeError, match='marker'):
        completed_checkpoint(tmp_path, 2000)
    marker.write_text(json.dumps(dict(step=2000, bytes=checkpoint.stat().st_size)))
    with pytest.raises(RuntimeError, match='Truncated'):
        completed_checkpoint(tmp_path, 2000)


@pytest.mark.parametrize('fail_warmup', [False, True])
def test_shell_runs_stages_sequentially_with_fresh_checkpoint(tmp_path, fail_warmup):
    foundation = tmp_path/'foundation'
    foundation.mkdir()
    (foundation/'config.yaml').write_text('{}')
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')
    fake_python = tmp_path/'python'
    fake_python.write_text('#!'+sys.executable+'\n'+r"""
import json,os,sys,struct
from pathlib import Path
if sys.argv[1] != '-m':
 os.execv(sys.executable,[sys.executable,*sys.argv[1:]])
args=sys.argv[1:]
config=next(x.split('=',1)[1] for x in args if x.startswith('--config-name='))
source=Path(next(x.split('=',1)[1] for x in args if x.startswith('training.finetune_ckpt_path=')))
output=Path(next(x.split('=',1)[1] for x in args if x.startswith('training.output_path=')))
assert '--nproc_per_node=4' in args
warm='warmup2k' in config
if warm and os.environ['FAIL_WARMUP']=='1':sys.exit(7)
if not warm:assert source.is_file() and source.name=='checkpoint_step_2000.safetensors'
step=2000 if warm else 21480
run=output/'fixture';run.mkdir(parents=True)
header=json.dumps({'w':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
checkpoint=run/f'checkpoint_step_{step}.safetensors'
checkpoint.write_bytes(struct.pack('<Q',len(header))+header+b'\0'*4)
checkpoint.with_suffix('.complete.json').write_text(json.dumps(dict(step=step,bytes=checkpoint.stat().st_size)))
for rank in range(4):
 (run/f'metrics_rank{rank}.jsonl').write_text(json.dumps(dict(optimizer_step=step,metrics={'loss':.1}))+'\n')
with Path(os.environ['CALLS']).open('a') as f:f.write(json.dumps(dict(config=config,source=str(source)))+'\n')
""")
    fake_python.chmod(0o755)
    calls=tmp_path/'calls.jsonl'
    output=tmp_path/'output'
    env=dict(os.environ,PYTHON_BIN=str(fake_python),OPENWAM_FOUNDATION=str(foundation),
             OPENWAM_MANIFEST=str(manifest),OPENWAM_OUTPUT=str(output),CALLS=str(calls),
             FAIL_WARMUP=str(int(fail_warmup)))
    result=subprocess.run(['bash','scripts/cluster/train_sidecar_grounded_two_stage.sh'],
                          env=env,capture_output=True,text=True)
    if fail_warmup:
        assert result.returncode != 0 and not calls.exists()
    else:
        assert result.returncode == 0, result.stderr
        rows=[json.loads(s) for s in calls.read_text().splitlines()]
        assert len(rows)==2
        assert 'warmup2k' in rows[0]['config'] and 'joint10epoch' in rows[1]['config']
        assert rows[1]['source'].startswith(str(output/'warmup/runs/'))
        assert (output/'final_checkpoint.txt').read_text().strip().endswith('checkpoint_step_21480.safetensors')
