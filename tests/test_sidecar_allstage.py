"""Stage-loss allocation and predecessor queue checks without launching GPU training."""
from pathlib import Path
import json
import sys

import pytest
import torch
from hydra import compose, initialize_config_dir

from tests.test_relational_sidecar import make, inputs
from tests.test_ea_rsig_lite import target


def test_allstage_coefficients_and_direct_gradients(make, monkeypatch):
    model = make(stage_supervision_weights=[.2, .3, .5], aux_loss_scale=.5)
    leaves = [{k: torch.tensor(float(i+1), requires_grad=True) for k in ('ground', 'relation', 'plan')}
              for i in range(3)]
    iterator = iter(leaves)
    monkeypatch.setattr(model.sidecar.heads, 'losses', lambda *_: next(iterator))
    x = inputs()
    result = model.compute_loss(actions=torch.randn(1, 32, 80), input_latents=x['latents'],
        first_frame_latents=x['first_frame_latents'], context=x['context'], context_mask=x['context_mask'],
        proprio=x['proprio'], proprio_mask=x['_proprio_sample_mask'],
        fuse_vae_embedding_in_latents=True, ea_targets=target())
    expected = sum(.5 * a * sum(model.loss_weights[k] * (1.1 if k != 'plan' else 1) * v
                   for k, v in row.items()) for a, row in zip([.2, .3, .5], leaves))
    torch.testing.assert_close(result['loss_ea'], expected)
    result['loss'].backward()
    for a, row in zip([.2, .3, .5], leaves):
        for k, v in row.items():
            assert v.grad.item() == pytest.approx(.5*a*model.loss_weights[k]*(1.1 if k != 'plan' else 1))
    assert sum(result[f'loss_ea_stage_{i}'] for i in range(3)) == result['loss_ea']


def test_allstage_preserves_checkpoint_and_inference(make):
    old, new = make(), make(stage_supervision_weights=[.2, .3, .5])
    new.load_state_dict(old.state_dict(), strict=True)
    assert list(old.state_dict()) == list(new.state_dict())
    x, actions = inputs(), torch.randn(1, 32, 80)
    for a, b in zip(old(actions, torch.ones(1), **x), new(actions, torch.ones(1), **x)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('weights', [[.5,.5], [.2,.3,.6], [0,.5,.5], [-.1,.5,.6], [float('nan'),.3,.7]])
def test_invalid_allstage_weights(make, weights):
    with pytest.raises(ValueError, match='Stage supervision'):
        make(stage_supervision_weights=weights)


def test_allstage_config_preserves_full_patchwam_and_batch():
    with initialize_config_dir(config_dir=str(Path('configs').resolve()), version_base=None):
        old = compose(config_name='train_sidecar_patchwam_warmstart_8gpu_10epochs')
        new = compose(config_name='train_sidecar_allstage_warmup2k_8gpu_10epochs')
    assert new.dataloader == old.dataloader
    assert 'cross_appearance' not in new.model.architecture.relational_sidecar
    assert list(new.model.architecture.relational_sidecar.stage_supervision_weights) == [.2,.3,.5]
    assert new.training.batch_size == 32 and new.training.gradient_accumulation_steps == 1
    assert new.training.learning_rate == old.training.learning_rate == 1e-4
    assert new.training.sidecar_lr == old.training.sidecar_lr == 5e-5
    assert new.training.num_epochs == 10 and new.training.max_steps is None
    assert new.training.resume_ckpt_path is None and new.training.sidecar_lr_screen is None
    assert new.training.save_optimizer_steps == 5000


def test_queue_waits_for_predecessor_and_launches_only_once(tmp_path, monkeypatch):
    from scripts import supervise_sidecar_xa as queue
    from scripts import supervise_sidecar_c2r_two_stage as dependency
    project, out = tmp_path/'project', tmp_path/'out'
    (project/'scripts').mkdir(parents=True)
    launcher = project/'scripts/train_sidecar_allstage_warmup2k_8gpu.sh'
    launcher.write_text('#!/bin/bash\n')
    calls=[]
    def completed(path, target):
        calls.append(('predecessor', target))
        if len(calls)==1:
            raise RuntimeError('Still training')
        return tmp_path/'predecessor.safetensors'
    monkeypatch.setattr(dependency, 'completed', completed)
    monkeypatch.setattr(queue.socket, 'gethostname', lambda: 'aigc24')
    monkeypatch.setattr(queue.time, 'sleep', lambda _: None)
    monkeypatch.setattr(queue.subprocess, 'check_output', lambda *a,**k: '\n'.join(f'{i}, 80000, 0' for i in range(8)))
    def popen(command, **kwargs):
        calls.append(('launch', command))
        run=out/'production/runs/fake';run.mkdir(parents=True)
        (run/'checkpoint_step_21480.safetensors').write_bytes(b'fixture')
        for rank in range(8):
            (run/f'metrics_rank{rank}.jsonl').write_text(json.dumps(dict(optimizer_step=21480))+'\n')
        class Process:
            pid=123
            returncode=0
            def poll(self): return 0
        return Process()
    monkeypatch.setattr(queue.subprocess, 'Popen', popen)
    monkeypatch.setattr(sys, 'argv', ['queue','--project',str(project),'--output',str(out),
        '--node','aigc24','--wait-for-stage',str(tmp_path/'before'), '--wait-for-updates','21480',
        '--launcher',launcher.name,'--skip-preflight','--lock-port','0'])
    queue.main()
    assert calls[:2] == [('predecessor',21480),('predecessor',21480)]
    assert len([c for c in calls if c[0]=='launch']) == 1
    assert json.loads((out/'complete.json').read_text())['optimizer_updates'] == 21480
    with pytest.raises(RuntimeError, match='already launched'):
        queue.main()
