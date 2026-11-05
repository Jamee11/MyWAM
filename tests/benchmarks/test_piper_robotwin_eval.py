import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import yaml
from benchmarks.robotwin.piper_protocol import (SLOTS, checkpoint_contract, piper_tasks,
                                               task_config, prepare_piper_runtime)
from benchmarks.robotwin.piper_eval import prepare_plan, summarize


def manifest(path):
    entries = [dict(group='robotwin_piper', split='train', task=t) for t in ['click_bell','click_bell','lift_pot']]
    entries += [dict(group='robotwin_aloha', split='train', task='aloha_only'),
                dict(group='robotwin_piper', split='val', task='val_only')]
    path.write_text(json.dumps(dict(episodes=entries)))
    return path


def checkpoint(path):
    path.mkdir()
    contract = dict(embodiment='robotwin_piper', action_frame='robotwin_endpose_reference',
                    action_semantics='next_achieved_absolute_eef', source_quaternion_order='wxyz',
                    action_slots=SLOTS, arm_present=[1,1], unified_action_dim=80)
    (path/'relational_sidecar_contract.json').write_text(json.dumps(contract))
    weight = path/'checkpoint_step_21952.safetensors'
    weight.write_bytes(b'only path metadata used by CPU prepare')
    return weight


def robotwin(path):
    (path/'task_config').mkdir(parents=True)
    for mode in ['demo_clean','demo_randomized']:
        config = dict(embodiment=['aloha-agilex'], eval_video_log=True,
                      domain_randomization=dict(random_light=mode=='demo_randomized'),
                      camera=dict(head_camera_type='D435', wrist_camera_type='D435'),
                      clear_cache_freq=5)
        (path/'task_config'/f'{mode}.yml').write_text(yaml.safe_dump(config))
    (path/'task_config'/'_camera_config.yml').write_text('D435: {h: 240, w: 320}\n')
    (path/'envs').mkdir()
    return path


def test_task_selection_uses_piper_train_only(tmp_path):
    assert piper_tasks(manifest(tmp_path/'manifest.json')) == ['click_bell','lift_pot']


def test_checkpoint_contract_rejects_wrong_frame(tmp_path):
    p = checkpoint(tmp_path/'ckpt')
    assert checkpoint_contract(p)['action_slots'] == list(range(10))+list(range(34,44))
    c = json.loads((p.parent/'relational_sidecar_contract.json').read_text())
    c['action_frame'] = 'robot_base_tcp'
    (p.parent/'relational_sidecar_contract.json').write_text(json.dumps(c))
    with pytest.raises(ValueError, match='action_frame'):
        checkpoint_contract(p)


@pytest.mark.parametrize('mode', ['demo_clean','demo_randomized'])
def test_only_embodiment_and_video_change(tmp_path, mode):
    root = robotwin(tmp_path/'RoboTwin')
    old = yaml.safe_load((root/'task_config'/f'{mode}.yml').read_text())
    new = task_config(root, mode)
    assert new['embodiment'] == ['piper','piper',.6]
    assert new['eval_video_log'] is False
    assert {k:v for k,v in new.items() if k not in ('embodiment','eval_video_log')} == {k:v for k,v in old.items() if k not in ('embodiment','eval_video_log')}


def test_private_runtime_leaves_aloha_sources_unchanged(tmp_path, monkeypatch):
    root = robotwin(tmp_path/'RoboTwin')
    source = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'task_config').glob('*')}
    policy = tmp_path/'policy.yml';policy.write_text('eval_video_log: false\n')
    monkeypatch.setenv('POLICY_CONFIG_PATH',str(policy));monkeypatch.setenv('ROBOTWIN_RUNTIME_ROOT',str(tmp_path/'runtime'))
    result = prepare_piper_runtime(root,tmp_path/'runtime')
    assert result['embodiment'] == ['piper','piper',.6]
    assert source == {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in source}
    assert yaml.safe_load((root/'task_config/demo_clean.yml').read_text())['embodiment'] == ['aloha-agilex']
    assert yaml.safe_load((tmp_path/'runtime/task_config/demo_clean.yml').read_text())['embodiment'] == ['aloha-agilex']
    assert yaml.safe_load((tmp_path/'runtime/task_config/piper_demo_clean.yml').read_text())['embodiment'] == ['piper','piper',.6]


def test_plan_and_unknown_task(tmp_path):
    args = SimpleNamespace(manifest=manifest(tmp_path/'manifest.json'),checkpoint=checkpoint(tmp_path/'ckpt'),
                           tasks='all',modes=['demo_clean','demo_randomized'],episodes=30,seed_group=0)
    plan = prepare_plan(args)
    assert len(plan['conditions']) == 4 and plan['episodes_per_condition'] == 30
    args.tasks = 'aloha_only'
    with pytest.raises(ValueError,match='absent'):
        prepare_plan(args)


def test_summary_excludes_failed_partial():
    state = dict(episodes_per_condition=30,conditions=[dict(status='complete',mode='demo_clean',result=dict(successes=20)),
                  dict(status='failed',mode='demo_randomized',result=dict(successes=10))])
    value = summarize(state)
    assert value['all']['episodes'] == 30 and value['all']['successes'] == 20
    assert value['demo_randomized']['episodes'] == 0


def test_piper_deployment_explicitly_keeps_all_three_views():
    from omegaconf import OmegaConf
    from openwam.deploy.obs_preprocess import ObsPreprocessor
    root = Path(__file__).resolve().parents[2]
    # The archive training config has no multiview field: deploy must restore it.
    saved = OmegaConf.create(dict(dataloader=dict(name='piper_archive')))
    cfg = OmegaConf.merge(saved, OmegaConf.load(root/'configs/deploy_piper_eval.yaml'))
    processor = ObsPreprocessor.from_cfg(cfg)
    assert processor.multiview is True
    assert processor.camera_layout == ['0','1','2']
    assert (processor.img_height, processor.img_width) == (384,320)
