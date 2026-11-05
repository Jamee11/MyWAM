import json
import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir


def test_original_wam_recipe_has_no_sidecar_or_patchwam():
    root=Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
        cfg=compose(config_name='train_maniskill_archive_baseline_2epochs')
    assert cfg.model.architecture.variant=='joint_self_attn'
    assert not cfg.model.architecture.get('relational_sidecar',{}).get('enabled',False)
    assert cfg.training.batch_size==64 and cfg.training.gradient_accumulation_steps==1
    assert cfg.training.num_epochs==2 and cfg.training.max_steps is None
    assert cfg.training.resume_ckpt_path is None
    assert 'Foundation-Model/checkpoint_step_154000' in cfg.training.finetune_ckpt_path
    assert cfg.dataloader.auxiliary is False
    assert cfg.dataloader.color_jitter.enabled and cfg.dataloader.color_jitter.hue==0
    assert not cfg.dataloader.patchwam_augmentation.enabled
    assert cfg.training.lambda_video==1 and cfg.training.lambda_action==1


def test_baseline_queue_defers_preload_and_reuses_handover(tmp_path,monkeypatch):
    from scripts import supervise_piper_after_maniskill as queue
    path=tmp_path/'spec.json'
    path.write_text(json.dumps(dict(root=str(tmp_path),python='/local/python',
        predecessor=dict(pid=123,start_ticks='1'),completion=dict(predecessor_run=str(tmp_path)),
        training_label='maniskill_baseline',predecessor_label='piper',
        preload_from_optimizer_update=90,output_env='MANISKILL_BASELINE_ARCHIVE_OUTPUT',
        launch_script='scripts/train_maniskill_archive_baseline_4gpu.sh')))
    events=[]
    ticks=iter([1,90])
    monkeypatch.setattr(queue,'tail',lambda path: json.dumps(dict(optimizer_step=next(ticks))))
    monkeypatch.setattr(queue.time,'sleep',lambda seconds: events.append(('sleep',seconds)))
    monkeypatch.setattr(queue,'same_process',lambda *args: not any(e[0]=='launch' for e in events))
    monkeypatch.setattr(queue,'completion_errors',lambda spec: [])

    class Child:
        pid=os.getpid()

        def __init__(self,command,**kwargs):
            events.append(('launch',command,kwargs['env']))
            (tmp_path/'handover').mkdir()
            (tmp_path/'handover/gpu_release.json').write_text('{}')

        def poll(self):
            return None

        def wait(self):
            return 0

    monkeypatch.setattr(queue.subprocess,'Popen',Child)
    monkeypatch.setattr(sys,'argv',['supervisor','--spec',str(path)])
    queue.main()
    assert events[0]==('sleep',30)
    launch=next(e for e in events if e[0]=='launch')
    assert launch[1]==['bash','scripts/train_maniskill_archive_baseline_4gpu.sh']
    assert launch[2]['MANISKILL_BASELINE_ARCHIVE_OUTPUT']==str(tmp_path/'runs')
    assert json.loads((tmp_path/'predecessor_complete.json').read_text())['status']=='complete'
