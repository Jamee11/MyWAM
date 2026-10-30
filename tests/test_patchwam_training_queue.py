"""Fresh training can preload early but must not inherit C's optimizer or progress."""
import json
import os
from types import SimpleNamespace

import pytest

from scripts import supervise_sidecar_four_arm as launcher
from scripts import supervise_sidecar_patchwam_after_c as queue
from tests.test_sidecar_c_full import completed_run


def cpu_contract(foundation='/foundation'):
    return dict(dataset_windows=549775, world_size=4, batch_per_gpu=32, accumulation=2,
                backbone_warmup_updates=2000, aux_shared_gradient_scale=1.0,
                aux_gradient_anneal_updates=0, completed_optimizer_updates=0,
                wam_ramp_updates=100, finetune_ckpt_path=foundation, resume_ckpt_path=None,
                patchwam_augmentation_active=True)


@pytest.mark.parametrize('change', [dict(wam_ramp_updates=300), dict(completed_optimizer_updates=4000),
                                   dict(resume_ckpt_path='/old_C'), dict(patchwam_augmentation_active=False)])
def test_rejects_wrong_fresh_training_contract(tmp_path, change):
    (tmp_path / 'handover').mkdir()
    queue.save(tmp_path / 'handover/cpu_contract.json', dict(cpu_contract(), **change))
    with pytest.raises(ValueError, match='Fresh Foundation'):
        queue.verify_cpu_contract(tmp_path, '/foundation')


def test_new_launcher_sets_formal_entry_and_completion_gate_without_changing_default(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda command, **kwargs:
                        calls.append((command, kwargs)) or SimpleNamespace(pid=os.getpid()))
    for name, plan in [('control', dict(python='/python')),
                       ('fresh', dict(python='/python', entrypoint='scripts/train_relational_sidecar_formal.py',
                                      predecessor_done_file=str(tmp_path / 'C_verified.json')))]:
        (tmp_path / name).mkdir()
        launcher.start(tmp_path, name, plan)
    assert 'scripts/train_sidecar_four_arm.py' in calls[0][0]
    assert 'EA_PREDECESSOR_DONE_FILE' not in calls[0][1]['env']
    assert 'scripts/train_relational_sidecar_formal.py' in calls[1][0]
    assert calls[1][1]['env']['EA_PREDECESSOR_DONE_FILE'] == str(tmp_path / 'C_verified.json')


def test_preloads_fresh_and_releases_after_C_exit_while_diagnostics_remain_parallel(tmp_path, monkeypatch):
    prior_root, root = tmp_path / 'prior', tmp_path / 'fresh'
    prior, source, marker = completed_run(prior_root)
    done = prior / 'completed.json'
    done.unlink()
    queue.save(prior_root / 'job_status.json', dict(status='training'))
    arm = root / 'F_patchwam_ramp100'
    run = arm / 'runs/new'
    (arm / 'handover').mkdir(parents=True)
    run.mkdir(parents=True)
    (run / 'config.yaml').write_text('{}')
    (root / 'engineering').mkdir()
    queue.save(root / 'engineering/acceptance.json', dict(status='pass'))
    queue.save(arm / 'handover/cpu_contract.json', cpu_contract())
    queue.save(arm / 'handover/cpu_ready.json', {})
    plan = dict(arms=['F_patchwam_ramp100'], entrypoint='scripts/train_relational_sidecar_formal.py',
                target_micro=42960, fresh_from_foundation=True, predecessor_root=str(prior_root),
                foundation='/foundation', required_state_files=['trainer_state.json'])
    queue.save(root / 'plan.json', plan)
    phase, starts = [0], []

    class Child:
        pid, returncode = -1, None

        def poll(self):
            self.returncode = 0 if phase[0] >= 3 else None
            return self.returncode

    def start(*args):
        assert not done.exists()
        starts.append(args)
        return Child()

    def latest(path):
        if path == prior:
            return dict(micro_step=7900)
        if phase[0] == 0:
            return None
        if phase[0] < 3:
            return dict(micro_step=2, optimizer_step=1, metrics=dict(ea_backbone_frozen_warmup=1))
        return dict(micro_step=42960, optimizer_step=21480)

    def advance(_):
        if phase[0] == 0:
            assert not (root / 'predecessor_verified.json').exists()
            queue.save(done, marker)
        elif phase[0] == 1:
            assert (root / 'predecessor_verified.json').exists()
            assert (arm / 'cpu_acceptance.json').exists()
        else:
            (run / 'checkpoint_step_42960.safetensors').write_bytes(b'final')
        phase[0] += 1

    monkeypatch.setattr(queue, 'start', start)
    monkeypatch.setattr(queue, 'latest', latest)
    monkeypatch.setattr(queue.time, 'sleep', advance)
    monkeypatch.setattr(queue.sys, 'argv', ['queue', str(root)])
    queue.main()
    assert len(starts) == 1
    assert json.loads((arm / 'startup_acceptance.json').read_text())['latest']['optimizer_step'] == 1
    assert json.loads((arm / 'completed.json').read_text())['opt_step'] == 21480
    assert json.loads((source / 'accel_state_step_8000/trainer_state.json').read_text())['opt_step'] == 4000
