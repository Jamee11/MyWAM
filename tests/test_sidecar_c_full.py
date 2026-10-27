"""Full continuation may preload early but must restore a complete, normally exited C."""
import json

import pytest

from scripts import supervise_sidecar_c_full as supervisor
from scripts.supervise_sidecar_c_full import completed_source, pin_intermediate, verify_state
from scripts.train_sidecar_four_arm import screen_source, verify_full_source


def full_plan(tmp_path):
    state = tmp_path / 'short_run/accel_state_step_8000'
    return dict(full_training=True, continuation=True, preload_scaffold=True,
                resume_micro=8000, target_micro=42960,
                sources={'C_wam_ramp': dict(weights_dir=str(tmp_path / 'scaffold'), full_state=str(state))})


def completed_run(tmp_path):
    arm, run = tmp_path / 'C_wam_ramp', tmp_path / 'short_run'
    arm.mkdir(parents=True)
    state = run / 'accel_state_step_8000'
    state.mkdir(parents=True)
    (state / 'trainer_state.json').write_text(json.dumps(dict(global_step=8000, opt_step=4000, epoch=1)))
    (run / 'checkpoint_step_8000.safetensors').write_bytes(b'fixed weights')
    (arm / 'output_run.json').write_text(json.dumps(dict(path=str(run))))
    marker = dict(returncode=0, global_step=8000, opt_step=4000, epoch=1, output_run=str(run))
    (arm / 'completed.json').write_text(json.dumps(marker))
    return arm, run, marker


def test_cpu_scaffold_cannot_authorize_missing_or_wrong_actual_state(tmp_path):
    plan = full_plan(tmp_path)
    result = screen_source(tmp_path / 'C_wam_ramp', plan)
    assert result[1:3] == (8000, 42960)
    with pytest.raises(FileNotFoundError):
        verify_full_source(result[4], result[5])
    result[4].mkdir(parents=True)
    marker = result[4] / 'trainer_state.json'
    marker.write_text(json.dumps(dict(global_step=6000, opt_step=3000, epoch=1)))
    with pytest.raises(ValueError, match='optimizer4000'):
        screen_source(tmp_path / 'C_wam_ramp', plan)
    marker.write_text(json.dumps(result[5]))
    verify_full_source(result[4], result[5])
    assert screen_source(tmp_path / 'C_wam_ramp', plan) == result


@pytest.mark.parametrize('changes', [dict(resume_micro=6000), dict(target_micro=50960), dict(continuation=False)])
def test_rejects_restart_or_extra_ten_epochs(tmp_path, changes):
    plan = full_plan(tmp_path)
    # Invalid continuation=False would otherwise lack a common source; it still must not launch.
    plan['common'] = str(tmp_path / 'scaffold')
    plan.update(changes)
    with pytest.raises(ValueError, match='original ten-epoch'):
        screen_source(tmp_path / 'C_wam_ramp', plan)


def test_completed_marker_needs_exit_zero_and_every_optimizer_shard(tmp_path):
    arm, run, marker = completed_run(tmp_path)
    marker['returncode'] = 7
    (arm / 'completed.json').write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='exit normally'):
        completed_source(arm, ['trainer_state.json'])
    marker['returncode'] = 0
    (arm / 'completed.json').write_text(json.dumps(marker))
    required = ['trainer_state.json', 'pytorch_model/rank3.pt']
    with pytest.raises(ValueError, match='Incomplete four-rank state'):
        completed_source(arm, required)
    shard = run / 'accel_state_step_8000/pytorch_model/rank3.pt'
    shard.parent.mkdir()
    shard.write_bytes(b'optimizer')
    restored, inventory = completed_source(arm, required)
    assert restored == run and inventory[required[1]]['size'] == 9
    assert verify_state(run, 8000, 1, required) == inventory


def test_intermediate_pin_survives_native_weight_pruning_without_state_copy(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'tokenizer').mkdir()
    for filename in ('checkpoint_step_32000.safetensors', 'config.yaml', 'normalization_stats.npy',
                     'relational_sidecar_contract.json'):
        (run / filename).write_bytes(b'fixed data')
    (run / 'tokenizer/token.json').write_text('{}')
    destination = tmp_path / 'snapshot'
    pin_intermediate(run, destination, 32000)
    weight = destination / 'checkpoint_step_32000.safetensors'
    assert weight.stat().st_ino == (run / weight.name).stat().st_ino
    (run / weight.name).unlink()
    assert weight.read_bytes() == b'fixed data'
    assert not list(destination.glob('accel_state*'))
    assert json.loads((destination / 'snapshot.json').read_text())['updates'] == 16000


def test_queue_preloads_before_completion_but_authorizes_only_after_exit_zero(tmp_path, monkeypatch):
    predecessor, root = tmp_path / 'prior', tmp_path / 'full'
    prior, source, marker = completed_run(predecessor)
    done = prior / 'completed.json'
    done.unlink()
    supervisor.save(predecessor / 'job_status.json', dict(status='training'))
    arm = root / 'C_wam_ramp'
    (arm / 'handover').mkdir(parents=True)
    (root / 'engineering').mkdir()
    supervisor.save(root / 'engineering/acceptance.json', dict(status='pass'))
    plan = dict(full_plan(tmp_path), predecessor_root=str(predecessor), arms=['C_wam_ramp'],
                required_state_files=['trainer_state.json'])
    supervisor.save(root / 'plan.json', plan)
    phase, starts = [0], []

    class Child:
        pid = -1
        returncode = None

        def poll(self):
            self.returncode = 0 if phase[0] >= 5 else None
            return self.returncode

    def start(*args):
        assert phase[0] == 1 and not done.exists()
        starts.append(args)
        supervisor.save(arm / 'handover/cpu_ready.json', {})
        return Child()

    def latest(path):
        if path == prior:
            return dict(micro_step=7400) if phase[0] else None
        return dict(micro_step=8002) if phase[0] >= 3 else None

    run = arm / 'runs/final'

    def advance(_):
        if phase[0] == 1:
            assert not (arm / 'gpu_authorized.json').exists()
            supervisor.save(done, marker)
        if phase[0] == 2:
            assert (arm / 'gpu_authorized.json').exists()
            run.mkdir(parents=True)
            supervisor.save(arm / 'output_run.json', dict(path=str(run)))
            supervisor.save(arm / 'resume_verified.json', dict(updates=4000))
        if phase[0] == 3:
            (run / 'accel_state_step_42960').mkdir()
            supervisor.save(run / 'accel_state_step_42960/trainer_state.json',
                            dict(global_step=42960, opt_step=21480, epoch=9))
            (run / 'checkpoint_step_42960.safetensors').write_bytes(b'final')
            supervisor.save(arm / 'screen_complete.json', {})
        if phase[0] == 4:
            assert (arm / 'release_exit.json').exists()
        phase[0] += 1

    monkeypatch.setattr(supervisor, 'start', start)
    monkeypatch.setattr(supervisor, 'latest', latest)
    monkeypatch.setattr(supervisor.time, 'sleep', advance)
    monkeypatch.setattr(supervisor.sys, 'argv', ['supervise', str(root)])
    supervisor.main()
    assert len(starts) == 1
    assert json.loads((arm / 'completed.json').read_text())['opt_step'] == 21480
    assert (arm / 'startup_acceptance.json').exists()
