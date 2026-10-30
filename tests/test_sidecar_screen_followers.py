"""The fifth arm must retain Control's starting state and exact update budget."""
import json
import subprocess

import pytest

from scripts import diagnose_sidecar_four_arm as diagnosis
from scripts.supervise_sidecar_gain_fifth import finish


def test_fifth_rejects_wrong_budget_or_optimizer_before_completion(tmp_path):
    control, arm, run = tmp_path / 'A_control', tmp_path / 'E_gain2', tmp_path / 'run'
    for path in (control, arm, run / 'accel_state_step_6000'):
        path.mkdir(parents=True)
    (arm / 'output_run.json').write_text(json.dumps(dict(path=str(run))))
    marker = run / 'accel_state_step_6000/trainer_state.json'
    marker.write_text(json.dumps(dict(global_step=6000, opt_step=2999, epoch=0)))
    with pytest.raises(RuntimeError, match='exactly 1000'):
        finish(tmp_path, arm)
    assert not (arm / 'completed.json').exists()
    marker.write_text(json.dumps(dict(global_step=6000, opt_step=3000, epoch=0)))
    for rank in range(4):
        for path in (control, arm):
            (path / f'initial_optimizer_rank{rank}.json').write_text('identical')
    (arm / 'initial_optimizer_rank3.json').write_text('changed')
    with pytest.raises(RuntimeError, match='differs from Control'):
        finish(tmp_path, arm)
    assert not (arm / 'completed.json').exists()
    (arm / 'initial_optimizer_rank3.json').write_text('identical')
    finish(tmp_path, arm)
    assert json.loads((arm / 'completed.json').read_text())['opt_step'] == 3000


def test_diagnostics_select_only_available_five_or_six(monkeypatch):
    def snapshot(five, six):
        return f'0, 80000\n1, 80000\n4, 80000\n5, {five}\n6, {six}\n7, 80000\n'
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **k: snapshot(27000, 27000))
    assert diagnosis.selected_gpu() is None
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **k: snapshot(50000, 60000))
    assert diagnosis.selected_gpu() == 6
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **k: snapshot(50000, 10000))
    assert diagnosis.selected_gpu() == 5


@pytest.mark.parametrize('target', [6000, 8000])
def test_diagnostics_pin_endpoint_and_preserve_fixed_samples(tmp_path, monkeypatch, target):
    reference, run, arm = tmp_path / 'reference', tmp_path / 'run', tmp_path / 'A_control'
    for path in (reference, run / f'accel_state_step_{target}', run / 'tokenizer', arm):
        path.mkdir(parents=True)
    (run / f'accel_state_step_{target}/trainer_state.json').write_text(
        json.dumps(dict(global_step=target, opt_step=target // 2, epoch=0)))
    (arm / 'completed.json').write_text(json.dumps(dict(output_run=str(run))))
    for name in (f'checkpoint_step_{target}.safetensors', 'config.yaml', 'normalization_stats.npy',
                 'relational_sidecar_contract.json'):
        (run / name).write_bytes(b'fixed data')
    (reference / 'plan.json').write_text('{}')
    (reference / 'inputs_ready.json').write_text('{"samples": ["fixed input"]}')
    (tmp_path / 'diagnostics').mkdir()
    (tmp_path / 'plan.json').write_text(json.dumps(dict(target_micro=target)))
    monkeypatch.setattr(diagnosis, 'REFERENCE', reference)
    out = diagnosis.prepare(tmp_path, 'A_control', 6)
    plan = json.loads((out / 'plan.json').read_text())
    assert (plan['optimizer_updates'], plan['scales'], plan['model_gpu'], plan['model_device']) == (
        target // 2, [0, 1], 6, 'cuda:0')
    assert (out / 'inputs_ready.json').read_bytes() == (reference / 'inputs_ready.json').read_bytes()
    pinned = out / f'checkpoint_step_{target}/checkpoint_step_{target}.safetensors'
    assert pinned.stat().st_ino == (run / pinned.name).stat().st_ino
    assert plan['checkpoint_sha256'] == diagnosis.sha256(pinned)
