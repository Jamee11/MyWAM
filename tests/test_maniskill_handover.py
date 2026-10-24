"""Completion must mean all ranks, final saved marker, and a complete weight file."""
import json
import os
import struct
from pathlib import Path

import pytest

from scripts import supervise_maniskill_after_robotwin as supervisor
from scripts.supervise_maniskill_after_robotwin import completion_errors, same_process


def finished(tmp_path):
    spec = dict(predecessor_run=str(tmp_path), predecessor_log=str(tmp_path / 'train.log'), total_micro=4)
    for rank in range(4):
        (tmp_path / f'metrics_rank{rank}.jsonl').write_text('{"micro_step":4}\n')
    checkpoint = tmp_path / 'checkpoint_step_4.safetensors'
    header = json.dumps({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
    checkpoint.write_bytes(struct.pack('<Q', len(header)) + header + struct.pack('<f', 1))
    (tmp_path / 'train.log').write_text(f'[checkpoint] Saved final: {checkpoint}\n')
    return spec


def test_complete_run(tmp_path):
    assert completion_errors(finished(tmp_path)) == []


def test_one_rank_incomplete(tmp_path):
    spec = finished(tmp_path)
    (tmp_path / 'metrics_rank2.jsonl').write_text('{"micro_step":3}\n')
    assert completion_errors(spec) == ['rank2 has not reached 4']


def test_missing_final_marker(tmp_path):
    spec = finished(tmp_path)
    (tmp_path / 'train.log').write_text('Traceback: process failed\n')
    assert 'final checkpoint saved marker missing' in completion_errors(spec)


def test_truncated_weight(tmp_path):
    spec = finished(tmp_path)
    p = tmp_path / 'checkpoint_step_4.safetensors'
    p.write_bytes(p.read_bytes()[:-1])
    assert 'final checkpoint size/header mismatch' in completion_errors(spec)


def test_process_identity_reuse():
    ticks = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]
    assert same_process(os.getpid(), ticks)
    assert not same_process(os.getpid(), '0')
    assert not same_process(999999999, ticks)


def test_one_shot_launch_after_completion(tmp_path, monkeypatch):
    spec = finished(tmp_path)
    (tmp_path / 'outputs/maniskill_sidecar').mkdir(parents=True)
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts/train_maniskill_sidecar_4gpu.sh').write_text('echo synthetic-launch\n')
    spec.update(root=str(tmp_path), hostname=os.uname().nodename,
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                pids_start_ticks={}, master_port=0, local_venv=str(tmp_path / 'missing'),
                fallback_venv='/usr', manifest='unused-by-synthetic-launch', gpus=[0, 1, 2, 3])
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec))
    readiness = iter([False, True])
    monkeypatch.setattr(supervisor, 'available_gpus', lambda spec: next(readiness))
    monkeypatch.setattr(supervisor.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr('sys.argv', ['supervisor', '--spec', str(path)])
    supervisor.main()
    state = json.loads((tmp_path / 'status.json').read_text())
    assert state['status'] == 'maniskill_exited' and state['returncode'] == 0
    assert (Path(state['output']) / 'train.log').read_text() == 'synthetic-launch\n'
    with pytest.raises(RuntimeError, match='already attempted'):
        supervisor.main()


def test_predecessor_failure_never_launches(tmp_path, monkeypatch):
    spec = finished(tmp_path)
    (tmp_path / 'train.log').write_text('interrupted\n')
    (tmp_path / 'outputs/maniskill_sidecar').mkdir(parents=True)
    spec.update(root=str(tmp_path), hostname=os.uname().nodename,
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), pids_start_ticks={})
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec))
    monkeypatch.setattr('sys.argv', ['supervisor', '--spec', str(path)])
    with pytest.raises(RuntimeError, match='did not finish normally'):
        supervisor.main()
    assert not (tmp_path / 'launched.json').exists()
