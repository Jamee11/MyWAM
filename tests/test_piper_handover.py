import json
import os
import sys

import pytest

from scripts import supervise_piper_after_maniskill as supervisor


@pytest.mark.parametrize('errors', [[], ['final save missing']])
def test_preload_then_release_only_after_completed_predecessor(tmp_path, monkeypatch, errors):
    path = tmp_path/'spec.json'
    path.write_text(json.dumps(dict(root=str(tmp_path), python='/local/python',
        predecessor=dict(pid=123, start_ticks='1'), completion={})))
    calls = []

    class Child:
        pid = os.getpid()

        def __init__(self, command, **kwargs):
            calls.append((command, kwargs['env']))
            (tmp_path/'handover').mkdir()
            (tmp_path/'handover/gpu_release.json').write_text('{}')

        def poll(self):
            return None

        def wait(self):
            return 0

    monkeypatch.setattr(supervisor.subprocess, 'Popen', Child)
    monkeypatch.setattr(supervisor, 'same_process', lambda *args: False)
    monkeypatch.setattr(supervisor, 'completion_errors', lambda spec: errors)
    monkeypatch.setattr(supervisor.os, 'killpg', lambda *args: calls.append(('terminated', args)))
    monkeypatch.setattr(sys, 'argv', ['supervisor', '--spec', str(path)])
    if errors:
        with pytest.raises(RuntimeError, match='final-save verification failed'):
            supervisor.main()
        assert not (tmp_path/'predecessor_complete.json').exists()
        assert calls[-1][0] == 'terminated'
    else:
        supervisor.main()
        assert json.loads((tmp_path/'predecessor_complete.json').read_text())['status'] == 'complete'
        assert json.loads((tmp_path/'status.json').read_text())['status'] == 'complete'
    assert calls[0][0] == ['bash', 'scripts/train_piper_archive_patchwam_4gpu.sh']
    assert calls[0][1]['CUDA_VISIBLE_DEVICES'] == '0,1,2,3'
