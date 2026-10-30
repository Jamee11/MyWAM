"""Two-stage boundaries, exact frozen-policy verification and shared-GPU admission."""
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts import supervise_sidecar_c2r_two_stage as queue


def test_resolved_stage_contracts(monkeypatch, tmp_path):
    import yaml
    monkeypatch.setenv('EA_SIDECAR_MANIFEST', str(tmp_path / 'manifest.json'))
    a = yaml.safe_load(queue.resolve(0, '/official/checkpoint_step_10740.safetensors', tmp_path / 'a'))
    b = yaml.safe_load(queue.resolve(1, '/warmup/checkpoint_step_2000.safetensors', tmp_path / 'b'))
    assert a['model']['architecture']['relational_sidecar']['foundation_warmstart']
    for key in ('learning_rate', 'action_lr', 'video_lr', 'sidecar_lr'):
        assert b['training'][key] == a['training'][key] / 4
    assert b['model']['architecture']['relational_sidecar']['backbone_warmup_updates'] == 0
    assert a['dataloader']['patchwam_augmentation'] == b['dataloader']['patchwam_augmentation']


def test_warmup_rejects_changed_official_weights(tmp_path):
    official, warmed = tmp_path / 'official.safetensors', tmp_path / 'warm.safetensors'
    save_file({'action.weight': torch.ones(3)}, official)
    save_file({'action.weight': torch.ones(3), 'sidecar.weight': torch.zeros(2)}, warmed)
    assert queue.verify_warmup(official, warmed) == dict(unchanged_policy_tensors=1, new_sidecar_tensors=1)
    save_file({'action.weight': torch.ones(3) + 1, 'sidecar.weight': torch.zeros(2)}, warmed)
    with pytest.raises(RuntimeError, match='Warmup changed official tensor'):
        queue.verify_warmup(official, warmed)


def test_predecessor_requires_final_save_and_all_eight_ranks(tmp_path):
    weights = tmp_path / 'checkpoint_step_21480.safetensors'
    save_file({'policy': torch.ones(1)}, weights)
    for rank in range(8):
        row = dict(optimizer_step=21480, metrics={'loss': 1.0})
        (tmp_path / f'metrics_rank{rank}.jsonl').write_text(json.dumps(row) + '\n')
    (tmp_path / 'launch.log').write_text(f'[checkpoint] Saved: {weights}\n')
    with pytest.raises(RuntimeError, match='completion was not logged'):
        queue.completed(tmp_path, 21480)
    (tmp_path / 'launch.log').write_text(f'[checkpoint] Saved final: {weights}\n')
    assert queue.completed(tmp_path, 21480) == weights
    (tmp_path / 'metrics_rank7.jsonl').write_text(json.dumps(dict(optimizer_step=21479, metrics={'loss': 1.0})))
    with pytest.raises(RuntimeError, match='eight ranks'):
        queue.completed(tmp_path, 21480)


@pytest.mark.parametrize('apps,used,expected', [('', 4, True), ('1234\n', 4, False), ('', 2000, False)])
def test_busy_gpu_is_never_taken(monkeypatch, apps, used, expected):
    def output(args, **kwargs):
        if '--query-compute-apps=pid' in args:
            return apps
        return '\n'.join(f'NVIDIA H100 80GB HBM3, {used}, {81559-used}' for _ in range(8))
    monkeypatch.setattr(queue.subprocess, 'check_output', output)
    assert queue.idle() is expected
