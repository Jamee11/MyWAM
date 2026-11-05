"""Piper-only metadata/runtime preparation; existing Aloha entrypoints are untouched."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / 'datasets/multisim_sidecar_archive_v2_20261025/manifest.ready.json'
DEFAULT_CHECKPOINT = ROOT / 'outputs/piper_sidecar/midtrain_patchwam_2epochs_20261102T101245Z/runs/2026-11-02_10-18-00/checkpoint_step_21952.safetensors'
DEFAULT_ROBOTWIN = ROOT / '.external/robotwin-official-20261018/RoboTwin'
MODES = ('demo_clean', 'demo_randomized')
EMBODIMENT = ['piper', 'piper', 0.6]
SLOTS = list(range(10)) + list(range(34, 44))


def piper_tasks(manifest: Path) -> list[str]:
    data = json.loads(Path(manifest).read_text())
    tasks = sorted({e['task'] for e in data['episodes'] if e['group'] == 'robotwin_piper' and e['split'] == 'train'})
    if not tasks:
        raise ValueError('No robotwin_piper train tasks in manifest')
    return tasks


def checkpoint_contract(checkpoint: Path) -> dict:
    checkpoint = Path(checkpoint)
    contract = json.loads((checkpoint.parent / 'relational_sidecar_contract.json').read_text())
    expected = dict(embodiment='robotwin_piper', action_frame='robotwin_endpose_reference',
                    action_semantics='next_achieved_absolute_eef', source_quaternion_order='wxyz',
                    action_slots=SLOTS, arm_present=[1, 1], unified_action_dim=80)
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f'Piper checkpoint contract mismatch: {key}')
    return contract


def task_config(robotwin: Path, mode: str) -> dict:
    if mode not in MODES:
        raise ValueError(mode)
    config = yaml.safe_load((Path(robotwin) / 'task_config' / f'{mode}.yml').read_text())
    config['embodiment'] = list(EMBODIMENT)
    config['eval_video_log'] = False
    return config


def prepare_piper_runtime(robotwin: Path, runtime: Path) -> dict:
    """Use existing private runtime copy, then add separately named Piper configs."""
    import os
    from benchmarks.robotwin import eval_policy_wrapper as wrapper
    robotwin, runtime = Path(robotwin).resolve(), Path(runtime).resolve()
    if runtime == robotwin or runtime.is_relative_to(robotwin):
        raise ValueError('Piper runtime must be outside the official checkout')
    os.environ['ROBOTWIN_RUNTIME_ROOT'] = str(runtime)
    wrapper._prepare_runtime_root(str(robotwin))
    records = {}
    for mode in MODES:
        target = runtime / 'task_config' / f'piper_{mode}.yml'
        config = task_config(robotwin, mode)
        target.write_text(yaml.safe_dump(config, sort_keys=False))
        records[mode] = dict(path=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    evidence = dict(embodiment=EMBODIMENT, eval_video_log=False, source_robotwin=str(robotwin),
                    runtime=str(runtime), configs=records,
                    changes='Only embodiment and video flag; official randomization/cameras/step budgets unchanged')
    (runtime / 'piper_runtime.json').write_text(json.dumps(evidence, indent=2)+'\n')
    return evidence
