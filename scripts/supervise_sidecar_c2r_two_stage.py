"""On aigc24, finish the existing run, then warm up a new Sidecar and finetune five epochs."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / 'assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Clean2Random'
WEIGHTS = OFFICIAL / 'checkpoint_step_10740.safetensors'
PROFILES = ('train_sidecar_c2r_warmup2k_8gpu', 'train_sidecar_c2r_joint5ep_8gpu')


def save(path, row):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(row, indent=2) + '\n')
    temporary.replace(path)


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def latest(run):
    rows = {}
    for path in sorted(run.rglob('metrics_rank*.jsonl')):
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 32000))
            lines = stream.read().splitlines()
        for line in reversed(lines):
            try:
                row = json.loads(line)
                break
            except ValueError:
                continue
        else:
            continue
        if not all(math.isfinite(value) for value in row['metrics'].values()):
            raise RuntimeError(f'Nonfinite metrics in {path}')
        if path.name in rows:
            raise RuntimeError('Ambiguous output directories')
        rows[path.name] = row
    return rows


def completed(run, target):
    rows = latest(run)
    if len(rows) != 8 or any(row['optimizer_step'] != target for row in rows.values()):
        raise RuntimeError(f'{run}: eight ranks must complete exactly {target} updates')
    candidates = list(run.rglob(f'checkpoint_step_{target}.safetensors'))
    if len(candidates) != 1:
        raise RuntimeError('Expected exactly one final checkpoint')
    with (run / 'launch.log').open('rb') as stream:
        stream.seek(max(0, (run / 'launch.log').stat().st_size - 100000))
        tail = stream.read().decode(errors='replace')
    if f'[checkpoint] Saved final: {candidates[0]}' not in tail:
        raise RuntimeError('Final checkpoint completion was not logged')
    from safetensors import safe_open
    with safe_open(candidates[0], framework='pt', device='cpu') as weights:
        if not weights.keys():
            raise RuntimeError('Empty checkpoint')
    return candidates[0]


def verify_warmup(official, warmed):
    """Gate stage two on exact equality of every original policy tensor."""
    import torch
    from safetensors import safe_open
    torch.set_num_threads(1)
    with safe_open(official, framework='pt', device='cpu') as before, safe_open(
        warmed, framework='pt', device='cpu'
    ) as after:
        old, new = set(before.keys()), set(after.keys())
        if any(k.startswith('sidecar.') for k in old) or not new - old or not old <= new:
            raise RuntimeError('Expected official policy plus newly initialized Sidecar')
        if any(not k.startswith('sidecar.') for k in new - old):
            raise RuntimeError('Unexpected new policy tensors')
        for key in sorted(old):
            left, right = before.get_tensor(key), after.get_tensor(key)
            if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
                raise RuntimeError(f'Warmup changed official tensor: {key}')
    return dict(unchanged_policy_tensors=len(old), new_sidecar_tensors=len(new - old))


def command(stage, source, output):
    return [str(ROOT / '.venv/bin/python'), '-m', 'torch.distributed.run',
            '--nnodes=1', '--nproc_per_node=8', '--master_addr=127.0.0.1', '--master_port=29631',
            'scripts/cluster/train_sidecar_metrics.py', f'--config-name={PROFILES[stage]}',
            f'training.finetune_ckpt_path={source}', f'training.output_path={output}']


def resolve(stage, source, output):
    import yaml
    args = command(stage, source, output)[8:]
    text = subprocess.check_output([str(ROOT / '.venv/bin/python'), 'scripts/train.py',
                                    *args, '--cfg', 'job', '--resolve'], cwd=ROOT, text=True)
    cfg = yaml.safe_load(text)
    t, r = cfg['training'], cfg['model']['architecture']['relational_sidecar']
    assert t['batch_size'] == 32 and t['gradient_accumulation_steps'] == 1
    assert t['resume_ckpt_path'] is None and t['sidecar_lr_screen'] is None
    assert r['stage'] == 'joint' and r['isolation_warmup_updates'] == 0
    assert r['backbone_warmup_updates'] == (2000 if stage == 0 else 0)
    assert t['num_epochs'] == (None if stage == 0 else 5)
    assert t['max_steps'] == (2000 if stage == 0 else None)
    assert t['warmup_ratio'] == 0.05 and not t['save_full_states_for_resume']
    assert (t['learning_rate'], t['action_lr'], t['video_lr'], t['sidecar_lr']) == (
        (1e-4, 1e-4, 1e-4, 5e-5) if stage == 0 else (2.5e-5, 2.5e-5, 2.5e-5, 1.25e-5))
    return text


def preflight():
    import numpy as np
    from safetensors import safe_open
    manifest = json.loads(Path(os.environ['EA_SIDECAR_MANIFEST']).read_text())
    assert len(manifest['tasks']) == 50 and sum(len(t['episodes']) for t in manifest['tasks']) == 2500
    stats_path = Path(manifest['normalization_stats_path'])
    assert hashlib.sha256(stats_path.read_bytes()).hexdigest() == manifest['normalization_stats_sha256']
    data = np.load(stats_path, allow_pickle=True).item()['eef']
    official = np.load(OFFICIAL / 'normalization_stats.npy', allow_pickle=True).item()['eef']
    # The files include different unused metadata/moments; min-max is the actual transform.
    assert all(np.array_equal(data[k], official[k]) for k in ('min', 'max'))
    assert shutil.disk_usage(ROOT).free >= 220 * 1024**3, 'Need 220GiB for the independent checkpoints'
    inv = json.loads((ROOT / 'transfers/bitdistill_20261028T130623Z/openwam_clean2random.tar.inventory.json').read_text())
    item = next(f for f in inv['files'] if f['path'].endswith('/checkpoint_step_10740.safetensors'))
    assert WEIGHTS.stat().st_size == item['size'] and WEIGHTS.stat().st_mtime_ns == item['mtime_ns']
    with safe_open(WEIGHTS, framework='pt', device='cpu') as weights:
        assert not any(k.startswith('sidecar.') for k in weights.keys())
    return dict(official_checkpoint=str(WEIGHTS), sha256_from_verified_archive=item['sha256'],
                tasks=50, episodes=2500, official_min_max_exact=True)


def idle():
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,memory.free',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    parsed = [line.split(',') for line in rows]
    if len(parsed) != 8 or any('H100' not in r[0] for r in parsed):
        raise RuntimeError('This experiment requires eight H100 GPUs')
    return not apps.strip() and all(int(r[1]) <= 128 and int(r[2]) >= 78000 for r in parsed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('queue', type=Path)
    parser.add_argument('--predecessor-state', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    queue = args.queue.resolve()
    if not queue.is_relative_to(ROOT / 'outputs/sidecar'):
        raise ValueError('Keep all experiment files inside project outputs/sidecar')
    os.environ.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', HDF5_USE_FILE_LOCKING='FALSE',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
                      PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', PYTHONNOUSERSITE='1',
                      OMP_NUM_THREADS='4', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    os.environ.pop('EA_STOP_AFTER_OPT_STEPS', None)
    os.environ.pop('PYTHONHOME', None)
    os.environ.pop('PYTHONPATH', None)
    configs = [resolve(0, WEIGHTS, queue / 'stage1/runs'),
               resolve(1, queue / 'stage1/PENDING/checkpoint_step_2000.safetensors', queue / 'stage2/runs')]
    if args.check_only:
        print(json.dumps(dict(status='config_checks_passed', profiles=PROFILES)))
        return
    if socket.gethostname() != 'aigc24':
        raise ValueError('Only the user-assigned aigc24 GPU0-7 is authorized')
    # NFS locks are unavailable; this node-local socket excludes duplicate queue controllers.
    lock = socket.socket()
    lock.bind(('127.0.0.1', 29630))
    lock.listen(1)
    queue.mkdir(parents=True, exist_ok=False)
    state = dict(status='preflight', pid=os.getpid(), queue=str(queue))

    def status(phase, **extra):
        state.update(status=phase, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra)
        save(queue / 'status.json', state)
        print(json.dumps(state), flush=True)

    child = None
    try:
        proof = preflight()
        prior = json.loads(args.predecessor_state.read_text())
        assert prior['node'] == 'aigc24' and prior['world_size'] == 8 and prior['num_epochs'] == 10
        assert Path(prior['run']).resolve().is_relative_to(ROOT / 'outputs/sidecar')
        token = identity(prior['pid'])
        save(queue / 'plan.json', dict(predecessor=prior, predecessor_identity=token, preflight=proof,
                                      profiles=PROFILES, stage1_updates=2000, stage2_updates=10740,
                                      world_size=8, global_batch=256, sidecar_write='action_hidden_only'))
        for i, text in enumerate(configs, 1):
            (queue / f'stage{i}_planned_config.yaml').write_text(text)
        while token is not None and identity(prior['pid']) == token:
            status('waiting_for_existing_training', predecessor_pid=prior['pid'])
            time.sleep(30)
        predecessor = completed(Path(prior['run']), 21480)
        save(queue / 'predecessor_completed.json', dict(checkpoint=str(predecessor)))
        for stage, target in enumerate((2000, 10740)):
            while not idle():
                status('waiting_for_eight_idle_gpus', next_stage=stage + 1)
                time.sleep(30)
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 29631))
            source = WEIGHTS if stage == 0 else warmed
            arm = queue / f'stage{stage + 1}'
            arm.mkdir()
            (arm / 'resolved_config.yaml').write_text(resolve(stage, source, arm / 'runs'))
            with (arm / 'launch.log').open('xb') as log:
                child = subprocess.Popen(command(stage, source, arm / 'runs'), cwd=ROOT,
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            save(arm / 'launch_state.json', dict(pid=child.pid, command=command(stage, source, arm / 'runs')))
            while child.poll() is None:
                rows = latest(arm)
                if stage == 0 and any(row['metrics'].get('ea_backbone_frozen_warmup') != 1 for row in rows.values()):
                    raise RuntimeError('Stage one escaped the full-backward/frozen-WAM warmup')
                status(f'stage{stage + 1}_running', child_pid=child.pid,
                       steps={k: v['optimizer_step'] for k, v in rows.items()})
                time.sleep(30)
            if child.returncode != 0:
                raise RuntimeError(f'Stage {stage + 1} exited {child.returncode}; inspect launch.log')
            warmed = completed(arm, target)
            save(arm / 'completed.json', dict(checkpoint=str(warmed), updates=target, returncode=0))
            if stage == 0:
                status('verifying_frozen_official_policy')
                save(queue / 'stage1_frozen_policy_verification.json', verify_warmup(WEIGHTS, warmed))
        status('complete', final_checkpoint=str(warmed))
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()  # torchrun forwards termination to this stage's workers.
            child.wait()
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
