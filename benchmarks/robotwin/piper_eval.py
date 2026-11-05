"""Independent manifest-derived 35-task Piper native evaluation queue."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import signal
import time
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.robotwin.piper_protocol import (DEFAULT_CHECKPOINT, DEFAULT_MANIFEST, DEFAULT_ROBOTWIN,
                                               MODES, checkpoint_contract, piper_tasks)
from benchmarks.robotwin.standard_eval import validate_result

RENDER_ERROR_MARKERS = ('OIDN Error', 'ErrorDeviceLost', 'VK_ERROR_DEVICE_LOST',
                        'illegal memory access', 'cannot create buffer', 'failed to create buffer')


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def prepare_plan(args):
    supported = piper_tasks(args.manifest)
    selected = supported if args.tasks == 'all' else list(dict.fromkeys(args.tasks.split(',')))
    unknown = set(selected) - set(supported)
    if unknown:
        raise ValueError(f'Tasks absent from Piper train group: {sorted(unknown)}')
    checkpoint_contract(args.checkpoint)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    return dict(checkpoint=str(args.checkpoint.resolve()), manifest=str(args.manifest.resolve()),
                manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                sim_environment={key: os.environ.get(key) for key in ('CUDA_HOME', 'PYTHONPATH',
                    '__EGL_VENDOR_LIBRARY_FILENAMES', 'TORCH_CUDA_ARCH_LIST', 'TORCH_EXTENSIONS_DIR',
                    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'SAPIEN_DISABLE_RAY_TRACING')},
                embodiment=['piper', 'piper', 0.6], episodes_per_condition=args.episodes,
                seed_group=args.seed_group, supported_tasks=supported, selected_tasks=selected,
                conditions=[dict(task=task, mode=mode, status='queued', attempts=[]) for task in selected for mode in args.modes],
                protocol='Native expert-validated sequential seeds; official RT/success/action budgets; complete Sidecar; no video; EEF20 wxyz-to-Rot6D,80D slots0:10/34:44')


def summarize(state):
    complete = [c for c in state['conditions'] if c['status'] == 'complete']
    summary = {}
    for mode in ['all'] + list(MODES):
        rows = [c for c in complete if mode == 'all' or c['mode'] == mode]
        total = state['episodes_per_condition'] * len(rows)
        good = sum(c['result']['successes'] for c in rows)
        summary[mode] = dict(conditions=len(rows), episodes=total, successes=good,
                             success_rate=good/total if total else None)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    p.add_argument('--robotwin', type=Path, default=DEFAULT_ROBOTWIN)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', default='all')
    p.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--seed-group', type=int, default=0)
    p.add_argument('--sim-python', type=Path, default=ROOT / '.conda/robotwin-official/bin/python')
    p.add_argument('--sim-gpus', default='0', help='One physical GPU index per worker; repetition is allowed')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--ports', default='43400', help='One independent Piper model endpoint per simulator worker')
    p.add_argument('--attempts', type=int, default=3)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--min-free-mib', type=int, default=0)
    p.add_argument('--resume-episodes', action='store_true')
    args = p.parse_args()
    if args.episodes < 1 or args.attempts < 1:
        p.error('episode/attempt counts must be positive')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = prepare_plan(args)
    state['status'] = 'prepared' if args.prepare_only else 'running'
    write(out / 'plan.json', state)
    print(f'Piper tasks={len(state["selected_tasks"])}/{len(state["supported_tasks"])}; conditions={len(state["conditions"])}; episodes={len(state["conditions"])*args.episodes}; output={out}', flush=True)
    if args.prepare_only:
        return
    gpus = args.sim_gpus.split(',')
    ports = [int(v) for v in args.ports.split(',')]
    if len(gpus) != len(ports) or len(set(ports)) != len(ports):
        raise ValueError('Give one unique model endpoint per simulator GPU slot')
    template = yaml.safe_load((ROOT / 'benchmarks/robotwin/policy_config_piper.yml').read_text())
    jobs = queue.Queue()
    for c in state['conditions']:
        jobs.put(c)
    lock = threading.Lock()
    reservations = []
    def persist():
        state['summary'] = summarize(state)
        write(out / 'state.json', state)
    def worker(index, gpu, port):
        while True:
            try:
                c = jobs.get_nowait()
            except queue.Empty:
                return
            for attempt in range(1, args.attempts + 1):
                if args.min_free_mib:
                    while True:
                        with lock:
                            now = time.monotonic()
                            reservations[:] = [(g, expiry) for g, expiry in reservations if expiry > now]
                            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
                            free = {g.strip(): int(m.strip()) for g, m in (line.split(',') for line in raw.splitlines())}
                            available = free[gpu] - args.min_free_mib * sum(g == gpu for g, _ in reservations)
                            if available >= args.min_free_mib:
                                reservations.append((gpu, now + 180))
                                break
                        time.sleep(5)
                d = out / 'jobs' / c['mode'] / c['task'] / f'attempt{attempt}'
                d.mkdir(parents=True)
                policy = d / 'policy.yml'
                policy.write_text(yaml.safe_dump(dict(template, host=args.host, port=port)))
                command = [str(args.sim_python.resolve()), '-u', str(ROOT / 'benchmarks/robotwin/run_piper_native.py'),
                           '--policy', str(policy), '--task', c['task'], '--mode', c['mode'],
                           '--runtime', str(d/'runtime'), '--robotwin', str(args.robotwin.resolve()),
                           '--episodes', str(args.episodes), '--seed-group', str(args.seed_group)]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
                if args.resume_episodes:
                    env['PIPER_EPISODE_RESUME'] = '1'
                with (d/'client.log').open('w') as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    with lock:
                        c.update(status='running', gpu=gpu, port=port)
                        record = dict(pid=child.pid, output=str(d), worker=index)
                        c['attempts'].append(record)
                        persist()
                    last_change = time.monotonic()
                    last_size = -1
                    terminated = None
                    while child.poll() is None:
                        size = (d/'client.log').stat().st_size
                        now = time.monotonic()
                        if size != last_size:
                            last_size, last_change = size, now
                        with (d/'client.log').open('rb') as stream:
                            stream.seek(max(0, size-65536))
                            tail = stream.read()
                        poisoned = tail.count(b"AttributeError: 'Robot' object has no attribute 'left_planner") + tail.count(b"AttributeError: 'Robot' object has no attribute 'right_planner") >= 3
                        bad = any(m.encode() in tail for m in RENDER_ERROR_MARKERS)
                        if (bad or poisoned or now-last_change > 1200) and terminated is None:
                            try:
                                os.killpg(child.pid, signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                            terminated = now
                        if terminated is not None and now-terminated > 15:
                            try:
                                os.killpg(child.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        time.sleep(3)
                    code = child.returncode
                text = (d/'client.log').read_text(errors='replace')
                result = validate_result(text, d/'runtime', args.episodes, code)
                if any(marker in text for marker in RENDER_ERROR_MARKERS):
                    result['complete'] = False
                    result['errors'].append('Rendering infrastructure error; rejected even exit0')
                if result['complete'] and args.resume_episodes:
                    from benchmarks.robotwin.native_episode_resume import verify_provenance
                    try:
                        verify_provenance(d)
                    except (ValueError, AssertionError) as exc:
                        result['complete'] = False
                        result['errors'].append(f'Recovery provenance: {exc}')
                write(d/'run.json', dict(returncode=code, result=result))
                with lock:
                    record.update(returncode=code)
                    c.update(status='complete' if result['complete'] else 'retrying', result=result)
                    persist()
                if result['complete']:
                    break
            with lock:
                if c['status'] != 'complete':
                    c['status'] = 'failed'
                persist()
            jobs.task_done()
    threads = [threading.Thread(target=worker, args=(i, g, port)) for i, (g, port) in enumerate(zip(gpus, ports))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    state['status'] = 'complete' if all(c['status'] == 'complete' for c in state['conditions']) else 'finished_with_errors'
    persist()
    write(out/'summary.json', dict(status=state['status'], **state['summary'], tasks=state['selected_tasks'], episodes_per_condition=args.episodes))


if __name__ == '__main__':
    main()
