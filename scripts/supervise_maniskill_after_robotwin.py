#!/usr/bin/env python3
"""One-shot, non-invasive handover from the pinned RoboTwin run to ManiSkill."""
import argparse
import fcntl
import json
import os
import socket
import struct
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def tail(path, size=65536):
    if not path.is_file():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - size))
        return stream.read().decode(errors='replace')


def same_process(pid, start_ticks):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == start_ticks
    except FileNotFoundError:
        return False


def completion_errors(spec):
    run, total = Path(spec['predecessor_run']), spec['total_micro']
    errors = []
    for rank in range(4):
        lines = tail(run / f'metrics_rank{rank}.jsonl').splitlines()
        try:
            row = json.loads(lines[-1])
            if row['micro_step'] != total:
                errors.append(f'rank{rank} has not reached {total}')
        except (IndexError, ValueError, KeyError):
            errors.append(f'rank{rank} final metrics missing')
    checkpoint = run / f'checkpoint_step_{total}.safetensors'
    marker = f'[checkpoint] Saved final: {checkpoint}'
    if marker not in tail(Path(spec['predecessor_log']), 2 * 1024 * 1024):
        errors.append('final checkpoint saved marker missing')
    try:
        with checkpoint.open('rb') as stream:
            header_size = struct.unpack('<Q', stream.read(8))[0]
            if not 0 < header_size < 100 * 1024 * 1024:
                raise ValueError('invalid checkpoint header size')
            header = json.loads(stream.read(header_size))
        end = max(v['data_offsets'][1] for k, v in header.items() if k != '__metadata__')
        if checkpoint.stat().st_size != 8 + header_size + end:
            errors.append('final checkpoint size/header mismatch')
    except (OSError, ValueError, KeyError, struct.error):
        errors.append('final checkpoint unavailable or incomplete')
    return errors


def available_gpus(spec):
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free',
         '--format=csv,noheader,nounits'], text=True, timeout=15)
    rows = {int(parts[0]): parts for line in output.splitlines()
            if (parts := [s.strip() for s in line.split(',')])}
    return all(g in rows and 'A100' in rows[g][1] and int(rows[g][2]) >= 75000
               and int(rows[g][3]) >= spec['min_free_mib'] for g in spec['gpus'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    root, out = Path(spec['root']), args.spec.resolve().parent
    if (os.uname().nodename != spec['hostname'] or
            Path('/proc/sys/kernel/random/boot_id').read_text().strip() != spec['boot_id']):
        raise RuntimeError('Pinned predecessor belongs to a different host/boot')
    lock = (root / 'outputs/maniskill_sidecar/.robotwin_handover.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'launched.json').exists():
        raise RuntimeError('This handover already attempted launch; inspect logs before retrying')
    status = {'supervisor_pid': os.getpid(), 'spec': str(args.spec.resolve())}

    def report(state, **kwargs):
        status.update(kwargs)
        status.update(status=state, utc=datetime.now(timezone.utc).isoformat())
        save(out / 'status.json', status)

    report('waiting_for_robotwin_completion')
    last_heartbeat = time.monotonic()
    while any(same_process(pid, ticks) for pid, ticks in spec['pids_start_ticks'].items()):
        if time.monotonic() - last_heartbeat >= 30:
            report('waiting_for_robotwin_completion')
            last_heartbeat = time.monotonic()
        time.sleep(5)
    errors = completion_errors(spec)
    if errors:
        report('predecessor_incomplete_no_launch', errors=errors)
        raise RuntimeError(f'RoboTwin did not finish normally: {errors}')
    report('waiting_for_gpu_release')
    while not available_gpus(spec):
        time.sleep(5)
    # The launcher repeats its own four-GPU capacity check immediately before torchrun.
    while True:
        try:
            with socket.socket() as check:
                check.bind(('127.0.0.1', spec['master_port']))
            break
        except OSError:
            report('waiting_for_master_port')
            time.sleep(5)
    local = Path(spec['local_venv'])
    environment = local if (local / 'bin/python').is_file() else Path(spec['fallback_venv'])
    output = root / 'outputs/maniskill_sidecar' / ('joint_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, spec['gpus'])),
               OPENWAM_VENV=str(environment), CHECK_ONLY='0', MASTER_PORT=str(spec['master_port']),
               MANISKILL_SIDECAR_MANIFEST=spec['manifest'], MANISKILL_SIDECAR_OUTPUT_DIR=str(output))
    command = ['bash', 'scripts/train_maniskill_sidecar_4gpu.sh']
    launch = dict(command=command, output=str(output), environment=str(environment),
                  utc=datetime.now(timezone.utc).isoformat(), gpus=spec['gpus'])
    save(out / 'launched.json', launch)
    with (output / 'train.log').open('x') as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        launch['pid'] = process.pid
        save(out / 'launched.json', launch)
        report('maniskill_process_started', **launch)
        code = process.wait()
    report('maniskill_exited' if code == 0 else 'maniskill_failed', returncode=code)


if __name__ == '__main__':
    main()
