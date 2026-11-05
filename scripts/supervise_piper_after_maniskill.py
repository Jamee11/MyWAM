#!/usr/bin/env python3
"""Preload Piper on CPU, then release it only after ManiSkill's final save and exit."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.supervise_maniskill_after_robotwin import completion_errors, same_process, save, tail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    out = args.spec.resolve().parent
    predecessor = spec['predecessor']
    training_label = spec.get('training_label', 'piper')
    predecessor_label = spec.get('predecessor_label', 'maniskill')
    threshold = int(spec.get('preload_from_optimizer_update', 0))
    while threshold and same_process(predecessor['pid'], predecessor['start_ticks']):
        lines = tail(Path(spec['completion']['predecessor_run'])/'metrics_rank0.jsonl').splitlines()
        if lines and json.loads(lines[-1])['optimizer_step'] >= threshold:
            break
        save(out/'status.json',dict(status='waiting_to_preload',supervisor_pid=os.getpid(),
             preload_from_optimizer_update=threshold,utc=datetime.now(timezone.utc).isoformat()))
        time.sleep(30)
    if threshold and not same_process(predecessor['pid'], predecessor['start_ticks']):
        errors = completion_errors(spec['completion'])
        if errors:
            raise RuntimeError(f'Predecessor final-save verification failed: {errors}')
    env = dict(os.environ, PYTHON_BIN=spec['python'], CUDA_VISIBLE_DEVICES='0,1,2,3',
               EA_HANDOVER_DIR=str(out/'handover'),
               EA_PREDECESSOR_DONE_FILE=str(out/'predecessor_complete.json'),
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env[spec.get('output_env', 'PIPER_ARCHIVE_OUTPUT')] = str(out/'runs')
    with (out/'train.log').open('x') as log:
        child = subprocess.Popen(['bash', spec.get('launch_script', 'scripts/train_piper_archive_patchwam_4gpu.sh')]
                                 + spec.get('launch_args', []),
            cwd=spec['root'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    stat = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()
    save(out/'launch.json', dict(pid=child.pid, start_ticks=stat[19], output=str(out/'runs')))

    def report(status, **kwargs):
        save(out/'status.json', dict(status=status, supervisor_pid=os.getpid(), training_pid=child.pid,
                                    utc=datetime.now(timezone.utc).isoformat(), **kwargs))

    try:
        report('cpu_preloading_waiting_for_'+predecessor_label)
        while same_process(predecessor['pid'], predecessor['start_ticks']):
            if child.poll() is not None:
                raise RuntimeError(f'{training_label} CPU preload failed; inspect train.log')
            time.sleep(5)
        errors = completion_errors(spec['completion'])
        if errors:
            raise RuntimeError(f'Predecessor final-save verification failed: {errors}')
        save(out/'predecessor_complete.json', dict(status='complete', utc=datetime.now(timezone.utc).isoformat(),
                                                  predecessor=predecessor, completion=spec['completion']))
        report(predecessor_label+'_complete_waiting_for_gpu_handover')
        while child.poll() is None and not (out/'handover/gpu_release.json').exists():
            time.sleep(5)
        if child.poll() is not None:
            raise RuntimeError(f'{training_label} exited before GPU handover')
        report(training_label+'_training')
        code = child.wait()
        report('complete' if code == 0 else 'failed', returncode=code)
        if code:
            raise RuntimeError(f'{training_label} exited with code {code}')
    except BaseException as exc:
        report('failed', error=repr(exc))
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait()
        raise


if __name__ == '__main__':
    main()
