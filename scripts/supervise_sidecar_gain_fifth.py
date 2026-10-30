"""Append the authorized Gain x2 arm without restarting the running four-arm queue."""
import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.supervise_sidecar_four_arm import latest, save, start


def finish(root, arm):
    run = Path(json.loads((arm / 'output_run.json').read_text())['path'])
    metadata = json.loads((run / 'accel_state_step_6000/trainer_state.json').read_text())
    if (metadata['global_step'], metadata['opt_step']) != (6000, 3000):
        raise RuntimeError('Gain arm did not save exactly 1000 additional optimizer updates')
    for rank in range(4):
        if (arm / f'initial_optimizer_rank{rank}.json').read_bytes() != (
                root / 'A_control' / f'initial_optimizer_rank{rank}.json').read_bytes():
            raise RuntimeError('Initial Gain optimizer differs from Control')
    save(arm / 'completed.json', dict(status='complete', returncode=0, output_run=str(run), **metadata))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    root = parser.parse_args().root.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    fifth = json.loads((root / 'fifth_plan.json').read_text())
    if fifth != dict(arm='E_gain2', main_task_gradient_gain=2, enabled=True):
        raise ValueError('Require the authorized single Gain x2 intervention')
    arm, predecessor = root / 'E_gain2', root / 'D_write2'
    child, authorized = None, False
    try:
        while True:
            queue = json.loads((root / 'job_status.json').read_text())
            if queue['status'] == 'failed':
                raise RuntimeError('Earlier four-arm queue failed; Gain will not take its allocation')
            row = latest(predecessor)
            if child is None and row and row['micro_step'] >= 5400:
                child = start(root, arm.name, plan)
            if child is not None:
                done = predecessor / 'completed.json'
                if not authorized and done.exists() and (arm / 'handover/cpu_ready.json').exists():
                    prior = json.loads(done.read_text())
                    if (prior['status'], prior['returncode'], prior['global_step'], prior['opt_step']) != (
                            'complete', 0, 6000, 3000):
                        raise RuntimeError('Fourth arm did not complete normally')
                    save(arm / 'gpu_authorized.json', dict(utc=datetime.now(timezone.utc).isoformat()))
                    authorized = True
                if (arm / 'screen_complete.json').exists():
                    save(arm / 'release_exit.json', dict(utc=datetime.now(timezone.utc).isoformat()))
                if child.poll() is not None:
                    if child.returncode != 0 or not (arm / 'screen_complete.json').exists():
                        raise RuntimeError(f'Gain arm failed: exit{child.returncode}; inspect launch.log')
                    finish(root, arm)
                    save(root / 'fifth_status.json', dict(status='complete', arm=arm.name,
                                                         utc=datetime.now(timezone.utc).isoformat()))
                    return
            save(root / 'fifth_status.json', dict(status='training' if authorized else (
                'preloading' if child else 'waiting_for_fourth_arm'), arm=arm.name,
                latest=latest(arm), utc=datetime.now(timezone.utc).isoformat()))
            time.sleep(5)
    except BaseException as error:
        if child and child.poll() is None:
            os.kill(child.pid, signal.SIGTERM)
        save(root / 'fifth_status.json', dict(status='failed', error=repr(error), arm=arm.name,
                                            utc=datetime.now(timezone.utc).isoformat()))
        raise


if __name__ == '__main__':
    main()
