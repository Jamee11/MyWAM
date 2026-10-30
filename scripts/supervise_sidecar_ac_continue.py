"""Finish E, then continue A and C from their own 3K states for 1K each."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    root = parser.parse_args().root.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    if (plan['arms'], plan['resume_micro'], plan['target_micro']) != (['A_control', 'C_wam_ramp'], 6000, 8000):
        raise ValueError('Only the authorized paired 3K->4K continuation is supported')
    if not (root / 'source_ready.json').exists() or not (root / 'engineering/acceptance.json').exists():
        raise RuntimeError('Require pinned source inventory and engineering checks')
    predecessor = Path(plan['predecessor_root'])
    children, completed = {}, []
    children['A_control'] = start(root, 'A_control', plan)
    authorized = set()
    try:
        for index, name in enumerate(plan['arms']):
            arm, child = root / name, children[name]
            while child.poll() is None:
                if index == 0 and name not in authorized:
                    prior = json.loads((predecessor / 'fifth_status.json').read_text())
                    if prior['status'] == 'failed':
                        raise RuntimeError('E failed; do not take its allocation')
                    done = predecessor / 'E_gain2/completed.json'
                    if done.exists() and (arm / 'handover/cpu_ready.json').exists():
                        marker = json.loads(done.read_text())
                        if (marker['returncode'], marker['global_step'], marker['opt_step']) != (0, 6000, 3000):
                            raise RuntimeError('Require a normally completed E with its final full state')
                        save(arm / 'gpu_authorized.json', dict(utc=datetime.now(timezone.utc).isoformat()))
                        authorized.add(name)
                row = latest(arm)
                if row and row['micro_step'] >= 7400 and index == 0 and 'C_wam_ramp' not in children:
                    children['C_wam_ramp'] = start(root, 'C_wam_ramp', plan)
                if row and row['micro_step'] >= 6002 and not (arm / 'startup_acceptance.json').exists():
                    save(arm / 'startup_acceptance.json', dict(status='first_real_continued_update', latest=row))
                if (arm / 'screen_complete.json').exists():
                    if index == 1 or (root / 'C_wam_ramp/handover/cpu_ready.json').exists():
                        save(arm / 'release_exit.json', dict(utc=datetime.now(timezone.utc).isoformat()))
                save(root / 'job_status.json', dict(status='training' if row else 'preloading_or_waiting_E',
                     active_arm=name, latest=row, completed_arms=completed, utc=datetime.now(timezone.utc).isoformat()))
                time.sleep(5)
            if child.returncode != 0 or not (arm / 'screen_complete.json').exists():
                raise RuntimeError(f'{name} continuation failed: exit{child.returncode}')
            run = Path(json.loads((arm / 'output_run.json').read_text())['path'])
            metadata = json.loads((run / 'accel_state_step_8000/trainer_state.json').read_text())
            if (metadata['global_step'], metadata['opt_step']) != (8000, 4000):
                raise RuntimeError('Continuation did not preserve its 1000-update budget')
            # Their initial optimizers intentionally differ: each resumes its own 3K trajectory.
            save(arm / 'completed.json', dict(status='complete', returncode=0, output_run=str(run), **metadata))
            completed.append(name)
            if index == 0:
                save(root / 'C_wam_ramp/gpu_authorized.json', dict(utc=datetime.now(timezone.utc).isoformat()))
        save(root / 'job_status.json', dict(status='complete_training_screen', completed_arms=completed,
                                           utc=datetime.now(timezone.utc).isoformat()))
    except BaseException as error:
        for child in children.values():
            if child.poll() is None:
                os.kill(child.pid, signal.SIGTERM)
        save(root / 'job_status.json', dict(status='failed', error=repr(error), completed_arms=completed))
        raise


if __name__ == '__main__':
    main()
