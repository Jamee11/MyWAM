"""Preload C during its 4K screen, then resume the original ten-epoch budget."""
import argparse
import json
import os
import shutil
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.supervise_sidecar_four_arm import latest, save, start


def verify_state(run, micro, epoch, required_files):
    state = run / f'accel_state_step_{micro}'
    expected = dict(global_step=micro, opt_step=micro // 2, epoch=epoch)
    if json.loads((state / 'trainer_state.json').read_text()) != expected:
        raise ValueError('Completed checkpoint counters differ from the authorized continuation')
    for relative in required_files:
        path = state / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Incomplete four-rank state: {relative}')
    if not (run / f'checkpoint_step_{micro}.safetensors').is_file():
        raise ValueError('Completed state must retain its matching weights')
    return {str(p.relative_to(state)): dict(size=p.stat().st_size, inode=p.stat().st_ino,
            device=p.stat().st_dev, mtime_ns=p.stat().st_mtime_ns) for p in sorted(state.rglob('*')) if p.is_file()}


def completed_source(arm, required_files):
    marker = json.loads((arm / 'completed.json').read_text())
    if (marker['returncode'], marker['global_step'], marker['opt_step']) != (0, 8000, 4000):
        raise ValueError('Wait for the short C launcher to exit normally at optimizer4000')
    run = Path(json.loads((arm / 'output_run.json').read_text())['path'])
    if Path(marker['output_run']) != run:
        raise ValueError('C completion and output-run identities differ')
    return run, verify_state(run, 8000, 1, required_files)


def pin_intermediate(run, destination, micro):
    """Protect the previously informative 32K weights without copying optimizer shards."""
    destination.mkdir()
    weight = run / f'checkpoint_step_{micro}.safetensors'
    os.link(weight, destination / weight.name)
    for filename in ('config.yaml', 'normalization_stats.npy', 'relational_sidecar_contract.json'):
        shutil.copy2(run / filename, destination / filename)
    shutil.copytree(run / 'tokenizer', destination / 'tokenizer')
    save(destination / 'snapshot.json', dict(micro=micro, updates=micro // 2, source=str(weight),
         size=weight.stat().st_size, inode=weight.stat().st_ino, method='hardlink_weights_only'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    root = parser.parse_args().root.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    if not plan.get('full_training') or (plan['resume_micro'], plan['target_micro'], plan['arms']) != (
            8000, 42960, ['C_wam_ramp']):
        raise ValueError('Only the user-authorized C 4K-to-ten-epoch continuation is supported')
    acceptance = root / 'engineering/acceptance.json'
    if not acceptance.is_file() or json.loads(acceptance.read_text()).get('status') != 'pass':
        raise ValueError('Engineering checks must pass before queuing full training')
    prior = Path(plan['predecessor_root']) / 'C_wam_ramp'
    arm, child, authorized = root / 'C_wam_ramp', None, False
    try:
        while True:
            prior_status = json.loads((prior.parent / 'job_status.json').read_text())
            if prior_status['status'] == 'failed':
                raise RuntimeError('A/C short queue failed; do not take its allocation')
            row = latest(prior)
            if child is None and row and row['micro_step'] >= 7400:
                source_run = Path(json.loads((prior / 'output_run.json').read_text())['path'])
                plan['sources']['C_wam_ramp']['full_state'] = str(source_run / 'accel_state_step_8000')
                plan['sources']['C_wam_ramp']['run'] = str(source_run)
                save(root / 'plan.json', plan)
                child = start(root, 'C_wam_ramp', plan)
            if child is not None and child.poll() is not None:
                break
            if child is not None and not authorized and (prior / 'completed.json').exists() and (
                    arm / 'handover/cpu_ready.json').exists():
                source_run, inventory = completed_source(prior, plan['required_state_files'])
                if str(source_run / 'accel_state_step_8000') != plan['sources']['C_wam_ramp']['full_state']:
                    raise ValueError('Preloaded source changed before GPU handover')
                save(root / 'source_ready.json', dict(run=str(source_run), inventory=inventory,
                     metadata=dict(global_step=8000, opt_step=4000, epoch=1)))
                save(arm / 'gpu_authorized.json', dict(utc=datetime.now(timezone.utc).isoformat(),
                     source='short_C_exit0_and_complete_4k_state'))
                authorized = True
            live = latest(arm) if child is not None else None
            if live and live['micro_step'] >= 8002 and not (arm / 'startup_acceptance.json').exists():
                if not (arm / 'resume_verified.json').exists():
                    raise RuntimeError('First full-training update lacks verified 4K resume')
                save(arm / 'startup_acceptance.json', dict(status='first_real_update_after_4k', latest=live))
            output = arm / 'output_run.json'
            snapshot = root / 'snapshots/checkpoint_step_32000'
            if output.exists() and not snapshot.exists():
                run = Path(json.loads(output.read_text())['path'])
                if (run / 'accel_state_step_32000/trainer_state.json').exists():
                    verify_state(run, 32000, 7, plan['required_state_files'])
                    pin_intermediate(run, snapshot, 32000)
            if (arm / 'screen_complete.json').exists():
                save(arm / 'release_exit.json', dict(utc=datetime.now(timezone.utc).isoformat()))
            status = 'training' if live else ('preloading_or_waiting_4k_exit' if child else 'waiting_for_C_preload_threshold')
            save(root / 'job_status.json', dict(status=status, latest=live, short_C_latest=row,
                 gpus=[0, 1, 2, 3], gpu_authorized=authorized, save_every_optimizer_updates=2000,
                 target_optimizer_updates=21480, utc=datetime.now(timezone.utc).isoformat()))
            time.sleep(5)
        if child.returncode != 0 or not (arm / 'screen_complete.json').exists():
            raise RuntimeError(f'Full C failed: exit{child.returncode}')
        run = Path(json.loads((arm / 'output_run.json').read_text())['path'])
        verify_state(run, 42960, 9, plan['required_state_files'])
        save(arm / 'completed.json', dict(status='complete', returncode=0, output_run=str(run),
             global_step=42960, opt_step=21480, epoch=9))
        save(root / 'job_status.json', dict(status='complete', optimizer_updates=21480,
             utc=datetime.now(timezone.utc).isoformat()))
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.kill(child.pid, signal.SIGTERM)
        save(root / 'job_status.json', dict(status='failed', error=repr(error),
             utc=datetime.now(timezone.utc).isoformat()))
        raise


if __name__ == '__main__':
    main()
