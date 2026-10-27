"""Preload a fresh PatchWAM/ramp100 run and take GPU0-3 after short C exits."""
import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.supervise_sidecar_c_full import completed_source, pin_intermediate, verify_state
from scripts.supervise_sidecar_four_arm import latest, save, start


def verify_cpu_contract(arm, foundation):
    contract = json.loads((arm / 'handover/cpu_contract.json').read_text())
    expected = dict(dataset_windows=549775, world_size=4, batch_per_gpu=32, accumulation=2,
                    backbone_warmup_updates=2000, aux_shared_gradient_scale=1.0,
                    aux_gradient_anneal_updates=0, completed_optimizer_updates=0,
                    wam_ramp_updates=100, finetune_ckpt_path=foundation, resume_ckpt_path=None,
                    patchwam_augmentation_active=True)
    if contract != expected:
        raise ValueError('Fresh Foundation/zero progress/global256/warmup2K/ramp100 contract failed')
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    root = parser.parse_args().root.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    if (plan['arms'], plan['entrypoint'], plan['target_micro'], plan['fresh_from_foundation']) != (
            ['F_patchwam_ramp100'], 'scripts/train_relational_sidecar_formal.py', 42960, True):
        raise ValueError('Only the user-authorized fresh ten-epoch PatchWAM/ramp100 run is supported')
    if json.loads((root / 'engineering/acceptance.json').read_text())['status'] != 'pass':
        raise ValueError('Require engineering checks before preloading')
    if (root / 'predecessor_verified.json').exists():
        raise ValueError('Use a fresh queue directory with no stale GPU authorization')
    arm, prior = root / 'F_patchwam_ramp100', Path(plan['predecessor_root']) / 'C_wam_ramp'
    child, released = start(root, arm.name, plan), False
    try:
        while child.poll() is None:
            if json.loads((prior.parent / 'job_status.json').read_text())['status'] == 'failed':
                raise RuntimeError('C short queue failed; keep evidence and do not take its GPUs')
            if (arm / 'handover/cpu_ready.json').exists() and not (arm / 'cpu_acceptance.json').exists():
                save(arm / 'cpu_acceptance.json', verify_cpu_contract(arm, plan['foundation']))
            if not released and (prior / 'completed.json').exists():
                run, inventory = completed_source(prior, plan['required_state_files'])
                save(root / 'predecessor_verified.json', dict(status='complete', run=str(run),
                     micro=8000, optimizer_updates=4000, returncode=0, state_inventory=inventory,
                     note='C state is preserved for diagnostics; the new job does not load it'))
                released = True
            runs = list((arm / 'runs').glob('*/config.yaml'))
            if len(runs) > 1:
                raise RuntimeError('Refuse ambiguous output run discovery')
            if runs and not (arm / 'output_run.json').exists():
                save(arm / 'output_run.json', dict(path=str(runs[0].parent)))
            row = latest(arm)
            if row and row['optimizer_step'] >= 1 and not (arm / 'startup_acceptance.json').exists():
                if row['optimizer_step'] > 10 or row['metrics']['ea_backbone_frozen_warmup'] != 1:
                    raise RuntimeError('The new experiment did not start fresh in delayed-WAM warmup')
                save(arm / 'startup_acceptance.json', dict(status='first_real_fresh_updates', latest=row))
            if runs and not (root / 'snapshots/checkpoint_step_32000').exists():
                run = runs[0].parent
                if (run / 'accel_state_step_32000/trainer_state.json').exists():
                    verify_state(run, 32000, 7, plan['required_state_files'])
                    pin_intermediate(run, root / 'snapshots/checkpoint_step_32000', 32000)
            state = 'training' if row else ('cpu_ready_waiting_C_exit' if (arm / 'cpu_acceptance.json').exists()
                                           else 'preloading_foundation_on_cpu')
            save(root / 'job_status.json', dict(status=state, latest=row, short_C_latest=latest(prior),
                 gpus=[0, 1, 2, 3], predecessor_verified=released, fresh_from_foundation=True,
                 wam_ramp_updates=100, target_optimizer_updates=21480, save_every_optimizer_updates=2000,
                 C_diagnostics='preserved_parallel_queue', utc=datetime.now(timezone.utc).isoformat()))
            time.sleep(5)
        row = latest(arm)
        run = Path(json.loads((arm / 'output_run.json').read_text())['path'])
        if child.returncode != 0 or not row or (row['micro_step'], row['optimizer_step']) != (42960, 21480):
            raise RuntimeError(f'Fresh ten-epoch training did not complete normally: exit{child.returncode}')
        if not (run / 'checkpoint_step_42960.safetensors').exists():
            raise RuntimeError('Final checkpoint missing')
        save(arm / 'completed.json', dict(status='complete', returncode=0, output_run=str(run),
             global_step=42960, opt_step=21480, fresh_from_foundation=True))
        save(root / 'job_status.json', dict(status='complete', optimizer_updates=21480,
             utc=datetime.now(timezone.utc).isoformat()))
    except BaseException as error:
        if child.poll() is None:
            os.kill(child.pid, signal.SIGTERM)
        save(root / 'job_status.json', dict(status='failed', error=repr(error),
             utc=datetime.now(timezone.utc).isoformat()))
        raise


if __name__ == '__main__':
    main()
