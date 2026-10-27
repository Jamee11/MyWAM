"""Sequential four-arm screen, overlapping CPU preloading and matching diagnostics."""
import argparse
import json
import os
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def latest(arm):
    marker = arm / 'output_run.json'
    if not marker.exists():
        return None
    file = Path(json.loads(marker.read_text())['path']) / 'metrics_rank0.jsonl'
    if not file.exists():
        return None
    with file.open('rb') as stream:
        stream.seek(max(0, file.stat().st_size - 8000))
        for line in reversed(stream.read().decode(errors='replace').splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass


def stop_predecessor(record, expected_root):
    identity = json.loads(record.read_text())
    proc = Path(f'/proc/{identity["pid"]}')
    if not proc.exists():
        return
    if proc.joinpath('stat').read_text().split()[21] != identity['start_ticks']:
        raise RuntimeError('Predecessor PID was reused')
    command = proc.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode()
    if 'torch.distributed.run' not in command or str(expected_root) not in command:
        raise RuntimeError('Predecessor does not match its unique training output')
    os.kill(identity['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 90
    while proc.exists() and time.monotonic() < deadline:
        time.sleep(1)
    if proc.exists():
        raise RuntimeError('Verified predecessor did not exit after SIGTERM; no broad process kill')


def start(root, name, plan):
    arm = root / name
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', SIDECAR_SCREEN_ARM=str(arm),
               EA_HANDOVER_DIR=str(arm / 'handover'), HF_ENDPOINT='https://hf-mirror.com', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled', TORCH_WARM_POOL='0', PYTHONNOUSERSITE='1',
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', NCCL_DEBUG='WARN', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               EA_SIDECAR_MANIFEST=str(ROOT / 'outputs/sidecar/aloha_4350/manifest.ready.json'))
    for key in ('PYTHONPATH', 'PYTHONHOME', 'EA_PREDECESSOR_DONE_FILE', 'EA_STOP_AFTER_OPT_STEPS'):
        env.pop(key, None)
    command = [plan['python'], '-m', 'torch.distributed.run', '--nproc_per_node=4', f'--master_port={port}',
               'scripts/train_sidecar_four_arm.py', f'--config-path={arm}', '--config-name=config']
    with (arm / 'launch.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(arm / 'launch.json', dict(pid=child.pid, start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],
                                 utc=datetime.now(timezone.utc).isoformat(), command=command, gpus=[0, 1, 2, 3]))
    return child


class AdoptedControl:
    """Keep Control training while replacing its paused supervisor, retaining exit status."""
    def __init__(self, arm):
        self.identity = json.loads((arm / 'launch.json').read_text())
        self.pid, self.returncode = self.identity['pid'], None
        command = Path(f'/proc/{self.pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        if 'torch.distributed.run' not in command or str(arm) not in command:
            raise RuntimeError('Adopt only the verified active Control launcher')
        self.poll()

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        fields = Path(f'/proc/{self.pid}/stat').read_text().split()
        if fields[21] != self.identity['start_ticks']:
            raise RuntimeError('Control PID was reused')
        if fields[2] == 'Z':
            # The old parent is paused until this kernel exit status has been read.
            self.returncode = os.waitstatus_to_exitcode(int(fields[51]))
        return self.returncode


def retire_paused_supervisor(root):
    record = root / 'retired_supervisor_process.json'
    identity = json.loads(record.read_text())
    proc = Path(f'/proc/{identity["pid"]}')
    if not proc.exists():
        return
    fields = proc.joinpath('stat').read_text().split()
    command = proc.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode()
    if fields[21] != identity['start_ticks'] or str(root) not in command or fields[2] not in ('T', 't'):
        raise RuntimeError('Refuse to retire a supervisor without its verified paused identity')
    os.kill(identity['pid'], signal.SIGTERM)
    os.kill(identity['pid'], signal.SIGCONT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--adopt-control', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    if not (root / 'engineering/acceptance.json').exists() or not (root / 'copy_ready.json').exists():
        raise RuntimeError('Require engineering acceptance and independent source copy before launch')
    children, completed = {}, []
    names = plan['arms']
    children[names[0]] = AdoptedControl(root / names[0]) if args.adopt_control else start(root, names[0], plan)
    handed_over = args.adopt_control
    try:
        for index, name in enumerate(names):
            arm, child = root / name, children[name]
            while child.poll() is None:
                row = latest(arm)
                cpu_ready = (arm / 'handover/cpu_ready.json').exists()
                if index == 0 and cpu_ready and not handed_over:
                    stop_predecessor(Path(plan['old_launch']), Path(plan['old_launch']).parent)
                    save(arm / 'gpu_authorized.json', dict(utc=datetime.now(timezone.utc).isoformat()))
                    handed_over = True
                    save(root / 'old_training_replacement.json', dict(status='user_requested_switch', source_full_state_preserved=True))
                if row and row['micro_step'] >= 5400 and index + 1 < len(names) and names[index + 1] not in children:
                    children[names[index + 1]] = start(root, names[index + 1], plan)
                if row and row['micro_step'] >= 4002 and not (arm / 'startup_acceptance.json').exists():
                    save(arm / 'startup_acceptance.json', dict(status='first_real_resumed_update', latest=row,
                         utc=datetime.now(timezone.utc).isoformat()))
                    if index == 0:
                        old_root = Path(plan['old_launch']).parent
                        old_status = json.loads((old_root / 'job_status.json').read_text())
                        save(old_root / 'job_status_before_four_arm_switch.json', old_status)
                        old_status.update(status='superseded_by_four_arm_screen', replacement_root=str(root),
                                          reason='User authorized switch after new model CPU preloading',
                                          source_4000_state_preserved=str(Path(plan['common']) / 'accel_state_step_4000'))
                        save(old_root / 'job_status.json', old_status)
                if (arm / 'screen_complete.json').exists():
                    successor_ready = index + 1 == len(names) or (root / names[index + 1] / 'handover/cpu_ready.json').exists()
                    if successor_ready:
                        save(arm / 'release_exit.json', dict(utc=datetime.now(timezone.utc).isoformat()))
                save(root / 'job_status.json', dict(utc=datetime.now(timezone.utc).isoformat(), active_arm=name,
                     status='training' if row else ('cpu_ready' if cpu_ready else 'preloading'), latest=row,
                     completed_arms=completed, preloaded_arms=[n for n in children if n not in completed and n != name]))
                time.sleep(5)
            if child.returncode != 0 or not (arm / 'screen_complete.json').exists():
                raise RuntimeError(f'{name} failed: exit{child.returncode}; inspect its launch.log')
            run = Path(json.loads((arm / 'output_run.json').read_text())['path'])
            metadata = json.loads((run / 'accel_state_step_6000/trainer_state.json').read_text())
            if (metadata['global_step'], metadata['opt_step']) != (6000, 3000):
                raise RuntimeError('Short-run final counters are incomplete')
            for rank in range(4):
                if name != names[0] and (arm / f'initial_optimizer_rank{rank}.json').read_bytes() != (
                        root / names[0] / f'initial_optimizer_rank{rank}.json').read_bytes():
                    raise RuntimeError('Initial restored optimizer differs from Control')
            completed.append(name)
            save(arm / 'completed.json', dict(status='complete', returncode=0, output_run=str(run), **metadata))
            if index == 0 and args.adopt_control:
                retire_paused_supervisor(root)
            if index + 1 < len(names):
                save(root / names[index + 1] / 'gpu_authorized.json', dict(utc=datetime.now(timezone.utc).isoformat()))
        save(root / 'job_status.json', dict(status='complete_training_screen', completed_arms=completed,
             utc=datetime.now(timezone.utc).isoformat(), diagnostics_status='see diagnostic supervisor'))
    except BaseException as error:
        for child in children.values():
            if child.poll() is None:
                os.kill(child.pid, signal.SIGTERM)
        save(root / 'job_status.json', dict(status='failed', error=repr(error), completed_arms=completed,
                                           utc=datetime.now(timezone.utc).isoformat()))
        if args.adopt_control:
            retire_paused_supervisor(root)
        raise


if __name__ == '__main__':
    main()
