"""Reuse the audited fixed-input GT/Full-Off FM diagnostic after each screen arm."""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.supervise_sidecar_four_arm import ROOT, save

REFERENCE = ROOT / 'outputs/robotwin_eval/sidecar_full_backward2k_step12000_diagnostics_20261023'
BOOTSTRAP = ROOT / 'outputs/robotwin_eval/sidecar_isolation2k_step24000_write_magnitudes_20261023/bootstrap.py'
ANALYSIS = ROOT / 'outputs/robotwin_eval/sidecar_full_backward2k_16k20k_diagnostics_20261024/analyze.py'


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def fixture_hashes():
    inputs = json.loads((REFERENCE / 'inputs_ready.json').read_text())['samples']
    fm = json.loads((REFERENCE / 'gradient32/inputs_ready.json').read_text())['samples']
    paths = {Path(row[key]) for row in inputs for key in ('payload', 'gt')}
    paths.update(Path(row[key]) for row in fm for key in ('payload', 'source'))
    return {str(p): sha256(p) for p in sorted(paths)}


def selected_gpu():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    free = {int(a): int(b) for a, b in (line.split(',') for line in raw.splitlines())}
    # Only these diagnostic cards; never evict a training/simulation/service process.
    eligible = [gpu for gpu in (5, 6) if free[gpu] >= 28000]
    return max(eligible, key=free.get) if eligible else None


def prepare(root, name, gpu):
    arm, out = root / name, root / 'diagnostics' / name
    out.mkdir(parents=True)
    config = json.loads((root / 'plan.json').read_text()) if (root / 'plan.json').exists() else {}
    target = config.get('target_micro', 6000)
    run = Path(json.loads((arm / 'completed.json').read_text())['output_run'])
    metadata = json.loads((run / f'accel_state_step_{target}/trainer_state.json').read_text())
    if (metadata['global_step'], metadata['opt_step']) != (target, target // 2):
        raise RuntimeError('Diagnostics require the exact completed endpoint')
    pinned = out / f'checkpoint_step_{target}'
    pinned.mkdir()
    weight = run / f'checkpoint_step_{target}.safetensors'
    os.link(weight, pinned / weight.name)  # Immutable endpoint; no duplicated 15GB model copy.
    for filename in ('config.yaml', 'normalization_stats.npy', 'relational_sidecar_contract.json'):
        shutil.copy2(run / filename, pinned / filename)
    shutil.copytree(run / 'tokenizer', pinned / 'tokenizer')
    shutil.copy2(REFERENCE / 'inputs_ready.json', out / 'inputs_ready.json')
    plan = json.loads((REFERENCE / 'plan.json').read_text())
    plan.update(scales=[0, 1], encoder_gpu=gpu, model_gpu=gpu, encoder_device='cuda:0', model_device='cuda:0',
                checkpoint_dir=str(pinned), checkpoint_source=str(weight), checkpoint_sha256=sha256(weight),
                optimizer_updates=target // 2, reference_run=str(REFERENCE), run_gradient32=False,
                recipe=f'Matched optimizer-update screen ending at {target // 2}: {name}')
    save(out / 'plan.json', plan)
    return out


def summarize(root):
    config = json.loads((root / 'plan.json').read_text())
    start, target = config.get('resume_micro', 4000), config.get('target_micro', 6000)
    analyzer = load_module(ANALYSIS, 'screen_diagnostic_analysis')
    reference_rows = json.loads((REFERENCE / 'results.json').read_text())['results']
    reference_fm = json.loads((REFERENCE / 'paired_action_fm.json').read_text())['rows']
    reports = {}
    for arm in sorted((root / 'diagnostics').iterdir()):
        if not arm.is_dir() or not (arm / 'complete.json').exists():
            continue
        report, _ = analyzer.collect(target, arm, reference_rows, reference_fm)
        source = root / arm.name
        run = Path(json.loads((source / 'completed.json').read_text())['output_run'])
        rows = [json.loads(line) for line in (run / 'metrics_rank0.jsonl').read_text().splitlines()]
        assert [r['micro_step'] for r in rows] == list(range(start + 1, target + 1))
        report['training'] = {}
        for low, high in ((start + 1, start + 200), (start + 201, start + 1000), (start + 1001, target)):
            subset = [r for r in rows if low <= r['micro_step'] <= high]
            report['training'][f'{low}-{high}'] = {
                key: mean(r['metrics'][key] for r in subset)
                for key in ('loss_action', 'loss_video', 'loss_ea')}
        report['actual_training_batches_equal_to_control'] = all(
            (source / f'batches_rank{rank}.jsonl').read_bytes() == (
                root / 'A_control' / f'batches_rank{rank}.jsonl').read_bytes() for rank in range(4))
        assert report['actual_training_batches_equal_to_control']
        reports[arm.name] = report
    result = dict(checkpoints=reports, optimizer_updates=target // 2,
                  scope='24 held-out overlapping windows from six episodes, 1 seen control, 32 fixed FM samples; not closed-loop SR')
    if config.get('continuation'):
        previous = json.loads((Path(config['predecessor_root']) / 'diagnostics/comparison.json').read_text())['checkpoints']
        result['reference_3k'] = {name: previous[name] for name in config['arms']}
    save(root / 'diagnostics/comparison.json', result)


def watch(root):
    plan = json.loads((root / 'plan.json').read_text())
    out = root / 'diagnostics'
    out.mkdir(exist_ok=True)
    hashes = fixture_hashes()
    save(out / 'fixture_sha256.json', hashes)
    names = plan['arms'][:]
    if (root / 'fifth_plan.json').exists():
        fifth = json.loads((root / 'fifth_plan.json').read_text())
        if fifth['enabled']:
            names.append(fifth['arm'])
    completed, failed, child = [], {}, None
    try:
        while len(completed) + len(failed) < len(names):
            for name in names:
                if name in completed or name in failed:
                    continue
                if not (root / name / 'completed.json').exists():
                    continue
                gpu = selected_gpu()
                if gpu is None:
                    continue
                destination = prepare(root, name, gpu)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT), PYTHONNOUSERSITE='1',
                           HF_ENDPOINT='https://hf-mirror.com', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
                with (destination / 'worker.log').open('xb') as log:
                    child = subprocess.Popen([plan['python'], str(Path(__file__).resolve()), str(destination), '--worker'],
                                             cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while child.poll() is None:
                    save(out / 'job_status.json', dict(status='running', active_arm=name, gpu=gpu,
                         pid=child.pid, completed=completed, failed=failed))
                    time.sleep(5)
                if child.returncode != 0 or not (destination / 'complete.json').exists():
                    failed[name] = f'Worker exit{child.returncode}; inspect {destination}/worker.log'
                    continue
                if fixture_hashes() != hashes:
                    raise RuntimeError('Fixed diagnostic fixtures changed; stop rather than compare different inputs')
                summarize(root)
                completed.append(name)
            queue = json.loads((root / 'job_status.json').read_text())
            if queue['status'] == 'failed':
                raise RuntimeError('Training queue failed; preserve partial diagnostics')
            if (root / 'fifth_status.json').exists():
                fifth = json.loads((root / 'fifth_status.json').read_text())
                if fifth['status'] == 'failed':
                    failed['E_gain2'] = fifth['error']
            save(out / 'job_status.json', dict(status='waiting_for_checkpoints_or_gpu', completed=completed, failed=failed,
                                             expected=names))
            time.sleep(10)
        save(out / 'job_status.json', dict(status='complete' if not failed else 'partial_failure', completed=completed,
                                         failed=failed, expected=names))
    except BaseException as error:
        if child and child.poll() is None:
            child.terminate()
        save(out / 'job_status.json', dict(status='failed', error=repr(error), completed=completed, failed=failed))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.worker:
        module = load_module(BOOTSTRAP, 'screen_fixed_input_worker')
        module.OUT = root
        module.main()
    elif args.summarize:
        summarize(root)
    else:
        watch(root)


if __name__ == '__main__':
    main()
