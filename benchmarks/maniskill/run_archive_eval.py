"""Run the archived 8-task held-out suite with existing isolated Sidecar endpoints."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def runtime_signature(plan, model_python, sim_python, model_gpu, sim_gpu, checkpoint):
    """Small identity for reusing completed environment/bridge/runtime checks."""
    import yaml
    packages = ['torch', 'numpy', 'sapien', 'gymnasium']
    code = ('import importlib.metadata,json; print(json.dumps({p:importlib.metadata.version(p) '
            'for p in '+repr(packages)+'}))')
    sim_versions = json.loads(subprocess.check_output([sim_python, '-c', code], text=True))
    sources = ['benchmarks/maniskill/archive_eval.py', 'benchmarks/maniskill/archive_protocol.py',
               'benchmarks/maniskill/single_eval.py', 'benchmarks/maniskill/serve_sidecar.py',
               'benchmarks/robotwin/sidecar_fast_runtime.py', 'openwam/maniskill_eef_contract.py',
               'configs/deploy.yaml', 'openwam/model/architectures/dual_system/relational_sidecar.py',
               'openwam/model/modules/relational_sidecar.py']
    gpu = lambda index: subprocess.check_output(['nvidia-smi', '-i', str(index),
        '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip()
    cfg = yaml.safe_load((Path(checkpoint).parent/'config.yaml').read_text())
    return dict(manifest_sha256=plan['manifest_sha256'], archive_root=plan['archive_root'],
                model_python=str(Path(model_python).resolve()), sim_python=str(Path(sim_python).resolve()),
                model_versions={p: importlib.metadata.version(p) for p in ('torch', 'triton')}, sim_versions=sim_versions,
                model_gpu=gpu(model_gpu), sim_gpu=gpu(sim_gpu), model=cfg['model'],
                sources={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})


def reuse_validation(reference, signature):
    """Reuse only a successful matching setup; different setup uses the original full check."""
    if reference is None:
        return False
    reference = Path(reference)
    stamp = reference/'runtime_signature.json'
    if not stamp.exists() or json.loads(stamp.read_text()) != signature:
        return False
    expert = json.loads((reference/'expert.json').read_text())
    validation = json.loads((reference/'server_manifest/validation/result.json').read_text())
    return (len(expert) == 24 and all(r['passed'] for r in expert) and validation['passed']
            and validation['selected_strategy'] == 'ffn' and 'variants' in validation)


def rendering_errors(path):
    markers = ('OIDN Error', 'ErrorDeviceLost', 'VK_ERROR_DEVICE_LOST', 'CUDA error: an illegal memory access')
    found = set()
    with Path(path).open(errors='replace') as stream:
        for line in stream:
            found.update(marker for marker in markers if marker in line)
    return sorted(found)


def aggregate(plan, records):
    expected = {r['id']: r for r in plan['cases']}
    if len(records) != len(expected) or len({r['id'] for r in records}) != len(records):
        raise ValueError('Incomplete or duplicate evaluation coverage')
    groups = defaultdict(list)
    for r in records:
        source = expected[r['id']]
        if any(r[k] != source[k] for k in ('task', 'condition', 'split', 'domain', 'seed')):
            raise ValueError('Evaluation identity differs from frozen plan')
        groups[(r['task'], r['condition'])].append(r)
    tasks = [dict(task=t, condition=c, episodes=len(rows), successes=sum(r['success'] for r in rows),
                  success_rate=sum(r['success'] for r in rows)/len(rows)) for (t, c), rows in sorted(groups.items())]
    domains = []
    for condition in sorted({r['condition'] for r in records}):
        rows = [r for r in records if r['condition'] == condition]
        domains.append(dict(condition=condition, episodes=len(rows), successes=sum(r['success'] for r in rows),
                            success_rate=sum(r['success'] for r in rows)/len(rows)))
    return dict(complete=True, episodes=len(records), tasks=tasks, conditions=domains,
                split_policy=plan['split_policy'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-python', default=sys.executable)
    p.add_argument('--sim-python', default='/remote-home/jinminghao/miniconda3/envs/maniskill-eort-v1/bin/python')
    p.add_argument('--model-gpu', type=int, default=0)
    p.add_argument('--sim-gpu', type=int, default=1)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--port', type=int, default=22400)
    p.add_argument('--validated-run', type=Path, help='Reuse matching successful setup; keep one action smoke check')
    args = p.parse_args()
    if args.workers < 1:
        p.error('Positive worker count required')
    for port in range(args.port, args.port+args.workers):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', port))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.plan, output / 'plan.json')
    plan = json.loads((output / 'plan.json').read_text())
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
               HF_ENDPOINT=env.get('HF_ENDPOINT', 'https://hf-mirror.com'), HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', TORCH_WARM_POOL='0', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    sim_env = dict(env, CUDA_VISIBLE_DEVICES=str(args.sim_gpu))
    compiler = Path(env.get('MANISKILL_GPUCOMP_LIBRARY', '/usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.570.86.10'))
    if env.get('MANISKILL_GPUCOMP_LIBRARY') and not compiler.is_file():
        raise FileNotFoundError(f'Configured graphics compiler library does not exist: {compiler}')
    if compiler.exists():
        sim_env['LD_PRELOAD'] = str(compiler) + (':'+env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
    entry = str(ROOT / 'benchmarks/maniskill/archive_eval.py')
    signature = runtime_signature(plan, args.model_python, args.sim_python, args.model_gpu,
                                  args.sim_gpu, args.checkpoint)
    (output/'runtime_signature.json').write_text(json.dumps(signature, indent=2)+'\n')
    reused = reuse_validation(args.validated_run, signature)
    # The full 24-scene controller gate is independent of model availability.
    if reused:
        shutil.copy2(args.validated_run/'expert.json', output/'expert.json')
        print(f'Reusing completed data/environment checks: {args.validated_run}', flush=True)
    else:
        with (output / 'expert.log').open('w') as log:
            subprocess.run([args.sim_python, entry, 'expert', '--plan', str(output/'plan.json'),
                            '--output', str(output/'expert.json')], cwd=ROOT, env=sim_env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([args.model_python, entry, 'prepare', '--checkpoint', str(args.checkpoint.resolve()),
                    '--output', str(output/'deployment')], cwd=ROOT, env=env, check=True)
    token = uuid.uuid4().hex
    server = None
    children = []
    try:
        model_env = dict(env, CUDA_VISIBLE_DEVICES=str(args.model_gpu),
                         OPENWAM_MANISKILL_EVAL_PLAN=str(output/'plan.json'),
                         TORCHINDUCTOR_CACHE_DIR=str(output/'inductor'))
        if reused:
            model_env['OPENWAM_MANISKILL_VALIDATED_RUN'] = str(args.validated_run.resolve())
            model_env['TORCHINDUCTOR_CACHE_DIR'] = str(args.validated_run.resolve()/'inductor')
        with (output/'server.log').open('w') as log:
            server = subprocess.Popen([args.model_python, str(ROOT/'benchmarks/maniskill/serve_sidecar.py'),
                '--ckpt-dir', str(output/'deployment/checkpoint'), '--config', str(output/'deployment/deploy.yaml'),
                '--manifest', str(output/'server_manifest.json'), '--port', str(args.port), '--slots', str(args.workers),
                '--readiness-token', token], cwd=ROOT, env=model_env, stdout=log, stderr=subprocess.STDOUT)
        from benchmarks.utils.transport import WSPolicyClient
        deadline = time.monotonic()+86400
        while True:
            if server.poll() is not None:
                raise RuntimeError('Model server exited; see server.log')
            try:
                with WSPolicyClient(f'ws://127.0.0.1:{args.port}', timeout=5) as client:
                    if client.ping().get('readiness_token') != token:
                        raise RuntimeError('Policy endpoint ownership mismatch')
                break
            except (ConnectionError, OSError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Model capacity/readiness wait exceeded 24 hours')
                time.sleep(10)
        if not reused:
            (ROOT/'outputs/maniskill_eval/validated_run.txt').write_text(str(output)+'\n')
        commands = []
        for shard in range(args.workers):
            command = [args.sim_python, entry, 'rollout', '--plan', str(output/'plan.json'),
                       '--output', str(output/f'worker{shard}'), '--shard', str(shard), '--shards', str(args.workers),
                       '--port', str(args.port+shard), '--readiness-token', token]
            commands.append(command)
            with (output/f'worker{shard}.log').open('w') as log:
                children.append(subprocess.Popen(command, cwd=ROOT, env=sim_env, stdout=log, stderr=subprocess.STDOUT))
        paths = []
        for shard, child in enumerate(children[:args.workers]):
            path = output/f'worker{shard}'
            if child.wait() != 0 or rendering_errors(output/f'worker{shard}.log'):
                # One full-shard retry preserves the failed attempt and excludes its partial results.
                path = output/f'worker{shard}_retry'
                command = list(commands[shard])
                command[command.index('--output')+1] = str(path)
                with (output/f'worker{shard}_retry.log').open('w') as log:
                    retry = subprocess.Popen(command, cwd=ROOT, env=sim_env, stdout=log, stderr=subprocess.STDOUT)
                children.append(retry)
                if retry.wait() != 0 or rendering_errors(output/f'worker{shard}_retry.log'):
                    raise RuntimeError('Evaluation retry failed; logs retained, no final SR')
            paths.append(path)
        records = []
        for path in paths:
            records.extend(json.loads(l) for l in (path/'episodes.jsonl').read_text().splitlines())
        result = aggregate(plan, records)
        (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as exc:
        (output/'failure.json').write_text(json.dumps(dict(error=repr(exc)))+'\n')
        raise
    finally:
        for process in children+([server] if server is not None else []):
            if process.poll() is None:
                process.terminate()
        for process in children+([server] if server is not None else []):
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    main()
