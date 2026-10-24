"""Pin one checkpoint and run six-task expert gates and policy evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.maniskill.single_eval import TASKS, seed_manifest, seeds_for


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda:f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def partitions(tasks, episodes, workers):
    jobs = []
    for task in tasks:
        for shard in range(min(workers, episodes)):
            start = episodes * shard // min(workers, episodes)
            end = episodes * (shard + 1) // min(workers, episodes)
            jobs.append(dict(task=task, start=start, count=end-start))
    return jobs


def combine(task, summaries, seeds):
    rows = sorted([r for s in summaries for r in s['records']], key=lambda r:r['episode'])
    if (not all(s['complete'] for s in summaries)
            or [r['episode'] for r in rows] != list(range(len(seeds)))
            or [r['seed'] for r in rows] != seeds):
        raise ValueError(f'Incomplete, duplicate or mismatched shards: {task}')
    successes = sum(r['success'] for r in rows)
    return dict(task=task, complete=True, episodes=len(rows), successes=successes,
                success_rate=successes/len(rows), mean_steps=sum(r['steps'] for r in rows)/len(rows), records=rows)


def assign_pools(jobs, gpus, port):
    pools = []
    for index, gpu in enumerate(gpus):
        selected = jobs[index::len(gpus)]
        if not selected:
            continue
        pools.append(dict(gpu=gpu, port=port, slots=len(selected)))
        for slot, job in enumerate(selected):
            job.update(port=port+slot, pool_slot=slot, sim_gpu=gpu)
        port += len(selected)
    return pools


def check_baseline_reference(cfg, checkpoint, reference, args):
    """Keep policy inputs and the recorded baseline evaluation protocol fixed."""
    from omegaconf import OmegaConf
    baseline = json.loads((reference/'run.json').read_text())
    base_cfg = OmegaConf.load(reference/'checkpoint/config.yaml')
    keys = ('action_mode','unify_action','unify_action_map','unify_state_map','normalize_mode',
            'num_frames','video_stride','height','width','multiview','camera_layout','color_jitter')
    for key in keys:
        if cfg.dataloader.get(key) != base_cfg.dataloader.get(key):
            raise ValueError(f'Sidecar/baseline input contract differs: {key}')
    for field in ('episodes_per_task','max_steps','replan_every','denoise_steps'):
        actual = args.episodes if field=='episodes_per_task' else getattr(args,field)
        if actual != baseline[field]:
            raise ValueError(f'Sidecar/baseline evaluation budget differs: {field}')
    stats_equal = sha256(checkpoint.parent/'normalization_stats.npy') == sha256(reference/'checkpoint/normalization_stats.npy')
    if not stats_equal:
        raise ValueError('Sidecar and baseline normalization statistics differ')
    for task, row in baseline['seed_manifests'].items():
        if seeds_for(task,args.episodes) != row['seeds']:
            raise ValueError(f'Sidecar/baseline test seeds differ: {task}')
    for name in ('benchmarks/maniskill/single_eval.py','openwam/maniskill_eef_contract.py'):
        if sha256(ROOT/name) != baseline['source_hashes'][name]:
            raise ValueError(f'Baseline observation/action bridge changed: {name}')
    state_file = checkpoint.parent/f'accel_state_step_{checkpoint.stem.split("_")[-1]}'/'trainer_state.json'
    counter = json.loads(state_file.read_text()) if state_file.exists() else None
    return dict(baseline_reference=str(reference),identical_input_contract=True,identical_stats=True,
        identical_test_seeds=True,identical_observation_action_bridge=True,
        baseline_optimizer_updates=14000,sidecar_trainer_state=counter,
        compile_difference='baseline full MoT compile; Sidecar FFN-only compile validated against its own eager',
        interpretation='matched evaluation protocol; checkpoints are not matched optimizer-update counts')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpus', type=int, nargs='+', default=[5, 6])
    p.add_argument('--sim-gpus', type=int, nargs='+', help='Override simulator placement without changing model pools')
    p.add_argument('--policy', choices=('baseline','sidecar'), default='baseline')
    p.add_argument('--baseline-reference',type=Path,default=ROOT/'outputs/maniskill_eval/baseline14000_six_task_100_replan8_max200_gpu56_20261019')
    p.add_argument('--workers-per-task', type=int, default=2)
    p.add_argument('--expert-report', type=Path, help='Explicitly reuse a previously verified six-task expert report')
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--replan-every', type=int, default=8)
    p.add_argument('--denoise-steps', type=int, default=10)
    p.add_argument('--task', choices=['all', *TASKS], default='all')
    p.add_argument('--port', type=int, default=21700)
    p.add_argument('--model-python', default=str(ROOT / '.venv/bin/python'))
    p.add_argument('--sim-python', default='/remote-home/jinminghao/miniconda3/envs/maniskill-eort-v1/bin/python')
    p.add_argument('--maniskill-root', type=Path, default=Path('/remote-home/jinminghao/WAMs/ManiSkill'))
    args = p.parse_args()
    if min(args.episodes, args.max_steps, args.denoise_steps, args.replan_every, args.workers_per_task) < 1 or args.replan_every > 32:
        p.error('Positive budgets required; replan-every must be <=32')
    checkpoint, output = args.checkpoint.absolute(), args.output.absolute()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    tasks = list(TASKS) if args.task == 'all' else [args.task]
    jobs = partitions(tasks, args.episodes, args.workers_per_task)
    if len(args.gpus)!=len(set(args.gpus)):
        p.error('GPU IDs must be unique')
    pools = assign_pools(jobs, args.gpus, args.port)
    if args.sim_gpus:
        for index,job in enumerate(jobs):
            job['sim_gpu']=args.sim_gpus[index%len(args.sim_gpus)]
    for task in tasks:
        seeds_for(task, args.episodes)
    for slot in range(len(jobs)):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', args.port + slot))
    output.mkdir(parents=True, exist_ok=False)
    from omegaconf import OmegaConf
    training_cfg = OmegaConf.load(checkpoint.parent/'config.yaml')
    if args.policy=='sidecar':
        if training_cfg.dataloader.type != 'maniskill_sidecar':
            raise ValueError('Sidecar policy requires a ManiSkill Sidecar checkpoint')
        fairness=check_baseline_reference(training_cfg,checkpoint,args.baseline_reference,args)
        (output/'fairness.json').write_text(json.dumps(fairness,indent=2)+'\n')
    snapshot = output / 'checkpoint'
    snapshot.mkdir()
    # Training prunes old checkpoint names. A read-only hardlink pins this exact inode.
    os.link(checkpoint, snapshot / checkpoint.name)
    for name in ('config.yaml', 'normalization_stats.npy'):
        shutil.copy2(checkpoint.parent / name, snapshot / name)
    shutil.copytree(checkpoint.parent / 'tokenizer', snapshot / 'tokenizer')
    token = uuid.uuid4().hex
    deploy = OmegaConf.load(ROOT/'configs/deploy.yaml')
    deploy.inference.inference_horizon = args.replan_every
    deploy.inference.denoise_steps = args.denoise_steps
    if args.policy=='sidecar':
        OmegaConf.update(deploy,'inference.cfg_scale',1.0)
        OmegaConf.update(deploy,'optimization.compile.sidecar_strategy','ffn')
    OmegaConf.save(deploy, output/'deploy.yaml')
    run = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(snapshot / checkpoint.name),
        snapshot=str(snapshot), readiness_token=token, policy=args.policy,tasks=tasks, episodes_per_task=args.episodes,
        max_steps=args.max_steps, replan_every=args.replan_every, denoise_steps=args.denoise_steps,
        model_pools=pools, workers=len(jobs), shards=jobs, control_mode='pd_ee_pose',
        cache=True, compile=True, condition='front+wrist RGB, task text, actual base-frame TCP10',
        source_hashes={str(q.relative_to(ROOT)):sha256(q) for q in
            [Path(__file__), ROOT/'benchmarks/maniskill/single_eval.py', ROOT/'openwam/maniskill_eef_contract.py',
             ROOT/'benchmarks/robotwin/serve_parallel.py', ROOT/'scripts/check_maniskill_expert_replay.py']},
        seed_manifests={t:dict(path=str(seed_manifest(t)), sha256=sha256(seed_manifest(t)), seeds=seeds_for(t,args.episodes)) for t in tasks},
        artifacts_sha256={n:sha256(snapshot/n) for n in ('config.yaml','normalization_stats.npy')},
        deploy_sha256=sha256(output/'deploy.yaml'))
    (output / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    if args.policy=='sidecar':
        run['source_hashes'].update({str(p.relative_to(ROOT)):sha256(p) for p in
            [ROOT/'benchmarks/maniskill/serve_sidecar.py',ROOT/'benchmarks/robotwin/sidecar_fast_runtime.py',
             ROOT/'benchmarks/robotwin/check_sidecar_fast_runtime.py']})
        (output/'run.json').write_text(json.dumps(run,indent=2)+'\n')
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', PYTHONPATH=str(ROOT),
        HF_ENDPOINT='https://hf-mirror.com', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TORCH_WARM_POOL='0',
        TORCHINDUCTOR_CACHE_DIR=str(output / 'inductor'), TORCH_EXTENSIONS_DIR=str(output / 'extensions'))
    sim_env = dict(env,
        PYTHONPATH=f'{ROOT}:{args.maniskill_root}',
        VK_ICD_FILENAMES=os.environ.get('VK_ICD_FILENAMES', '/usr/share/vulkan/icd.d/nvidia_icd.json'))
    # Current A100 image needs this existing graphics compiler preloaded (process-local).
    gpucomp = Path('/usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.570.86.10')
    if gpucomp.is_file():
        sim_env['LD_PRELOAD'] = str(gpucomp) + (':' + env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
    servers, children = [], []
    try:
        # Check the thin simulator's imports before paying for full model loading.
        subprocess.run([args.sim_python,'-c',
            'from benchmarks.utils.client import build_payload, encode_numpy_b64; '
            'from benchmarks.utils.transport import WSPolicyClient; '
            'from benchmarks.robotwin.prompt_template import format_prompt_for_inference; '
            'from openwam.maniskill_eef_contract import achieved_states; '
            'from websockets.sync.client import connect; import gymnasium, mani_skill.envs'],
            cwd=ROOT,env=sim_env,check=True)
        print('Simulator client import preflight passed',flush=True)
        for pool in pools:
            gpu=pool['gpu']
            server_entry = 'benchmarks/maniskill/serve_sidecar.py' if args.policy=='sidecar' else 'benchmarks/robotwin/serve_parallel.py'
            with (output / f'server_gpu{gpu}.log').open('w') as log:
                server = subprocess.Popen([args.model_python, str(ROOT/server_entry),
                    '--ckpt-dir', str(snapshot), '--port', str(pool['port']), '--slots', str(pool['slots']),
                    '--config', str(output/'deploy.yaml'), '--manifest', str(output/f'server_gpu{gpu}_manifest.json'),
                    '--readiness-token', token], cwd=ROOT,
                    env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu),TORCHINDUCTOR_CACHE_DIR=str(output/f'inductor_gpu{gpu}')),
                    stdout=log, stderr=subprocess.STDOUT)
            servers.append(server)
            print(f'Model GPU{gpu} PID={server.pid}, {pool["slots"]} isolated endpoints',flush=True)
        report = output/'expert_replay.json'
        if args.expert_report:
            shutil.copy2(args.expert_report, report)
        else:
            with (output/'expert_replay.log').open('w') as log:
                subprocess.run([args.sim_python, str(ROOT/'scripts/check_maniskill_expert_replay.py'),
                    '--output', str(report)], cwd=ROOT,
                    env=dict(sim_env,CUDA_VISIBLE_DEVICES=str(args.gpus[0])),
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        gates = json.loads(report.read_text())
        if (len(gates)!=6 or {r['task'] for r in gates}!=set(TASKS)
            or not all(all(r['success'].values()) and r['maximum_tcp_position_difference_m']<.02 for r in gates)):
            raise RuntimeError('Six-task expert gate incomplete or failed')
        print('Six-task expert gate verified', flush=True)
        from benchmarks.utils.transport import WSPolicyClient
        deadline = time.monotonic() + 1800
        while True:
            if any(s.poll() is not None for s in servers):
                raise RuntimeError('Model server exited; see server_gpu*.log')
            try:
                for job in jobs:
                    with WSPolicyClient(f'ws://127.0.0.1:{job["port"]}', timeout=5, open_timeout=2) as client:
                        pong = client.ping()
                        if pong.get('readiness_token') != token or pong.get('pool_slot') != job['pool_slot']:
                            raise RuntimeError('Wrong server ownership token/slot')
                break
            except (OSError, TimeoutError):
                if time.monotonic() > deadline:
                    raise TimeoutError('Model server handshake timeout')
                time.sleep(5)
        for slot, job in enumerate(jobs):
            task, start, count = job['task'], job['start'], job['count']
            target = output/'policy'/task/f'shard_{start:04d}'
            job.update(output=str(target))
            with (output/f'{task}_{start:04d}.log').open('w') as log:
                child = subprocess.Popen([args.sim_python,str(ROOT/'benchmarks/maniskill/single_eval.py'),
                    '--task',task,'--output',str(target),'--episodes',str(count),'--seed-offset',str(start),
                    '--max-steps',str(args.max_steps),'--port',str(job['port']),'--readiness-token',token],
                    cwd=ROOT,env=dict(sim_env,CUDA_VISIBLE_DEVICES=str(job['sim_gpu'])),stdout=log,stderr=subprocess.STDOUT)
            children.append(child)
            job['pid']=child.pid
            print(f'Worker {slot}: {task} seeds[{start}:{start+count}] GPU{job["sim_gpu"]} PID={child.pid}',flush=True)
            # Stagger graphics initialization without changing the evaluation protocol.
            time.sleep(2)
        (output/'workers.json').write_text(json.dumps(jobs,indent=2)+'\n')
        completed = set()
        while len(completed)<len(jobs):
            for index,child in enumerate(children):
                if index in completed or child.poll() is None:
                    continue
                if child.returncode != 0:
                    raise RuntimeError(f'Worker {index} exited {child.returncode}: {jobs[index]}')
                completed.add(index)
            if any(s.poll() is not None for s in servers):
                raise RuntimeError('Model pool exited')
            summaries=[]
            for task in tasks:
                shards=[j for j in jobs if j['task']==task]
                files=[Path(j['output'])/'summary.json' for j in shards]
                if all(f.is_file() for f in files):
                    summary=combine(task,[json.loads(f.read_text()) for f in files],seeds_for(task,args.episodes))
                    (output/'policy'/task/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
                    summaries.append(summary)
            total = sum(s['episodes'] for s in summaries)
            successes = sum(s['successes'] for s in summaries)
            result = dict(complete=len(completed)==len(jobs) and len(summaries)==len(tasks), tasks=summaries, episodes=total,
                          successes=successes, success_rate=successes/total if total else None)
            (output/'summary.json').write_text(json.dumps(result, indent=2) + '\n')
            time.sleep(5)
        print(json.dumps(result),flush=True)
        if args.policy=='sidecar':
            baseline=json.loads((args.baseline_reference/'summary.json').read_text())
            comparisons=[]
            for task in result['tasks']:
                base=next(b for b in baseline['tasks'] if b['task']==task['task'])
                by_seed={r['seed']:r['success'] for r in base['records']}
                comparisons.append(dict(task=task['task'],baseline_successes=base['successes'],
                    sidecar_successes=task['successes'],episodes=task['episodes'],
                    delta_pp=100*(task['success_rate']-base['success_rate']),
                    baseline_only=sum(by_seed[r['seed']] and not r['success'] for r in task['records']),
                    sidecar_only=sum(r['success'] and not by_seed[r['seed']] for r in task['records'])))
            baseline_successes=sum(r['baseline_successes'] for r in comparisons)
            comparison=dict(baseline_successes=baseline_successes,sidecar_successes=result['successes'],
                episodes=result['episodes'],tasks=comparisons,fairness=fairness)
            (output/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
            lines=['# ManiSkill baseline / Sidecar comparison','',
                '| Task | Baseline | Sidecar | Delta pp |','| --- | ---: | ---: | ---: |']
            lines += [f'| {r["task"]} | {r["baseline_successes"]}/{r["episodes"]} | {r["sidecar_successes"]}/{r["episodes"]} | {r["delta_pp"]:+.1f} |' for r in comparisons]
            lines += ['',f'Total: baseline {baseline_successes}/{result["episodes"]}; Sidecar {result["successes"]}/{result["episodes"]}.',
                '',fairness['interpretation'],fairness['compile_difference'],
                'Next: inspect failed episodes before attributing differences to a specific mechanism.']
            (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    except BaseException as exc:
        (output/'failure.json').write_text(json.dumps(dict(error=repr(exc))) + '\n')
        raise
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for server in servers:
            if server.poll() is not None:
                continue
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == '__main__':
    main()
