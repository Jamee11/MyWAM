"""Prepare a single-arm deployment, replay experts, or evaluate a frozen archive plan."""
import argparse
import errno
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.maniskill.archive_protocol import make_case_env, register_environments, reset_case


def single_arm_stats(stats):
    import numpy as np
    if np.asarray(stats['eef']['min']).shape != (20,):
        raise ValueError('Expected the 20D midtrain normalization contract')
    result = dict(stats)
    result['eef'] = {}
    for key, value in stats['eef'].items():
        array = np.asarray(value)
        if array.ndim and array.shape[-1] == 20:
            value = array[..., :10].copy()
        result['eef'][key] = value
    if np.asarray(result['eef']['min']).shape != (10,):
        raise ValueError('Expected the 20D midtrain normalization contract')
    return result


def prepare(args):
    import numpy as np
    from omegaconf import OmegaConf
    source = args.checkpoint.resolve()
    cfg = OmegaConf.load(source.parent / 'config.yaml')
    if cfg.dataloader.type != 'maniskill_archive' or list(cfg.dataloader.unify_action_map) != ['0-9', '34-43']:
        raise ValueError('Expected the current archived-ManiSkill finetune checkpoint')
    if cfg.model.architecture.action_dim != 80:
        raise ValueError('Keep the trained 80D model intact')
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / 'checkpoint'
    snapshot.mkdir()
    identity = 'read-only shared inode; weights unchanged'
    try:
        os.link(source, snapshot / source.name)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copy2(source, snapshot / source.name)
        identity = 'copied across filesystems; source weights unchanged'
    shutil.copy2(source.parent / 'config.yaml', args.output / 'training_config.yaml')
    shutil.copy2(source.parent / 'normalization_stats.npy', args.output / 'training_normalization_stats.npy')
    cfg.dataloader.unify_action_map = ['0-9']
    cfg.dataloader.multiview = True
    cfg.dataloader.camera_layout = ['front_camera', 'hand_camera', None]
    if cfg.dataloader.get('unify_state_map') is not None:
        cfg.dataloader.unify_state_map = ['0-9']
    OmegaConf.save(cfg, snapshot / 'config.yaml')
    stats = np.load(source.parent / 'normalization_stats.npy', allow_pickle=True).item()
    np.save(snapshot / 'normalization_stats.npy', single_arm_stats(stats))
    shutil.copytree(source.parent / 'tokenizer', snapshot / 'tokenizer')
    if (source.parent / 'relational_sidecar_contract.json').exists():
        contract = json.loads((source.parent / 'relational_sidecar_contract.json').read_text())
        contract.update(unify_action_map=['0-9'], camera_layout=['front_camera', 'hand_camera', None])
        (snapshot / 'relational_sidecar_contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    deploy = OmegaConf.load(ROOT / 'configs/deploy.yaml')
    deploy.inference.inference_horizon = 8
    deploy.inference.denoise_steps = 10
    OmegaConf.update(deploy, 'inference.cfg_scale', 1.0)
    OmegaConf.update(deploy, 'optimization.compile.sidecar_strategy', 'ffn')
    OmegaConf.save(deploy, args.output / 'deploy.yaml')
    proof = dict(checkpoint=str(source), bytes=source.stat().st_size,
                 identity=identity,
                 source_config_sha256=hashlib.sha256((source.parent / 'config.yaml').read_bytes()).hexdigest(),
                 raw_proprio_dim=10, model_dim=80, active_slots=list(range(10)),
                 conversion='Deploy only: select first 10 raw stats and scatter map; inactive arm stays masked',
                 cache=True, compile='local FFN; numerical validation required before serving',
                 denoise_steps=10, replan_every=8)
    (args.output / 'deployment_projection.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof, indent=2))


def expert(args, plan):
    import h5py
    import numpy as np
    from benchmarks.maniskill.single_eval import native_action, numpy, robot_state
    root = Path(plan['archive_root'])
    groups = {}
    for case in plan['cases']:
        groups.setdefault((case['task'], case['condition'], case['domain']), case)
    results = []
    for case in list(groups.values())[:args.limit or None]:
        with h5py.File(root / case['media'], 'r') as f:
            delta = np.asarray(f['traj_0/actions'])
        with np.load(root / case['numeric'], allow_pickle=False) as f:
            absolute = np.stack([native_action(a[:10]) for a in f['action']])
        traces, successes = {}, {}
        for mode, commands in [('pd_ee_delta_pose', delta), ('pd_ee_pose', absolute)]:
            env = make_case_env(case, control_mode=mode)
            try:
                reset_case(env, case, root)
                trace, success = [], False
                for command in commands:
                    _, _, _, _, info = env.step(command[None])
                    trace.append(robot_state(env.unwrapped)[:3])
                    success |= bool(numpy(info['success']).item())
                traces[mode], successes[mode] = np.asarray(trace), success
            finally:
                env.close()
        distance = float(np.linalg.norm(traces['pd_ee_pose'] - traces['pd_ee_delta_pose'], axis=-1).max())
        row = dict(id=case['id'], task=case['task'], condition=case['condition'], domain=case['domain'],
                   success=successes, maximum_tcp_difference_m=distance,
                   passed=all(successes.values()) and distance < .02)
        results.append(row)
        args.output.write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(row), flush=True)
        if not row['passed']:
            raise RuntimeError('Expert absolute-EEF bridge replay failed; retain this log for diagnosis')


def rollout(args, plan):
    import numpy as np
    from benchmarks.maniskill.single_eval import camera, native_action, numpy, robot_state
    from benchmarks.robotwin.prompt_template import format_prompt_for_inference
    from benchmarks.utils.client import build_payload, encode_numpy_b64
    from benchmarks.utils.transport import WSPolicyClient
    selected = plan['cases'][args.shard::args.shards]
    rows = []
    with WSPolicyClient(f'ws://127.0.0.1:{args.port}', timeout=600) as client:
        if client.ping().get('readiness_token') != args.readiness_token:
            raise ValueError('Wrong model service ownership token')
        for case in selected:
            env = make_case_env(case, render=True)
            try:
                obs = reset_case(env, case, plan['archive_root'])
                control = env.unwrapped.agent.controller.controllers['arm'].config
                if control.use_delta or control.normalize_action or control.ee_link != 'panda_hand_tcp':
                    raise ValueError('Absolute base-frame TCP controller mismatch')
                client.reset()
                success = False
                for step in range(case['max_steps']):
                    payload = build_payload(encode_numpy_b64(camera(obs, 'front_camera')),
                                            encode_numpy_b64(camera(obs, 'hand_camera')),
                                            prompt=format_prompt_for_inference(case['prompt']),
                                            state=robot_state(env.unwrapped).tolist())
                    response = client.predict_once(payload)
                    if response.get('type') != 'action':
                        raise RuntimeError(f'Policy service error: {response}')
                    command = native_action(np.asarray(response['action']))
                    if not env.unwrapped.single_action_space.contains(command):
                        raise ValueError('Prediction outside absolute native action bounds')
                    obs, _, terminated, truncated, info = env.step(command[None])
                    success = bool(numpy(info['success']).item())
                    if success or bool(numpy(terminated).item()) or bool(numpy(truncated).item()):
                        break
                row = dict(id=case['id'], task=case['task'], condition=case['condition'], split=case['split'],
                           domain=case['domain'], seed=case['seed'], success=success, steps=step+1)
                rows.append(row)
                with (args.output / 'episodes.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
            finally:
                env.close()
    (args.output / 'summary.json').write_text(json.dumps(dict(complete=len(rows)==len(selected),
        episodes=len(rows), successes=sum(r['success'] for r in rows)), indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'expert', 'rollout'])
    p.add_argument('--plan', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--port', type=int, default=22400)
    p.add_argument('--readiness-token')
    args = p.parse_args()
    if args.mode == 'prepare':
        if args.checkpoint is None:
            p.error('--checkpoint is required')
        prepare(args)
        return
    if args.plan is None or args.limit < 0 or not 0 <= args.shard < args.shards:
        p.error('Need --plan, nonnegative limit and valid shard/shards')
    plan = json.loads(args.plan.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    register_environments(plan['archive_root'])
    if args.mode == 'expert':
        expert(args, plan)
    else:
        args.output.mkdir()
        try:
            rollout(args, plan)
        except BaseException as exc:
            (args.output / 'failure.json').write_text(json.dumps(dict(error=repr(exc))) + '\n')
            raise


if __name__ == '__main__':
    main()
