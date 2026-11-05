"""Held-out archive identities and native scene reconstruction; no model inputs from labels."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def seed_from_record(record):
    match = re.search(r'/seed(\d+)_', record['media'])
    if not match:
        raise ValueError(f"Missing physical seed: {record['media']}")
    return int(match.group(1))


def select_records(manifest, count=100):
    if count < 1:
        raise ValueError('Positive episode count required')
    records = [r for r in manifest['episodes'] if r['group'] == 'maniskill_franka']
    train = [r for r in records if r['split'] == 'train' and r.get('selected_for_midtrain', True)]
    forbidden = {key: {r[key] for r in train} for key in ('id', 'family_id', 'numeric', 'media')}
    train_seeds = {seed_from_record(r) for r in train}
    selected = []
    for task in sorted({r['task'] for r in records}):
        clean = sorted([r for r in records if r['task'] == task and r['collection'] == 'maniskill_base'
                        and r['domain'] == 'Clean' and r['split'] in ('val', 'test')],
                       key=lambda r: (r['split'] != 'test', seed_from_record(r), r['id']))
        hard = {d: sorted([r for r in records if r['task'] == task and r['collection'] == 'maniskill_hard'
                           and r['split'] == 'test' and r['domain'] == d], key=seed_from_record)
                for d in ('D', 'M')}
        # Preserve the native hard-test mixture: 25% D, 75% M at count=100.
        nd = round(count * .25)
        if len(clean) < count or len(hard['D']) < nd or len(hard['M']) < count - nd:
            raise ValueError(f'Insufficient distinct held-out scenes for {task}')
        for condition, subset in [('Clean', clean[:count]), ('HardRandom', hard['D'][:nd] + hard['M'][:count-nd])]:
            if len({seed_from_record(r) for r in subset}) != count:
                raise ValueError(f'Duplicate physical scene in {task}/{condition}')
            for r in subset:
                if seed_from_record(r) in train_seeds or any(r[k] in forbidden[k] for k in forbidden):
                    raise ValueError(f"Training overlap: {r['id']}")
                selected.append(dict(r, condition=condition))
    return selected


def build_plan(manifest_path, count=100):
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    selected = select_records(manifest, count)
    cases = []
    for row in selected:
        media = root / row['media']
        metadata = json.loads(media.with_suffix('.json').read_text())
        episode = metadata['episodes'][0]
        seed = seed_from_record(row)
        if len(metadata['episodes']) != 1 or not episode['success'] or episode['episode_seed'] != seed:
            raise ValueError(f'Unaccepted or mismatched expert episode: {media}')
        info = metadata['env_info']
        kwargs = dict(info['env_kwargs'])
        if kwargs['domain'] != row['domain'] or ('Hard' in info['env_id'] and 'hard_index' not in kwargs):
            raise ValueError(f'Incomplete native environment identity: {media}')
        cases.append(dict(row, seed=seed, env_id=info['env_id'], env_kwargs=kwargs,
                          reset_kwargs=episode['reset_kwargs'], max_steps=info['max_episode_steps']))
    groups = Counter((r['task'], r['condition'], r['split'], r['domain']) for r in cases)
    return dict(schema='openwam_maniskill_archive_eval_v1', archive_root=str(root),
                manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                episodes_per_task_condition=count, episodes=len(cases),
                split_policy={'Clean': 'held-out val+test; test first', 'HardRandom': 'test only; D25% M75%'},
                training_overlap=dict(episode_ids=0, family_ids=0, physical_seeds=0, numeric_paths=0, media_paths=0),
                counts=[dict(task=k[0], condition=k[1], split=k[2], domain=k[3], episodes=n)
                        for k, n in sorted(groups.items())], cases=cases)


def reuse_plan(reference, manifest_path, count):
    """An unchanged archive can use its already selected held-out scenes."""
    reference, manifest_path = Path(reference), Path(manifest_path).resolve()
    if not reference.is_file():
        return None
    plan = json.loads(reference.read_text())
    if (plan['manifest_sha256'] != hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            or Path(plan['archive_root']).resolve() != manifest_path.parent
            or plan['episodes_per_task_condition'] != count):
        return None
    return plan


def register_environments(root):
    import sys
    root = Path(root)
    sys.path.insert(0, str(root / 'replay/DiT4DiT'))
    sys.path.insert(0, str(root / 'replay/ManiSkill'))
    from examples.ManiSkill3 import franka_hard10k_envs  # noqa: F401


def make_case_env(case, *, render=False, control_mode='pd_ee_pose'):
    import gymnasium as gym
    kwargs = dict(case['env_kwargs'])
    kwargs.update(control_mode=control_mode, obs_mode='state_dict+rgb' if render else 'none',
                  render_mode='rgb_array' if render else None,
                  # Native URDF materials require a RenderSystem even for state-only physics replay.
                  render_backend='sapien_cuda',
                  sim_backend='physx_cpu', num_envs=1, max_episode_steps=case['max_steps'])
    return gym.make(case['env_id'], **kwargs)


def reset_case(env, case, root):
    import h5py
    from mani_skill.trajectory.utils import index_dict
    env.reset(**case['reset_kwargs'])
    with h5py.File(Path(root) / case['media'], 'r') as f:
        env.unwrapped.set_state_dict(index_dict(f['traj_0/env_states'], 0))
    return env.unwrapped.get_obs()


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=100)
    p.add_argument('--reuse-plan', type=Path)
    args = p.parse_args()
    plan = reuse_plan(args.reuse_plan, args.manifest, args.count) if args.reuse_plan else None
    if plan is None:
        plan = build_plan(args.manifest, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(plan, f, indent=2)
    print(json.dumps({k: v for k, v in plan.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
