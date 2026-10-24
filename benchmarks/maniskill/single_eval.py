"""Six-task EORT evaluation using OpenWAM absolute base-frame TCP commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OLD_DATA = Path('/remote-home/jinminghao/datasets/maniskill_eort_color_distractors_v4_20260728')
SUITE_DATA = Path('/remote-home/jinminghao/datasets/maniskill_eort_task_suite_v5_20260730')
TASKS = {
    'push_cube': ('PushCube', 'push the {color} cube to the target region'),
    'pick_cube': ('PickCube', 'pick up the {color} cube and move it to the elevated goal'),
    'stack_cube': ('StackCube', 'stack the {color} cube on the gray cube'),
    'place_sphere': ('PlaceSphere', 'place the {color} sphere in the bin'),
    'pick_cuboid': ('PickCuboid', 'pick up the {color} cuboid and move it to the elevated goal'),
    'pick_sphere': ('PickSphere', 'pick up the {color} sphere and move it to the elevated goal'),
}
COLORS = ('red', 'blue', 'green', 'yellow', 'orange', 'purple')


def seed_manifest(task):
    name = TASKS[task][0] + 'EORTColorDistractors-v1'
    root, count = (OLD_DATA, 400) if task in ('push_cube', 'pick_cube') else (SUITE_DATA, 200)
    return root / task / 'test/color_distractors/raw' / name / 'motionplanning' / f'{task}_test_color_distractors_seed2000000_n{count}.seed_manifest.json'


def seeds_for(task, count):
    rows = json.loads(seed_manifest(task).read_text())['saved_trajectory_seeds']
    seeds = [int(r['seed']) for r in rows if r.get('success', False)]
    if len(seeds) != len(set(seeds)) or len(seeds) < count or min(seeds) < 2000000:
        raise ValueError(f'Invalid held-out seed manifest for {task}')
    return seeds[:count]


def native_action(value):
    """Project predicted column-6D rotation to SO(3), then native XYZ Euler."""
    a = np.asarray(value, np.float64)
    if a.shape != (10,) or not np.isfinite(a).all():
        raise ValueError(f'Expected finite physical EEF10, got {a}')
    x = a[3:6]
    if np.linalg.norm(x) < 1e-6:
        raise ValueError('Degenerate first rotation column')
    x = x / np.linalg.norm(x)
    y = a[6:9] - x * np.dot(x, a[6:9])
    if np.linalg.norm(y) < 1e-6:
        raise ValueError('Degenerate second rotation column')
    y /= np.linalg.norm(y)
    r = Rotation.from_matrix(np.stack((x, y, np.cross(x, y)), axis=-1))
    return np.r_[a[:3], r.as_euler('XYZ'), 2 * np.clip(a[9], 0, 1) - 1].astype(np.float32)


def numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def robot_state(env):
    from openwam.maniskill_eef_contract import achieved_states
    return achieved_states(numpy(env.agent.tcp.pose.raw_pose), numpy(env.agent.robot.pose.raw_pose),
                           numpy(env.agent.robot.get_qpos()))[0]


def make_env(task, control_mode, max_steps):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    return gym.make(TASKS[task][0] + 'EORTColorDistractors-v1',
                    obs_mode='state_dict+rgb', control_mode=control_mode,
                    sim_backend='physx_cpu', render_backend='sapien_cuda', num_envs=1,
                    sensor_configs={'shader_pack': 'default'}, render_mode='rgb_array',
                    max_episode_steps=max_steps)


def camera(obs, name):
    return numpy(obs['sensor_data'][name]['rgb'])[0].astype(np.uint8)


class Video:
    def __init__(self, path, frame):
        h, w = frame.shape[:2]
        self.process = subprocess.Popen(['/usr/bin/ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', '20',
            '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-pix_fmt', 'yuv420p', str(path)], stdin=subprocess.PIPE)

    def write(self, frame):
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError('Video encoding failed')


def evaluate(args):
    from benchmarks.utils.client import build_payload, encode_numpy_b64
    from benchmarks.utils.transport import WSPolicyClient
    from benchmarks.robotwin.prompt_template import format_prompt_for_inference
    seeds = seeds_for(args.task, args.seed_offset + args.episodes)[args.seed_offset:]
    records = []
    with WSPolicyClient(f'ws://127.0.0.1:{args.port}', timeout=600) as client:
        pong = client.ping()
        if pong.get('readiness_token') != args.readiness_token:
            raise ValueError('Wrong policy server ownership token')
        env = make_env(args.task, 'pd_ee_pose', args.max_steps)
        try:
            controller = env.unwrapped.agent.controller.controllers['arm'].config
            if controller.use_delta or controller.normalize_action or controller.ee_link != 'panda_hand_tcp':
                raise ValueError('Absolute TCP execution controller mismatch')
            for episode, seed in enumerate(seeds, start=args.seed_offset):
                obs, _ = env.reset(seed=seed)
                color_id = int(numpy(obs['extra']['target_color_id']).item())
                if not 0 <= color_id < len(COLORS):
                    raise ValueError('Invalid target color')
                language = TASKS[args.task][1].format(color=COLORS[color_id])
                client.reset()
                video = None
                clipped_grippers, success = 0, False
                try:
                    for step in range(args.max_steps):
                        front, hand = camera(obs, 'front_camera'), camera(obs, 'hand_camera')
                        frame = np.concatenate((front, hand), axis=1)
                        if video is None:
                            video = Video(args.output / f'episode_{episode:04d}.mp4', frame)
                        video.write(frame)
                        payload = build_payload(encode_numpy_b64(front), encode_numpy_b64(hand),
                            prompt=format_prompt_for_inference(language), state=robot_state(env.unwrapped).tolist())
                        response = client.predict_once(payload)
                        if response.get('type') != 'action':
                            raise RuntimeError(f'Unexpected response: {response}')
                        action = np.asarray(response['action'])
                        native = native_action(action)
                        clipped_grippers += int(not 0 <= action[9] <= 1)
                        if not env.unwrapped.single_action_space.contains(native):
                            raise ValueError(f'Absolute action outside native controller bounds: {native}')
                        obs, _, terminated, truncated, info = env.step(native[None])
                        success = bool(numpy(info['success']).item())
                        if success or bool(numpy(terminated).item()) or bool(numpy(truncated).item()):
                            break
                    video.write(np.concatenate((camera(obs, 'front_camera'), camera(obs, 'hand_camera')), axis=1))
                finally:
                    if video is not None:
                        video.close()
                row = dict(episode=episode, seed=seed, success=success, steps=step + 1,
                           instruction=language, clipped_grippers=clipped_grippers)
                records.append(row)
                with (args.output / 'episodes.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
        finally:
            env.close()
    summary = dict(task=args.task, complete=len(records) == args.episodes, episodes=len(records),
        successes=sum(r['success'] for r in records), success_rate=sum(r['success'] for r in records) / len(records),
        mean_steps=float(np.mean([r['steps'] for r in records])), records=records)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'records'}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--seed-offset', type=int, default=0)
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--port', type=int, default=21700)
    p.add_argument('--readiness-token')
    args = p.parse_args()
    if args.episodes < 1 or args.max_steps < 1 or args.seed_offset < 0:
        p.error('episodes and max-steps must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'run_config.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, indent=2) + '\n')
    try:
        evaluate(args)
    except BaseException as exc:
        (args.output / 'failure.json').write_text(json.dumps(dict(error=repr(exc))) + '\n')
        raise


if __name__ == '__main__':
    main()
