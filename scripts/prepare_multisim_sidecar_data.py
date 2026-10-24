#!/usr/bin/env python3
"""Freeze accepted sources, then prepare a self-contained Sidecar sim mixture offline."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULTS = dict(
    rlbench='/remote-home/jinminghao/datasets/rlbench18_franka_ea_rsig_v3_absolute20k_20261020/nvidia_xorg_livefk_v1_gripper_v5/production',
    maniskill='/remote-home/jinminghao/datasets/maniskill3_franka_realwhite_8tasks_20k_20261017/raw',
    hard='/remote-home/jinminghao/datasets/maniskill3_franka_hard_random10k_camerav2_20261019/production',
    robotwin='/remote-home/jinminghao/DiT4DiT/datasets/robotwin2_ea_rsig_v3_wrist020_20k_20261001',
    clutter='/remote-home/jinminghao/DiT4DiT/datasets/robotwin2_fresh_clutter_randomized_4000_20261014',
)
BUILD_FINGERPRINT = hashlib.sha256(b''.join(
    path.read_bytes() for path in (Path(__file__), ROOT / 'openwam/multisim_sidecar_contract.py',
                                  ROOT / 'openwam/maniskill_eef_contract.py', ROOT / 'openwam/dataloader/ea_rsig_lite.py')
)).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=256)
def _small_evidence_hash(path, size, mtime_ns):
    return sha(path)


def verify_evidence(path, expected):
    stat = Path(path).stat()
    if _small_evidence_hash(str(path), stat.st_size, stat.st_mtime_ns) != expected:
        raise ValueError(f'Accepted-source evidence changed: {path}')


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def marker_records(root, pattern):
    paths = sorted(Path(root).glob(pattern))
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(zip(paths, pool.map(read_json, paths)))


def identify(record):
    identity = '\0'.join(str(record[k]) for k in ('collection', 'family_id', 'domain'))
    record['id'] = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return record


def inventory(roots):
    records = []
    for marker, accepted in marker_records(roots['rlbench'], '*/*/*/ACCEPTED.json'):
        if accepted['status'] != 'family_audited' or sha(accepted['audit']) != accepted['audit_sha256']:
            raise ValueError(f'Unsealed RLBench family: {marker}')
        job = accepted['job']
        folder = Path(accepted['folder'])
        family = 'rlbench/' + '/'.join(str(job[k]) for k in ('task', 'split', 'kind', 'slot'))
        for domain in job['domains']:
            records.append(identify(dict(collection='rlbench', group='rlbench_franka', task=job['task'],
                split=job['split'], family_id=family, domain=domain, source_h5=str(folder / 'raw.h5'),
                source_numeric=str(folder / 'exports' / domain / 'openwam32.npz'), marker=str(marker),
                marker_sha256=sha(marker), audit=accepted['audit'], audit_sha256=accepted['audit_sha256'])))
    print(f'RLBench: {len(records)} accepted visual episodes (20k is the collection target)', flush=True)

    base = []
    for marker, accepted in marker_records(roots['maniskill'], '*/*/*/*/RAW_COLLECTED.json'):
        family_path = Path(accepted['family_report'])
        proof = read_json(family_path.parent / 'RAW_FAMILY_AUDITED.json')
        if proof['status'] != 'raw_family_audited' or sha(family_path) != accepted['family_report_sha256']:
            raise ValueError(f'Unsealed ManiSkill family: {marker}')
        if proof['family_report_sha256'] != accepted['family_report_sha256']:
            raise ValueError(f'ManiSkill audit identity changed: {marker}')
        family = read_json(family_path)
        job = accepted['job']
        identity = 'maniskill/' + '/'.join(str(job[k]) for k in ('task', 'split', 'kind', 'slot'))
        for domain in job['domains']:
            base.append(identify(dict(collection='maniskill_base', group='maniskill_franka', task=job['task'],
                split=job['split'], family_id=identity, domain=domain, source_h5=family[domain]['replay'],
                marker=str(marker), marker_sha256=sha(marker), audit=str(family_path.parent / 'RAW_FAMILY_AUDITED.json'))))
    counts = Counter(x['domain'] for x in base)
    if counts != dict(Clean=5000, B=3000, C=3000, O=3000, D=3000, M=3000):
        raise ValueError(f'Original ManiSkill collection is incomplete: {counts}')
    records.extend(x for x in base if x['domain'] == 'Clean')
    tasks = sorted({x['task'] for x in base})
    if len(tasks) != 8:
        raise ValueError('Expected eight ManiSkill tasks')
    for di, domain in enumerate(('B', 'C', 'O', 'D', 'M')):
        for ti, task in enumerate(tasks):
            val = 12 + ((ti + di) % 2)
            for split, quota in [('train', 100), ('val', val), ('test', 25 - val)]:
                candidates = [x for x in base if (x['task'], x['domain'], x['split']) == (task, domain, split)]
                candidates.sort(key=lambda x: hashlib.sha256(('42/' + x['family_id']).encode()).digest())
                if len(candidates) < quota:
                    raise ValueError(f'Missing quota: {task}/{domain}/{split}')
                records.extend(candidates[:quota])
    hard = marker_records(roots['hard'], '*/*/*/*/ACCEPTED_RAW.json')
    if len(hard) != 10000:
        raise ValueError(f'Expected all 10000 hard-random episodes, got {len(hard)}')
    for marker, accepted in hard:
        if sha(accepted['audit']) != accepted['audit_sha256']:
            raise ValueError(f'Hard audit changed: {marker}')
        proof, job = read_json(accepted['audit']), accepted['job']
        if proof['status'] != 'raw_hard_audited':
            raise ValueError(f'Hard episode is not accepted: {marker}')
        family = 'maniskill_hard/' + '/'.join(str(job[k]) for k in ('task', 'split', 'domain', 'slot'))
        records.append(identify(dict(collection='maniskill_hard', group='maniskill_franka', task=job['task'],
            split=job['split'], family_id=family, domain=job['domain'], source_h5=proof['replay'],
            marker=str(marker), marker_sha256=sha(marker), audit=accepted['audit'], audit_sha256=accepted['audit_sha256'])))
    print('ManiSkill: Clean5000 + B/C/O/D/M1000 each + hard10000 = 20000', flush=True)

    audit_path = Path(roots['robotwin']) / 'TRANSFER_AUDIT/ACCEPTED_WRIST020_20000_SCHEMA5.json'
    audit = read_json(audit_path)
    if audit['status'] != 'passed' or audit['episodes'] != 20000:
        raise ValueError('Need the accepted RoboTwin wrist020 collection')
    paths = []
    for current, dirs, files in os.walk(Path(roots['robotwin']) / 'raw'):
        dirs[:] = [x for x in dirs if x not in ('.cache', '_traj_data', 'video', 'instructions')]
        if Path(current).name == 'data':
            paths.extend(Path(current) / name for name in files if name.startswith('episode') and name.endswith('.hdf5'))
            dirs[:] = []
    if len(paths) != 20000:
        raise ValueError(f'Expected 20000 accepted RoboTwin raw episodes, got {len(paths)}')
    audit_hash = sha(audit_path)
    for path in sorted(paths):
        parts = path.relative_to(Path(roots['robotwin']) / 'raw').parts
        records.append(identify(dict(collection='robotwin_base', group='robotwin_' + parts[0], task=parts[-4],
            split=None, family_id=str(path), domain='/'.join(parts[2:-4]), source_h5=str(path),
            marker=str(audit_path), marker_sha256=audit_hash)))
    clutter = marker_records(Path(roots['clutter']) / 'run/state', '*.json')
    if len(clutter) != 4000:
        raise ValueError(f'Expected 4000 clutter states, got {len(clutter)}')
    for marker, state in clutter:
        if state['status'] != 'passed' or state['returncode'] != 0:
            raise ValueError(f'Unaccepted clutter candidate: {marker}')
        folder = Path(state['attempt_root'])
        proof = read_json(folder / 'FRESH_RANDOMIZED_ACCEPTED.json')
        sources = list((folder / 'raw/data').glob('episode*.hdf5'))
        if len(sources) != 1:
            raise ValueError(f'Clutter trajectory inventory changed: {folder}')
        robot = 'aloha' if proof['embodiment'].startswith('aloha') else proof['embodiment']
        records.append(identify(dict(collection='robotwin_clutter', group='robotwin_' + robot, task=proof['task'],
            split=None, family_id=str(folder), domain='fresh_clutter_randomized', source_h5=str(sources[0]),
            marker=str(marker), marker_sha256=sha(marker))))
    if len({x['id'] for x in records}) != len(records):
        raise ValueError('Duplicate episode identity')
    return records


def robotwin_numeric(file, record):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from openwam.dataloader.ea_rsig_lite import targets_from_oracle
    from openwam.multisim_sidecar_contract import decode_rgb

    values = []
    for arm in ('left', 'right'):
        pose = np.asarray(file[f'endpose/{arm}_endpose'], np.float32)
        q = pose[:, [4, 5, 6, 3]]
        if not np.allclose(np.linalg.norm(q, axis=1), 1, atol=.01):
            raise ValueError('RoboTwin source requires unit wxyz quaternion')
        matrix = Rotation.from_quat(q).as_matrix()
        grip = np.asarray(file[f'endpose/{arm}_gripper']).reshape(-1, 1)
        values.append(np.concatenate((pose[:, :3], matrix[:, :, 0], matrix[:, :, 1], grip), axis=1))
    state = np.concatenate(values, axis=1).astype(np.float32)
    cameras = [f'observation/{c}/rgb' for c in ('head_camera', 'left_camera', 'right_camera')]
    markers = [b'XPL-RGB1' in bytes(file[c][0]) for c in cameras]
    if len(set(markers)) != 1:
        raise ValueError('Mixed camera JPEG color contracts')
    codec = 'standard_rgb_jpeg' if markers[0] else 'robotwin_legacy_rgb_jpeg'
    sizes = [decode_rgb(file[c][0], codec).size for c in cameras]
    oracle = {k: np.asarray(v) for k, v in file['ea_rsig_oracle_v3'].items()}
    targets = targets_from_oracle(oracle, sizes, record['task'])[:-1]
    # A non-spatial goal must never receive spatial grounding supervision.
    targets[:, 121:124] *= targets[:, 14:15]
    targets[:, 127:130] *= targets[:, 15:16]
    instructions = Path(record['source_h5']).parent.parent / 'instructions' / (Path(record['source_h5']).stem + '.json')
    prompts = read_json(instructions).get('seen', [])
    if not prompts:
        raise ValueError(f'Missing training instruction: {instructions}')
    family = record['group'] + '/' + record['task'] + '/' + hashlib.sha256(state.tobytes()).hexdigest()
    arms = np.asarray(oracle['arm_present'])
    if not (arms == arms[0]).all():
        raise ValueError('Observed robot embodiment cannot switch within one episode')
    return state, state[1:].copy(), targets, arms[0].astype(bool), cameras, codec, prompts[0], family


def maniskill_numeric(file, record):
    import numpy as np
    from openwam.maniskill_eef_contract import absolute_command_targets, achieved_states
    from openwam.multisim_sidecar_contract import pad_single_arm, segmentation_boxes, single_arm_targets

    metadata = read_json(Path(record['source_h5']).with_suffix('.json'))
    cfg = metadata['env_info']['env_kwargs']
    if cfg['control_mode'] != 'pd_ee_delta_pose' or cfg['sim_backend'] != 'physx_cpu':
        raise ValueError('Unsupported ManiSkill command controller or physics backend')
    controller = read_json(Path(record['source_h5']).parent / 'TARGET_CONTROLLER.json')
    expected = dict(pos_lower=-.1, pos_upper=.1, rot_lower=-.1, rot_upper=.1,
                    frame='root_translation:root_aligned_body_rotation', normalize_action=True,
                    use_delta=True, use_target=False, ee_link='panda_hand_tcp')
    if any(controller.get(k) != v for k, v in expected.items()):
        raise ValueError('ManiSkill controller differs from the audited absolute conversion')
    tr = file['traj_0']
    if not bool(np.asarray(tr['success']).any()):
        raise ValueError('Unsuccessful source replay')
    extra = tr['obs/extra']
    tcp, base, obj, goal = (np.asarray(extra[k]) for k in ('tcp_pose', 'robot_base_pose', 'obj_pose', 'goal_pos'))
    state = pad_single_arm(achieved_states(tcp, base, np.asarray(tr['obs/agent/qpos'])))
    native_actions = np.asarray(tr['actions'])
    with np.load(Path(record['source_h5']).parent / 'backup_actions.npz', allow_pickle=False) as backup:
        np.testing.assert_array_equal(native_actions, backup['pd_ee_delta_pose_command7'])
        times = backup['replay_time_seconds']
        if len(times) != len(native_actions) + 1 or not np.allclose(np.diff(times), .05, atol=2e-5):
            raise ValueError('ManiSkill replay time axis is not the verified 20Hz command axis')
    action = pad_single_arm(absolute_command_targets(tcp[:-1], base[:-1], native_actions))
    n = len(tcp)
    ids, gids = (np.asarray(extra[k]).reshape(n) for k in ('obj_segmentation_id', 'goal_segmentation_id'))
    boxes, visible = np.zeros((n, 2, 2, 4), np.float32), np.zeros((n, 2, 2), np.float32)
    cameras = [f'traj_0/obs/sensor_data/{c}/rgb' for c in ('front_camera', 'hand_camera')]
    for view, name in enumerate(('front_camera', 'hand_camera')):
        sensor = tr[f'obs/sensor_data/{name}']
        if sensor['rgb'].shape != (n, 256, 256, 3) or sensor['rgb'].dtype != np.uint8:
            raise ValueError('Unexpected native ManiSkill RGB geometry')
        boxes[:, :, view], visible[:, :, view] = segmentation_boxes(sensor['segmentation'], ids, gids)
    force = np.asarray(extra['robot_obj_contact_force_norm']).reshape(n) if 'robot_obj_contact_force_norm' in extra else None
    if force is not None and (not np.isfinite(force).all() or (force < 0).any()):
        raise ValueError('Invalid physical contact-force norm')
    contact = force > 1e-6 if force is not None else None
    gvalid = np.asarray(extra['franka20k_spatial_goal_valid']).reshape(n)
    targets = single_arm_targets(tcp, obj, goal, ids, contact, boxes, visible, gvalid)
    colors = ('red', 'blue', 'green', 'yellow', 'orange', 'purple')
    color_id = int(extra['target_color_id'][0, 0])
    descriptions = dict(push_cube='push the {color} cube to the target region',
        pick_cube='pick up the {color} cube and move it to the visible goal',
        pick_cuboid='pick up the {color} cuboid and move it to the visible goal',
        pick_sphere='pick up the {color} sphere and move it to the visible goal',
        stack_cube='stack the {color} cube on the gray cube', place_sphere='place the {color} sphere in the bin',
        pull_cube='pull the {color} cube to the target region', lift_peg_upright='stand the peg upright on the table')
    if record['task'] != 'lift_peg_upright' and not 0 <= color_id < len(colors):
        raise ValueError('Unknown target color')
    prompt = descriptions[record['task']].format(color='' if record['task'] == 'lift_peg_upright' else colors[color_id])
    return state, action, targets, np.array([True, False]), cameras, 'uint8_rgb', prompt, record['family_id']


def _prepare_episode(payload):
    record, destination = payload
    import h5py
    import numpy as np
    from openwam.multisim_sidecar_contract import decode_rgb, pad_single_arm, validate_numeric

    out = Path(destination)
    verify_evidence(record['marker'], record['marker_sha256'])
    numeric = out / 'numeric' / record['group'] / (record['id'] + '.npz')
    proof_path = numeric.with_suffix('.json')
    if proof_path.exists():
        proof = read_json(proof_path)
        if proof['id'] != record['id'] or sha(numeric) != proof['numeric_sha256']:
            raise ValueError(f'Prepared artifact changed: {numeric}')
        stat = (out / proof['media']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (proof['media_size'], proof['media_mtime_ns']):
            raise ValueError('Prepared source changed during resume')
        if proof.get('preparation_sha256') == BUILD_FINGERPRINT:
            return proof
    source = Path(record['source_h5']).resolve()
    stat = source.stat()
    with h5py.File(source, 'r') as file:
        if record['collection'] == 'rlbench':
            path = Path(record['source_numeric'])
            meta = read_json(path.with_suffix('.json'))
            if meta['source_oracle_sha256'] != sha(ROOT / 'openwam/dataloader/ea_rsig_lite.py') or sha(path) != meta['numeric_sha256']:
                raise ValueError('RLBench cached130D source or oracle definition changed')
            if meta['plan_steps'] != [8, 16, 32] or meta['video_anchors'] != list(range(0, 33, 4)):
                raise ValueError('Wrong RLBench temporal contract')
            with np.load(path, allow_pickle=False) as data:
                state, action, targets = pad_single_arm(data['state']), pad_single_arm(data['action']), data['targets']
            np.testing.assert_allclose(state[:, :10], np.asarray(file['observation/effector_state10']), atol=1e-6)
            np.testing.assert_allclose(action[:, :10], np.asarray(file['action/absolute_base_eef10']), atol=1e-6)
            if not np.allclose(np.diff(file['observation/time'][:]), .05, atol=2e-5):
                raise ValueError('RLBench native command time axis is not 20Hz')
            arms = np.array([True, False])
            cameras = [f'domains/{record["domain"]}/{c}/rgb' for c in ('front', 'wrist')]
            codec, family = 'uint8_rgb', record['family_id']
            prompt = read_json(path.parent / 'meta/tasks.jsonl')['task']
        elif record['collection'].startswith('maniskill'):
            state, action, targets, arms, cameras, codec, prompt, family = maniskill_numeric(file, record)
        else:
            state, action, targets, arms, cameras, codec, prompt, family = robotwin_numeric(file, record)
        state, action, targets, arms = validate_numeric(state, action, targets, arms)
        for camera in cameras:
            images = file[camera]
            if not os.path.samefile(images.file.filename, source):
                raise ValueError('Raw camera payload uses external HDF5 links; materialize it before release')
            if len(images) != len(state):
                raise ValueError('RGB and observed-state time axes differ')
            for t in (0, len(state) // 2, len(state) - 1):
                decode_rgb(images[t], codec)
        if source.stat().st_mtime_ns != stat.st_mtime_ns or source.stat().st_size != stat.st_size:
            raise ValueError('Source changed while preparing')
    media = out / 'media' / (hashlib.sha256(str(source).encode()).hexdigest() + source.suffix)
    media.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, media)
    except FileExistsError:
        if not os.path.samefile(source, media):
            raise ValueError('Canonical media identity collision')
    # These hardlinks are only opened read-only; chmod or writes would affect the original.
    numeric.parent.mkdir(parents=True, exist_ok=True)
    temporary = numeric.with_suffix('.npz.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, state=state, action=action, targets=targets, arm_present=arms)
    temporary.replace(numeric)
    proof = dict(record, family_id=family, numeric=str(numeric.relative_to(out)), numeric_sha256=sha(numeric),
                 media=str(media.relative_to(out)), media_size=stat.st_size, media_mtime_ns=stat.st_mtime_ns,
                 camera_paths=cameras, rgb_codec=codec, prompt=prompt, steps=len(action),
                 arm_present=arms.astype(int).tolist(),
                 action_semantics='next_achieved_absolute_eef' if record['collection'].startswith('robotwin') else 'absolute_controller_target',
                 action_frame='robotwin_endpose_reference' if record['collection'].startswith('robotwin') else 'robot_base_tcp',
                 source_numeric_reused=record['collection'] == 'rlbench', preparation_sha256=BUILD_FINGERPRINT,
                 control_hz=None if record['collection'].startswith('robotwin') else 20)
    save(proof_path, proof)
    return proof


def prepare_episode(payload):
    try:
        return _prepare_episode(payload)
    except Exception as exc:
        record, destination = payload
        save(Path(destination) / 'errors' / (record['id'] + '.json'),
             dict(id=record['id'], source=record['source_h5'], error=repr(exc), traceback=traceback.format_exc()))
        raise


def initialize_worker(destination):
    import faulthandler
    import signal
    global _trace_stream
    _trace_stream = Path(destination, f'worker_{os.getpid()}.trace.log').open('a')
    faulthandler.register(signal.SIGUSR1, file=_trace_stream, all_threads=True)


def finalize(records, out, smoke):
    import numpy as np
    from openwam.multisim_sidecar_contract import CONTRACT, PLAN_STEPS, VIDEO_ANCHORS
    from openwam.dataloader.utils.stats_computation.robotwin_stats_computation import _ModeAccumulator
    from openwam.dataloader.utils.normalization import pin_rot6d_identity

    families = defaultdict(set)
    for record in records:
        if record.get('preparation_sha256') not in (None, BUILD_FINGERPRINT):
            raise ValueError('Preparation code changed during the build; resume with one frozen revision')
        if record['split'] is None:
            families[(record['group'], record['task'])].add(record['family_id'])
    assignments = {}
    for key, items in families.items():
        ordered = sorted(items, key=lambda x: hashlib.sha256(('42/' + x).encode()).digest())
        n = len(ordered)
        train, val = max(1, int(.8 * n)), max(1, int(.1 * n))
        for i, family in enumerate(ordered):
            assignments[family] = 'train' if i < train else 'val' if i < train + val else 'test'
    seen = {}
    accumulators = [_ModeAccumulator(10), _ModeAccumulator(10)]
    for record in records:
        record['split'] = record['split'] or assignments[record['family_id']]
        previous = seen.setdefault(record['family_id'], record['split'])
        if previous != record['split']:
            raise ValueError('Physical family leaked across splits')
        if record['split'] == 'train':
            with np.load(out / record['numeric'], allow_pickle=False) as data:
                for arm, present in enumerate(record['arm_present']):
                    if present:
                        for key in ('state', 'action'):
                            accumulators[arm].update(data[key][:, 10 * arm:10 * (arm + 1)].copy())
    pieces = [a.finalize() for a in accumulators]
    stats = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    pin_rot6d_identity(stats, list(range(3, 9)) + list(range(13, 19)))
    stats['min'][[9, 19]], stats['max'][[9, 19]] = 0, 1
    stats_path = out / 'normalization_stats.npy'
    np.save(stats_path, {'eef': stats})
    manifest = dict(contract=CONTRACT, ready=True, release_kind='smoke' if smoke else 'accepted_available_sources',
        created_utc=datetime.now(timezone.utc).isoformat(), episodes=records,
        preparation_sha256=BUILD_FINGERPRINT,
        counts=dict(Counter(x['collection'] for x in records)), splits=dict(Counter(x['split'] for x in records)),
        group_counts=dict(Counter(x['group'] for x in records)),
        domain_counts=dict(Counter(x['collection'] + '/' + x['domain'] for x in records)),
        independent_physical_families=len(seen),
        total_episodes=len(records), total_windows=sum(x['steps'] for x in records if x['split'] == 'train'),
        normalization_stats='normalization_stats.npy', normalization_stats_sha256=sha(stats_path),
        normalization_source='train_only_present_arm_state_and_command', plan_steps=list(PLAN_STEPS),
        video_anchors=list(VIDEO_ANCHORS), physical_family_no_split_leak=True,
        media_policy='hardlinked immutable accepted HDF5; dataset is movable as a whole; never mutate media',
        rlbench_collection_target=20000, rlbench_available=sum(x['collection'] == 'rlbench' for x in records))
    save(out / 'manifest.ready.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'episodes'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key, default in DEFAULTS.items():
        parser.add_argument('--' + key, default=default)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--sample-per-source', type=int, default=0)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.ready.json').exists():
        raise FileExistsError('Release already finalized; use a new version directory')
    plan = out / 'selection.json'
    if plan.exists():
        selected = read_json(plan)
    else:
        selected = inventory({key: getattr(args, key) for key in DEFAULTS})
        if args.sample_per_source:
            groups = defaultdict(list)
            for record in selected:
                groups[(record['collection'], record['group'])].append(record)
            selected = [record for group in groups.values()
                        for record in [r for r in group if r['split'] in (None, 'train')][:args.sample_per_source]]
        save(plan, selected)
    print('Frozen selection:', len(selected), dict(Counter(x['collection'] for x in selected)), flush=True)
    if args.inventory_only:
        return
    completed = []
    # Fresh processes avoid inherited GPFS/HDF5 native locks; completion order must not affect release order.
    with multiprocessing.get_context('spawn').Pool(args.workers, initializer=initialize_worker, initargs=(str(out),)) as pool:
        for index, record in enumerate(pool.imap_unordered(prepare_episode, ((r, str(out)) for r in selected), chunksize=1), 1):
            completed.append(record)
            if index % 100 == 0 or index == len(selected):
                save(out / 'progress.json', dict(completed=index, selected=len(selected), phase='numeric_media_preparation', time=datetime.now(timezone.utc).isoformat()))
                print(f'Prepared {index}/{len(selected)}', flush=True)
    positions = {record['id']: index for index, record in enumerate(selected)}
    completed.sort(key=lambda record: positions[record['id']])
    finalize(completed, out, bool(args.sample_per_source))


if __name__ == '__main__':
    main()
