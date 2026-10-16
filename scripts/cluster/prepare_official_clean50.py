#!/usr/bin/env python3
"""Audit the downloaded official Aloha Clean50; fit stats without modifying HDF5."""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def prepare(raw_root, output_dir):
    import h5py
    import numpy as np
    from PIL import Image

    from openwam.dataloader.ea_rsig_lite import CAMERAS, LiteRoboTwinSource
    from openwam.dataloader.robotwin import ROBOTWIN_ALL_TASKS
    from openwam.dataloader.utils.stats_computation.robotwin_stats_computation import (
        _ModeAccumulator,
        _pin_eef_rot6d_identity,
    )

    raw_root, output_dir = Path(raw_root).resolve(), Path(output_dir).resolve()
    source_root = raw_root / 'raw/aloha-agilex'
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    accumulator = _ModeAccumulator(20)
    entries = []
    for task in sorted(ROBOTWIN_ALL_TASKS):
        task_root = source_root / task / 'aloha-agilex_clean_50/aloha-agilex_clean_50'
        files = sorted((task_root / 'data').glob('episode*.hdf5'))
        if len(files) != 50:
            raise ValueError(f'Need exactly 50 official Clean trajectories: {task}: {len(files)}')
        episodes, image_size = [], None
        for index, path in enumerate(files):
            before = path.stat()
            with h5py.File(path, 'r') as f:
                # Official RoboTwin serializes SAPIEN Pose.q as wxyz. Reuse the
                # existing explicit conversion, never the old xyzw assumption.
                eef = LiteRoboTwinSource._read_eef_actions(None, f, 0, len(f['endpose/left_endpose']))
                if not np.isfinite(eef).all() or eef.shape[1] != 20:
                    raise ValueError(f'Invalid EEF: {path}')
                if np.any((eef[:, [9, 19]] < -1e-5) | (eef[:, [9, 19]] > 1 + 1e-5)):
                    raise ValueError(f'Unexpected gripper range: {path}')
                length = len(eef)
                for camera in CAMERAS:
                    if len(f[f'observation/{camera}/rgb']) != length:
                        raise ValueError(f'Camera/action length mismatch: {path}')
                if index == 0:
                    with Image.open(io.BytesIO(bytes(f['observation/head_camera/rgb'][0]))) as image:
                        image_size = [image.height, image.width]
                accumulator.update(eef)
            if (path.stat().st_size, path.stat().st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                raise ValueError(f'Source changed during audit: {path}')
            episodes.append(dict(path=str(path), frames=length, size=before.st_size, mtime_ns=before.st_mtime_ns))
        instructions = {p.name: json.loads(p.read_text()) for p in sorted((task_root / 'instructions').glob('*.json'))}
        scene = task_root / 'scene_info.json'
        entries.append(dict(name=task, data_root=str(task_root / 'data'), episodes=episodes,
                            startup_metadata=dict(version=1, obs_image_size=image_size,
                                                  instructions=instructions,
                                                  scene_info=json.loads(scene.read_text()) if scene.exists() else {})))
        print(f'Official Clean metadata: {task}: 50 trajectories; instructions={len(instructions)}', flush=True)
    stats_path = output_dir / 'normalization_stats.npy'
    np.save(stats_path, {'eef': _pin_eef_rot6d_identity(accumulator.finalize())})
    manifest = dict(contract='openwam_alpha_eef32_ea_lite_v1', split='Clean',
                    source_root=str(source_root), source_quaternion_order='wxyz',
                    ea_context='observed_endpose_plus_local_x_0.12m_TCP',
                    auxiliary_labels_available=False, dataset_source='official_download',
                    official_inventory=json.loads((raw_root / 'inventory.json').read_text()),
                    action_semantics='absolute_eef_t_plus_1_to_t_plus_32',
                    normalization_stats_path=str(stats_path),
                    normalization_stats_sha256=hashlib.sha256(stats_path.read_bytes()).hexdigest(),
                    tasks=entries, total_episodes=sum(len(t['episodes']) for t in entries),
                    total_frames=sum(ep['frames'] for t in entries for ep in t['episodes']))
    target = output_dir / 'manifest.ready.json'
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(manifest=str(target), total_episodes=manifest['total_episodes'],
                          total_frames=manifest['total_frames'],
                          windows=manifest['total_frames']-manifest['total_episodes'],
                          source_bytes=sum(ep['size'] for t in entries for ep in t['episodes']),
                          ea_labels=False), indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    prepare(**vars(p.parse_args()))
