#!/usr/bin/env python3
"""Fast prepared-data startup check; opt-in full artifact audit stays offline."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(manifest, full=False, allow_smoke=False):
    import numpy as np

    from openwam.dataloader.multisim_sidecar import PreparedSimDataset
    from openwam.multisim_sidecar_contract import SOURCE_GROUPS

    manifest = Path(manifest).resolve()
    data = json.loads(manifest.read_text())
    results = []
    for group in SOURCE_GROUPS:
        dataset = PreparedSimDataset(manifest, group, allow_smoke=allow_smoke)
        for index in sorted({0, len(dataset) // 2, len(dataset) - 1}):
            sample = dataset[index]
            if sample['action'].shape != (32, 80) or sample['proprio'].shape != (1, 80):
                raise ValueError('Wrong model action/state contract')
            if sample['ea_targets'].shape != (1, 130) or len(sample['video']) != 9:
                raise ValueError('Wrong Sidecar/video contract')
            if any(image.size != (320, 384) for image in sample['video']):
                raise ValueError('Wrong L-canvas geometry')
            for key in ('action', 'proprio', 'ea_targets'):
                if not np.isfinite(sample[key].numpy()).all():
                    raise ValueError('Nonfinite training input')
            inactive = ~sample['proprio_mask'][0]
            if sample['action'][:, inactive].any() or sample['proprio'][:, inactive].any():
                raise ValueError('Absent/unified-padding dimensions must be zero')
        results.append(dict(group=group, episodes=len(dataset.records), windows=len(dataset), checked_anchors=3))
    if full:
        for i, record in enumerate(data['episodes'], 1):
            path = manifest.parent / record['numeric']
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['numeric_sha256']:
                raise ValueError(f'Numeric hash mismatch: {path}')
            stat = (manifest.parent / record['media']).stat()
            if (stat.st_size, stat.st_mtime_ns) != (record['media_size'], record['media_mtime_ns']):
                raise ValueError('Media changed')
            if i % 1000 == 0:
                print(f'Audited {i}/{len(data["episodes"])}', flush=True)
    result = dict(manifest=str(manifest), sources=results, scope='all_numeric_hashes_and_media_stats' if full else 'prepared_manifest_and_12_training_anchors',
                  source_counts=data['counts'], splits=data['splits'], total_episodes=data['total_episodes'],
                  gpu_model_forward_tested=False, formal_training_started=False)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args()
    check(args.manifest, args.full, args.allow_smoke)
