"""CPU check: actual archive first-frame canvas equals the deployed observation canvas."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from omegaconf import OmegaConf
    import numpy as np
    from openwam.dataloader.multisim_sidecar import PreparedSimDataset
    from openwam.deploy.obs_preprocess import ObsPreprocessor
    from openwam.deploy.server import merge_deploy_cfg
    from benchmarks.robotwin.piper_protocol import DEFAULT_CHECKPOINT, DEFAULT_MANIFEST

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--deploy-config', type=Path, default=ROOT/'configs/deploy_piper_eval.yaml')
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True,
                        help='raw_contract_checks.json identifying each fixture source record, in order')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    training = OmegaConf.load(args.checkpoint.parent/'config.yaml')
    deploy = OmegaConf.load(args.deploy_config)
    preprocessor = ObsPreprocessor.from_cfg(merge_deploy_cfg(training, deploy))
    if not preprocessor.multiview or preprocessor.camera_layout != ['0', '1', '2']:
        raise ValueError('Piper must use the PreparedSimDataset three-camera layout')
    data = PreparedSimDataset(args.manifest, 'robotwin_piper', auxiliary=False)
    fixtures = json.loads(args.fixtures.read_text())
    references = json.loads(args.references.read_text())['records']
    if len(fixtures) != len(references):
        raise ValueError('Fixture/source reference count differs')
    records = []
    for fixture, reference in zip(fixtures, references):
        episode = next(i for i, record in enumerate(data.records) if record['id'] == reference['record_id'])
        index = data.ends[episode-1] if episode else 0
        sample = data[index]  # Actual unaugmented training reader, including legacy JPEG decoder.
        if sample['prompt'] != fixture['prompt']:
            raise ValueError('Fixture does not match the referenced training episode')
        expected = np.asarray(sample['first_frame_image'][0])
        actual = np.asarray(preprocessor.preprocess(copy.deepcopy(fixture))['image'])
        difference = np.abs(actual.astype(np.int16)-expected.astype(np.int16))
        old = ObsPreprocessor(multiview=False, camera_layout=preprocessor.camera_layout,
                              img_height=384, img_width=320)
        old_image = np.asarray(old.preprocess(copy.deepcopy(fixture))['image'])
        records.append(dict(task=reference['task'], domain=reference['domain'],
            record_id=reference['record_id'], shape=list(actual.shape),
            identical=bool(np.array_equal(actual, expected)), max_abs=int(difference.max()),
            different_values=int(np.count_nonzero(difference)),
            canvas_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
            rejected_single_view_different_values=int(np.count_nonzero(old_image != expected))))
    result = dict(passed=all(row['identical'] for row in records), multiview=preprocessor.multiview,
        camera_layout=preprocessor.camera_layout,
        deploy_config_sha256=hashlib.sha256(args.deploy_config.read_bytes()).hexdigest(),
        records=records, scope='Two actual unaugmented archive first frames; CPU preprocessing only, no GPU inference or rollout')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
