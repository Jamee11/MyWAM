"""Canonical recipe, temporal consistency and the actual PIL camera bridge."""
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from openwam.dataloader.transforms.patchwam_appearance import AppearanceRandomizer
from openwam.dataloader.transforms.patchwam_video import PatchWAMVideoAugment

ROOT = Path(__file__).resolve().parents[1]
RECIPE = yaml.safe_load((ROOT / 'configs/train_relational_sidecar_aloha50_patchwam_c2r_10epochs.yaml').read_text())[
    'dataloader']['patchwam_augmentation']


def test_vendored_algorithm_is_byte_exact_upstream_and_recipe_enables_all_stages():
    source = ROOT / 'openwam/dataloader/transforms/patchwam_appearance.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == '1d9719cd379bca55085938183f4d6dbb068b07f1015225ad38cb5d1f653acf27'
    assert RECIPE['p'] == .8
    assert RECIPE['style']['enabled'] and RECIPE['fourier']['enabled']
    assert not RECIPE['background']['enabled']
    assert RECIPE['photometric']['hue'] == .1


@pytest.mark.parametrize('seed', [0, 2, 6, 11])
def test_per_camera_temporal_consistency_range_and_seed_reproducibility(seed):
    frame = torch.linspace(.05, .95, 3 * 16 * 16).reshape(1, 3, 16, 16)
    views = [frame.repeat(9, 1, 1, 1) for _ in range(3)]
    randomizer = AppearanceRandomizer(**{k: v for k, v in RECIPE.items() if k != 'enabled'})
    torch.manual_seed(seed)
    result = randomizer(views)
    torch.manual_seed(seed)
    again = randomizer(views)
    for clip, repeated in zip(result, again):
        assert clip.shape == (9, 3, 16, 16)
        assert torch.isfinite(clip).all() and 0 <= clip.min() <= clip.max() <= 1
        assert torch.equal(clip, repeated)
        assert all(torch.equal(clip[0], t) for t in clip)


def make_sample():
    frame = Image.fromarray(np.arange(48 * 32 * 3, dtype=np.uint8).reshape(48, 32, 3))
    video = [frame.copy() for _ in range(9)]
    return dict(video=video, first_frame_image=[video[0]], action=torch.randn(32, 80),
                ea_targets=torch.randn(1, 130), prompt='move the object')


def test_pil_adapter_matches_native_per_camera_randomizer_and_keeps_supervision():
    sample = make_sample()
    original = dict(sample)
    adapter = PatchWAMVideoAugment(dict(RECIPE, p=1))
    boxes = [(0, 0, 32, 32), (0, 32, 16, 48), (16, 32, 32, 48)]
    views = [torch.from_numpy(np.stack([np.asarray(f.crop(box)) for f in sample['video']]))
             .permute(0, 3, 1, 2).float() / 255 for box in boxes]
    torch.manual_seed(42)
    expected = adapter.randomizer(views)
    torch.manual_seed(42)
    result = adapter(sample, [True, True, True])
    for box, clip in zip(boxes, expected):
        pixels = clip.mul(255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
        assert all(np.array_equal(np.asarray(frame.crop(box)), row) for frame, row in zip(result['video'], pixels))
    assert result['first_frame_image'][0] is result['video'][0]
    for key in ('action', 'ea_targets', 'prompt'):
        assert result[key] is original[key]
    assert all(frame.size == (32, 48) for frame in result['video'])
    assert not np.array_equal(np.asarray(original['video'][0]), np.asarray(result['video'][0]))


def test_skipped_sample_has_no_quantization_and_does_not_replace_first_frame():
    sample = make_sample()
    video, conditioning = sample['video'], sample['first_frame_image']
    adapter = PatchWAMVideoAugment(dict(RECIPE, p=0))
    assert adapter(sample, [True, True, True]) is sample
    assert sample['video'] is video and sample['first_frame_image'] is conditioning


def test_missing_slot_is_not_styled_and_real_black_camera_is_not_masked():
    frames = [Image.new('RGB', (32, 48), 'black') for _ in range(9)]
    sample = dict(video=frames, first_frame_image=[frames[0]])
    adapter = PatchWAMVideoAugment(dict(RECIPE, p=1))
    torch.manual_seed(42)
    result = adapter(sample, [True, True, False])
    image = np.asarray(result['video'][0])
    assert image[:32].max() > 0  # Real camera black pixels remain augmentable.
    assert not image[32:, 16:].any()
    with pytest.raises(ValueError, match='at least one real camera'):
        adapter(sample, [False, False, False])


@pytest.mark.parametrize('unsupported', [dict(background=dict(enabled=True)), dict(rotate=dict(degrees=5))])
def test_background_and_geometry_are_not_silently_enabled(unsupported):
    with pytest.raises((NotImplementedError, TypeError)):
        PatchWAMVideoAugment(dict(RECIPE, **unsupported))


def test_launcher_preserves_allocated_gpus_and_uses_new_config(tmp_path):
    binaries = tmp_path / 'venv/bin'
    binaries.mkdir(parents=True)
    python = binaries / 'python'
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$CUDA_VISIBLE_DEVICES" "$@"\n')
    python.chmod(0o755)
    env = dict(os.environ, OPENWAM_VENV=str(binaries.parent), CUDA_VISIBLE_DEVICES='GPU-a,GPU-b,GPU-c,GPU-d',
               CHECK_ONLY='1', OPENWAM_OUTPUT=str(tmp_path / 'independent_output'))
    result = subprocess.check_output(['bash', str(ROOT / 'scripts/train_sidecar_patchwam_c2r_4gpu.sh')],
                                     env=env, text=True)
    assert result.splitlines()[0] == env['CUDA_VISIBLE_DEVICES']
    assert '--config-name=train_relational_sidecar_aloha50_patchwam_c2r_10epochs' in result
    assert f'training.output_path={tmp_path / "independent_output"}' in result
    assert '--cfg\njob\n--resolve' in result
