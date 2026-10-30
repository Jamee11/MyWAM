"""Appearance-only changes are deterministic, geometry preserving and side-effect free."""
import numpy as np
import pytest
from PIL import Image

from benchmarks.robotwin.sidecar_appearance_diagnostics import CONDITIONS, feature_distance, perturb


@pytest.mark.parametrize('condition', CONDITIONS[:4])
def test_deterministic_independent_images(condition):
    pixels = np.random.default_rng(1).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    left, right = perturb(image, condition, 19), perturb(image, condition, 19)
    assert left.size == image.size and left.mode == 'RGB'
    np.testing.assert_array_equal(np.asarray(image), pixels)
    np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    assert left is not image
    assert np.array_equal(np.asarray(left), pixels) == (condition == 'clean')


def test_sensor_seed_and_feature_distance():
    image = Image.fromarray(np.full((48, 64, 3), 127, np.uint8))
    assert not np.array_equal(np.asarray(perturb(image, 'sensor_strong', 1)),
                              np.asarray(perturb(image, 'sensor_strong', 2)))
    x = np.array([[1., 2.], [0., 0.]])
    d = feature_distance(x, x)
    assert d['relative_l2'] == 0 and d['cosine'] == pytest.approx(1) and d['valid_tokens'] == 1
    with pytest.raises(ValueError):
        feature_distance(x, np.full_like(x, np.nan))
    with pytest.raises(ValueError):
        perturb(image, 'move_object', 1)


def test_capture_preserves_forward_values():
    import torch
    from types import SimpleNamespace
    from benchmarks.robotwin.sidecar_appearance_diagnostics import Capture
    heads = torch.nn.Identity()
    hidden = torch.ones(1, 32, 4)
    driver = SimpleNamespace(step=lambda layer: (None, SimpleNamespace(payload=SimpleNamespace(x_action=hidden))))
    aux = dict(tokens=torch.ones(1, 15, 4), grid=(1, 2),
               grounding=dict(attention=torch.full((1, 4, 2), .5), presence_logits=torch.zeros(1, 4)),
               track_valid_logits=torch.zeros(1, 2), near_logits=torch.zeros(1, 2),
               contact_logits=torch.zeros(1, 2), goal_proximity=torch.zeros(1, 2),
               plan_directions=torch.zeros(1, 3, 2, 2, 3), plan_moving_logits=torch.zeros(1, 3, 2, 2))

    def forward(actions, action_time, **kwargs):
        for layer in (8, 20):
            driver.step(layer)
        for _ in range(3):
            heads(aux)
        return torch.zeros(1), actions + 1

    arch = SimpleNamespace(mot_driver=driver, forward=forward, sidecar=SimpleNamespace(heads=heads), tap_blocks=(7, 15, 23))
    capture = Capture(arch)
    capture.clear(np.zeros(130), np.ones(2))
    actions = torch.randn(1, 32, 80)
    result = arch.forward(actions, torch.ones(1), timestep=torch.ones(1), latents=torch.ones(1, 1, 3, 1, 1))
    torch.testing.assert_close(result[1], actions + 1, rtol=0, atol=0)
    assert set(capture.action) == {'8', '20'} and len(capture.sidecar) == 3
    np.testing.assert_array_equal(capture.velocity, (actions + 1)[0].numpy())
    assert 'video_future' in capture.noise
