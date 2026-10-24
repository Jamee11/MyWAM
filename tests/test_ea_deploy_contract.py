"""Deployment adapter checks; model generation is covered by the full roundtrip."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from openwam.deploy import engine as engine_module
from openwam.deploy.obs_preprocess import ObsPreprocessor
from openwam.deploy.policy import WAMPolicy


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(engine_module, "make_schedule", lambda *a, **k: [(1000, 1000), (0, 0)])
    class Architecture:
        ea_bridge = object()
        device = torch.device("cpu")
        dtype = torch.float32
        video_scheduler = action_scheduler = object()
        def normalize_deploy_proprio(self, value):
            return torch.as_tensor(value)
        def generate(self, **kwargs):
            self.inputs = kwargs
            return {"actions": np.zeros((40, 14)), "video": None}
    cfg = SimpleNamespace(inference=SimpleNamespace(denoise_steps=2, num_frames=41, video_num_frames=5))
    return engine_module.JointInferenceEngine(cfg, architecture=Architecture())


def observation():
    return {"images": {k: Image.new("RGB", (320, 240), c) for k, c in (
        ("head_camera", "red"), ("left_wrist_camera", "green"), ("right_wrist_camera", "blue"))},
        "prompt": "task", "state": np.zeros(14), "ea_effectors": np.arange(20).reshape(2, 10),
        "ea_arm_present": [1, 1]}


def test_preprocessed_current_context_reaches_engine(engine):
    obs = observation()
    processed = ObsPreprocessor(multiview=True, camera_layout=["head", "left", "right"],
                                img_height=384, img_width=320, requires_proprio=True).preprocess(obs)
    image = processed["image"]
    assert image.size == (320, 384)
    assert image.getpixel((10, 10)) == (255, 0, 0)
    assert image.getpixel((10, 300)) == (0, 128, 0)
    assert image.getpixel((200, 300)) == (0, 0, 255)
    policy = WAMPolicy.__new__(WAMPolicy)
    engine.generate(policy._build_conditions(processed))
    actual = engine.architecture.inputs
    torch.testing.assert_close(actual["ea_effectors"], torch.arange(20).float().reshape(1, 2, 10))
    assert actual["ea_arm_present"].shape == (1, 2)
    assert actual["num_frames"] == 5 and actual["action_num_frames"] == 41
    assert "ea_targets" not in actual


@pytest.mark.parametrize("field,value", [("ea_targets", np.zeros(130)),
    ("ea_arm_present", [0, 0]), ("ea_arm_present", [0.5, 1]),
    ("ea_effectors", np.full((2, 10), np.nan)), ("ea_effectors", None)])
def test_engine_rejects_invalid_ea_observation(engine, field, value):
    obs = observation()
    obs[field] = value
    with pytest.raises(ValueError, match="EA|ea_effectors"):
        engine.generate({"observation": obs})
    assert not hasattr(engine.architecture, "inputs")
