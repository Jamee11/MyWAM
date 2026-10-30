"""Actual deployment config/engine routing, with a stub generator (CPU only)."""
from pathlib import Path

import numpy as np
import torch


def test_real_lite_deploy_config_and_engine_horizons(monkeypatch):
    from openwam.deploy import engine as module
    from openwam.deploy.server import _load_deploy_yaml
    cfg = _load_deploy_yaml(str(Path(__file__).resolve().parents[1] / "configs/deploy_ea_rsig_lite.yaml"))
    assert cfg.server.port == 8848 and not cfg.optimization.dit_cache.enabled
    assert not cfg.optimization.compile.enabled
    monkeypatch.setattr(module, "make_schedule", lambda *a, **k: [(1000, 1000), (0, 0)])
    class Architecture:
        ea_bridge = object()
        device, dtype = torch.device("cpu"), torch.float32
        video_scheduler = action_scheduler = object()
        def normalize_deploy_proprio(self, value): return torch.as_tensor(value)
        def generate(self, **kwargs):
            self.arguments = kwargs
            return {"actions": np.zeros((32, 20)), "video": None}
    model = Architecture()
    engine = module.JointInferenceEngine(cfg, architecture=model)
    result = engine.generate(dict(proprio=np.zeros(20), ea_effectors=np.zeros((2, 10)), ea_arm_present=[1, 1]))
    assert result["actions"].shape == (32, 20)
    assert model.arguments["num_frames"] == 9
    assert model.arguments["action_num_frames"] == 33
    assert "ea_targets" not in model.arguments
