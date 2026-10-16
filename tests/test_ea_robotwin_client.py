import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from benchmarks.robotwin import openwam2robotwin_interface as interface
from benchmarks.robotwin.ea_fk import AlohaEEF


def test_fk_rejects_actual_joint_contract(tmp_path):
    root = Path(__file__).resolve().parents[1]
    summary = json.loads((root / "assets/ea_robotwin/training_contract.json").read_text())
    summary["deployable_effector_contract"] = "actual-joint FK"
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="drive-target"):
        AlohaEEF(root / "assets/ea_robotwin/aloha.urdf", root / "assets/ea_robotwin/aloha_config.yml", path)


def test_client_sends_only_deployable_context(monkeypatch, tmp_path):
    class Client:
        def __init__(self, *args, **kwargs):
            self.payload = None
        def predict(self, payload):
            self.payload = payload
            return {"action": [0.0] * 14}
        def connect(self):
            pass
        def ping(self):
            return {"type": "pong"}
        def close(self):
            pass
    monkeypatch.setattr(interface, "WSPolicyClient", Client)
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "benchmarks/robotwin/policy_config_ea_rsig_v3.yml").read_text())
    monkeypatch.chdir(tmp_path)
    model = interface.get_model(cfg)
    qpos = np.zeros(14, dtype=np.float32)
    qpos[[6, 13]] = [0.5, 0.7]
    model.step({"cams": {"head": np.zeros((240, 320, 3), dtype=np.uint8)}, "state": qpos})
    payload = model._client.payload
    assert payload["state"] == qpos.tolist()
    assert np.asarray(payload["ea_effectors"]).shape == (2, 10)
    assert payload["ea_arm_present"] == [1, 1]
    assert "ea_targets" not in payload
    np.testing.assert_allclose(np.asarray(payload["ea_effectors"])[:, 9], qpos[[6, 13]])


def test_fk_matches_real_schema5_current_row():
    import pyarrow.parquet as pq
    root = Path(__file__).resolve().parents[1]
    data = Path("/remote-home/jinminghao/DiT4DiT/datasets/robotwin2_ea_rsig_v3_aloha_10000_schema5_20260928/EA_RSIG_v3/Clean/adjust_bottle")
    if not data.exists():
        pytest.skip("real schema5 data is not available")
    fk = AlohaEEF(root / "assets/ea_robotwin/aloha.urdf", root / "assets/ea_robotwin/aloha_config.yml",
                  root / "assets/ea_robotwin/training_contract.json")
    table = pq.read_table(data / "data/chunk-000/episode_000000.parquet").to_pydict()
    for row in (0, 15, len(table["observation.state"]) - 1):
        predicted = fk(table["observation.state"][row])
        recorded = np.asarray(table["observation.ea_rsig_v3.effector_state"][row], dtype=np.float32).reshape(2, 10)
        np.testing.assert_allclose(predicted, recorded, rtol=0, atol=2e-6)
