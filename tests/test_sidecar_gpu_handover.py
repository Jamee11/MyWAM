"""GPU transfer must wait for both the notice window and every allocated card."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/train_relational_sidecar_formal.py"


def load_gate():
    spec = importlib.util.spec_from_file_location("sidecar_handover", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("busy_until,expected", [(0, 90), (120, 120)])
def test_notice_and_all_four_cards(monkeypatch, tmp_path, busy_until, expected):
    gate = load_gate()
    clock = [0]

    def sleep(seconds):
        assert not (tmp_path / "gpu_release.json").exists()
        clock[0] += seconds

    monkeypatch.setattr(gate, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
    monkeypatch.setattr(gate, "selected_gpu_memory", lambda: {
        "0": 80000, "1": 80000, "2": 80000, "3": 10000 if clock[0] < busy_until else 80000,
    })
    gate.wait_for_gpu_release(tmp_path)
    assert clock[0] == expected
    assert json.loads((tmp_path / "cpu_ready.json").read_text())["notice_seconds"] == 90
    assert min(json.loads((tmp_path / "gpu_release.json").read_text())["gpu_free_mib"].values()) >= 70000


def test_reject_wrong_gpu_selection(monkeypatch):
    gate = load_gate()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    with pytest.raises(ValueError, match="GPU0,1,2,3"):
        gate.selected_gpu_memory()
