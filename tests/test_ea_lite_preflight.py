"""GPU selection/parser gates with fake NVML text; no hardware execution."""
import pytest

from scripts.preflight_ea_lite import check_four_gpus, check_gpus


@pytest.mark.parametrize("selection", ["", "0,1", "0,1,2,2", "GPU-a,GPU-b,GPU-c,GPU-d"])
def test_reject_non_four_distinct_indices(monkeypatch, selection):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", selection)
    with pytest.raises(ValueError, match="four distinct"):
        check_four_gpus()


def test_busy_and_wrong_hardware_are_rejected(monkeypatch):
    import scripts.preflight_ea_lite as module
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    text = "\n".join(f"{i}, NVIDIA A100-SXM4-80GB, 81155, 80000" for i in range(4))
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: text)
    assert check_four_gpus() == ["0", "1", "2", "3"]
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: text.replace("80000", "10000"))
    with pytest.raises(ValueError, match="free-memory gate"):
        check_four_gpus()
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: text.replace("A100", "H200"))
    with pytest.raises(ValueError, match="free-memory gate"):
        check_four_gpus()
    smaller = text.replace("80GB", "40GB").replace("81155", "40536").replace("80000", "39000")
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: smaller)
    assert check_four_gpus(40) == ["0", "1", "2", "3"]


@pytest.mark.parametrize("count,name", [(2, "A100"), (8, "H200")])
def test_explicit_multi_gpu_count_and_name(monkeypatch, count, name):
    import scripts.preflight_ea_lite as module
    selected = [str(i) for i in range(count)]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(selected))
    text = "\n".join(f"{i}, NVIDIA {name}, 141000, 130000" for i in selected)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: text)
    assert check_gpus(count, 80, name) == selected
    with pytest.raises(ValueError, match="distinct"):
        check_gpus(count + 1, 80, name)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(selected[:-1] + ["99"]))
    with pytest.raises(ValueError, match="absent"):
        check_gpus(count, 80, name)
