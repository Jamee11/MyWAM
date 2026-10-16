"""Result accounting and independent-copy checks without SAPIEN or a GPU."""
import importlib.util
import subprocess
from pathlib import Path

import pytest

from benchmarks.robotwin.eval_policy_wrapper import _install_test_num_override
from benchmarks.robotwin.standard_eval import select_ffmpeg, task_names, validate_result
from scripts.copy_robotwin_runtime import copy_runtime


def load_fake(tmp_path, count=100):
    source = tmp_path / "fake_upstream.py"
    source.write_text(f"""
seen = []
def eval_policy(*args, test_num=100, **kwargs):
    seen.append(test_num)
    return test_num
def main(args):
    test_num = {count}
    success = eval_policy(test_num=test_num)
    return success / test_num
""")
    spec = importlib.util.spec_from_file_location("fake_robotwin_denominator", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


@pytest.mark.parametrize("count", [1, 5, 100])
def test_episode_count_and_native_denominator_match(tmp_path, monkeypatch, count):
    module, source = load_fake(tmp_path)
    original = source.read_bytes()
    monkeypatch.setenv("ROBOTWIN_TEST_NUM", str(count))
    _install_test_num_override(module)
    assert module.main({}) == 1.0
    assert module.seen == [count]
    assert source.read_bytes() == original


def test_unknown_upstream_denominator_fails_before_rollout(tmp_path, monkeypatch):
    module, _ = load_fake(tmp_path, 80)
    monkeypatch.setenv("ROBOTWIN_TEST_NUM", "1")
    with pytest.raises(RuntimeError, match="Unrecognized"):
        _install_test_num_override(module)
    assert not module.seen


def test_default_upstream_main_is_untouched(tmp_path, monkeypatch):
    module, _ = load_fake(tmp_path)
    original = module.main
    monkeypatch.delenv("ROBOTWIN_TEST_NUM", raising=False)
    _install_test_num_override(module)
    assert module.main is original
    assert module.main({}) == 1
    assert module.seen == [100]


@pytest.mark.parametrize("success", [0, 1])
def test_actual_policy_failure_still_completes(tmp_path, success):
    result = tmp_path / "eval_result/task/_result.txt"
    result.parent.mkdir(parents=True)
    result.write_text(f"Timestamp: now\nInstruction Type: unseen\n{success}.0")
    log = (f"Success rate: \x1b[96m{success}/1\x1b[0m => x, current seed: 100002\n"
           "Data has been saved to eval_result/task/_result.txt\n")
    checked = validate_result(log, tmp_path, 1, 0)
    assert checked["complete"]
    assert checked["success_rate"] == success
    assert checked["accepted_seeds"] == [100002]


@pytest.mark.parametrize("native_rate,exit_code,terminal", [(0.01, 0, True), ("nan", 0, True), (1, 1, True), (1, 0, False)])
def test_incomplete_or_wrong_denominator_never_passes(tmp_path, native_rate, exit_code, terminal):
    (tmp_path / "_result.txt").write_text(str(native_rate))
    log = "Success rate: 1/1 => 100%, current seed: 100000\n"
    if terminal:
        log += "Data has been saved to _result.txt\n"
    assert not validate_result(log, tmp_path, 1, exit_code)["complete"]


def test_task_list_is_existing_50_unique_tasks():
    assert len(task_names()) == len(set(task_names())) == 50


def test_ffmpeg_must_run_and_support_native_codec(tmp_path):
    good = tmp_path / "ffmpeg"
    good.write_text('#!/bin/sh\nif [ "$1" = "-version" ]; then echo "ffmpeg fixture"; else echo " V..... libx264 encoder"; fi\n')
    good.chmod(0o755)
    assert select_ffmpeg(good)[0] == good
    bad = tmp_path / "broken_ffmpeg"
    bad.write_text('#!/bin/sh\nexit 1\n')
    bad.chmod(0o755)
    with pytest.raises(RuntimeError, match="No working ffmpeg"):
        select_ffmpeg(bad)


def test_copy_materializes_assets_without_mutating_source(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    (source / "script").mkdir(parents=True)
    (source / "script/eval_policy.py").write_text("# official evaluator\n")
    (source / "script/__pycache__").mkdir()
    (source / "script/__pycache__/tracked.pyc").write_bytes(b"tracked baseline content")
    (source / "data").mkdir()
    (source / "data/helper.py").write_text("# tracked helper, not episode data\n")
    assets = tmp_path / "external_assets"
    assets.mkdir()
    (assets / "object.bin").write_bytes(b"object")
    (source / "assets").symlink_to(assets, target_is_directory=True)
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], check=True)
    manifest = copy_runtime(source, target)
    assert not (target / "assets").is_symlink()
    assert (target / "assets/object.bin").stat().st_ino != (assets / "object.bin").stat().st_ino
    (target / "assets/object.bin").write_bytes(b"changed")
    assert (assets / "object.bin").read_bytes() == b"object"
    assert manifest["file_count"] > 1
    assert (target / "OPENWAM_COPY_MANIFEST.json").is_file()
    assert (target / "script/__pycache__/tracked.pyc").read_bytes() == b"tracked baseline content"
    assert (target / "data/helper.py").is_file()
