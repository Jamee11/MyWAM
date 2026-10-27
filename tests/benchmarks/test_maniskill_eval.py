import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from benchmarks.maniskill.run_eval import assign_pools, combine, partitions
from benchmarks.maniskill.single_eval import native_action, seeds_for
from openwam.maniskill_eef_contract import absolute_command_targets


def test_native_absolute_target_roundtrip():
    rng = np.random.default_rng(42)
    n = 100
    q = Rotation.random(n, random_state=rng).as_quat()[:, [3, 0, 1, 2]]
    base = np.c_[rng.uniform(-1, 1, (n, 3)), q]
    tcp = np.c_[rng.uniform(-1, 1, (n, 3)), q]
    delta = rng.uniform(-2, 2, (n, 7))
    target = absolute_command_targets(tcp, base, delta)
    native = np.stack([native_action(a) for a in target])
    np.testing.assert_allclose(native[:, :3], target[:, :3], atol=1e-6)
    r = Rotation.from_euler("XYZ", native[:, 3:6]).as_matrix()
    np.testing.assert_allclose(r[:, :, 0], target[:, 3:6], atol=1e-6)
    np.testing.assert_allclose(r[:, :, 1], target[:, 6:9], atol=1e-6)
    np.testing.assert_allclose(native[:, 6], np.clip(delta[:, 6], -1, 1), atol=1e-6)


def test_degenerate_or_wrong_action_rejected():
    for a in (np.zeros(10), np.ones(10), np.full(10, np.nan), np.zeros(7)):
        with pytest.raises(ValueError):
            native_action(a)


def test_seeds_are_ordered_successful_and_held_out(tmp_path, monkeypatch):
    path = tmp_path / "seeds.json"
    path.write_text(
        '{"saved_trajectory_seeds":[{"seed":2000000,"success":true},{"seed":2000001,"success":false},{"seed":2000002,"success":true}]}'
    )
    monkeypatch.setattr("benchmarks.maniskill.single_eval.seed_manifest", lambda _: path)
    assert seeds_for("push_cube", 2) == [2000000, 2000002]
    with pytest.raises(ValueError):
        seeds_for("push_cube", 3)


def test_server_canvas_matches_training_layout():
    from PIL import Image

    from benchmarks.utils.client import build_payload, encode_numpy_b64
    from openwam.dataloader.transforms.multiview import assemble_multiview_layout
    from openwam.deploy.obs_preprocess import ObsPreprocessor

    rng = np.random.default_rng(42)
    front = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    hand = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    layout = ["front_camera", "hand_camera", None]
    expected = assemble_multiview_layout(
        {"front_camera": Image.fromarray(front), "hand_camera": Image.fromarray(hand)}, layout, 384, 320
    )
    obs = ObsPreprocessor(multiview=True, camera_layout=layout, img_height=384, img_width=320).preprocess(
        build_payload(encode_numpy_b64(front), encode_numpy_b64(hand))
    )
    np.testing.assert_array_equal(np.asarray(obs["image"]), np.asarray(expected))


def test_parallel_shards_cover_each_seed_once():
    for count in (1, 49, 50, 100):
        jobs = partitions(["push_cube", "pick_cube"], count, 2)
        for task in ("push_cube", "pick_cube"):
            covered = [i for j in jobs if j["task"] == task for i in range(j["start"], j["start"] + j["count"])]
            assert covered == list(range(count))
    summaries = [
        dict(complete=True, records=[dict(episode=i, seed=2000000 + i, steps=12, success=True)]) for i in (1, 0)
    ]
    result = combine("push_cube", summaries, [2000000, 2000001])
    assert result["successes"] == 2 and result["success_rate"] == 1
    with pytest.raises(ValueError):
        combine("push_cube", [summaries[0], summaries[0]], [2000000, 2000001])


def test_two_models_have_isolated_worker_ports():
    jobs = partitions(["push_cube", "pick_cube", "stack_cube", "place_sphere", "pick_cuboid", "pick_sphere"], 100, 2)
    pools = assign_pools(jobs, [5, 6], 21700)
    assert pools == [dict(gpu=5, port=21700, slots=6), dict(gpu=6, port=21706, slots=6)]
    assert sorted(j["port"] for j in jobs) == list(range(21700, 21712))
    for task in {j["task"] for j in jobs}:
        assert [(j["sim_gpu"], j["start"], j["count"]) for j in jobs if j["task"] == task] == [(5, 0, 50), (6, 50, 50)]


def test_sidecar_validation_uses_native_rgb_and_actual_eef(tmp_path):
    import json

    import h5py

    from benchmarks.maniskill.serve_sidecar import validation_inputs

    source = tmp_path / "observations.h5"
    with h5py.File(source, "w") as f:
        for index in range(2):
            obs = f.create_group(f"traj_{index}/obs")
            obs["extra/tcp_pose"] = np.array([[0.2, 0.1, 0.3, 1, 0, 0, 0]] * 2)
            obs["extra/robot_base_pose"] = np.array([[0, 0, 0, 1, 0, 0, 0]] * 2)
            obs["agent/qpos"] = np.array([[0] * 7 + [0.02, 0.02]] * 2)
            for camera in ("front_camera", "hand_camera"):
                obs[f"sensor_data/{camera}/rgb"] = np.zeros((2, 16, 16, 3), np.uint8)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                episodes=[
                    dict(task=task, source_h5=str(source), trajectory=f"traj_{i}", steps=1, prompt=task)
                    for i, task in enumerate(("push_cube", "pick_cube"))
                ]
            )
        )
    )
    values = validation_inputs(manifest)
    assert len(values) == 2
    for value in values:
        np.testing.assert_allclose(value["state"], [0.2, 0.1, 0.3, 1, 0, 0, 0, 1, 0, 0.5])
        assert set(value) == {"images", "state", "prompt"}
        assert value["images"]["right_wrist_camera"] is None


def test_comparison_gate_rejects_changed_budget(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from benchmarks.maniskill.run_eval import ROOT, check_baseline_reference, sha256

    baseline = tmp_path / "baseline"
    (baseline / "checkpoint").mkdir(parents=True)
    side = tmp_path / "sidecar"
    side.mkdir()
    for parent in (baseline / "checkpoint", side):
        (parent / "normalization_stats.npy").write_bytes(b"same-statistics")
    cfg = OmegaConf.create(dict(dataloader=dict(action_mode="eef")))
    OmegaConf.save(cfg, baseline / "checkpoint/config.yaml")
    record = dict(
        episodes_per_task=100,
        max_steps=200,
        replan_every=8,
        denoise_steps=10,
        seed_manifests=dict(push_cube=dict(seeds=[2000000])),
        source_hashes={
            name: sha256(ROOT / name)
            for name in ("benchmarks/maniskill/single_eval.py", "openwam/maniskill_eef_contract.py")
        },
    )
    (baseline / "run.json").write_text(json.dumps(record))
    monkeypatch.setattr("benchmarks.maniskill.run_eval.seeds_for", lambda *a: [2000000])
    args = SimpleNamespace(episodes=100, max_steps=200, replan_every=8, denoise_steps=10)
    result = check_baseline_reference(cfg, side / "checkpoint_step_4.safetensors", baseline, args)
    assert result["identical_input_contract"] and result["identical_stats"]
    args.max_steps = 201
    with pytest.raises(ValueError, match="budget differs"):
        check_baseline_reference(cfg, side / "checkpoint_step_4.safetensors", baseline, args)
