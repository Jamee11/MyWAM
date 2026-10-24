"""Semantic checks for the new single-arm supervision and its actual Sidecar gradient path."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from openwam.maniskill_sidecar_contract import build_targets
from scripts.train_maniskill_sidecar import restore_saved_counter, validate_save_boundaries, write_maniskill_contract
from tests.test_relational_sidecar import inputs

pytest_plugins = ["tests.test_relational_sidecar"]


def trajectory():
    t = np.arange(41, dtype=np.float32)
    tcp = np.zeros((41, 7), np.float32)
    tcp[:, 0], tcp[:, 3:] = 0.01 * t, Rotation.from_euler("z", 90, degrees=True).as_quat()[[3, 0, 1, 2]]
    objects = tcp.copy()
    objects[:, 0] += 0.05
    goals = np.tile([0.5, 0, 0], (41, 1)).astype(np.float32)
    boxes = np.tile([64, 64, 128, 128], (40, 2, 2, 1))
    visible = np.ones((40, 2, 2))
    return dict(
        tcp=tcp,
        objects=objects,
        goals=goals,
        object_ids=np.ones(41),
        goal_ids=np.ones(41) * 2,
        contact_force=np.zeros(41),
        boxes=boxes,
        visible=visible,
    )


def test_local_direction_and_terminal_validity():
    target = build_targets(**trajectory())
    assert target.shape == (40, 130)
    # World +x becomes local -y for a role whose current orientation is Rz(90).
    np.testing.assert_allclose(target[8, 16:19], [0, -1, 0], atol=1e-6)
    assert target[8, 68] == 1  # t+32=40 uses the real terminal observation.
    assert target[9, 68] == 0
    assert target[32, 64] == 1 and target[33, 64] == 0
    assert not target[:, 1].any()
    assert not target[:, 118:130].reshape(-1, 4, 3)[:, :, 2].any()
    assert not target[:, 118:130].reshape(-1, 4, 3)[:, 2:].any()


def test_unknown_contact_and_virtual_goal():
    data = trajectory()
    data.update(contact_force=None, goal_ids=np.zeros(41))
    target = build_targets(**data)
    assert not target[:, 10:12].any()
    assert (target[:, 14] == 1).all()  # Geometry still known for proximity.
    assert not target[:, 118:130].reshape(-1, 4, 3)[:, 1].any()
    data["contact_force"] = np.ones(41) * 2e-6
    target = build_targets(**data)
    assert (target[:, 8] == 1).all() and (target[:, 10] == 1).all()


def test_rejects_role_switch_and_bad_bbox():
    data = trajectory()
    data["object_ids"][20:] = 3
    with pytest.raises(ValueError, match="persistent"):
        build_targets(**data)
    data = trajectory()
    data["boxes"][0, 0, 0, 2] = 999
    with pytest.raises(ValueError, match="bbox"):
        build_targets(**data)


def test_save_boundary_guard_and_exported_frame(tmp_path):
    import json

    validate_save_boundaries(4235, 2, 4235, 42350)
    validate_save_boundaries(4235, 2, 40, 40)
    validate_save_boundaries(2118, 1, 2000, 21180)
    with pytest.raises(ValueError, match="inside gradient"):
        validate_save_boundaries(4235, 2, 4000, 42350)
    with pytest.raises(ValueError, match="inside gradient"):
        validate_save_boundaries(4235, 2, 40, 41)
    path = tmp_path / "relational_sidecar_contract.json"
    path.write_text(json.dumps(dict(action_frame="official_robotwin_endpose_reference", tokens=15)))
    write_maniskill_contract(tmp_path)
    saved = json.loads(path.read_text())
    assert saved["action_frame"] == "robot_base" and saved["eef_reference"] == "panda_hand_tcp"
    assert saved["tool_offset_m"] == 0 and saved["tokens"] == 15
    (tmp_path / "trainer_state.json").write_text(json.dumps(dict(global_step=4235, opt_step=2118)))
    assert restore_saved_counter((4235, 2117, 1, 0), tmp_path) == (4235, 2118, 1, 0)
    with pytest.raises(ValueError, match="aligned backward"):
        restore_saved_counter((4234, 2117, 0, 4234), tmp_path)


def test_single_arm_joint_loss_gradients_and_privileged_isolation(make):
    model, x = make(stage="joint"), inputs()
    x["_proprio_sample_mask"][:, :, 34:44] = False
    x["proprio"][:, 34:44] = 0
    targets = torch.from_numpy(build_targets(**trajectory())[0:1])[None]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        loss = model.compute_loss(
            actions=torch.randn(1, 32, 80),
            input_latents=x["latents"],
            first_frame_latents=x["first_frame_latents"],
            context=x["context"],
            context_mask=x["context_mask"],
            proprio=x["proprio"],
            proprio_mask=x["_proprio_sample_mask"],
            fuse_vae_embedding_in_latents=True,
            ea_targets=targets,
            action_is_pad=~x["_proprio_sample_mask"].expand(-1, 32, -1),
        )
        assert all(torch.isfinite(value).all() for value in loss.values())
        loss["loss"].backward()
        for stage in model.sidecar.stages:
            assert stage.projector[1].weight.grad.norm() > 0
            assert stage.write.beta.grad.abs() > 0
        assert model.sidecar.heads.plan.weight.grad.norm() > 0
        assert model.sidecar.heads.contact.weight.grad.norm() > 0
        optimizer.step()
    with pytest.raises(ValueError, match="labels"):
        model(torch.zeros(1, 32, 80), torch.ones(1), ea_targets=targets, **x)
