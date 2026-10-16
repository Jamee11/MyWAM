"""CPU engineering gates using actual tiny Wan/ActionDiT blocks; no GPU claim."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.dual_system.ea_rsig_lite import EARSIGLiteArchitecture
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.modules.ea_rsig_lite import EARSIGLiteBridge
from openwam.model.video_backbone.wan.models.dit import WanModel
from openwam.model.video_backbone.wan.shared.diffusion.flow_match import FlowMatchScheduler
from openwam.model.video_backbone.wan_backbone import Wan22Ti2v
from openwam.dataloader.ea_rsig_lite import targets_from_oracle


@pytest.fixture
def model_factory(monkeypatch):
    torch.set_num_threads(2)
    def init_video(self, cfg):
        dit = WanModel(has_image_input=False, patch_size=(1, 2, 2), in_dim=4,
                       dim=32, ffn_dim=64, freq_dim=16, text_dim=32, out_dim=4,
                       num_heads=4, num_layers=2, eps=1e-6, seperated_timestep=True,
                       fuse_vae_embedding_in_latents=True)
        scheduler = FlowMatchScheduler(template="Wan")
        scheduler.set_timesteps(10, training=True, shift=5)
        self.video_backbone = Wan22Ti2v(SimpleNamespace(dit=dit, scheduler=scheduler))
        self.video_backbone._device = torch.device("cpu")
        self.video_backbone._dtype = torch.float32
    monkeypatch.setattr(BaseWAMArchitecture, "_init_video_backbone", init_video)
    def make(stage="joint", conditioning=True, cls=EARSIGLiteArchitecture):
        cfg = dict(dim=32, ffn_dim=64, action_dim=80, state_dim=80, bridge_interval=1,
                   use_proprioception=True, attention_mask_mode="mutual",
                   video_attention_mask_mode="first_frame_causal", mot_checkpoint_mixed_attn=True,
                   ea_lite=dict(slot_dim=32, num_heads=4, grid_height=2, view_grid_width=2,
                                stage=stage, action_conditioning=conditioning, adapter_layer=0,
                                foundation_warmstart=True))
        model = cls(cfg)
        model.set_dtype_device(torch.float32, torch.device("cpu"))
        model.action_backbone.scheduler.set_timesteps(10, training=True)
        return model
    return make


def inputs():
    clean = torch.randn(1, 4, 1, 12, 8)
    return dict(first_frame_latents=clean, latents=torch.randn(1, 4, 3, 12, 8),
                context=torch.randn(1, 3, 32), context_mask=torch.ones(1, 3, dtype=torch.bool),
                timestep=torch.tensor([111.]), proprio=torch.randn(1, 80),
                ea_effectors=torch.randn(1, 2, 10), ea_arm_present=torch.ones(1, 2),
                fuse_vae_embedding_in_latents=True)


def target():
    value = torch.zeros(1, 1, 130)
    value[..., :2] = 1
    value[..., 8:16] = 1
    value[..., 16:52].reshape(1, 1, 3, 2, 2, 3)[..., 0] = 1
    value[..., 52:70] = 1
    value[..., 70:118] = torch.tensor([.2, .2, .8, .8]).repeat(12)
    value[..., 118:130] = 1
    return value


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("conditioning", [False, True])
def test_T1_official_identity(model_factory, checkpointed, conditioning):
    model = model_factory(conditioning=conditioning)
    baseline = model_factory(cls=DualSystemSelfAttnArchitecture)
    baseline.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("ea_bridge.")})
    model.eval(); baseline.eval()
    if checkpointed:
        model.train(); baseline.train()
    x, actions = inputs(), torch.randn(1, 32, 80)
    expected = baseline(actions, torch.ones(1), use_gradient_checkpointing=checkpointed,
                        **{k: v for k, v in x.items() if not k.startswith("ea_")})
    actual = model(actions, torch.ones(1), use_gradient_checkpointing=checkpointed, **x)
    for ref, out in zip(expected, actual):
        torch.testing.assert_close(ref, out, rtol=0, atol=0)


def test_T2_independent_future_noise_target_interventions(model_factory):
    model, x = model_factory(), inputs()
    reference = model._observe(x, x["ea_effectors"], x["ea_arm_present"])
    for key in ("latents", "timestep", "future_actions", "diffusion_noise", "ea_targets"):
        changed = dict(x)
        changed[key] = torch.randn_like(x[key]) * 1000 if key in x else torch.randn(1, 32, 80) * 1000
        observed = model._observe(changed, x["ea_effectors"], x["ea_arm_present"])
        for output in ("action_tokens", "plan_directions", "plan_moving_logits", "eef_object_relations"):
            torch.testing.assert_close(reference[output], observed[output], rtol=0, atol=0)
    with pytest.raises(ValueError, match="Privileged"):
        model(torch.randn(1, 32, 80), torch.ones(1), ea_targets=target(), **x)


def test_T3_train_eval_and_current_input_sensitivity(model_factory):
    model, x = model_factory(), inputs()
    model.train()
    train = model._observe(x, x["ea_effectors"], x["ea_arm_present"])
    model.eval()
    deploy = model._observe(x, x["ea_effectors"], x["ea_arm_present"])
    torch.testing.assert_close(train["action_tokens"], deploy["action_tokens"], rtol=0, atol=0)
    for key in ("first_frame_latents", "context"):
        changed = dict(x); changed[key] = x[key] + torch.randn_like(x[key]) * 3
        out = model._observe(changed, x["ea_effectors"], x["ea_arm_present"])
        assert not torch.equal(train["action_tokens"], out["action_tokens"])
    changed = model._observe(x, x["ea_effectors"] + 1, x["ea_arm_present"])
    assert not torch.equal(train["action_tokens"], changed["action_tokens"])


def test_three_camera_crop_and_no_video_block_observation(model_factory, monkeypatch):
    model, x = model_factory(), inputs()
    monkeypatch.setattr(model.video_backbone, "run_block", lambda *a: pytest.fail("Observation ran a DiT block"))
    assert model._observe(x, x["ea_effectors"], x["ea_arm_present"])["action_tokens"].shape == (1, 15, 32)
    bridge = EARSIGLiteBridge(32, 32, 32, slot_dim=32, num_heads=4, grid_height=2, view_grid_width=2)
    bridge.patch_projection = torch.nn.Identity()
    torch.nn.init.zeros_(bridge.text_projection.weight); torch.nn.init.zeros_(bridge.text_projection.bias)
    monkeypatch.setattr(bridge.core, "forward", lambda hidden, *args: {"packed": hidden})
    patch = torch.zeros(1, 32, 1, 6, 4)
    patch[:, :, :, :4] = 11
    patch[:, :, :, 4:, :2] = 22
    patch[:, :, :, 4:, 2:] = 33
    packed = bridge.observe(patch, x["context"], None, x["ea_effectors"], x["ea_arm_present"])["packed"]
    grid = packed.reshape(1, 2, 6, 32)
    assert torch.all(grid[:, :, :2] == 11)
    assert torch.all(grid[:, :, 2:4] == 22)
    assert torch.all(grid[:, :, 4:] == 33)
    patch[:, :, :, 0, 0] += 5
    changed = bridge.observe(patch, x["context"], None, x["ea_effectors"], x["ea_arm_present"])["packed"]
    assert not torch.equal(packed, changed)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_warmup_two_steps_gradient_paths_and_token_intervention(model_factory, checkpointed):
    model, x = model_factory(stage="warmup"), inputs()
    assert all(not p.requires_grad for n, p in model.named_parameters() if not n.startswith("ea_bridge."))
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for step in range(2):
        losses = model.compute_loss(input_latents=x["latents"], first_frame_latents=x["first_frame_latents"],
            context=x["context"], context_mask=x["context_mask"], proprio=x["proprio"],
            actions=torch.randn(1, 32, 80), ea_effectors=x["ea_effectors"],
            ea_arm_present=x["ea_arm_present"], ea_targets=target(),
            fuse_vae_embedding_in_latents=True, use_gradient_checkpointing=checkpointed)
        assert all(torch.isfinite(v).all() for v in losses.values())
        losses["loss"].backward()
        assert model.ea_bridge.action_out.weight.grad.abs().sum() > 0
        assert model.ea_bridge.core.ground_key.weight.grad.abs().sum() > 0
        if step == 0:
            assert model.ea_bridge.action_query.weight.grad.abs().sum() == 0
        else:
            assert model.ea_bridge.action_query.weight.grad.abs().sum() > 0
            assert model.ea_bridge.core.eef_plan_to_hidden.weight.grad.abs().sum() > 0
        assert model._ea_training_output is None and model.ea_bridge.core._video_output is None
        optimizer.step(); optimizer.zero_grad()
    model.eval()
    actions = torch.randn(1, 32, 80)
    enabled = model(actions, torch.ones(1), **x)[1]
    model.ea_bridge.action_conditioning = False
    disabled = model(actions, torch.ones(1), **x)[1]
    assert not torch.equal(enabled, disabled)


def test_aux_only_reaches_shared_patch_in_joint_stage(model_factory):
    model, x = model_factory(conditioning=False), inputs()
    output = model._observe(x, x["ea_effectors"], x["ea_arm_present"])
    sum(model.ea_bridge.core.losses(output, target()).values()).backward()
    assert model.video_backbone.dit.patch_embedding.weight.grad.abs().sum() > 0
    assert model.ea_bridge.action_out.weight.grad is None
    assert all(p.grad is not None for p in model.ea_bridge.parameters() if p.requires_grad)


def test_foundation_heads_preserved_and_strict_lite_roundtrip(model_factory, tmp_path):
    source = model_factory(cls=DualSystemSelfAttnArchitecture)
    path = tmp_path / "foundation.safetensors"
    save_file(source.state_dict(), path)
    model = model_factory()
    model.load_checkpoint(path)
    assert model.foundation_load_report["reset_policy_heads"] == []
    for key, value in source.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    with torch.no_grad(): model.ea_bridge.action_out.weight.fill_(.01)
    save_file(model.state_dict(), tmp_path / "lite.safetensors")
    restored = model_factory()
    restored.load_checkpoint(tmp_path / "lite.safetensors")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


def oracle(length=40):
    pose = np.zeros((length, 2, 7), np.float32); pose[..., 3] = 1
    pose[:, :, 0] = np.arange(length)[:, None] * .002
    obj = pose.copy(); obj[..., 1] = .05
    boxes = np.tile(np.array([20, 10, 60, 30], np.float32), (length, 4, 3, 1))
    return dict(eef_pose_world_wxyz=pose, track_object_pose_world_wxyz=obj,
        track_goal_pose_world_wxyz=obj.copy(), track_valid=np.ones((length, 2), np.float32),
        track_role_id=np.ones((length, 2), np.float32), track_contact=np.zeros((length, 2), np.float32),
        goal_valid=np.ones((length, 2), np.float32), arm_present=np.ones((length, 2), np.float32),
        ground_bbox_pixels=boxes, ground_bbox_valid=np.ones((length, 4, 3), np.float32))


def test_plan_horizons_tail_and_role_identity():
    raw = oracle()
    labels = targets_from_oracle(raw, [(100, 50)] * 3, "adjust_bottle")
    np.testing.assert_allclose(labels[0, 16:52].reshape(3, 2, 2, 3), np.tile([1, 0, 0], (3, 2, 2, 1)))
    future = labels[:, 64:70].reshape(40, 3, 2)
    assert future[7, 2].all() and not future[8, 2].any()  # t+32 < 40
    assert future[23, 1].all() and not future[24, 1].any()
    assert future[31, 0].all() and not future[32, 0].any()
    raw["track_role_id"][8, 0] = 2
    changed = targets_from_oracle(raw, [(100, 50)] * 3, "adjust_bottle")
    assert changed[0, 64] == 0
    raw = oracle(); raw["eef_pose_world_wxyz"][8, :, 1] += .08
    changed = targets_from_oracle(raw, [(100, 50)] * 3, "adjust_bottle")
    assert not np.array_equal(changed[0, 16:28], labels[0, 16:28])
    np.testing.assert_array_equal(changed[0, 28:52], labels[0, 28:52])


def test_per_view_boxes_virtual_goals_and_goal_mask():
    raw = oracle()
    raw["ground_bbox_valid"][:, 1::2] = 0
    labels = targets_from_oracle(raw, [(100, 50), (200, 100), (400, 200)], "blocks_ranking_rgb")
    box = labels[0, 70:118].reshape(4, 3, 4)
    np.testing.assert_allclose(box[0, 0], [.2, .2, .6, .6])
    np.testing.assert_allclose(box[0, 1], [.1, .1, .3, .3])
    assert not box[1::2].any() and not labels[:, 14:16].any()
    ordinary = targets_from_oracle(raw, [(100, 50)] * 3, "adjust_bottle")
    assert ordinary[:, 14:16].all()


def test_deploy_eef_conversion_and_client_lite_payload(monkeypatch):
    from benchmarks.robotwin import openwam2robotwin_interface as interface
    from benchmarks.utils import action_conversion
    from openwam.ea_lite_contract import tcp_context_from_eef20
    class Client:
        def __init__(self, *a, **k): pass
        def connect(self): pass
        def ping(self): return {"type": "pong"}
        def close(self): pass
        def predict(self, payload): self.payload = payload; return {"action": [0.] * 20}
    monkeypatch.setattr(interface, "WSPolicyClient", Client)
    client = interface.ModelClient(state_dim=20, action_type="ee", ea_context="official_eef")
    state = np.tile(np.array([.1, .2, .3, 1, 0, 0, 0, 1, 0, .3], np.float32), 2)
    client.step({"cams": {"head": np.zeros((240, 320, 3), np.uint8)}, "state": state})
    payload = client._client.payload
    np.testing.assert_array_equal(np.asarray(payload["ea_effectors"]), tcp_context_from_eef20(state))
    assert "ea_targets" not in payload
    converted = np.stack([action_conversion.eef20d_to_ee16d(state) for _ in range(32)])
    assert converted.shape == (32, 16) and np.isfinite(converted).all()
    np.testing.assert_allclose(converted[:, [7, 15]], .3)


def test_wxyz_controller_pose_roundtrip(monkeypatch):
    from scipy.spatial.transform import Rotation
    from benchmarks.robotwin import openwam2robotwin_interface as interface
    from openwam.dataloader.ea_rsig_lite import LiteRoboTwinSource
    rotation = Rotation.from_euler("xyz", [.3, -.7, 1.1])
    q = rotation.as_quat()[[3, 0, 1, 2]]
    pose = np.concatenate(([.2, -.1, .9], q)).astype(np.float32)
    endpose = dict(left_endpose=pose, right_endpose=pose, left_gripper=.3, right_gripper=.7)
    state = interface._extract_eef_proprio({"endpose": endpose}, "wxyz")
    file = {f"endpose/{key}": np.asarray([value]) for key, value in endpose.items()}
    training = LiteRoboTwinSource._read_eef_actions(None, file, 0, 1)[0]
    np.testing.assert_allclose(state, training, atol=1e-6)
    np.testing.assert_allclose(state[3:9], np.concatenate((rotation.as_matrix()[:, 0], rotation.as_matrix()[:, 1])), atol=1e-6)
    class Model:
        _action_type = "ee"
        _send_state = True
        _source_quaternion_order = "wxyz"
        def step(self, observation): return state.copy()
    class Env:
        def get_instruction(self): return "test task"
        def take_action(self, action, action_type): self.action = action
    monkeypatch.setattr(interface, "_apply_step_lim_override", lambda env: None)
    env = Env()
    obs = {"endpose": endpose, "observation": {"head_camera": {"rgb": np.zeros((4, 4, 3), np.uint8)}}}
    interface.eval(env, Model(), obs)
    np.testing.assert_allclose(env.action[:3], pose[:3])
    actual_q = env.action[[4, 5, 6, 3]]
    np.testing.assert_allclose(Rotation.from_quat(actual_q).as_matrix(), rotation.as_matrix(), atol=1e-6)


def test_two_actual_cpu_accelerate_save_restore_boundaries(model_factory, tmp_path):
    from accelerate import Accelerator
    from openwam.train.utils.checkpointing import save_full_state, load_full_state
    accelerator = Accelerator(cpu=True)
    model = model_factory(stage="warmup")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    x, actions = inputs(), torch.randn(1, 32, 80)
    def step():
        optimizer.zero_grad()
        losses = model.compute_loss(input_latents=x["latents"], first_frame_latents=x["first_frame_latents"],
            context=x["context"], context_mask=x["context_mask"], proprio=x["proprio"],
            actions=actions, ea_effectors=x["ea_effectors"], ea_arm_present=x["ea_arm_present"],
            ea_targets=target(), fuse_vae_embedding_in_latents=True)
        accelerator.backward(losses["loss"])
        optimizer.step(); scheduler.step()
        return losses["loss"].detach().clone()
    step()
    for saved_step in (1, 2):
        save_full_state(accelerator, str(tmp_path), saved_step, saved_step, 0)
        reference = step()
        expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
        expected_lr = optimizer.param_groups[0]["lr"]
        restored = load_full_state(accelerator, str(tmp_path / f"accel_state_step_{saved_step}"))
        assert restored["opt_step"] == saved_step
        actual = step()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert optimizer.param_groups[0]["lr"] == expected_lr
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    accelerator.end_training()


def test_retained_resume_states_opt_in_only(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    import openwam.train.openwam_trainer as module
    trainer = module.OpenWAMTrainer.__new__(module.OpenWAMTrainer)
    trainer.accelerator = SimpleNamespace(wait_for_everyone=lambda: None)
    trainer.cfg = OmegaConf.create({"training": {"keep_final_resume_state": True}})
    called = []
    monkeypatch.setattr(module, "finalize_keep_weights_only", lambda *a: called.append(a))
    trainer.finish_training(str(tmp_path), 0, 0, 2, True, None)
    assert not called
    trainer.cfg.training.keep_final_resume_state = False
    trainer.finish_training(str(tmp_path), 0, 0, 2, True, None)
    assert len(called) == 1


def test_cpu_bf16_zero_init_and_inference_identity(model_factory):
    model = model_factory()
    baseline = model_factory(cls=DualSystemSelfAttnArchitecture)
    baseline.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("ea_bridge.")})
    for module in (model, baseline):
        module.set_dtype_device(torch.bfloat16, torch.device("cpu"))
        module.eval()
    x = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in inputs().items()}
    actions, time = torch.randn(1, 32, 80, dtype=torch.bfloat16), torch.ones(1, dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = baseline(actions, time, **{k: v for k, v in x.items() if not k.startswith("ea_")})
        actual = model(actions, time, **x)
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_joint_sft_keeps_video_action_and_ea_gradients(model_factory, checkpointed):
    model, x = model_factory(stage="joint"), inputs()
    losses = model.compute_loss(input_latents=x["latents"], first_frame_latents=x["first_frame_latents"],
        context=x["context"], context_mask=x["context_mask"], proprio=x["proprio"],
        actions=torch.randn(1, 32, 80), ea_effectors=x["ea_effectors"], ea_arm_present=x["ea_arm_present"],
        ea_targets=target(), fuse_vae_embedding_in_latents=True, use_gradient_checkpointing=checkpointed)
    losses["loss"].backward()
    for parameter in (model.video_backbone.dit.patch_embedding.weight,
                      model.action_backbone.action_encoder.weight, model.proprio_encoder.weight,
                      model.ea_bridge.action_out.weight, model.ea_bridge.core.ground_key.weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_evaluation_loss_does_not_retain_auxiliary_graph(model_factory):
    model, x = model_factory(), inputs()
    model.eval()
    with torch.inference_mode():
        losses = model.compute_loss(input_latents=x["latents"], first_frame_latents=x["first_frame_latents"],
            context=x["context"], context_mask=x["context_mask"], proprio=x["proprio"],
            actions=torch.randn(1, 32, 80), ea_effectors=x["ea_effectors"], ea_arm_present=x["ea_arm_present"],
            ea_targets=target(), fuse_vae_embedding_in_latents=True, use_gradient_checkpointing=True)
    assert torch.isfinite(losses["loss"])
    assert model._ea_training_output is None and not model._capture_ea_output
