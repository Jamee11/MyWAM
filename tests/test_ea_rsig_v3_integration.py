"""CPU checks on real Wan blocks; these are not foundation or rollout evidence."""
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.dual_system.ea_rsig_v3 import EARSIGV3Architecture
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.video_backbone.wan.models.dit import WanModel
from openwam.model.video_backbone.wan.shared.diffusion.flow_match import FlowMatchScheduler
from openwam.model.video_backbone.wan_backbone import Wan22Ti2v
from openwam.dataloader.ea_rsig_v3 import EARSIGV3Dataset
from openwam.dataloader.transforms.normalize import Normalizer
import numpy as np


@pytest.fixture
def model(monkeypatch):
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
    cfg = dict(dim=32, ffn_dim=64, action_dim=14, state_dim=14, bridge_interval=1,
               use_proprioception=True, attention_mask_mode="mutual",
               video_attention_mask_mode="first_frame_causal", mot_checkpoint_mixed_attn=False,
               ea_rsig_v3=dict(layer_index=0, action_horizon=4, video_anchors=[0, 1, 2, 3, 4],
                               slot_dim=32, num_heads=4, grid_height=2, view_grid_width=2,
                               layout="robotwin", num_tracks=2, video_residual=True,
                               foundation_warmstart=True))
    m = EARSIGV3Architecture(cfg)
    m.set_dtype_device(torch.float32, torch.device("cpu"))
    m.action_backbone.scheduler.set_timesteps(10, training=True)
    return m


def inputs():
    latent = torch.randn(2, 4, 2, 12, 8)
    return dict(input_latents=latent, first_frame_latents=latent[:, :, :1].clone(),
                fuse_vae_embedding_in_latents=True, context=torch.randn(2, 3, 32),
                context_mask=torch.ones(2, 3, dtype=torch.bool), proprio=torch.randn(2, 14),
                actions=torch.randn(2, 4, 14), ea_effectors=torch.randn(2, 2, 10),
                ea_arm_present=torch.ones(2, 2), ea_targets=torch.zeros(2, 1, 130))


def test_ea_follows_runtime_dtype_device(model):
    model.set_dtype_device(torch.float64, torch.device("cpu"))
    assert all(p.dtype == torch.float64 for p in model.ea_bridge.parameters())


def test_original_binary_gripper_contract(model):
    data = EARSIGV3Dataset.__new__(EARSIGV3Dataset)
    low, high = np.zeros(14, dtype=np.float32), np.ones(14, dtype=np.float32)
    low[[6, 13]] = -1
    data.stats = {"observation.state": {"min": low, "max": high}}
    raw = np.full(14, 0.3, dtype=np.float32)
    raw[[6, 13]] = [0.5, 0.5001]
    expected = data.normalize(raw, "observation.state")
    assert expected[6] == 0 and expected[13] == 1
    model.attach_normalizer(Normalizer(mode="min_max", stats=data.stats["observation.state"]))
    actual = model.normalize_deploy_proprio(raw)
    np.testing.assert_allclose(actual.numpy().reshape(-1), expected, atol=1e-6)


def test_disabled_ea_preserves_original_joint_forward(model):
    baseline = DualSystemSelfAttnArchitecture(dict(model.cfg))
    baseline.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("ea_bridge.")})
    baseline.set_dtype_device(torch.float32, torch.device("cpu"))
    baseline.eval()
    model.eval()
    model.ea_bridge.action_conditioning = False
    model.ea_bridge.video_residual = False
    x = inputs()
    x.pop("ea_targets")
    actions = x.pop("actions")
    x["latents"] = x.pop("input_latents")
    x["timestep"] = torch.ones(2)
    with torch.no_grad():
        adapted = model(actions, torch.ones(2), **x)
        reference = baseline(actions, torch.ones(2), **{k: v for k, v in x.items() if not k.startswith("ea_")})
    for actual, expected in zip(adapted, reference):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_detached_observation_ignores_future_and_actions(model):
    x = inputs()
    first = model._observe(x, x["ea_effectors"], x["ea_arm_present"])
    x["input_latents"][:, :, 1:] += 1000
    x["actions"] += 1000
    second = model._observe(x, x["ea_effectors"], x["ea_arm_present"])
    torch.testing.assert_close(first["action_tokens"], second["action_tokens"], rtol=0, atol=0)
    first["action_tokens"].square().mean().backward()
    assert model.ea_bridge.core.ground_key.weight.grad is not None
    assert all(p.grad is None for p in model.video_backbone.parameters())


@pytest.mark.parametrize("checkpointed", [False, True])
def test_all_losses_backward_and_clear_state(model, checkpointed):
    x = inputs()
    model.train()
    loss = model.compute_loss(use_gradient_checkpointing=checkpointed, **x)
    assert {"loss", "loss_action", "loss_video", "loss_ea_ground", "loss_ea_relation", "loss_ea_plan"} <= loss.keys()
    assert all(torch.isfinite(v) for v in loss.values())
    loss["loss"].backward()
    assert model.video_backbone.dit.patch_embedding.weight.grad is not None
    assert model.action_backbone.action_encoder.weight.grad is not None
    assert model.ea_bridge.action_projection.weight.grad is not None
    assert model.ea_bridge.core.video_alpha.grad is not None
    assert model._ea_training_output is None
    assert model.ea_bridge.core._video_output is None


def test_oracle_labels_cannot_change_forward(model):
    x = inputs()
    x.pop("ea_targets")
    actions = x.pop("actions")
    x["latents"] = x.pop("input_latents")
    x["timestep"] = torch.ones(2)
    model.eval()
    with torch.no_grad():
        _, pred = model(actions, torch.ones(2), **x)
        model.ea_bridge.core.video_alpha.fill_(100)
        _, no_train_residual = model(actions, torch.ones(2), **x)
    torch.testing.assert_close(pred, no_train_residual, rtol=0, atol=0)
    with pytest.raises(ValueError, match="Privileged"):
        model(actions, torch.ones(2), ea_targets=torch.randn(2, 1, 130), **x)


def test_foundation_warmstart_and_strict_reload(model, tmp_path):
    source = {k: v for k, v in model.state_dict().items() if not k.startswith("ea_bridge.")}
    source["action_backbone.action_encoder.weight"] = torch.randn(32, 80)
    source["action_backbone.action_decoder.weight"] = torch.randn(80, 32)
    source["action_backbone.action_decoder.bias"] = torch.randn(80)
    source["proprio_encoder.weight"] = torch.randn(32, 80)
    foundation = tmp_path / "foundation.safetensors"
    save_file(source, str(foundation))
    model.load_checkpoint(str(foundation))
    assert len(model.foundation_load_report["reset_policy_heads"]) == 4
    source.pop("action_backbone.blocks.0.self_attn.q.weight")
    save_file(source, str(foundation))
    with pytest.raises(RuntimeError, match="Invalid foundation"):
        model.load_checkpoint(str(foundation))
    ckpt = tmp_path / "ea.safetensors"
    model.save_checkpoint(str(ckpt))
    model.load_checkpoint(str(ckpt))
    full = model.state_dict()
    full.pop("ea_bridge.core.video_alpha")
    save_file(full, str(ckpt))
    with pytest.raises(RuntimeError, match="Strict load"):
        model.load_checkpoint(str(ckpt))
