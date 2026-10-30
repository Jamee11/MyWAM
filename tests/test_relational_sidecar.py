"""Actual tiny Wan/ActionDiT engineering gates, without GPU or pretrained-quality claims."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.architectures.dual_system.relational_sidecar import RelationalSidecarArchitecture
from openwam.model.modules.relational_sidecar import canvas_bbox_targets
from openwam.model.video_backbone.wan.models.dit import WanModel
from openwam.model.video_backbone.wan.shared.diffusion.flow_match import FlowMatchScheduler
from openwam.model.video_backbone.wan_backbone import Wan22Ti2v
from tests.test_ea_rsig_lite import target


@pytest.fixture
def make(monkeypatch):
    torch.set_num_threads(2)

    def init_video(self, cfg):
        dit = WanModel(
            has_image_input=False,
            patch_size=(1, 2, 2),
            in_dim=4,
            dim=32,
            ffn_dim=64,
            freq_dim=16,
            text_dim=32,
            out_dim=4,
            num_heads=4,
            num_layers=4,
            eps=1e-6,
            seperated_timestep=True,
            fuse_vae_embedding_in_latents=True,
        )
        scheduler = FlowMatchScheduler(template="Wan")
        scheduler.set_timesteps(10, training=True, shift=5)
        self.video_backbone = Wan22Ti2v(SimpleNamespace(dit=dit, scheduler=scheduler))
        self.video_backbone._device, self.video_backbone._dtype = torch.device("cpu"), torch.float32

    monkeypatch.setattr(BaseWAMArchitecture, "_init_video_backbone", init_video)

    def build(stage="joint", enabled=True, conditioning=True, cls=RelationalSidecarArchitecture, dtype=torch.float32,
              isolation_updates=0, aux_loss_scale=1.0, backbone_updates=0, heads_only=False,
              shared_gradient_scale=1.0, gradient_anneal_updates=0, main_task_gain=1.0):
        cfg = dict(
            dim=32,
            ffn_dim=64,
            action_dim=80,
            state_dim=80,
            bridge_interval=1,
            use_proprioception=True,
            attention_mask_mode="mutual",
            video_attention_mask_mode="first_frame_causal",
            mot_checkpoint_mixed_attn=True,
            relational_sidecar=dict(
                latent_dim=32,
                heads=4,
                tap_blocks=[0, 1, 2],
                enabled=enabled,
                action_conditioning=conditioning,
                stage=stage,
                foundation_warmstart=True,
                isolation_warmup_updates=isolation_updates,
                backbone_warmup_updates=backbone_updates,
                aux_loss_scale=aux_loss_scale,
                aux_heads_only_after_warmup=heads_only,
                aux_shared_gradient_scale=shared_gradient_scale,
                aux_gradient_anneal_updates=gradient_anneal_updates,
                main_task_gradient_gain=main_task_gain,
            ),
        )
        model = cls(cfg)
        model.set_dtype_device(dtype, torch.device("cpu"))
        model.action_backbone.scheduler.set_timesteps(10, training=True)
        return model

    return build


@pytest.mark.parametrize("checkpointed", [False, True])
def test_isolated_warmup_preserves_backbone_forward_and_gradients(make, checkpointed):
    model = make(isolation_updates=2000)
    base = make(cls=DualSystemSelfAttnArchitecture)
    base.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("sidecar.")})
    # Even a nonzero gate must be completely bypassed during isolation.
    for stage in model.sidecar.stages:
        stage.write.beta.data.fill_(0.3)
    model.set_training_progress(1999)
    x, actions = inputs(), torch.randn(1, 32, 80)
    video, action, aux = model.forward_with_aux(
        actions, torch.ones(1), use_gradient_checkpointing=checkpointed, **x
    )
    expected = base(actions, torch.ones(1), use_gradient_checkpointing=checkpointed, **x)
    for actual, reference in zip((video, action), expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    auxiliary = sum(model.sidecar.heads.losses(aux[-1], target()).values())
    auxiliary.backward()
    assert model.sidecar.heads.plan.weight.grad.norm() > 0
    assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith("sidecar."))
    assert all(p.grad is None for stage in model.sidecar.stages for p in stage.write.parameters())
    (video.square().mean() + action.square().mean()).backward()
    sum(value.square().mean() for value in expected).backward()
    for name, parameter in base.named_parameters():
        actual = model.get_parameter(name).grad
        if parameter.grad is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, parameter.grad, rtol=0, atol=0, msg=name)


def test_isolation_boundary_resume_and_inference(make, tmp_path):
    model = make(isolation_updates=2000)
    for stage in model.sidecar.stages:
        stage.write.beta.data.fill_(0.3)
    x, actions = inputs(), torch.randn(1, 32, 80)
    model.set_training_progress(1999)
    warm = model(actions, torch.ones(1), **x)
    model.set_training_progress(2000)
    assert not model.isolation_warmup_active
    video, action, aux = model.forward_with_aux(actions, torch.ones(1), **x)
    assert not torch.equal(action, warm[1])
    auxiliary = sum(model.sidecar.heads.losses(aux[-1], target()).values())
    auxiliary.backward(retain_graph=True)
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.video_backbone.parameters())
    model.zero_grad()
    action.square().mean().backward()
    assert all(stage.write.beta.grad.abs() > 0 for stage in model.sidecar.stages)
    path = tmp_path / "phase.pt"
    torch.save(dict(model=model.state_dict(), opt_step=2000), path)
    saved = torch.load(path, weights_only=True)
    restored = make(isolation_updates=2000)
    restored.load_state_dict(saved["model"], strict=True)
    restored.set_training_progress(saved["opt_step"])
    torch.testing.assert_close(restored(actions, torch.ones(1), **x)[1], action, rtol=0, atol=0)
    restored.set_training_progress(1999)
    assert restored.isolation_warmup_active
    restored.eval()  # Deployment always uses the complete trained branch, regardless of counter.
    assert not restored.isolation_warmup_active
    torch.testing.assert_close(restored(actions, torch.ones(1), **x)[1], action, rtol=0, atol=0)


def test_auxiliary_scale_includes_deep_supervision(make):
    original, reduced = make(), make(aux_loss_scale=0.5)
    reduced.load_state_dict(original.state_dict(), strict=True)
    x = inputs()
    kwargs = dict(actions=torch.randn(1, 32, 80), input_latents=x["latents"],
                  first_frame_latents=x["first_frame_latents"], context=x["context"],
                  context_mask=x["context_mask"], proprio=x["proprio"],
                  proprio_mask=x["_proprio_sample_mask"], fuse_vae_embedding_in_latents=True,
                  ea_targets=target())
    torch.manual_seed(42)
    old = original.compute_loss(**kwargs)
    torch.manual_seed(42)
    new = reduced.compute_loss(**kwargs)
    torch.testing.assert_close(new["loss_ea"], old["loss_ea"] * 0.5)
    torch.testing.assert_close(new["loss"], old["loss"] - old["loss_ea"] * 0.5)


def inputs(dtype=torch.float32):
    mask = torch.zeros(1, 80, dtype=torch.bool)
    mask[:, :10] = mask[:, 34:44] = True
    clean = torch.randn(1, 4, 1, 12, 8, dtype=dtype)
    latents = torch.randn(1, 4, 3, 12, 8, dtype=dtype)
    latents[:, :, :1] = clean
    return dict(
        first_frame_latents=clean,
        latents=latents,
        context=torch.randn(1, 3, 32, dtype=dtype),
        context_mask=torch.tensor([[True, True, False]]),
        timestep=torch.tensor([111.0], dtype=dtype),
        proprio=torch.randn(1, 80, dtype=dtype) * mask,
        _proprio_sample_mask=mask[:, None],
        fuse_vae_embedding_in_latents=True,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_identity(make, dtype, checkpointed, enabled):
    model, base = make(enabled=enabled, dtype=dtype), make(cls=DualSystemSelfAttnArchitecture, dtype=dtype)
    base.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("sidecar.")})
    model.train()
    base.train()
    x, actions = inputs(dtype), torch.randn(1, 32, 80, dtype=dtype)
    expected = base(actions, torch.ones(1, dtype=dtype), use_gradient_checkpointing=checkpointed, **x)
    actual = model(actions, torch.ones(1, dtype=dtype), use_gradient_checkpointing=checkpointed, **x)
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("intervention", ["future_video", "action_noise", "video_time", "action_time", "padded_text"])
def test_causality_and_deploy_parity(make, intervention):
    model, x = make(), inputs()
    model.eval()
    for stage in model.sidecar.stages:
        stage.write.beta.data.fill_(0.3)
    actions = torch.randn(1, 32, 80)
    ref = model.forward_with_aux(actions, torch.ones(1), **x)
    changed = deepcopy(x)
    other_actions, other_time = actions, torch.ones(1)
    if intervention == "future_video":
        changed["latents"][:, :, 1:] = torch.randn_like(changed["latents"][:, :, 1:]) * 50
    elif intervention == "action_noise":
        other_actions = actions * 100
    elif intervention == "video_time":
        changed["timestep"] = torch.tensor([999.0])
    elif intervention == "action_time":
        other_time = torch.tensor([999.0])
    else:
        changed["context"][:, 2] = 1000
    with torch.inference_mode():
        other = model.forward_with_aux(other_actions, other_time, **changed)
    for a, b in zip(ref[2], other[2]):
        for key in ("read_hidden", "tokens", "plan_directions", "plan_moving_logits", "near_logits"):
            torch.testing.assert_close(a[key], b[key], rtol=2e-5, atol=2e-6)
    if intervention == "action_noise":
        assert not torch.allclose(ref[1], other[1])
    model.train()
    training = model.forward_with_aux(other_actions, other_time, **changed)
    for a, b in zip(ref[2], training[2]):
        torch.testing.assert_close(a["tokens"], b["tokens"], rtol=2e-5, atol=2e-6)
    with pytest.raises(ValueError, match="labels"):
        model(actions, torch.ones(1), ea_targets=target(), **x)
    model.apply_compile_optimizations(dict(enabled=True))
    assert model._compiled_mot_run_joint_loop is None


@pytest.mark.parametrize("stage", ["warmup", "action_only", "joint"])
def test_full_loss_optimizer_and_resume(make, tmp_path, stage):
    model, x = make(stage=stage), inputs()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    old = model.sidecar.role_tokens.detach().clone()
    for step in range(2):
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
            ea_targets=target(),
        )
        assert all(torch.isfinite(v).all() for v in loss.values())
        loss["loss"].backward()
        for block in model.sidecar.stages:
            assert block.projector[1].weight.grad.norm() > 0
            assert block.write.beta.grad.abs() > 0
            if step:
                assert block.write.projection.weight.grad.norm() > 0
        assert model.sidecar.heads.plan.weight.grad.norm() > 0
        assert model.sidecar.heads.near.weight.grad.norm() > 0
        if stage != "joint":
            assert all(p.grad is None for p in model.video_backbone.parameters())
        optimizer.step()
    assert not torch.equal(old, model.sidecar.role_tokens)
    path = tmp_path / "resume.pt"
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), path)
    restored = make(stage=stage)
    saved = torch.load(path, weights_only=True)
    restored.load_state_dict(saved["model"])
    opt = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=1e-3)
    opt.load_state_dict(saved["optimizer"])
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, restored.state_dict()[k], rtol=0, atol=0)
    assert opt.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()


def test_checkpoint_gradients(make):
    model, other = make(), make()
    for block in model.sidecar.stages:
        block.write.beta.data.fill_(0.1)
    other.load_state_dict(model.state_dict())
    x, actions = inputs(), torch.randn(1, 32, 80)
    for current, checkpointed in ((model, False), (other, True)):
        v, a, aux = current.forward_with_aux(actions, torch.ones(1), use_gradient_checkpointing=checkpointed, **x)
        (
            v.square().mean() + a.square().mean() + sum(current.sidecar.heads.losses(aux[-1], target()).values())
        ).backward()
    for (name, p), (_, q) in zip(model.named_parameters(), other.named_parameters()):
        if p.grad is not None:
            assert q.grad is not None, name
            torch.testing.assert_close(p.grad, q.grad, rtol=5e-4, atol=1e-5, msg=name)


def test_canvas_and_invalid_masks(make):
    boxes = torch.tensor([0.2, 0.2, 0.8, 0.8]).repeat(1, 4, 3, 1)
    for view in range(3):
        valid = torch.zeros(1, 4, 3)
        valid[:, :, view] = 1
        density = canvas_bbox_targets(boxes, valid, 12, 10)
        torch.testing.assert_close(density.sum((-3, -2, -1)), torch.ones(1, 4))
        y, x = divmod(int(density[0, 0].argmax()), 10)
        assert y < 8 if view == 0 else y >= 8 and (x < 5 if view == 1 else x >= 5)
    model, x = make(), inputs()
    x["_proprio_sample_mask"][:, :, 34:44] = False
    x["proprio"][:, 34:44] = 0
    _, _, aux = model.forward_with_aux(torch.randn(1, 32, 80), torch.ones(1), **x)
    labels = torch.zeros_like(target())
    parts = model.sidecar.heads.losses(aux[-1], labels)
    assert parts["plan"] == 0 and parts["relation"] == 0
    assert all(torch.isfinite(v) for v in parts.values())
    assert (aux[-1]["tokens"][:, [2, 3, 6, 7, 10, 12, 14]] == 0).all()
    labels[:, :, :2] = 1
    labels[:, :, 64:70] = 1  # Valid future, stationary roles: direction supervision is disabled.
    stationary = model.sidecar.heads.losses(aux[-1], labels)["plan"]
    labels[:, :, 16:52] = torch.randn(1, 1, 36)
    torch.testing.assert_close(model.sidecar.heads.losses(aux[-1], labels)["plan"], stationary)
    labels = target()
    labels[:, :, 14:16] = 0  # Invisible/virtual goal: its box cannot affect grounding.
    first = model.sidecar.heads.losses(aux[-1], labels)["ground"]
    boxes = labels[:, :, 70:118].reshape(1, 1, 4, 3, 4)
    boxes[:, :, [1, 3]] = torch.tensor([0.0, 0.0, 0.1, 0.1])
    torch.testing.assert_close(model.sidecar.heads.losses(aux[-1], labels)["ground"], first)


def test_aux_only_and_persistent_read(make):
    model, x = make(conditioning=False), inputs()
    _, _, outputs = model.forward_with_aux(torch.randn(1, 32, 80), torch.ones(1), **x)
    model.sidecar.heads.losses(outputs[-1], target())["plan"].backward()
    assert model.sidecar.stages[0].projector[1].weight.grad.norm() > 0
    assert all(p.grad is None for s in model.sidecar.stages for p in s.write.parameters())
    for s in model.sidecar.stages[1:]:
        s.alpha.data.zero_()
    first = model.forward_with_aux(torch.randn(1, 32, 80), torch.ones(1), **x)[2][-1]["tokens"]
    model.sidecar.stages[0].alpha.data.fill_(0.5)
    second = model.forward_with_aux(torch.randn(1, 32, 80), torch.ones(1), **x)[2][-1]["tokens"]
    assert not torch.allclose(first, second)  # Later zero read gates do not reset earlier information.


def test_strict_foundation_warmstart(make, tmp_path):
    from safetensors.torch import save_file

    base = make(cls=DualSystemSelfAttnArchitecture)
    path = tmp_path / "foundation.safetensors"
    save_file(base.state_dict(), path)
    model = make()
    model.load_checkpoint(str(path))
    assert model.foundation_load_report["reset_policy_heads"] == []
    for name, tensor in base.state_dict().items():
        torch.testing.assert_close(tensor, model.state_dict()[name], rtol=0, atol=0)
    state = base.state_dict()
    state.pop(next(iter(state)))
    save_file(state, path)
    with pytest.raises(RuntimeError, match="keys/shapes"):
        model.load_checkpoint(str(path))


def test_actual_generate_loop(make, monkeypatch):
    model, x = make(), inputs()
    pipeline = {k: v for k, v in x.items() if k not in ("proprio", "_proprio_sample_mask", "timestep")}
    monkeypatch.setattr(model.video_backbone, "preprocess_input_for_inference", lambda **kwargs: deepcopy(pipeline))
    calls = []
    handle = model.sidecar.stages[2].register_forward_hook(lambda *args: calls.append(True))
    result = model.generate(
        prompt="test task",
        schedule=[(1000, 1000), (500, 500), (0, 0)],
        proprio=x["proprio"],
        active_action_mask=x["_proprio_sample_mask"][0, 0],
        action_num_frames=33,
        decode_video=False,
    )
    handle.remove()
    assert len(calls) == 2
    assert torch.isfinite(torch.as_tensor(result["actions"])).all()
    assert result["actions"].shape[-2:] == (32, 80)


def test_actual_accelerate_resume(make, tmp_path):
    from accelerate import Accelerator

    from openwam.train.utils.checkpointing import load_full_state, save_full_state

    accelerator = Accelerator(cpu=True)
    model = make(stage="warmup")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    x = inputs()

    def step():
        optimizer.zero_grad()
        losses = model.compute_loss(
            actions=torch.randn(1, 32, 80),
            input_latents=x["latents"],
            first_frame_latents=x["first_frame_latents"],
            context=x["context"],
            context_mask=x["context_mask"],
            proprio=x["proprio"],
            proprio_mask=x["_proprio_sample_mask"],
            ea_targets=target(),
            fuse_vae_embedding_in_latents=True,
        )
        accelerator.backward(losses["loss"])
        optimizer.step()
        scheduler.step()
        return losses["loss"].detach().clone()

    step()
    save_full_state(accelerator, str(tmp_path), 1, 1, 0)
    reference = step()
    expected = deepcopy(model.state_dict())
    expected_opt = deepcopy(optimizer.state_dict())
    restored = load_full_state(accelerator, str(tmp_path / "accel_state_step_1"))
    assert restored["opt_step"] == 1
    torch.testing.assert_close(step(), reference, rtol=0, atol=0)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    for key, values in optimizer.state_dict()["state"].items():
        for field, value in values.items():
            torch.testing.assert_close(value, expected_opt["state"][key][field], rtol=0, atol=0)
    accelerator.end_training()


def test_deploy_engine_routes_state_and_rejects_oracle(monkeypatch):
    from pathlib import Path

    import numpy as np

    from openwam.deploy import engine as module
    from openwam.deploy.server import _load_deploy_yaml

    cfg = _load_deploy_yaml(str(Path(__file__).resolve().parents[1] / "configs/deploy_relational_sidecar.yaml"))
    assert not cfg.optimization.compile.enabled and not cfg.optimization.dit_cache.enabled
    monkeypatch.setattr(module, "make_schedule", lambda *a, **k: [(1000, 1000), (0, 0)])

    class Architecture:
        sidecar = object()
        device, dtype = torch.device("cpu"), torch.float32
        video_scheduler = action_scheduler = object()

        def normalize_deploy_proprio(self, value):
            return torch.as_tensor(value)

        def generate(self, **kwargs):
            self.arguments = kwargs
            return {"actions": np.zeros((32, 20)), "video": None}

    architecture = Architecture()
    engine = module.JointInferenceEngine(cfg, architecture=architecture)
    engine.generate(dict(proprio=np.zeros(20)))
    assert architecture.arguments["num_frames"] == 9
    assert architecture.arguments["action_num_frames"] == 33
    assert "ea_effectors" not in architecture.arguments
    for observation in ({"ea_targets": np.zeros(130)},):
        with pytest.raises(ValueError, match="oracle"):
            engine.generate({"observation": observation})
