"""Actual tiny WAM: full gradients, frozen updates, joint transition and resume."""

from types import SimpleNamespace

import pytest
import torch

from openwam.train.openwam_trainer import OpenWAMTrainer
from tests.test_relational_sidecar import inputs, target

pytest_plugins = ["tests.test_relational_sidecar"]


def optimizer_for(model):
    cfg = SimpleNamespace(
        training=SimpleNamespace(
            learning_rate=1e-3, action_lr=1e-3, video_lr=1e-3, sidecar_lr=5e-4, weight_decay=0.1, adam_fused=False
        )
    )
    return OpenWAMTrainer.build_optimizer(SimpleNamespace(architecture=model, cfg=cfg))


def loss(model, checkpointed=False):
    x = inputs()
    x = {
        k: v.to(device=model.device, dtype=model.dtype if v.is_floating_point() else v.dtype)
        if isinstance(v, torch.Tensor)
        else v
        for k, v in x.items()
    }
    return model.compute_loss(
        actions=torch.randn(1, 32, 80, device=model.device, dtype=model.dtype),
        input_latents=x["latents"],
        first_frame_latents=x["first_frame_latents"],
        context=x["context"],
        context_mask=x["context_mask"],
        proprio=x["proprio"],
        proprio_mask=x["_proprio_sample_mask"],
        ea_targets=target().to(model.device, model.dtype),
        use_gradient_checkpointing=checkpointed,
        fuse_vae_embedding_in_latents=True,
    )["loss"]


@pytest.mark.parametrize("checkpointed", [False, True])
def test_full_forward_sidecar_gradients_and_frozen_wam(make, checkpointed):
    model = make(backbone_updates=2000)
    reference = make()
    for stage in model.sidecar.stages:
        stage.write.beta.data.fill_(0.1)
    reference.load_state_dict(model.state_dict())
    opt = optimizer_for(model)
    reference_opt = optimizer_for(reference)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    model.set_training_progress(1999)
    torch.manual_seed(11)
    value = loss(model, checkpointed)
    value.backward()
    torch.manual_seed(11)
    expected = loss(reference, checkpointed)
    expected.backward()
    torch.testing.assert_close(value, expected, rtol=0, atol=0)
    for n, p in model.named_parameters():
        expected_grad = reference.get_parameter(n).grad
        if expected_grad is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, expected_grad, rtol=0, atol=0, msg=n)
    assert any(
        p.grad is not None and p.grad.count_nonzero() > 0
        for n, p in model.named_parameters()
        if not n.startswith("sidecar.")
    )
    assert all(s.write.projection.weight.grad.norm() > 0 for s in model.sidecar.stages)
    # Force clipping to exercise WAM's contribution to the shared scaling factor.
    max_norm = (
        float(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])))
        / 2
    )
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), max_norm)
    assert norm > max_norm
    torch.testing.assert_close(norm, expected_norm, rtol=0, atol=0)
    for n, p in model.named_parameters():
        if p.grad is not None:
            torch.testing.assert_close(p.grad, reference.get_parameter(n).grad, rtol=0, atol=0, msg=n)
    opt.step()
    reference_opt.step()
    for n, p in model.named_parameters():
        if not n.startswith("sidecar."):
            torch.testing.assert_close(p, before[n], rtol=0, atol=0)
            assert not opt.state.get(p)
        else:
            torch.testing.assert_close(p, reference.get_parameter(n), rtol=0, atol=0, msg=n)
    assert any(not torch.equal(p, before[n]) for n, p in model.sidecar.named_parameters(prefix="sidecar"))
    opt.zero_grad(set_to_none=True)
    model.set_training_progress(2000)
    loss(model, checkpointed).backward()
    opt.step()
    assert any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if not n.startswith("sidecar."))
    assert any(opt.state.get(p) for n, p in model.named_parameters() if not n.startswith("sidecar."))


@pytest.mark.parametrize("saved_step", [1, 2, 3])
def test_full_optimizer_resume_before_at_after_boundary(make, tmp_path, saved_step):
    model = make(backbone_updates=2)
    opt = optimizer_for(model)
    for step in range(saved_step):
        model.set_training_progress(step)
        torch.manual_seed(step)
        loss(model).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
    path = tmp_path / "full.pt"
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), opt_step=saved_step), path)
    restored = make(backbone_updates=2)
    resumed_opt = optimizer_for(restored)
    state = torch.load(path, weights_only=True)
    restored.load_state_dict(state["model"])
    resumed_opt.load_state_dict(state["optimizer"])
    for m, o in [(model, opt), (restored, resumed_opt)]:
        m.set_training_progress(state["opt_step"])
        torch.manual_seed(101)
        loss(m).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        o.step()
        o.zero_grad()
    for n, p in model.named_parameters():
        torch.testing.assert_close(p, restored.get_parameter(n), rtol=0, atol=0)
    torch.testing.assert_close(opt.state_dict(), resumed_opt.state_dict(), rtol=0, atol=0)


def test_both_warmups_rejected_and_eval_ignores_counter(make):
    with pytest.raises(ValueError, match="overlap"):
        make(isolation_updates=2, backbone_updates=2)
    with pytest.raises(ValueError, match="nonnegative"):
        make(backbone_updates=-1)
    with pytest.raises(ValueError, match="joint"):
        make(stage="warmup", backbone_updates=2)
    with pytest.raises(ValueError, match="enabled"):
        make(enabled=False, backbone_updates=2)
    model = make(backbone_updates=2)
    assert model.backbone_warmup_active and not model.isolation_warmup_active
    model.eval()
    assert not model.backbone_warmup_active
