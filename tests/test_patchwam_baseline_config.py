"""The no-Sidecar control must retain the joint experiment's data and WAM budget."""
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from tests.test_cross_appearance import loss_inputs
from tests.test_relational_sidecar import make


def test_baseline_matches_joint_recipe_and_preserves_old_baseline():
    with initialize_config_dir(config_dir=str(Path('configs').resolve()), version_base=None):
        baseline = compose(config_name='train_alpha_patchwam_baseline_10epoch_4gpu')
        joint = compose(config_name='train_sidecar_grounded_joint10epoch_4gpu')
        old = compose(config_name='train_alpha_baseline_clean50')
    assert baseline.model.architecture.variant == 'joint_self_attn'
    assert baseline.model.architecture.relational_sidecar is None
    assert baseline.model.freeze == joint.model.freeze
    a, b = (OmegaConf.to_container(c.dataloader, resolve=False) for c in (baseline, joint))
    assert a.pop('auxiliary') is False and b.pop('auxiliary') is True
    assert a == b
    a, b = (OmegaConf.to_container(c.training, resolve=False) for c in (baseline, joint))
    assert a.pop('sidecar_lr') is None and b.pop('sidecar_lr') == 5e-5
    a.pop('output_path'); b.pop('output_path')
    assert a == b
    assert baseline.training.num_epochs == 10 and baseline.training.save_optimizer_steps == 5000
    assert baseline.training.batch_size * baseline.training.gradient_accumulation_steps * 4 == 256
    assert old.training.num_epochs == 5 and old.dataloader.color_jitter.enabled


def test_original_model_trains_without_sidecar_or_ea(make):
    model = make(cls=DualSystemSelfAttnArchitecture)
    assert not any('sidecar' in key for key in model.state_dict())
    batch = {k: v for k, v in loss_inputs().items() if not k.startswith(('ea_', 'xa_'))}
    losses = model.compute_loss(**batch)
    assert all('ea' not in key for key in losses)
    assert all(torch.isfinite(v).all() for v in losses.values())
    before = model.action_backbone.action_decoder.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses['loss'].backward()
    optimizer.step()
    assert not torch.equal(before, model.action_backbone.action_decoder.weight)
