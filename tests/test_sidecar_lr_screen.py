"""Effective LR interventions must preserve moments, unrelated params and schedules."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from openwam.train.openwam_trainer import OpenWAMTrainer
from openwam.train.utils.sidecar_lr_screen import SidecarLRScreen
from tests.test_sidecar_backbone_warmup import loss, optimizer_for

pytest_plugins = ["tests.test_relational_sidecar"]


@pytest.mark.parametrize("mode,ramp_updates", [("control", 300), ("wam_ramp", 300), ("wam_ramp", 100), ("write2", 300)])
def test_matches_actual_per_parameter_lr_and_resume(make, tmp_path, mode, ramp_updates):
    original = make(backbone_updates=2)
    optimizer = optimizer_for(original)
    for u in range(2):
        original.set_training_progress(u)
        torch.manual_seed(u)
        loss(original).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    expected = make(backbone_updates=2)
    expected.load_state_dict(original.state_dict())
    reference_groups = []
    reference_state = {}
    for group in optimizer.param_groups:
        for parameter in group['params']:
            name = next(n for n, p in original.named_parameters() if p is parameter)
            ref = expected.get_parameter(name)
            reference_groups.append(dict(**{k: v for k, v in group.items() if k != 'params'}, params=[ref]))
            if parameter in optimizer.state:
                reference_state[ref] = deepcopy(optimizer.state[parameter])
    reference = torch.optim.AdamW(reference_groups)
    reference.state.update(reference_state)
    controller = SidecarLRScreen(optimizer, original, mode, ramp_updates=ramp_updates)
    for u in (2, 3, 2 + ramp_updates):
        original.set_training_progress(u)
        expected.set_training_progress(u)
        for model in (original, expected):
            torch.manual_seed(u)
            loss(model).backward()
        for group in reference.param_groups:
            ref = group['params'][0]
            name = next(n for n, p in expected.named_parameters() if p is ref)
            base = 1e-3 if group['ea_backbone_group'] else 5e-4
            group['lr'] = base * (controller.factor() if mode == 'wam_ramp' and group['ea_backbone_group'] else 1)
            if mode == 'write2' and name.startswith('sidecar.stages.') and '.write.' in name:
                group['lr'] = 2 * base
        optimizer.step()
        reference.step()
        for n, p in original.named_parameters():
            torch.testing.assert_close(p, expected.get_parameter(n), rtol=0, atol=0, msg=n)
            if p in optimizer.state:
                for key, value in optimizer.state[p].items():
                    torch.testing.assert_close(value, reference.state[expected.get_parameter(n)][key], rtol=0, atol=0)
        assert all(g['lr'] == (1e-3 if g['ea_backbone_group'] else 5e-4) for g in optimizer.param_groups)
        optimizer.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
    path = tmp_path / 'state.pt'
    torch.save(dict(model=original.state_dict(), optimizer=optimizer.state_dict()), path)
    restored = make(backbone_updates=2)
    restored_opt = optimizer_for(restored)
    SidecarLRScreen(restored_opt, restored, mode, ramp_updates=ramp_updates)
    data = torch.load(path, weights_only=True)
    restored.load_state_dict(data['model'])
    restored_opt.load_state_dict(data['optimizer'])
    for model, opt in ((original, optimizer), (restored, restored_opt)):
        model.set_training_progress(303)
        torch.manual_seed(303)
        loss(model).backward()
        opt.step()
    for n, p in original.named_parameters():
        torch.testing.assert_close(p, restored.get_parameter(n), rtol=0, atol=0, msg=n)


@pytest.mark.parametrize('ramp_updates', [None, 100])
def test_training_factory_passes_selected_ramp_and_preserves_old_default(make, ramp_updates):
    model = make(backbone_updates=2000)
    options = dict(learning_rate=1e-3, action_lr=1e-3, video_lr=1e-3, sidecar_lr=5e-4,
                   weight_decay=.1, adam_fused=False, sidecar_lr_screen='wam_ramp')
    if ramp_updates is not None:
        options['sidecar_wam_ramp_updates'] = ramp_updates
    trainer = SimpleNamespace(architecture=model, cfg=SimpleNamespace(training=SimpleNamespace(**options)))
    OpenWAMTrainer.build_optimizer(trainer)
    controller = trainer.sidecar_lr_screen
    assert controller.ramp_updates == (300 if ramp_updates is None else ramp_updates)
    model.set_training_progress(2000)
    assert controller.factor() == .1
    model.set_training_progress(2050)
    assert controller.factor() == pytest.approx(.55 if ramp_updates == 100 else .25)
    model.set_training_progress(2100)
    assert controller.factor() == pytest.approx(1 if ramp_updates == 100 else .4)
