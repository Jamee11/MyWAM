"""Effective LR interventions must preserve moments, unrelated params and schedules."""
from copy import deepcopy

import pytest
import torch

from openwam.train.utils.sidecar_lr_screen import SidecarLRScreen
from tests.test_sidecar_backbone_warmup import loss, optimizer_for

pytest_plugins = ["tests.test_relational_sidecar"]


@pytest.mark.parametrize("mode", ["control", "wam_ramp", "write2"])
def test_matches_actual_per_parameter_lr_and_resume(make, tmp_path, mode):
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
    controller = SidecarLRScreen(optimizer, original, mode)
    for u in (2, 3, 302):
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
    SidecarLRScreen(restored_opt, restored, mode)
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
