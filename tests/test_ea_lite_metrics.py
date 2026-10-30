"""Optional EA metrics preserve the baseline reduction contract."""
from types import SimpleNamespace

import torch

from openwam.train.utils.training_utils import reduce_step_metrics


def test_baseline_and_ea_distributed_metrics():
    base = dict(total=torch.tensor(3.), video=torch.tensor(1.), action=torch.tensor(2.))
    gradient = torch.tensor(.5)
    expected = dict(loss_total=3., loss_video=1., loss_action=2., grad_norm=.5)
    assert reduce_step_metrics(None, base, gradient) == expected
    accelerator = SimpleNamespace(num_processes=2, gather=lambda tensor: torch.cat((tensor, tensor)))
    assert reduce_step_metrics(accelerator, base, gradient) == expected
    extras = dict(base, loss_ea_ground=torch.tensor(.2), ea_valid_plan_fraction=torch.tensor(.5))
    actual = reduce_step_metrics(accelerator, extras, gradient)
    assert actual["loss_total"] == 3. and actual["ea_valid_plan_fraction"] == .5
    torch.testing.assert_close(torch.tensor(actual["loss_ea_ground"]), torch.tensor(.2))
