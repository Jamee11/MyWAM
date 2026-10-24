"""Exercise the actual trainer loop's restored counter across accumulated updates."""

from contextlib import contextmanager
import torch
from omegaconf import OmegaConf

from openwam.train.openwam_trainer import OpenWAMTrainer


def test_restored_optimizer_counter_is_constant_within_accumulation(tmp_path):
    class Accelerator:
        num_processes = 1
        is_main_process = False
        sync_gradients = False
        micro = 0

        @contextmanager
        def accumulate(self, model):
            self.micro += 1
            self.sync_gradients = self.micro % 2 == 0
            yield

        def backward(self, loss):
            (loss / 2).backward()

        def unwrap_model(self, model):
            return model

    trainer = object.__new__(OpenWAMTrainer)
    trainer.cfg = OmegaConf.create(dict(training=dict(
        batch_size=1, gradient_accumulation_steps=2, num_epochs=None, max_steps=4004,
        resume_ckpt_path="existing_state", save_steps=None,
    )))
    trainer.accelerator = Accelerator()
    trainer.architecture = torch.nn.Linear(1, 1)
    seen = []
    trainer.architecture.set_training_progress = seen.append
    trainer._run_seed, trainer._rank, trainer.dataset = None, 0, None
    trainer.build_optimizer = lambda: torch.optim.SGD(trainer.architecture.parameters(), lr=0.01)
    trainer.build_dataloader = lambda batch_size: list(range(6))
    trainer.build_lr_scheduler = lambda *args, **kwargs: None
    trainer.setup_output_dir = lambda *args: (str(tmp_path), "existing_state")
    trainer.prepare_accelerate = lambda *args: args
    trainer.resume_if_configured = lambda *args: (3998, 1999, 0, 0)
    trainer.compute_loss = lambda batch: {
        "total": trainer.architecture(torch.ones(1, 1)).square().sum(), "video": 0.0, "action": 0.0
    }
    trainer.log_step = lambda **kwargs: None
    trainer.finish_training = lambda *args: None
    trainer.train()
    assert seen == [1999, 1999, 2000, 2000, 2001, 2001]
