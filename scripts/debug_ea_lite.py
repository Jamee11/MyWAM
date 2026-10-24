"""Official trainer with explicit diagnostics and clean planned save/resume stops."""
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import torch  # noqa: E402 -- source-tree imports follow sys.path bootstrap
import train as entry  # noqa: E402 -- source-tree imports follow sys.path bootstrap

import openwam.train.openwam_trainer as trainer_module  # noqa: E402 -- source-tree imports follow sys.path bootstrap

Trainer = trainer_module.OpenWAMTrainer
original_init, original_log = Trainer.__init__, Trainer.log_step
original_save, original_train = trainer_module.save_full_state, entry._train


class PlannedStop(Exception):
    pass


def diagnostic_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self._debug_gradients = {}
    wanted = ("ea_bridge.patch_projection.1.weight", "ea_bridge.core.ground_key.weight",
              "ea_bridge.core.relation_encoder.0.weight", "ea_bridge.core.eef_plan_head.0.weight",
              "ea_bridge.action_attention.in_proj_weight", "ea_bridge.action_out.weight",
              "video_backbone.dit.patch_embedding.weight", "action_backbone.action_encoder.weight")
    for name, parameter in self.architecture.named_parameters():
        if name in wanted and parameter.requires_grad:
            def record(grad, name=name):
                self._debug_gradients[name] = grad.detach().float().norm()
            parameter.register_hook(record)


def diagnostic_log(self, **kwargs):
    metrics = kwargs["metrics"]
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise RuntimeError("Nonfinite trainer metrics")
    row = dict(micro_step=kwargs["global_step"], optimizer_step=kwargs["opt_step"],
               seconds_micro=1 / kwargs["steps_per_sec"], metrics=metrics,
               batch_per_gpu=int(self.cfg.training.batch_size), world_size=self.accelerator.num_processes,
               accumulation=int(self.cfg.training.gradient_accumulation_steps),
               gradient_norms={k: float(v) for k, v in self._debug_gradients.items()},
               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
               peak_reserved_bytes=torch.cuda.max_memory_reserved())
    path = Path(kwargs["output_path"]) / f"diagnostics_rank{self._rank}.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    self._debug_gradients.clear()
    original_log(self, **kwargs)
    if kwargs["global_step"] % int(self.cfg.training.gradient_accumulation_steps) == 0:
        torch.cuda.reset_peak_memory_stats()


def saved_stop(accelerator, output_path, global_step, opt_step, epoch):
    original_save(accelerator, output_path, global_step, opt_step, epoch)
    accelerator.wait_for_everyone()
    requested = int(os.environ.get("EA_STOP_AFTER_OPT_STEPS", "0"))
    if requested and opt_step >= requested:
        raise PlannedStop(f"Saved all ranks at optimizer_step={opt_step}, micro_step={global_step}")


def guarded_train(cfg):
    requested = int(os.environ.get("EA_STOP_AFTER_OPT_STEPS", "0"))
    if requested:
        micro = requested * int(cfg.training.gradient_accumulation_steps)
        save_steps = int(cfg.training.save_steps or 0)
        if not cfg.training.save_full_states_for_resume or not save_steps or micro % save_steps:
            raise ValueError("Planned stop must coincide with a full-state save boundary")
    try:
        original_train(cfg)
    except PlannedStop as stop:
        print(str(stop), flush=True)
    except BaseException:
        # Exit promptly on one-rank failure; do not hang collective teardown.
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)


Trainer.__init__, Trainer.log_step = diagnostic_init, diagnostic_log
trainer_module.save_full_state, entry._train = saved_stop, guarded_train
if __name__ == "__main__":
    entry.main()
