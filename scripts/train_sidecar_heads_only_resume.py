"""Resume the isolated 40k fork, reusing the original GPU gate and metrics."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_resume_identity(run, output, state_dir):
    run = Path(run).resolve()
    if Path(output).resolve() != run or Path(state_dir).resolve() != run / "accel_state_step_40000":
        raise ValueError("Heads-only fork must resume its own exact 40000 state and write to its own run")
    metadata = json.loads(Path(state_dir, "trainer_state.json").read_text())
    if metadata != dict(global_step=40000, opt_step=20000, epoch=9):
        raise ValueError("Expected micro40000 / optimizer20000 / epoch9")
    return metadata


def main():
    import train_relational_sidecar_formal as formal

    from openwam.train.openwam_trainer import OpenWAMTrainer

    run = Path(os.environ["HEADS_ONLY_FORK_RUN"]).resolve()
    fork_root = Path(os.environ["HEADS_ONLY_FORK_ROOT"]).resolve()
    plan = json.loads((fork_root / "fork_plan.json").read_text())
    if run != Path(plan["run"]).resolve() or run.is_relative_to(Path(plan["source"]).resolve()):
        raise ValueError("Fork output must be independent of the source run")
    original_setup = OpenWAMTrainer.setup_output_dir
    original_resume = OpenWAMTrainer.resume_if_configured

    def setup(self, debug, resume_path):
        output, state_dir = original_setup(self, debug, resume_path)
        metadata = validate_resume_identity(run, output, state_dir)
        architecture = self.architecture
        training = self.cfg.training
        if not architecture.aux_heads_only_after_warmup or not architecture.sidecar.action_conditioning:
            raise ValueError("The saved fork config must enable heads-only EA and normal Action Write")
        if training.num_epochs != 10 or training.max_steps is not None:
            raise ValueError("Keep the original ten-epoch schedule; do not reset the training budget")
        if training.batch_size != 32 or training.gradient_accumulation_steps != 2:
            raise ValueError("Keep original bs32/acc2 for four-rank optimizer state")
        if self.accelerator.num_processes != 4:
            raise ValueError("The saved ZeRO optimizer requires its original four-rank topology")
        if self.dataset is None or len(self.dataset) != 549775:
            raise ValueError("The fork must retain the original Aloha2500 / 549775-window dataset")
        if self.accelerator.is_main_process:
            (fork_root / "cpu_contract.json").write_text(json.dumps(dict(
                output_path=output, state_dir=state_dir, source_state=metadata,
                auxiliary_head_parameters=sum(p.numel() for p in architecture.sidecar.heads.parameters()),
                sidecar_parameters=sum(p.numel() for p in architecture.sidecar.parameters()),
                dataset_windows=len(self.dataset),
                aux_heads_only_after_warmup=True, target_microstep=42960, target_optimizer_updates=21480,
                status="cpu_contract_passed_gpu_and_resume_pending",
            ), indent=2) + "\n")
        return output, state_dir

    def resume(self, state_dir, dataloader, accumulation):
        result = original_resume(self, state_dir, dataloader, accumulation)
        micro, updates, epoch, skip = result
        if (micro, updates, epoch) != (40000, 20000, 9):
            raise ValueError("Restored counters differ from the isolated 40k source state")
        architecture = self.accelerator.unwrap_model(self.architecture)
        architecture.set_training_progress(updates)
        if not architecture.aux_heads_only_active or architecture.backbone_warmup_active:
            raise ValueError("40k must resume in joint training with EA routed to heads")
        if self.accelerator.is_main_process:
            (fork_root / "resume_verified.json").write_text(json.dumps(dict(
                micro_step=micro, optimizer_step=updates, epoch=epoch, skip_first_batches=skip,
                aux_heads_only_active=True, backbone_warmup_active=False,
                output_path=str(run), status="full_accelerate_state_restored",
            ), indent=2) + "\n")
        return result

    OpenWAMTrainer.setup_output_dir = setup
    OpenWAMTrainer.resume_if_configured = resume
    formal.main()


if __name__ == "__main__":
    main()
