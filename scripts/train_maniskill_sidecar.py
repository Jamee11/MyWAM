#!/usr/bin/env python3
"""Register ManiSkill auxiliary reader; reuse Sidecar architecture and baseline metrics/trainer."""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_WARM_POOL", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def write_maniskill_contract(output_path):
    """Correct the existing Robotwin-specific descriptive export, without changing model math."""
    path = Path(output_path, "relational_sidecar_contract.json")
    contract = json.loads(path.read_text())
    contract.update(action_frame="robot_base", eef_reference="panda_hand_tcp", tool_offset_m=0.0,
                    action_slots=list(range(10)), source_quaternion_order="wxyz",
                    camera_layout=["front_camera", "hand_camera", None],
                    plan_frame="each_role_current_local_rotation", embodiment="panda_wristcam",
                    action_semantics="absolute_controller_target_at_t_for_transition_t_to_t_plus_1")
    path.write_text(json.dumps(contract, indent=2) + "\n")


def validate_save_boundaries(micro_per_epoch, accumulation, save_steps, total_micro):
    """The shared trainer saves by global microstep; reject resumable saves mid-accumulation."""
    points = list(range(save_steps, total_micro + 1, save_steps)) + [total_micro] if save_steps else []
    for point in points:
        within_epoch = point % micro_per_epoch
        if within_epoch and within_epoch % accumulation:
            raise ValueError(
                f"Save at microstep {point} is inside gradient accumulation: epoch has {micro_per_epoch} "
                f"microsteps, accumulation={accumulation}. Set save_steps to the epoch length "
                "and end short runs on a synchronized update."
            )


def restore_saved_counter(result, state_dir):
    """Epoch-end partial accumulation means opt_step is not simply global_micro // accumulation."""
    meta = json.loads(Path(state_dir, "trainer_state.json").read_text())
    global_step, _, epoch, skip = result
    if global_step != meta["global_step"]:
        raise ValueError("Checkpoint was aligned backward; resume only synchronized Sidecar checkpoints")
    return global_step, int(meta["opt_step"]), epoch, skip


def main():
    import train_maniskill_baseline as baseline

    from openwam.dataloader.maniskill_sidecar import ManiSkillSidecarDataset
    from openwam.dataloader.registry import register_dataset
    from openwam.model.architectures.dual_system.relational_sidecar import RelationalSidecarArchitecture
    from openwam.train.openwam_trainer import OpenWAMTrainer

    register_dataset("maniskill_sidecar")(ManiSkillSidecarDataset)
    original = RelationalSidecarArchitecture.save_assets_for_deployment

    def save(self, output_path, cfg):
        original(self, output_path, cfg)
        write_maniskill_contract(output_path)

    RelationalSidecarArchitecture.save_assets_for_deployment = save
    original_prepare = OpenWAMTrainer.prepare_accelerate

    def prepare(self, optimizer, dataloader, scheduler):
        result = original_prepare(self, optimizer, dataloader, scheduler)
        t = self.cfg.training
        if t.save_full_states_for_resume:
            epoch_micro = len(result[1])
            total = epoch_micro * int(t.num_epochs) if t.num_epochs is not None else int(t.max_steps)
            if t.max_steps:
                total = min(total, int(t.max_steps))
            validate_save_boundaries(epoch_micro, int(t.gradient_accumulation_steps), int(t.save_steps or 0), total)
        return result

    OpenWAMTrainer.prepare_accelerate = prepare
    original_resume = OpenWAMTrainer.resume_if_configured

    def resume(self, resume_state_dir, dataloader, grad_accum):
        result = original_resume(self, resume_state_dir, dataloader, grad_accum)
        return restore_saved_counter(result, resume_state_dir)

    OpenWAMTrainer.resume_if_configured = resume
    baseline.main()


if __name__ == "__main__":
    main()
