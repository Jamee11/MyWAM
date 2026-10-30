#!/usr/bin/env python3
"""Register the isolated sim mixture and reuse the unchanged Sidecar trainer."""

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TORCH_WARM_POOL', '0')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def data_contract(cfg):
    manifest = Path(cfg.dataloader.manifest_path).resolve()
    data = json.loads(manifest.read_text())
    return dict(dataset_contract=data['contract'], dataset_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                normalization_stats_sha256=data['normalization_stats_sha256'],
                groups=['rlbench_franka', 'maniskill_franka', 'robotwin_aloha', 'robotwin_piper'],
                weights=list(cfg.dataloader.weights), seed=int(cfg.project.seed),
                epoch_alignment=int(cfg.dataloader.epoch_alignment), episodes=data['total_episodes'],
                world_size=int(os.environ.get('WORLD_SIZE', '1')),
                batch_size=int(cfg.training.batch_size), accumulation=int(cfg.training.gradient_accumulation_steps),
                planned_microsteps=cfg.training.max_steps, planned_epochs=cfg.training.num_epochs,
                source_counts=data['counts'], retained_splits=data['splits'],
                source_action_semantics=['absolute_controller_target', 'next_achieved_absolute_eef'],
                arm_slots={'left': list(range(10)), 'right': list(range(34, 44))},
                plan_steps=[8, 16, 32], video_anchors=list(range(0, 33, 4)))


def validate_warmup_optimizer(training, backbone_updates):
    if backbone_updates and str(getattr(training, "offload_optimizer_device", "none")) != "none":
        raise ValueError("CPU optimizer offload replaces AdamW and drops WAM warmup hooks; use offload_optimizer_device=none")


def main():
    import train_maniskill_baseline as baseline
    from train_maniskill_sidecar import restore_saved_counter, validate_save_boundaries

    from openwam.dataloader.multisim_sidecar import MultiSimSidecarDataset
    from openwam.dataloader.registry import register_dataset
    from openwam.model.architectures.dual_system.relational_sidecar import RelationalSidecarArchitecture
    from openwam.train.openwam_trainer import OpenWAMTrainer

    register_dataset('multisim_sidecar')(MultiSimSidecarDataset)
    original_save = RelationalSidecarArchitecture.save_assets_for_deployment

    def save(self, output_path, cfg):
        original_save(self, output_path, cfg)
        path = Path(output_path, 'relational_sidecar_contract.json')
        contract = json.loads(path.read_text())
        contract.update(embodiment='multi_simulator', action_frame='source_specific_see_midtrain_data_contract',
                        action_semantics='source_specific_absolute_EEF', policy_rotation='first_two_matrix_columns',
                        privileged_inputs=False)
        path.write_text(json.dumps(contract, indent=2) + '\n')
        Path(output_path, 'midtrain_data_contract.json').write_text(json.dumps(data_contract(cfg), indent=2) + '\n')

    RelationalSidecarArchitecture.save_assets_for_deployment = save
    original_prepare = OpenWAMTrainer.prepare_accelerate

    def prepare(self, optimizer, dataloader, scheduler):
        validate_warmup_optimizer(self.cfg.training, self.architecture.backbone_warmup_updates)
        result = original_prepare(self, optimizer, dataloader, scheduler)
        t = self.cfg.training
        epoch_micro = len(result[1])
        accumulation = int(t.gradient_accumulation_steps)
        if epoch_micro % accumulation:
            raise ValueError('Mixture epoch ends mid-accumulation; align epoch_alignment with global batch')
        total = int(t.max_steps) if t.max_steps else epoch_micro * int(t.num_epochs)
        if t.save_full_states_for_resume:
            validate_save_boundaries(epoch_micro, accumulation, int(t.save_steps or 0), total)
        return result

    OpenWAMTrainer.prepare_accelerate = prepare
    original_resume = OpenWAMTrainer.resume_if_configured

    def resume(self, state_dir, dataloader, accumulation):
        saved = json.loads(Path(state_dir).parent.joinpath('midtrain_data_contract.json').read_text())
        if saved != data_contract(self.cfg):
            raise ValueError('Resume dataset identity, mixture, or seed differs; use finetune for a new stage')
        result = original_resume(self, state_dir, dataloader, accumulation)
        return restore_saved_counter(result, state_dir)

    OpenWAMTrainer.resume_if_configured = resume
    baseline.main()


if __name__ == '__main__':
    main()
