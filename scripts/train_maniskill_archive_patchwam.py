#!/usr/bin/env python3
"""Fine-tune the existing 80D midtrain policy on only the archived ManiSkill train split."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import train_maniskill_baseline as baseline
    from openwam.dataloader.maniskill_archive import ManiSkillArchiveDataset
    from openwam.dataloader.registry import register_dataset
    from openwam.model.architectures.dual_system.relational_sidecar import RelationalSidecarArchitecture

    register_dataset('maniskill_archive')(ManiSkillArchiveDataset)
    original = RelationalSidecarArchitecture.save_assets_for_deployment

    def save(self, output_path, cfg):
        original(self, output_path, cfg)
        path = Path(output_path, 'relational_sidecar_contract.json')
        contract = json.loads(path.read_text())
        contract.update(embodiment='maniskill_franka', action_frame='robot_base_tcp',
                        action_semantics='absolute_controller_target', privileged_inputs=False,
                        action_slots=list(range(10)), unified_action_dim=80,
                        unify_action_map=['0-9', '34-43'], arm_present=[1, 0],
                        camera_layout=['front_camera', 'hand_camera', None])
        path.write_text(json.dumps(contract, indent=2) + '\n')

    RelationalSidecarArchitecture.save_assets_for_deployment = save
    baseline.main()


if __name__ == '__main__':
    main()
