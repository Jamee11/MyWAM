#!/usr/bin/env python3
"""One fresh two-epoch Piper finetune, with the existing CPU preload/handover path."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_piper_contract(output_path):
    path = Path(output_path, 'relational_sidecar_contract.json')
    contract = json.loads(path.read_text())
    contract.update(embodiment='robotwin_piper', action_frame='robotwin_endpose_reference',
                    action_semantics='next_achieved_absolute_eef', privileged_inputs=False,
                    action_slots=list(range(10))+list(range(34,44)), unified_action_dim=80,
                    unify_action_map=['0-9', '34-43'], arm_present=[1,1],
                    camera_layout=['head_camera', 'left_camera', 'right_camera'])
    path.write_text(json.dumps(contract, indent=2)+'\n')


def main():
    from openwam.dataloader.piper_archive import PiperArchiveDataset
    from openwam.dataloader.registry import register_dataset
    from openwam.model.architectures.dual_system.relational_sidecar import RelationalSidecarArchitecture
    register_dataset('piper_archive')(PiperArchiveDataset)
    original = RelationalSidecarArchitecture.save_assets_for_deployment

    def save(self, output_path, cfg):
        original(self, output_path, cfg)
        write_piper_contract(output_path)

    RelationalSidecarArchitecture.save_assets_for_deployment = save
    if '--cfg' in sys.argv:
        from train import main as train
    else:
        from train_relational_sidecar_formal import main as train
    train()


if __name__ == '__main__':
    main()
