#!/usr/bin/env python3
"""Original WAM only: same archive train split, no Sidecar or auxiliary-label reads."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from openwam.dataloader.maniskill_archive import ManiSkillArchiveDataset
    from openwam.dataloader.registry import register_dataset
    register_dataset('maniskill_archive')(ManiSkillArchiveDataset)
    if '--cfg' in sys.argv:
        from train import main as train
    else:
        from train_relational_sidecar_formal import main as train
    train()


if __name__ == '__main__':
    main()
