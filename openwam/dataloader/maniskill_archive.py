"""The archived ManiSkill subset, without mixed-source resampling."""
from openwam.dataloader.multisim_sidecar import PreparedSimDataset


class ManiSkillArchiveDataset(PreparedSimDataset):
    source_group = 'maniskill_franka'

    @classmethod
    def from_config(cls, cfg, split='train'):
        data = cls(cfg.manifest_path, cls.source_group, split=split,
                   color_jitter=cfg.get('color_jitter'),
                   patchwam_augmentation=cfg.get('patchwam_augmentation') if split == 'train' else None,
                   auxiliary=bool(cfg.get('auxiliary', True)))
        data.epoch_alignment = int(cfg.epoch_alignment) if split == 'train' else 1
        if data.epoch_alignment < 1:
            raise ValueError('epoch_alignment must be positive')
        return data

    def __len__(self):
        length = super().__len__()
        alignment = getattr(self, 'epoch_alignment', 1)
        return ((length + alignment - 1) // alignment) * alignment

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        # Pad only the final global batch, rather than resampling the multi-sim epoch.
        return super().__getitem__(index % self.ends[-1])
