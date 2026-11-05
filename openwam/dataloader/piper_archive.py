"""Select the archived Piper train split with the existing aligned subset reader."""
from openwam.dataloader.maniskill_archive import ManiSkillArchiveDataset


class PiperArchiveDataset(ManiSkillArchiveDataset):
    source_group = 'robotwin_piper'
