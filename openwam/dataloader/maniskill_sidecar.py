"""Add loss-only labels to the exact ManiSkill absolute-EEF baseline reader."""

import hashlib
import io
import json
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from openwam.dataloader.maniskill_absolute import ManiSkillAbsoluteDataset
from openwam.maniskill_sidecar_contract import CONTRACT


class ManiSkillSidecarDataset(ManiSkillAbsoluteDataset):
    def __init__(self, manifest_path, color_jitter=None):
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest.get("sidecar_contract") != CONTRACT:
            raise ValueError("Prepare independent ManiSkill 130D labels first")
        if manifest.get("plan_steps") != [8, 16, 32] or manifest.get("camera_layout") != [
            "front_camera", "hand_camera", None
        ]:
            raise ValueError("Wrong temporal or camera contract")
        super().__init__(manifest_path, color_jitter)

    @lru_cache(maxsize=64)
    def _labels(self, episode):
        record = self.records[episode]
        content = Path(record["labels"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != record["labels_sha256"]:
            raise ValueError("Prepared ManiSkill auxiliary labels changed")
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            targets = archive["targets"]
        if targets.shape != (record["steps"], 130) or not np.isfinite(targets).all():
            raise ValueError("Invalid prepared 130D supervision")
        return targets

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        episode = bisect_right(self.ends, index)
        start = index - (self.ends[episode - 1] if episode else 0)
        sample.update(
            ea_targets=torch.from_numpy(self._labels(episode)[start:start + 1].copy()),
            ea_plan_steps=[8, 16, 32], ea_video_anchors=list(range(0, 33, 4)),
        )
        return sample
