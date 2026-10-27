"""Adapt the upstream C2R randomizer to OpenWAM's PIL L-shaped camera canvas."""
import numpy as np
import torch
from PIL import Image

from openwam.dataloader.transforms.patchwam_appearance import AppearanceRandomizer


class PatchWAMVideoAugment:
    def __init__(self, config):
        self.randomizer = AppearanceRandomizer(**{k: v for k, v in config.items() if k != 'enabled'})

    def __call__(self, sample, camera_present):
        frames = sample['video']
        width, height = frames[0].size
        if len(camera_present) != 3 or any(frame.size != (width, height) for frame in frames):
            raise ValueError('PatchWAM adapter requires a consistent three-view L-shaped clip')
        top = round(height * 2 / 3)
        boxes = [(0, 0, width, top), (0, top, width // 2, height), (width // 2, top, width, height)]
        present_boxes = [box for box, present in zip(boxes, camera_present) if present]
        if not present_boxes:
            raise ValueError('PatchWAM augmentation requires at least one real camera')
        views = [torch.from_numpy(np.stack([np.asarray(frame.crop(box)) for frame in frames]))
                 .permute(0, 3, 1, 2).float().div_(255) for box in present_boxes]
        augmented = self.randomizer(views)
        if augmented is views:
            return sample
        result = [frame.copy() for frame in frames]
        for box, clip in zip(present_boxes, augmented):
            pixels = clip.mul(255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
            for frame, image in zip(result, pixels):
                frame.paste(Image.fromarray(image), box)
        sample['video'] = result
        sample['first_frame_image'] = [result[0]]
        return sample
