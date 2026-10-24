"""Prepared absolute-EEF sim data with the unchanged Alpha/Sidecar sample contract."""

import hashlib
import json
import os
from bisect import bisect_right
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import torch

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.mixture import MixtureDataset
from openwam.dataloader.transforms.multiview import assemble_multiview_layout, format_prompt_for_inference
from openwam.dataloader.transforms.video import VideoColorJitter, color_jitter_enabled
from openwam.dataloader.utils.normalization import apply_normalization
from openwam.multisim_sidecar_contract import ACTION_SLOTS, CONTRACT, SOURCE_GROUPS, VIDEO_ANCHORS, decode_rgb


class PreparedSimDataset(BaseDataset):
    action_dim = 80

    def __init__(self, manifest_path, group, split='train', color_jitter=None, allow_smoke=False):
        self.manifest_path = str(Path(manifest_path).resolve())
        self.root = Path(self.manifest_path).parent
        manifest = json.loads(Path(self.manifest_path).read_text())
        if manifest.get('contract') != CONTRACT or not manifest.get('ready'):
            raise ValueError('Run the offline multi-simulator preparer first')
        if manifest.get('release_kind') == 'smoke' and not allow_smoke:
            raise ValueError('A sample release is for checks only, not formal training')
        if manifest['plan_steps'] != [8, 16, 32] or manifest['video_anchors'] != list(VIDEO_ANCHORS):
            raise ValueError('Wrong temporal alignment')
        self.records = [x for x in manifest['episodes'] if x['group'] == group and x['split'] == split]
        if not self.records:
            raise ValueError(f'Empty prepared source/split: {group}/{split}')
        self.normalization_stats_path = str(self.root / manifest['normalization_stats'])
        if hashlib.sha256(Path(self.normalization_stats_path).read_bytes()).hexdigest() != manifest['normalization_stats_sha256']:
            raise ValueError('Normalization artifact changed')
        self.normalization_stats = np.load(self.normalization_stats_path, allow_pickle=True).item()
        self.stats = self.normalization_stats['eef']
        rotation = list(range(3, 9)) + list(range(13, 19))
        if not (np.asarray(self.stats['min'])[rotation] == -1).all() or not (np.asarray(self.stats['max'])[rotation] == 1).all():
            raise ValueError('Rot6D normalization must be identity')
        self.ends = np.cumsum([x['steps'] for x in self.records]).tolist()
        self._handles, self._pid = OrderedDict(), os.getpid()
        self._color_jitter = None
        if color_jitter_enabled(color_jitter):
            self._color_jitter = VideoColorJitter(**{k: float(color_jitter.get(k, v)) for k, v in
                [('brightness', .2), ('contrast', .2), ('saturation', .2), ('hue', 0)]})

    def __len__(self):
        return self.ends[-1]

    @lru_cache(maxsize=32)
    def _numeric(self, episode):
        import io
        record = self.records[episode]
        content = (self.root / record['numeric']).read_bytes()
        if hashlib.sha256(content).hexdigest() != record['numeric_sha256']:
            raise ValueError('Prepared numeric/auxiliary data changed')
        with np.load(io.BytesIO(content), allow_pickle=False) as data:
            state = apply_normalization(data['state'], self.stats, 'min-max').astype(np.float32)
            action = apply_normalization(data['action'], self.stats, 'min-max').astype(np.float32)
            targets = data['targets']
        return state, action, targets

    def _file(self, record):
        if self._pid != os.getpid():
            for file in self._handles.values():
                file.close()
            self._handles, self._pid = OrderedDict(), os.getpid()
        path = str(self.root / record['media'])
        if path in self._handles:
            self._handles.move_to_end(path)
            return self._handles[path]
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (record['media_size'], record['media_mtime_ns']):
            raise ValueError('Accepted source media changed after preparation')
        self._handles[path] = h5py.File(path, 'r')
        if len(self._handles) > 8:
            _, previous = self._handles.popitem(last=False)
            previous.close()
        return self._handles[path]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = bisect_right(self.ends, index)
        start = index - (self.ends[episode - 1] if episode else 0)
        record = self.records[episode]
        state, action, targets = self._numeric(episode)
        length = record['steps']
        offsets = np.asarray(VIDEO_ANCHORS)
        indices = np.minimum(start + offsets, length)
        unique, inverse = np.unique(indices, return_inverse=True)
        file = self._file(record)
        views = {}
        for view, path in enumerate(record['camera_paths']):
            frames = file[path][unique.tolist()]
            images = [decode_rgb(frame, record['rgb_codec']) for frame in frames]
            views[str(view)] = [images[i] for i in inverse]
        layout = ['0', '1', '2' if len(views) == 3 else None]
        video = [assemble_multiview_layout({key: value[t] for key, value in views.items()}, layout, 384, 320) for t in range(9)]
        if self._color_jitter is not None:
            video = self._color_jitter.apply({'video': video})['video']
        present = np.repeat(np.asarray(record['arm_present'], bool), 10)
        actions, proprio = np.zeros((32, 80), np.float32), np.zeros((1, 80), np.float32)
        amask, pmask = np.zeros((32, 80), bool), np.zeros((1, 80), bool)
        native = action[np.minimum(start + np.arange(32), length - 1)].copy()
        native[:, ~present] = 0
        current = state[start].copy()
        current[~present] = 0
        actions[:, ACTION_SLOTS], proprio[0, ACTION_SLOTS] = native, current
        amask[:, ACTION_SLOTS] = (start + np.arange(32) < length)[:, None] & present[None]
        pmask[0, ACTION_SLOTS] = present
        return dict(video=video, first_frame_image=[video[0]], vace_video=None,
                    action=torch.from_numpy(actions), proprio=torch.from_numpy(proprio),
                    action_mask=torch.from_numpy(amask), proprio_mask=torch.from_numpy(pmask),
                    video_mask=torch.from_numpy(start + offsets <= length),
                    ea_targets=torch.from_numpy(targets[start:start + 1].copy()),
                    ea_plan_steps=[8, 16, 32], ea_video_anchors=list(VIDEO_ANCHORS),
                    prompt=format_prompt_for_inference(record['prompt']), task_name=record['task'],
                    episode_index=episode, episode_path=record['media'], start_frame=start,
                    end_frame=min(start + 33, length + 1), episode_length=length + 1,
                    embodiment=record['group'], source_collection=record['collection'], domain=record['domain'])


class MultiSimSidecarDataset(MixtureDataset):
    """Reuse the native mixture interface, with full-source reshuffling and aligned epochs."""

    def __init__(self, manifest_path, *, weights=None, seed=42, epoch_alignment=256,
                 split='train', color_jitter=None, allow_smoke=False):
        self.manifest_path = str(Path(manifest_path).resolve())
        self.epoch_alignment = int(epoch_alignment)
        if self.epoch_alignment < 1:
            raise ValueError('epoch_alignment must be positive')
        datasets = [PreparedSimDataset(manifest_path, name, split, color_jitter, allow_smoke) for name in SOURCE_GROUPS]
        super().__init__(datasets, weights=weights or [1, 1, 1, 1], seed=seed, names=SOURCE_GROUPS, strict_action_dim=True)
        self.normalization_stats_path = datasets[0].normalization_stats_path

    @property
    def normalization_stats(self):
        return self.datasets[0].normalization_stats

    def _build_index_map(self):
        # The generic mixture cycles source prefixes; mid-training must rotate coverage too.
        rng = np.random.RandomState(self._seed)
        total = sum(len(dataset) for dataset in self._datasets)
        parts = []
        for i, (dataset, weight) in enumerate(zip(self._datasets, self._weights)):
            if weight <= 0:
                continue
            count = max(1, round(total * weight))
            sample = np.resize(rng.permutation(len(dataset)).astype(np.int32), count)
            parts.append(np.stack((np.full(count, i, np.int32), sample), axis=1))
        combined = np.concatenate(parts)
        combined = combined[rng.permutation(len(combined))]
        extra = (-len(combined)) % self.epoch_alignment
        if extra:
            combined = np.concatenate((combined, np.resize(combined, (extra, 2))))
        self._index_map = combined

    @classmethod
    def from_config(cls, cfg, split='train'):
        return cls(cfg['manifest_path'], weights=list(cfg.get('weights', [1, 1, 1, 1])),
                   seed=int(cfg.get('seed', 42)), epoch_alignment=int(cfg.get('epoch_alignment', 256)),
                   split=split, color_jitter=cfg.get('color_jitter'), allow_smoke=bool(cfg.get('allow_smoke', False)))
