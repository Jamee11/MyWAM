"""Prepared single-arm absolute controller targets + original ManiSkill RGB.

Keeps the original Alpha sample/head/loss contract; no EA/oracle model inputs.
"""
from bisect import bisect_right
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import torch

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.transforms.multiview import assemble_multiview_layout, format_prompt_for_inference
from openwam.dataloader.transforms.video import VideoColorJitter, color_jitter_enabled
from openwam.dataloader.utils.normalization import apply_normalization


class ManiSkillAbsoluteDataset(BaseDataset):
    action_dim = 80

    def __init__(self, manifest_path, color_jitter=None):
        self.manifest_path = str(Path(manifest_path).resolve())
        data = json.loads(Path(self.manifest_path).read_text())
        if data['contract'] != 'maniskill_panda_absolute_command_eef_v1':
            raise ValueError('Need separately prepared absolute controller-target dataset')
        self.records, self.sources = data['episodes'], data['sources']
        self.normalization_stats_path = data['normalization_stats_path']
        if hashlib.sha256(Path(self.normalization_stats_path).read_bytes()).hexdigest() != data['normalization_stats_sha256']:
            raise ValueError('Statistics changed')
        self.normalization_stats = np.load(self.normalization_stats_path, allow_pickle=True).item()
        self.stats = self.normalization_stats['eef']
        if not (np.asarray(self.stats['min'])[3:9] == -1).all() or not (np.asarray(self.stats['max'])[3:9] == 1).all():
            raise ValueError('Rotation6D must use identity normalization')
        self.ends = np.cumsum([r['steps'] for r in self.records]).tolist()
        self._handles, self._pid = {}, os.getpid()
        self._color_jitter = None
        if color_jitter_enabled(color_jitter):
            self._color_jitter = VideoColorJitter(**{k:float(color_jitter.get(k,v)) for k,v in
                [('brightness',.2),('contrast',.2),('saturation',.2),('hue',0)]})

    @classmethod
    def from_config(cls, cfg, split='train'):
        if split != 'train':
            raise ValueError('Held-out data must be explicitly prepared; no implicit training validation')
        expected=dict(action_mode='eef', num_frames=33, video_stride=4, height=384, width=320,
                      normalize_mode='min-max', unify_action=True)
        for key, value in expected.items():
            if cfg.get(key,value) != value:
                raise ValueError(f'Unsupported baseline contract: {key}')
        return cls(cfg['manifest_path'], cfg.get('color_jitter'))

    def __len__(self):
        return self.ends[-1]

    @lru_cache(maxsize=64)
    def _numeric(self, index):
        record=self.records[index]
        content=Path(record['numeric']).read_bytes()
        if hashlib.sha256(content).hexdigest()!=record['numeric_sha256']:
            raise ValueError('Numeric absolute-target sidecar changed')
        import io
        with np.load(io.BytesIO(content)) as f:
            state=apply_normalization(f['state'],self.stats,'min-max')
            action=apply_normalization(f['action'],self.stats,'min-max')
        return state.astype(np.float32),action.astype(np.float32)

    def _file(self, path):
        if self._pid != os.getpid():
            for h in self._handles.values():
                h.close()
            self._handles,self._pid={},os.getpid()
        if path not in self._handles:
            stat=Path(path).stat();record=self.sources[path]
            if (stat.st_size,stat.st_mtime_ns)!=(record['size'],record['mtime_ns']):
                raise ValueError('Source HDF5 changed after preparation')
            self._handles[path]=h5py.File(path,'r')
        return self._handles[path]

    def __getitem__(self, index):
        if not 0<=index<len(self):
            raise IndexError(index)
        episode=bisect_right(self.ends,index)
        start=index-(self.ends[episode-1] if episode else 0)
        record=self.records[episode];length=record['steps']
        state,action=self._numeric(episode)
        offsets=np.arange(0,33,4);indices=np.minimum(start+offsets,length)
        unique,inverse=np.unique(indices,return_inverse=True)
        tr=self._file(record['source_h5'])[record['trajectory']]
        cameras={}
        for camera in ('front_camera','hand_camera'):
            # Native HDF5 is uint8 RGB, unlike RoboTwin's legacy JPEG encoding.
            values=np.asarray(tr[f'obs/sensor_data/{camera}/rgb'][unique.tolist()])[inverse]
            cameras[camera]=[Image.fromarray(x) for x in values]
        video=[assemble_multiview_layout({k:v[t] for k,v in cameras.items()},
               ['front_camera','hand_camera',None],384,320) for t in range(9)]
        if self._color_jitter is not None:
            video=self._color_jitter.apply({'video':video})['video']
        actions=np.zeros((32,80),np.float32);proprio=np.zeros((1,80),np.float32)
        actions[:,:10]=action[np.minimum(start+np.arange(32),length-1)]
        proprio[0,:10]=state[start]
        amask=np.zeros((32,80),bool);pmask=np.zeros((1,80),bool)
        amask[:,:10]=(start+np.arange(32)<length)[:,None];pmask[:,:10]=True
        return dict(video=video,first_frame_image=[video[0]],vace_video=None,
                    action=torch.from_numpy(actions),proprio=torch.from_numpy(proprio),
                    action_mask=torch.from_numpy(amask),proprio_mask=torch.from_numpy(pmask),
                    video_mask=torch.from_numpy(start+offsets<=length),
                    prompt=format_prompt_for_inference(record['prompt']),task_name=record['task'],
                    episode_index=episode,episode_path=record['source_h5']+':'+record['trajectory'],
                    start_frame=start,end_frame=min(start+33,length+1),episode_length=length+1,
                    active_arm='left')
