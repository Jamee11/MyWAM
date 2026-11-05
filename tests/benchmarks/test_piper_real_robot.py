import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from benchmarks.real_robot.piper.client_repo.inference.contract import (
    SCHEMA, Frames, action_chunk, guard_targets, matrix_to_eef, eef_to_matrix,
)
from benchmarks.real_robot.piper.serve_piper_flask import create_app

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'benchmarks/real_robot/piper/client_repo/robot.json'


def test_frame_roundtrip_and_no_delta_accumulation():
    frames = Frames(json.loads(CONFIG.read_text()))
    poses = [np.eye(4), np.eye(4)]
    poses[0][:3, 3], poses[1][:3, 3] = [.2, .1, .3], [.3, -.1, .2]
    state = frames.state(poses, [.035, .07])
    targets = frames.targets(state)
    np.testing.assert_allclose(targets[:, :3], [p[:3, 3] for p in poses], atol=1e-7)
    np.testing.assert_allclose(targets[:, 3:6], 0, atol=1e-7)
    np.testing.assert_allclose(targets[:, 6], [.035, .07], atol=1e-7)
    for i in range(2):
        np.testing.assert_allclose(eef_to_matrix(state[10*i:10*i+10]),
                                   frames.base[i] @ poses[i], atol=1e-7)
    action_chunk(np.tile(state, (32, 1)))
    with pytest.raises(ValueError):
        action_chunk(np.tile(state, (50, 1)))


def test_motion_limits():
    cfg = json.loads(CONFIG.read_text())
    for side in ('left', 'right'):
        cfg[side]['workspace_xyz'] = [[-.5]*3, [.5]*3]
    current = np.zeros((2, 7))
    target = current.copy()
    target[:, 2] = .01
    guard_targets(current, target, cfg)
    target[1, 0] = .03
    with pytest.raises(ValueError, match='right EEF'):
        guard_targets(current, target, cfg)


def test_tool_offset_and_rotation_are_inverted():
    cfg = json.loads(CONFIG.read_text())
    cfg['left']['feedback_from_tool'] = [[0,-1,0,0],[1,0,0,0],[0,0,1,.035],[0,0,0,1]]
    frames = Frames(cfg)
    measured = np.eye(4)
    measured[:3,3] = [.2,-.1,.3]
    targets = frames.targets(frames.state([measured,measured],[.035,.035]))
    np.testing.assert_allclose(targets[:,:3], [[.2,-.1,.3]]*2,atol=1e-7)
    np.testing.assert_allclose(targets[:,3:6],0,atol=1e-7)


def test_original_temporal_buffer_accepts_eef20():
    from benchmarks.real_robot.piper.client_repo.inference.temporal import StreamActionBuffer
    state = np.tile(matrix_to_eef(np.eye(4), .5), 2)
    buffer = StreamActionBuffer(state_dim=20)
    buffer.integrate_new_chunk(np.tile(state,(32,1)),max_k=0,min_m=12)
    np.testing.assert_array_equal(buffer.pop_next_action(),state)
    moved = np.tile(state,(32,1)); moved[:,0] += .01
    buffer.integrate_new_chunk(moved,max_k=0,min_m=12)
    first = buffer.pop_next_action()
    assert first.shape == (20,) and first[0] == 0
    for _ in range(30):
        last = buffer.pop_next_action()
    assert np.isclose(last[0],.01)


def test_http_auth_images_state_actions_and_records(tmp_path):
    state = np.tile(matrix_to_eef(np.eye(4), .5), 2)
    seen = []
    def infer(images, raw_state, prompt):
        seen.append((images, raw_state, prompt))
        return np.tile(raw_state, (32, 1))
    app = create_app(infer, 'test-token-0123456789', {'checkpoint':'fixture'}, tmp_path)
    client = app.test_client()
    auth = {'Authorization':'Bearer test-token-0123456789'}
    assert client.get('/health').status_code == 401
    assert client.get('/health', headers=auth).json['schema'] == SCHEMA
    query = dict(schema=SCHEMA, state=state.tolist(), prompt='stack the blocks')
    def payload():
        result = {'json':(io.BytesIO(json.dumps(query).encode()), 'request.json')}
        for name, color in zip(('front', 'left', 'right'), ('red','green','blue')):
            out = io.BytesIO()
            Image.new('RGB', (40,30), color).save(out, format='PNG')
            result[name] = (io.BytesIO(out.getvalue()), name+'.png')
        return result
    response = client.post('/api/inference', data=payload(), headers=auth)
    assert response.status_code == 200, response.json
    assert np.asarray(response.json['actions']).shape == (32,20)
    np.testing.assert_array_equal(seen[0][1], state)
    assert [x.getpixel((0,0)) for x in seen[0][0]] == [(255,0,0),(0,128,0),(0,0,255)]
    folder = next(tmp_path.iterdir())
    assert json.loads((folder/'request.json').read_text()) == query
    assert json.loads((folder/'response.json').read_text()) == response.json
    assert all((folder/(name+'.png')).is_file() for name in ('front','left','right'))
    query['state'] = [0]*14
    assert client.post('/api/inference', data=payload(), headers=auth).status_code == 400


def test_training_layout_and_mask_normalization_match():
    from omegaconf import OmegaConf
    from openwam.deploy.obs_preprocess import ObsPreprocessor
    from openwam.deploy.model_loader import _build_normalizer
    from openwam.deploy.server import merge_deploy_cfg
    from openwam.dataloader.transforms.multiview import assemble_multiview_layout
    ckpt = ROOT/'outputs/piper_sidecar/midtrain_patchwam_2epochs_20261102T101245Z/runs/2026-11-02_10-18-00'
    cfg = OmegaConf.load(ckpt/'config.yaml')
    merged = merge_deploy_cfg(cfg, OmegaConf.load(ROOT/'configs/deploy.yaml'))
    assert merged.inference.num_frames == 33
    assert merged.inference.video_num_frames == 9
    images = [Image.new('RGB', (40,30), c) for c in ('red','green','blue')]
    pp = ObsPreprocessor(multiview=True, camera_layout=['head_camera','left_camera','right_camera'],
                         img_height=384,img_width=320,requires_proprio=True)
    state = np.tile(matrix_to_eef(np.eye(4), .5), 2)
    state[[2,12]] = .95
    obs = pp.preprocess(dict(images=dict(head_camera=images[0],left_wrist_camera=images[1],right_wrist_camera=images[2]),state=state))
    reference = assemble_multiview_layout(dict(zip(['0','1','2'], images)), ['0','1','2'],384,320)
    np.testing.assert_array_equal(obs['image'], reference)
    normalizer = _build_normalizer(cfg, str(ckpt))
    normalized = normalizer.normalize(state)
    from openwam.dataloader.utils.normalization import apply_normalization
    stats = np.load(ckpt/'normalization_stats.npy',allow_pickle=True).item()['eef']
    active = list(range(10))+list(range(34,44))
    np.testing.assert_allclose(normalized[active], apply_normalization(state, stats, 'min-max'),atol=1e-6)
    np.testing.assert_allclose(normalizer.unnormalize(normalized), state, atol=1e-6)
    assert normalized.shape == (80,)
    np.testing.assert_array_equal(normalized[np.setdiff1d(np.arange(80),active)],0)
