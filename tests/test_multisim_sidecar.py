"""Data and sampling invariants that matter for mixed single/bimanual training."""

import hashlib
import json

import cv2
import h5py
import numpy as np
import pytest

from openwam.multisim_sidecar_contract import (
    CONTRACT,
    SOURCE_GROUPS,
    decode_rgb,
    segmentation_boxes,
    validate_numeric,
)

pytest_plugins = ["tests.test_relational_sidecar"]


def arrays(steps=40, dual=False):
    state = np.zeros((steps + 1, 20), np.float32)
    state[:, 3], state[:, 7], state[:, 9] = 1, 1, .5
    state[:, 0] = np.arange(steps + 1) / 100
    if dual:
        state[:, 13], state[:, 17], state[:, 19] = 1, 1, .5
    action = state[1:].copy()
    targets = np.zeros((steps, 130), np.float32)
    targets[:, 0] = 1
    targets[:, 1] = dual
    return state, action, targets, np.array([True, dual])


def make_release(root):
    episodes = []
    for i, group in enumerate(SOURCE_GROUPS):
        state, action, targets, arms = arrays(40, group.startswith('robotwin'))
        numeric = root / f'{group}.npz'
        np.savez_compressed(numeric, state=state, action=action, targets=targets, arm_present=arms)
        media = root / f'{group}.h5'
        with h5py.File(media, 'w') as f:
            for view in range(3 if arms[1] else 2):
                images = np.zeros((41, 8, 8, 3), np.uint8)
                images[..., 0] = np.arange(41)[:, None, None]
                f.create_dataset(f'rgb/{view}', data=images)
        stat = media.stat()
        episodes.append(dict(id=group, group=group, split='train', task='pick', prompt='pick the object',
            collection=group, domain='Clean', family_id=group, steps=40, arm_present=arms.astype(int).tolist(),
            numeric=numeric.name, numeric_sha256=hashlib.sha256(numeric.read_bytes()).hexdigest(),
            media=media.name, media_size=stat.st_size, media_mtime_ns=stat.st_mtime_ns,
            camera_paths=[f'rgb/{v}' for v in range(3 if arms[1] else 2)], rgb_codec='uint8_rgb'))
    stats = {k: np.full(20, val, np.float32) for k, val in dict(min=-1, max=1, mean=0, std=1, q01=-1, q99=1).items()}
    np.save(root / 'normalization_stats.npy', {'eef': stats})
    manifest = dict(contract=CONTRACT, ready=True, release_kind='smoke', episodes=episodes,
        normalization_stats='normalization_stats.npy', normalization_stats_sha256=hashlib.sha256((root / 'normalization_stats.npy').read_bytes()).hexdigest(),
        plan_steps=[8, 16, 32], video_anchors=list(range(0, 33, 4)))
    path = root / 'manifest.ready.json'
    path.write_text(json.dumps(manifest))
    return path


def test_masked_actions_keep_true_command_alignment_and_terminal_masks(tmp_path):
    from openwam.dataloader.multisim_sidecar import PreparedSimDataset
    path = make_release(tmp_path)
    data = PreparedSimDataset(path, 'rlbench_franka', allow_smoke=True)
    first = data[0]
    assert first['proprio'][0, 0] == 0
    assert first['action'][0, 0] == pytest.approx(.01)
    assert first['action'][31, 0] == pytest.approx(.32)
    assert not first['proprio_mask'][0, 34:44].any()
    assert not first['action'][:, 34:44].any()
    last = data[len(data) - 1]
    assert last['action_mask'][:, :10].sum().item() == 10
    assert last['video_mask'].tolist() == [True] + [False] * 8
    assert first['video'][0].size == (320, 384)
    assert first['ea_targets'].shape == (1, 130)
    dual = PreparedSimDataset(path, 'robotwin_piper', allow_smoke=True)[0]
    assert dual['proprio_mask'][0, 34:44].all()
    assert dual['action_mask'].sum().item() == 32 * 20


def test_smoke_release_rejected_and_media_mutation_detected(tmp_path):
    from openwam.dataloader.multisim_sidecar import PreparedSimDataset
    path = make_release(tmp_path)
    with pytest.raises(ValueError, match='sample release'):
        PreparedSimDataset(path, 'rlbench_franka')
    data = PreparedSimDataset(path, 'rlbench_franka', allow_smoke=True)
    with (tmp_path / 'rlbench_franka.h5').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='media changed'):
        data[0]


def test_mixture_reshuffles_source_coverage_not_only_prefix(tmp_path):
    from openwam.dataloader.multisim_sidecar import MultiSimSidecarDataset
    data = MultiSimSidecarDataset(make_release(tmp_path), weights=[.05, .05, .05, .85],
                                epoch_alignment=64, allow_smoke=True)
    assert len(data) % 64 == 0
    first = set(data._index_map[data._index_map[:, 0] == 0, 1].tolist())
    data.set_epoch(1)
    second = set(data._index_map[data._index_map[:, 0] == 0, 1].tolist())
    assert first != second
    snapshot = data._index_map.copy()
    data.set_epoch(1)
    np.testing.assert_array_equal(snapshot, data._index_map)
    assert all(data[i]['action'].shape == (32, 80) for i in [0, len(data) // 2, len(data) - 1])


def test_rgb_codec_preserves_both_source_conventions():
    rgb = np.full((16, 16, 3), [210, 40, 15], np.uint8)
    _, legacy = cv2.imencode('.jpg', rgb)
    _, standard = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    for encoded, codec in [(legacy, 'robotwin_legacy_rgb_jpeg'), (standard, 'standard_rgb_jpeg')]:
        actual = np.asarray(decode_rgb(encoded, codec)).mean((0, 1))
        np.testing.assert_allclose(actual, rgb[0, 0], atol=3)


def test_bbox_same_frame_and_virtual_goal_mask():
    seg = np.zeros((2, 8, 8, 1), np.int16)
    seg[0, 1:3, 2:5, 0] = 7
    seg[1, 4:6, 3:7, 0] = 7
    boxes, visible = segmentation_boxes(seg, [7, 7], [-1, -1])
    np.testing.assert_array_equal(boxes[:, 0], [[2, 1, 5, 3], [3, 4, 7, 6]])
    assert not visible[:, 1].any()


def test_numeric_rejects_invalid_rotations_and_absent_arm_values():
    state, action, targets, arms = arrays()
    validate_numeric(state, action, targets, arms)
    state[0, 3:9] = 0
    with pytest.raises(ValueError, match='orthonormal'):
        validate_numeric(state, action, targets, arms)
    state, action, targets, arms = arrays()
    action[0, 10] = 1
    with pytest.raises(ValueError, match='Absent-arm'):
        validate_numeric(state, action, targets, arms)


def test_mixed_embodiment_loss_update_and_warmup_boundary(make, tmp_path):
    import torch

    from openwam.dataloader.multisim_sidecar import PreparedSimDataset
    from tests.test_relational_sidecar import inputs
    from tests.test_sidecar_backbone_warmup import optimizer_for

    manifest = make_release(tmp_path)
    samples = [PreparedSimDataset(manifest, group, allow_smoke=True)[0] for group in SOURCE_GROUPS]
    model = make(backbone_updates=2, aux_loss_scale=.5)
    optimizer = optimizer_for(model)
    for stage in model.sidecar.stages:
        stage.write.beta.data.fill_(.01)
    base = inputs()
    count = len(samples)
    context = base['context'].repeat(count, 1, 1)
    kwargs = dict(actions=torch.stack([x['action'] for x in samples]),
        action_is_pad=~torch.stack([x['action_mask'] for x in samples]),
        input_latents=base['latents'].repeat(count, 1, 1, 1, 1),
        first_frame_latents=base['first_frame_latents'].repeat(count, 1, 1, 1, 1),
        context=context, context_mask=base['context_mask'].repeat(count, 1),
        proprio=torch.stack([x['proprio'] for x in samples]),
        proprio_mask=torch.stack([x['proprio_mask'] for x in samples]),
        ea_targets=torch.stack([x['ea_targets'] for x in samples]),
        fuse_vae_embedding_in_latents=True)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    model.set_training_progress(1)
    result = model.compute_loss(**kwargs)
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    assert all(stage.write.projection.weight.grad.norm() > 0 for stage in model.sidecar.stages)
    optimizer.step()
    for name, parameter in model.named_parameters():
        if not name.startswith('sidecar.'):
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    optimizer.zero_grad()
    model.set_training_progress(2)
    model.compute_loss(**kwargs)['loss'].backward()
    optimizer.step()
    assert any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if not n.startswith('sidecar.'))


def test_train_statistics_do_not_read_heldout_values(tmp_path):
    from copy import deepcopy

    from scripts.prepare_multisim_sidecar_data import finalize

    manifest = make_release(tmp_path)
    records = json.loads(manifest.read_text())['episodes']
    heldout = deepcopy(records[0])
    heldout.update(id='heldout', family_id='heldout', split='test', numeric='heldout.npz')
    state, action, targets, arms = arrays()
    state[:, :3], action[:, :3] = 999, 999
    np.savez_compressed(tmp_path / 'heldout.npz', state=state, action=action, targets=targets, arm_present=arms)
    finalize(records + [heldout], tmp_path, True)
    stats = np.load(tmp_path / 'normalization_stats.npy', allow_pickle=True).item()['eef']
    assert stats['max'][0] < 1
    assert stats['min'][3] == -1 and stats['max'][3] == 1


def test_resume_identity_includes_world_and_update_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts.train_multisim_sidecar import data_contract

    path = make_release(tmp_path)
    manifest = json.loads(path.read_text())
    manifest.update(total_episodes=4, counts={'fixture': 4}, splits={'train': 4})
    path.write_text(json.dumps(manifest))
    cfg = SimpleNamespace(dataloader=SimpleNamespace(manifest_path=str(path), weights=[1, 1, 1, 1], epoch_alignment=256),
                          project=SimpleNamespace(seed=42), training=SimpleNamespace(batch_size=32, gradient_accumulation_steps=2,
                                                                                 max_steps=100000, num_epochs=None))
    monkeypatch.setenv('WORLD_SIZE', '4')
    expected = data_contract(cfg)
    monkeypatch.setenv('WORLD_SIZE', '8')
    assert data_contract(cfg) != expected
    monkeypatch.setenv('WORLD_SIZE', '4')
    cfg.training.max_steps = 200000
    assert data_contract(cfg) != expected


def test_full_archive_keeps_unselected_test_but_excludes_unselected_train(tmp_path):
    from openwam.dataloader.multisim_sidecar import PreparedSimDataset

    path = make_release(tmp_path)
    data = json.loads(path.read_text())
    original = data['episodes'][0]
    data['episodes'].extend([
        dict(original, id='archive_train', selected_for_midtrain=False),
        dict(original, id='full_test', split='test', selected_for_midtrain=False),
    ])
    path.write_text(json.dumps(data))
    train = PreparedSimDataset(path, 'rlbench_franka', allow_smoke=True)
    test = PreparedSimDataset(path, 'rlbench_franka', split='test', allow_smoke=True)
    assert len(train.records) == 1 and len(test.records) == 1
    assert test.records[0]['id'] == 'full_test'


def test_cpu_offload_cannot_silently_drop_backbone_warmup_hooks():
    from types import SimpleNamespace

    from scripts.train_multisim_sidecar import validate_warmup_optimizer

    with pytest.raises(ValueError, match="drops WAM warmup hooks"):
        validate_warmup_optimizer(SimpleNamespace(offload_optimizer_device="cpu"), 2000)
    validate_warmup_optimizer(SimpleNamespace(offload_optimizer_device="none"), 2000)
    validate_warmup_optimizer(SimpleNamespace(offload_optimizer_device="cpu"), 0)
