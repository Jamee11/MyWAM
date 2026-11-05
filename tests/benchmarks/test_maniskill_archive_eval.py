import copy

import numpy as np
import pytest

from benchmarks.maniskill.archive_eval import single_arm_stats
from benchmarks.maniskill.archive_protocol import select_records
from benchmarks.maniskill.run_archive_eval import aggregate, rendering_errors


def manifest():
    rows = []
    for split, collection, domain, n, offset in [
        ('train', 'maniskill_base', 'Clean', 10, 100),
        ('test', 'maniskill_base', 'Clean', 62, 1000),
        ('val', 'maniskill_base', 'Clean', 63, 2000),
        ('test', 'maniskill_hard', 'D', 31, 3000),
        ('test', 'maniskill_hard', 'M', 94, 4000),
    ]:
        for i in range(n):
            identity = f'{split}_{collection}_{domain}_{i}'
            rows.append(dict(group='maniskill_franka', task='push_cube', split=split, collection=collection,
                             domain=domain, id=identity, family_id=identity, numeric=identity+'.npz',
                             media=f'{identity}/seed{offset+i}_speed0.5/replay.h5'))
    return dict(episodes=rows)


def test_exact_heldout_coverage_and_domain_mix():
    rows = select_records(manifest())
    assert len(rows) == 200
    clean = [r for r in rows if r['condition'] == 'Clean']
    assert sum(r['split'] == 'test' for r in clean) == 62
    assert sum(r['split'] == 'val' for r in clean) == 38
    hard = [r for r in rows if r['condition'] == 'HardRandom']
    assert all(r['split'] == 'test' for r in hard)
    assert sum(r['domain'] == 'D' for r in hard) == 25
    assert sum(r['domain'] == 'M' for r in hard) == 75
    assert len({r['id'] for r in rows}) == 200


@pytest.mark.parametrize('key', ['id', 'family_id', 'numeric', 'media'])
def test_training_identity_overlap_rejected(key):
    data = manifest()
    train = data['episodes'][0]
    heldout = next(r for r in data['episodes'] if r['split'] == 'test')
    train[key] = heldout[key]
    with pytest.raises(ValueError, match='Training overlap'):
        select_records(data)


def test_not_enough_clean_is_not_silently_repeated():
    data = manifest()
    data['episodes'] = [r for r in data['episodes'] if r['split'] != 'val']
    with pytest.raises(ValueError, match='Insufficient'):
        select_records(data)


def test_deploy_stats_are_exact_active_prefix_without_mutating_training_stats():
    original = dict(eef={k: np.arange(20, dtype=np.float32)+i for i, k in enumerate(['min', 'max', 'mean', 'std'])})
    before = copy.deepcopy(original)
    projected = single_arm_stats(original)
    for key in original['eef']:
        np.testing.assert_array_equal(projected['eef'][key], original['eef'][key][:10])
        np.testing.assert_array_equal(original['eef'][key], before['eef'][key])


def test_aggregate_requires_exact_identity_and_counts_successes():
    from benchmarks.maniskill.archive_protocol import seed_from_record
    cases = [dict(r, seed=seed_from_record(r)) for r in select_records(manifest())]
    plan = dict(cases=cases, split_policy={'Clean': 'val+test', 'HardRandom': 'test'})
    rows = [dict(r, success=i % 2 == 0) for i, r in enumerate(cases)]
    result = aggregate(plan, rows)
    assert result['complete'] and result['episodes'] == 200
    assert all(c['success_rate'] == .5 for c in result['conditions'])
    with pytest.raises(ValueError, match='Incomplete or duplicate'):
        aggregate(plan, rows[:-1])
    bad = copy.deepcopy(rows)
    bad[0]['seed'] += 1
    with pytest.raises(ValueError, match='identity differs'):
        aggregate(plan, bad)


@pytest.mark.parametrize('cross_filesystem', [False, True])
def test_deployment_projects_normalizer_and_masks_the_missing_arm(tmp_path, monkeypatch, cross_filesystem):
    from types import SimpleNamespace
    import torch
    from omegaconf import OmegaConf
    from safetensors.numpy import save_file
    from benchmarks.maniskill.archive_eval import prepare
    from openwam.deploy.model_loader import _build_normalizer
    from openwam.model.architectures.base import BaseWAMArchitecture
    source = tmp_path/'source'
    source.mkdir()
    weight = source/'checkpoint_step_20000.safetensors'
    save_file({'weight': np.zeros(1, np.float32)}, str(weight))
    cfg = OmegaConf.create(dict(dataloader=dict(type='maniskill_archive', action_mode='eef', unify_action=True,
        unify_action_map=['0-9', '34-43'], normalize_mode='min-max'), model=dict(architecture=dict(action_dim=80))))
    OmegaConf.save(cfg, source/'config.yaml')
    np.save(source/'normalization_stats.npy', {'eef': dict(min=-np.ones(20), max=np.ones(20),
                                                        mean=np.zeros(20), std=np.ones(20))})
    (source/'tokenizer').mkdir()
    output = tmp_path/'deploy'
    if cross_filesystem:
        import errno
        def cross_device(*args):
            raise OSError(errno.EXDEV, 'different filesystem')
        monkeypatch.setattr('benchmarks.maniskill.archive_eval.os.link', cross_device)
    prepare(SimpleNamespace(checkpoint=weight, output=output))
    assert (output/'checkpoint'/weight.name).read_bytes() == weight.read_bytes()
    derived = OmegaConf.load(output/'checkpoint/config.yaml')
    from openwam.deploy.obs_preprocess import ObsPreprocessor
    observer = ObsPreprocessor.from_cfg(derived)
    assert observer.multiview and observer.camera_layout == ['front_camera', 'hand_camera', None]
    normalizer = _build_normalizer(derived, str(output/'checkpoint'))
    assert normalizer._dst_index.tolist() == list(range(10))
    model = SimpleNamespace(normalizer=normalizer, action_dim=80)
    inactive = BaseWAMArchitecture._resolve_inactive_action_dims(model, None, torch.device('cpu'))
    assert not inactive[:10].any() and inactive[10:].all()
    raw = np.linspace(-.8, .8, 10).astype(np.float32)[None]
    unified = normalizer.normalize(raw)
    assert unified.shape == (1,80)
    np.testing.assert_allclose(normalizer.unnormalize(unified), raw, atol=1e-6)


@pytest.mark.parametrize('marker', ['OIDN Error', 'ErrorDeviceLost', 'VK_ERROR_DEVICE_LOST',
                                   'CUDA error: an illegal memory access'])
def test_renderer_errors_are_not_valid_successful_rollouts(tmp_path, marker):
    path = tmp_path/'worker.log'
    path.write_text('episode complete success=True\n'+marker+'\n')
    assert rendering_errors(path) == [marker]
    path.write_text('No initial pose set for actor builder\nepisode complete success=False\n')
    assert rendering_errors(path) == []


def test_reuse_plan_keeps_scenes_but_rejects_changed_data_or_budget(tmp_path):
    import hashlib
    import json
    from benchmarks.maniskill.archive_protocol import reuse_plan
    source = tmp_path/'manifest.ready.json'
    source.write_text('{"episodes": []}')
    plan = dict(manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                archive_root=str(tmp_path), episodes_per_task_condition=100, cases=[{'id': 'heldout'}])
    reference = tmp_path/'plan.json'
    reference.write_text(json.dumps(plan))
    assert reuse_plan(reference, source, 100) == plan
    assert reuse_plan(reference, source, 30) is None
    source.write_text('{"episodes": [1]}')
    assert reuse_plan(reference, source, 100) is None


def test_runtime_reuse_requires_matching_setup_and_successful_full_validation(tmp_path):
    import json
    from benchmarks.maniskill.run_archive_eval import reuse_validation
    signature = {'data': 'fixed', 'runtime': 'same'}
    (tmp_path/'runtime_signature.json').write_text(json.dumps(signature))
    (tmp_path/'expert.json').write_text(json.dumps([{'passed': True}]*24))
    out = tmp_path/'server_manifest/validation'
    out.mkdir(parents=True)
    result = dict(passed=True, selected_strategy='ffn', variants={})
    (out/'result.json').write_text(json.dumps(result))
    assert reuse_validation(tmp_path, signature)
    assert not reuse_validation(tmp_path, dict(signature, runtime='changed'))
    result['passed'] = False
    (out/'result.json').write_text(json.dumps(result))
    assert not reuse_validation(tmp_path, signature)


@pytest.mark.parametrize('finite', [True, False])
def test_reused_runtime_keeps_ffn_cache_and_checks_new_checkpoint_actions(tmp_path, finite):
    import json
    from types import SimpleNamespace
    from benchmarks.maniskill.serve_sidecar import reuse_checked_runtime
    reference = tmp_path/'reference'
    fixtures = reference/'server_manifest/validation'
    fixtures.mkdir(parents=True)
    (fixtures/'inputs.json').write_text(json.dumps([{'state': [0]*10}]))
    settings = []
    arch = SimpleNamespace(apply_compile_optimizations=lambda cfg: settings.append((cfg.enabled, cfg.sidecar_strategy)))
    engine = SimpleNamespace(architecture=arch)
    def generate(conditions):
        assert engine._dit_cache is not None
        return {'actions': np.full((32, 10), 0 if finite else np.nan)}
    engine.generate = generate
    root = SimpleNamespace(engine=engine, _init_policy=lambda: None, reset=lambda: None,
        _obs_preprocessor=SimpleNamespace(preprocess=lambda x: x),
        _policy=SimpleNamespace(_build_conditions=lambda x: x))
    if finite:
        result = reuse_checked_runtime(root, reference, tmp_path/'output')
        assert result['passed'] and result['action_shape'] == [32, 10]
    else:
        with pytest.raises(ValueError, match='finite physical EEF10'):
            reuse_checked_runtime(root, reference, tmp_path/'output')
    assert settings == [(True, 'ffn')]
