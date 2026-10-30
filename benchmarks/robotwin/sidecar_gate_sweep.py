"""Inference-only gate multiplier sweep using the audited appearance inputs."""
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from benchmarks.robotwin.sidecar_appearance_diagnostics import CONDITIONS, save


def xyz_distance(a, b, stop=32):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != (32, 20) or b.shape != a.shape or not np.isfinite([a, b]).all():
        raise ValueError('Expected finite physical EEF chunks [32,20]')
    return float(np.mean([np.linalg.norm(a[:stop, j:j+3] - b[:stop, j:j+3], axis=-1)
                          for j in (0, 10)]) * 1000)


def summarize(rows):
    lookup = {(r['sample_id'], r['condition'], r['scale']): r for r in rows}
    assert len(lookup) == len(rows)
    groups = []
    for scale in sorted({r['scale'] for r in rows}):
        for condition in CONDITIONS:
            rr = [r for r in rows if r['scale'] == scale and r['condition'] == condition]
            if not rr:
                continue
            paired = [(r, lookup[(r['sample_id'], 'clean', scale)]) for r in rr
                      if (r['sample_id'], 'clean', scale) in lookup]
            drift = [xyz_distance(r['actions'], c['actions']) for r, c in paired]
            groups.append(dict(scale=scale, condition=condition, windows=len(rr),
                gt_xyz_mm=float(np.mean([r['gt_metrics']['position_difference_mm']['mean'] for r in rr])),
                gt_rotation_degrees=float(np.mean([r['gt_metrics']['rotation_difference_degrees']['mean'] for r in rr])),
                drift_xyz_mm=float(np.mean(drift)) if drift else None,
                drift_median_mm=float(np.median(drift)) if drift else None,
                worst_window_drift_mm=max(drift, default=None),
                drift_first8_mm=float(np.mean([xyz_distance(r['actions'], c['actions'], 8) for r,c in paired])) if paired else None,
                write_ratio_mean=float(np.mean([w['ratio'] for r in rr for w in r['write_measurements']])),
                write_ratio_max=max(w['ratio'] for r in rr for w in r['write_measurements'])))
    return dict(generations=len(rows), groups=groups)


def run_gate_sweep(server, probe, samples, plan, out, capacity):
    import torch
    from benchmarks.robotwin.sidecar_gt_diagnostics import score_actions
    arch, engine = server.engine.architecture, server.engine
    scales = plan['gate_scales']
    assert scales == [0, 1, 2, 5, 10, 100]
    reference = Path(plan['reference_diagnostic'])
    old = json.loads((reference / 'results.json').read_text())['rows']
    old_lookup = {(r['sample_id'], r['condition'], r['mode']): r for r in old}

    def fingerprint():
        h = hashlib.sha256()
        for name, p in arch.sidecar.named_parameters():
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    before = fingerprint()
    versions = {n:p._version for n,p in arch.named_parameters() if not n.endswith('.write.beta')}
    save(out / 'original_gates.json', {str(tap):float(stage.write.beta) for tap,stage in zip(arch.tap_blocks,arch.sidecar.stages)})
    rows, checks = [], 0
    started = time.monotonic()
    for index, sample in enumerate(samples):
        obs = server._obs_preprocessor.preprocess(copy.deepcopy(json.loads(Path(sample['payload']).read_text())))
        with np.load(sample['gt']) as gt:
            target = gt['actions'].copy()
        for condition in CONDITIONS:
            image_path = reference / 'images' / f'{index:02d}_{condition}.png'
            with Image.open(image_path) as image:
                image = image.convert('RGB')
            assert image.size == obs['image'].size
            for scale in scales:
                capacity(8192)
                server.reset()
                probe.rows = []
                conditions = server._policy._build_conditions(dict(obs, image=image.copy()))
                conditions['seed'] = plan['seed']
                with torch.inference_mode(), probe.use('full', collect=True, scale=scale):
                    actions = np.asarray(engine.generate(conditions)['actions']).copy()
                xyz_distance(actions, actions)  # Reject nonfinite output, including the 100x stress case.
                if scale in (0, 1):
                    expected = old_lookup[index, condition, 'full' if scale else 'write_off']
                    assert np.array_equal(actions, np.asarray(expected['actions'])), 'Canonical baseline changed'
                    checks += 1
                if index == 0 and condition == 'clean' and scale == 100:
                    server.reset()
                    with torch.inference_mode(), probe.use('full', scale=100):
                        repeated = engine.generate(conditions)['actions']
                    assert np.array_equal(actions, repeated), '100x repeat differs'
                    save(out / 'repeat_check.json', dict(scale=100, exact=True))
                row = dict(sample_id=index, sample=sample, condition=condition, scale=scale,
                           actions=actions.tolist(), gt_metrics=score_actions(actions, target, arch.normalizer),
                           write_measurements=list(probe.rows))
                rows.append(row)
                save(out / 'results.json', dict(rows=rows))
                save(out / 'summary.json', summarize(rows))
                save(out / 'progress.json', dict(stage='running', completed=len(rows), total=360,
                    sample=index, condition=condition, scale=scale, elapsed_seconds=time.monotonic()-started))
                print('GATE_SWEEP',len(rows),index,condition,scale,flush=True)
    assert fingerprint() == before
    assert versions == {n:p._version for n,p in arch.named_parameters() if not n.endswith('.write.beta')}
    assert checks == 120 and len(rows) == 360
    save(out / 'complete.json', dict(status='complete', generations=len(rows), baseline_exact_checks=checks,
        sidecar_fingerprint_unchanged=True, non_gate_versions_unchanged=True, elapsed_seconds=time.monotonic()-started))
    save(out / 'progress.json', dict(stage='complete', completed=len(rows), total=360))
