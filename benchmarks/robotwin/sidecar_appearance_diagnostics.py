"""Fixed-state appearance stress tests; inference only, no training or simulator actions."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ('clean', 'light_mild', 'light_strong', 'sensor_strong', 'patchwam_stress')
ACTIVE = list(range(10)) + list(range(34, 44))


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def feature_distance(left, right):
    a, b = np.asarray(left, np.float64), np.asarray(right, np.float64)
    if a.shape != b.shape or not np.isfinite([a, b]).all():
        raise ValueError('Mismatched or nonfinite features')
    a, b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    an, bn = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    valid = (an > 1e-8) & (bn > 1e-8)
    return dict(relative_l2=float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-8)),
                cosine=float(np.mean(np.sum(a[valid] * b[valid], axis=-1) / (an[valid] * bn[valid])))
                if valid.any() else None, valid_tokens=int(valid.sum()))


def perturb(image, condition, seed, recipe=None):
    """Keep geometry, text and proprio fixed; mark strong/color changes as stress tests."""
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if condition == 'clean':
        return image.copy()
    if condition == 'patchwam_stress':
        import torch
        from openwam.dataloader.transforms.patchwam_video import PatchWAMVideoAugment
        cfg = copy.deepcopy(recipe)
        cfg['p'] = 1.0  # Always apply for this stress test; the original training probability is 0.8.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            return PatchWAMVideoAugment(cfg)({'video': [image.copy()]}, [True, True, True])['video'][0]
    result = image.copy()
    w, h = image.size
    top = round(h * 2 / 3)
    rng = np.random.default_rng(seed)
    for box in ((0, 0, w, top), (0, top, w // 2, h), (w // 2, top, w, h)):
        view = image.crop(box)
        if condition.startswith('light_'):
            brightness, contrast, gamma = (1.15, 0.9, 1.1) if condition == 'light_mild' else (1.8, 1.4, 0.65)
            view = ImageEnhance.Brightness(view).enhance(brightness)
            view = ImageEnhance.Contrast(view).enhance(contrast)
            pixels = ((np.asarray(view, np.float32) / 255) ** gamma * 255).clip(0, 255).astype(np.uint8)
        else:
            view = view.filter(ImageFilter.GaussianBlur(1.8))
            pixels = (np.asarray(view, np.float32) + rng.normal(0, 12, (*view.size[::-1], 3))).clip(0, 255).astype(np.uint8)
        result.paste(Image.fromarray(pixels), box)
    return result


class Capture:
    """Observe the first denoising step in this fresh eval instance, without changing tensors."""
    def __init__(self, arch):
        self.clear()
        step, forward = arch.mot_driver.step, arch.forward

        def observed_step(layer, *args, **kwargs):
            result = step(layer, *args, **kwargs)
            if layer in (8, 20) and str(layer) not in self.action:
                self.action[str(layer)] = result[1].payload.x_action[0].detach().float().cpu().numpy().copy()
            return result

        def observed_forward(actions, action_time, **kwargs):
            first = self.velocity is None
            if first:
                self.noise['action'] = self.tensor_hash(actions)
                latents = kwargs.get('latents')
                if latents is not None:
                    self.noise['video_future'] = self.tensor_hash(latents[:, :, 1:])
                self.noise['action_t'] = float(action_time.flatten()[0])
                self.noise['video_t'] = float(kwargs['timestep'].flatten()[0])
            result = forward(actions, action_time, **kwargs)
            if first:
                self.velocity = result[1][0].detach().float().cpu().numpy().copy()
            return result

        def observed_heads(module, args, output):
            if len(self.sidecar) >= 3:
                return
            from benchmarks.robotwin.sidecar_gt_diagnostics import representation_metrics
            index = len(self.sidecar)
            self.sidecar.append(output['tokens'][0].detach().float().cpu().numpy().copy())
            prediction = {key: output[key][0].detach().float().cpu().numpy() for key in
                          ('track_valid_logits', 'near_logits', 'contact_logits', 'goal_proximity',
                           'plan_directions', 'plan_moving_logits')}
            prediction.update(grid=list(output['grid']),
                              attention=output['grounding']['attention'][0].detach().float().cpu().numpy(),
                              presence_logits=output['grounding']['presence_logits'][0].detach().float().cpu().numpy())
            self.representation.append(dict(tap=arch.tap_blocks[index],
                                            metrics=representation_metrics(prediction, self.targets, self.arms)))

        arch.mot_driver.step = observed_step
        arch.forward = observed_forward
        self.hook = arch.sidecar.heads.register_forward_hook(observed_heads)

    @staticmethod
    def tensor_hash(tensor):
        return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()

    def clear(self, targets=None, arms=None):
        self.action, self.noise, self.sidecar, self.representation = {}, {}, [], []
        self.velocity, self.targets, self.arms = None, targets, arms


def summaries(rows, out):
    from benchmarks.robotwin.sidecar_write_diagnostics import action_metrics, distribution
    groups = {}
    for row in rows:
        groups.setdefault(row['sample_id'], {})[(row['condition'], row['mode'])] = row
    pairs = []
    normalizer = summaries.normalizer
    for sample, group in groups.items():
        for (condition, mode), row in group.items():
            if condition == 'clean' or ('clean', mode) not in group:
                continue
            clean = group[('clean', mode)]
            assert row['noise'] == clean['noise'], 'Paired initial action/video noise or timestep differs'
            a, b = np.asarray(row['actions']), np.asarray(clean['actions'])
            result = dict(sample_id=sample, task=row['sample']['task'], domain=row['sample']['domain'],
                          condition=condition, mode=mode, **action_metrics(a, b, normalizer))
            with np.load(out / row['features']) as aug, np.load(out / clean['features']) as nominal:
                result['action_hidden'] = {str(layer): feature_distance(aug[f'action_{layer}'], nominal[f'action_{layer}'])
                                           for layer in (8, 20)}
                result['velocity'] = feature_distance(aug['velocity'][:, ACTIVE], nominal['velocity'][:, ACTIVE])
                result['sidecar'] = {name: [feature_distance(x[start:end], y[start:end])
                                             for x, y in zip(aug['sidecar'], nominal['sidecar'])]
                                     for name, start, end in (('ground', 0, 4), ('relation', 4, 8),
                                                             ('summary', 8, 9), ('plan', 9, 15))}
            result['segments_xyz_drift_mm'] = {
                f'{start + 1}-{start + 8}': float(np.mean([
                    np.linalg.norm(a[start:start + 8, j:j + 3] - b[start:start + 8, j:j + 3], axis=-1)
                    for j in (0, 10)]) * 1000) for start in (0, 8, 16, 24)}
            pairs.append(result)
    grouped = []
    for condition in CONDITIONS[1:]:
        for mode in ('full', 'write_off'):
            selected = [r for r in pairs if r['condition'] == condition and r['mode'] == mode]
            if selected:
                values = [r['position_difference_mm']['mean'] for r in selected]
                grouped.append(dict(condition=condition, mode=mode, windows=len(selected),
                                    xyz_drift_mm=distribution(values), window_p95_mm=float(np.quantile(values, .95))))
    report = dict(scope='Paired predicted physical EEF chunks and first-denoising-step features; not executed drift or closed-loop SR. Small correlated window sample; quantiles are descriptive.',
                  completed_generations=len(rows), pairs=pairs, groups=grouped)
    save(out / 'summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    out = parser.parse_args().output.resolve()
    plan = json.loads((out / 'plan.json').read_text())
    import torch
    from omegaconf import OmegaConf
    from openwam.deploy.server import build_server_from_config
    from openwam.model.video_backbone.wan.encode import encode_text
    from benchmarks.robotwin.sidecar_write_diagnostics import WriteProbe
    from benchmarks.robotwin.sidecar_gt_diagnostics import score_actions
    gpu = plan['gpu']

    def capacity(minimum):
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
        free = dict(tuple(int(v.strip()) for v in line.split(',')) for line in raw.splitlines())
        if free[gpu] < minimum:
            raise RuntimeError(f'GPU{gpu} free={free[gpu]}MiB below {minimum}; stop this diagnostic only')

    capacity(40000)
    torch.cuda.set_per_process_memory_fraction(0.45, 0)
    save(out / 'progress.json', dict(stage='loading_cpu_model', completed=0))
    cfg = OmegaConf.load(ROOT / 'configs/deploy_relational_sidecar.yaml')
    cfg.optimization.compile.enabled = False
    cfg.optimization.dit_cache.enabled = False
    cfg.optimization.prompt_embed_cache.maxsize = 128
    server = build_server_from_config(cfg, plan['checkpoint'], device='cpu',ckpt_name=plan.get('checkpoint_name'))
    arch, engine = server.engine.architecture, server.engine
    vb = arch.video_backbone
    arch.eval()
    samples = json.loads((out / 'inputs_ready.json').read_text())['samples']
    prompts = sorted({json.loads(Path(row['payload']).read_text())['prompt'] for row in samples})
    save(out / 'progress.json', dict(stage='encoding_text', completed=0))
    vb.text_encoder.to(device='cuda:0', dtype=arch.dtype)
    with torch.inference_mode():
        encoded = {p: tuple(v.cpu() for v in encode_text([p], tokenizer=vb._tokenizer,
                    text_encoder=vb.text_encoder, device='cuda:0')) for p in prompts}
    vb.text_encoder.to('cpu')
    torch.cuda.empty_cache()
    encoder, vb.text_encoder = vb.text_encoder, None
    try:
        arch.set_dtype_device(arch.dtype, torch.device('cuda:0'))
    finally:
        vb.text_encoder = encoder
    arch.eval()
    engine._dit_cache = None
    engine._prompt_embed_cache = {p: tuple(v.to('cuda:0') for v in values) for p, values in encoded.items()}

    def forbid_uncached(*args, **kwargs):
        raise RuntimeError('Unexpected uncached prompt')

    vb.text_encoder.forward = forbid_uncached
    server._init_policy()
    probe = WriteProbe(arch)
    if 'gate_scales' in plan:
        from benchmarks.robotwin.sidecar_gate_sweep import run_gate_sweep
        return run_gate_sweep(server, probe, samples, plan, out, capacity)
    capture = Capture(arch)
    recipe = plan['patchwam_recipe']
    summaries.normalizer = arch.normalizer

    def fingerprint():
        h = hashlib.sha256()
        for name, p in arch.sidecar.named_parameters():
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    before = fingerprint()
    versions = {n: p._version for n, p in arch.named_parameters() if not n.endswith('.write.beta')}
    rows = []
    start = time.monotonic()
    for sample_index, sample in enumerate(samples):
        payload = json.loads(Path(sample['payload']).read_text())
        obs = server._obs_preprocessor.preprocess(copy.deepcopy(payload))
        with np.load(sample['gt']) as gt:
            target, labels, arms = gt['actions'], gt['targets'], gt['arms']
        for condition in CONDITIONS:
            image = perturb(obs['image'], condition, 103000 + sample_index, recipe)
            image.save(out / 'images' / f'{sample_index:02d}_{condition}.png')
            for mode in ('full', 'write_off'):
                capacity(8192)
                capture.clear(labels, arms)
                probe.rows = []
                server.reset()
                changed = dict(obs, image=image)
                conditions = server._policy._build_conditions(changed)
                conditions['seed'] = 42
                with torch.inference_mode(), probe.use(mode, collect=True):
                    actions = np.asarray(engine.generate(conditions)['actions']).copy()
                assert set(capture.action) == {'8', '20'} and len(capture.sidecar) == 3
                if sample_index == 0 and condition == 'clean' and mode == 'full':
                    expected = actions.copy()
                    capture.clear(labels, arms)
                    server.reset()
                    with torch.inference_mode(), probe.use('full'):
                        repeated = engine.generate(conditions)['actions']
                    assert np.array_equal(expected, repeated), 'Clean repeat is not deterministic'
                    save(out / 'repeat_check.json', dict(exact=True, seed=42))
                name = f'features/{sample_index:02d}_{condition}_{mode}.npz'
                np.savez_compressed(out / name, action_8=capture.action['8'], action_20=capture.action['20'],
                                    sidecar=np.stack(capture.sidecar), velocity=capture.velocity)
                row = dict(sample_id=sample_index, sample=sample, condition=condition, mode=mode,
                           actions=actions.tolist(), features=name, noise=dict(capture.noise),
                           gt_metrics=score_actions(actions, target, arch.normalizer),
                           representations=list(capture.representation), write_measurements=list(probe.rows))
                rows.append(row)
                save(out / 'results.json', dict(rows=rows))
                summaries(rows, out)
                save(out / 'progress.json', dict(stage='running', completed=len(rows), total=len(samples) * len(CONDITIONS) * 2,
                                                elapsed_seconds=time.monotonic() - start, sample=sample_index,
                                                condition=condition, mode=mode))
                print('DIAGNOSTIC', len(rows), sample_index, condition, mode, flush=True)
    assert fingerprint() == before
    assert versions == {n: p._version for n, p in arch.named_parameters() if not n.endswith('.write.beta')}
    save(out / 'complete.json', dict(status='complete', generations=len(rows), windows=len(samples),
                                    no_optimizer=True, sidecar_fingerprint_unchanged=True,
                                    non_gate_parameter_versions_unchanged=True, elapsed_seconds=time.monotonic() - start))
    save(out / 'progress.json', dict(stage='complete', completed=len(rows), total=len(rows)))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    try:
        main()
    except BaseException as error:
        # The output is an existing, dedicated experiment directory; preserve partial evidence.
        if len(sys.argv) == 2 and Path(sys.argv[1]).is_dir():
            save(Path(sys.argv[1]) / 'failed.json', dict(error=repr(error)))
        raise
