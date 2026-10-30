#!/usr/bin/env python3
"""Bundle current training code, identical prepared Aloha data and two requested models."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from package_training_transfer import ROOT, reuse_archive, write_archive

ARM = ROOT / 'outputs/sidecar/aloha50_patchwam_c2r_ramp100_10epochs_20261027T144939Z/F_patchwam_ramp100'
OFFICIAL = ROOT / 'assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Clean2Random'
PREPARED = ROOT / 'outputs/sidecar/aloha_4350'
DATA_ARCHIVE = ROOT / 'outputs/transfers/sidecar_isolated_20261021T071002Z/openwam_sidecar_aloha50_data.tar.gz'


def package(output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((PREPARED / 'manifest.ready.json').read_text())
    digest = hashlib.sha256((PREPARED / 'manifest.ready.json').read_bytes()).hexdigest()
    inventory = json.loads(DATA_ARCHIVE.with_name(DATA_ARCHIVE.name + '.inventory.json').read_text())
    source = next(r for r in inventory['files'] if r['path'].endswith('/manifest.source.json'))
    if source['sha256'] != digest or manifest['total_episodes'] != 2500:
        raise ValueError('Reusable data archive must match the current prepared manifest exactly')
    run = Path(json.loads((ARM / 'output_run.json').read_text())['path'])
    stats = run / 'normalization_stats.npy'
    if hashlib.sha256(stats.read_bytes()).hexdigest() != manifest['normalization_stats_sha256']:
        raise ValueError('2K model normalization must match the packaged training data')
    checkpoint = ARM / 'checkpoint_step_4000.safetensors'
    if not checkpoint.is_file() or checkpoint.stat().st_size < 1_000_000_000:
        raise ValueError('The pinned 2K checkpoint is absent/incomplete')
    code = []
    for directory in ('openwam', 'configs', 'scripts', 'tests', 'licenses', 'assets/ea_robotwin', 'benchmarks'):
        for p in sorted((ROOT / directory).rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix in {'.pyc', '.log'}:
                continue
            if p.stat().st_size > 5_000_000:
                raise ValueError(f'Unexpected large code asset: {p}')
            code.append((p, 'OpenWAM/' + str(p.relative_to(ROOT))))
    for name in ('AGENTS.md', 'LICENSE', 'EA_RSIG_DIT4DIT_LICENSE', 'EA_RSIG_README.md', 'README.md',
                 'pyproject.toml', 'requirements-ea-lock.txt', 'requirements-ci.txt', 'MODIFICATION_LOG.md',
                 'docs/思考与隐患.md', 'docs/PATCHWAM_C2R_AUGMENTATION_20261027.md',
                 'docs/BITDISTILL_TRANSFER_AND_8GPU_20261028.md'):
        code.append((ROOT / name, 'OpenWAM/' + name))
    official = [(p, 'OpenWAM/assets/openwam_ckpt/openwam_alpha/' + str(p.relative_to(OFFICIAL.parent)))
                for p in sorted(OFFICIAL.rglob('*')) if p.is_file() and '.cache' not in p.parts]
    warmroot = 'OpenWAM/assets/openwam_ckpt/openwam_alpha/Sidecar-PatchWAM-Warmup2K/'
    warm = [(checkpoint, warmroot + checkpoint.name)]
    for name in ('config.yaml', 'normalization_stats.npy', 'relational_sidecar_contract.json'):
        warm.append((run / name, warmroot + name))
    warm.extend((p, warmroot + str(p.relative_to(run))) for p in sorted((run / 'tokenizer').rglob('*')) if p.is_file())
    needed = sum(p.stat().st_size for p, _ in code + official + warm)
    if shutil.disk_usage(output).free < needed + (1 << 30):
        raise OSError('Insufficient space for the two model archives')
    reports = [reuse_archive(DATA_ARCHIVE, output / 'openwam_aloha50_data.tar.gz')]
    reports.append(write_archive(output / 'openwam_code.tar.gz', code))
    # Weight tensors are already dense; uncompressed tar avoids slow, low-gain gzip work.
    for name, files in [('openwam_warmup2k.tar', warm), ('openwam_clean2random.tar', official)]:
        reports.append(write_archive(output / name, files, compress=False))
    (output / 'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['archive']}\n" for r in reports))
    info = dict(utc=datetime.now(timezone.utc).isoformat(), target='/mnt/100T/lixiangyu/cuigm/bitdistill/OpenWAM',
                manifest_sha256=digest, episodes=2500, tasks=50, raw_bytes=sum(e['size'] for t in manifest['tasks'] for e in t['episodes']),
                warmup_weight_source=str(checkpoint), new_run='weights only; zero counters/fresh optimizer; no repeated 2K warmup',
                data_archive='Reused existing regular archive; exact current source-manifest SHA match; not recompressed',
                second_training_version='not configured; awaiting user specification',
                archives=[{k: r[k] for k in ('archive', 'size_bytes', 'file_count', 'sha256')} for r in reports])
    (output / 'TRANSFER_INFO.json').write_text(json.dumps(info, indent=2) + '\n')
    shutil.copyfile(ROOT / 'docs/BITDISTILL_TRANSFER_AND_8GPU_20261028.md', output / 'README_TRANSFER.md')
    print(json.dumps(info, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    package(parser.parse_args().output.resolve())
