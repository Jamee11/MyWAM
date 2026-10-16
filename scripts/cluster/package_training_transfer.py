#!/usr/bin/env python3
"""Build dereferenced, checksummed code/data/Foundation archives for this migration."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]


class HashWriter:
    def __init__(self, stream):
        self.stream, self.sha, self.size = stream, hashlib.sha256(), 0

    def write(self, value):
        count = self.stream.write(value)
        self.sha.update(value[:count])
        self.size += count
        return count

    def tell(self):
        return self.size

    def flush(self):
        self.stream.flush()


class HashReader:
    def __init__(self, stream):
        self.stream, self.sha = stream, hashlib.sha256()

    def read(self, size=-1):
        value = self.stream.read(size)
        self.sha.update(value)
        return value


def write_archive(destination, files, generated=(), expected=None):
    partial = destination.with_suffix(destination.suffix + '.partial')
    if destination.exists() or partial.exists():
        raise FileExistsError(destination)
    records = []
    with partial.open('xb', buffering=8 << 20) as out:
        writer = HashWriter(out)
        with tarfile.open(fileobj=writer, mode='w|gz', format=tarfile.PAX_FORMAT,
                          dereference=True, compresslevel=1, copybufsize=8 << 20,
                          bufsize=8 << 20) as archive:
            for i, (source, name) in enumerate(files):
                source = Path(source)
                before = source.stat()  # follows symlinks deliberately
                info = archive.gettarinfo(str(source), arcname=name)
                if not info.isfile():
                    raise ValueError(f'Only regular-file content is allowed: {source}')
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                info.pax_headers['mtime'] = f'{before.st_mtime_ns // 10**9}.{before.st_mtime_ns % 10**9:09d}'
                with source.open('rb') as stream:
                    reader = HashReader(stream)
                    archive.addfile(info, reader)
                after = source.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError(f'Source changed while packaging: {source}')
                digest = reader.sha.hexdigest()
                if expected and name in expected and expected[name] != digest:
                    raise ValueError(f'Upstream hash mismatch: {name}')
                records.append(dict(path=name, size=before.st_size, mtime_ns=before.st_mtime_ns, sha256=digest))
                if i % 100 == 0:
                    print(f'{destination.name}: {i + 1}/{len(files)} files', flush=True)
            for name, value in generated:
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(value), 0o644
                archive.addfile(info, io.BytesIO(value))
                records.append(dict(path=name, size=len(value), sha256=hashlib.sha256(value).hexdigest()))
        writer.flush()
        digest, size = writer.sha.hexdigest(), writer.size
    partial.rename(destination)
    report = dict(archive=destination.name, sha256=digest, size_bytes=size,
                  file_count=len(records), symlinks=0, hardlinks=0, files=records)
    destination.with_name(destination.name + '.inventory.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'COMPLETE {destination.name}: {size} bytes SHA256={digest}', flush=True)
    return report


def reuse_archive(source, destination):
    source = Path(source).resolve()
    inventory = source.with_name(source.name + '.inventory.json')
    while not (source.is_file() and inventory.is_file()):
        time.sleep(2)
    record = json.loads(inventory.read_text())
    if source.stat().st_size != record['size_bytes']:
        raise ValueError(f'Incomplete reusable archive: {source}')
    # A hard link is a real regular archive file, not a symlink. Its members
    # remain dereferenced regular files. Fall back to byte-copy across devices.
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)
    record['archive'] = destination.name
    destination.with_name(destination.name + '.inventory.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'REUSED Foundation archive: {destination}', flush=True)
    return record


def package(output, manifest=None, reuse_foundation=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = Path(manifest).resolve() if manifest else ROOT / 'outputs/ea_lite/data/manifest.ready.json'
    manifest = json.loads(manifest_path.read_text())
    raw_root = Path(manifest['source_root'])
    data_files = []
    for task in manifest['tasks']:
        for ep in task['episodes']:
            source = Path(ep['path'])
            st = source.stat()
            if (st.st_size, st.st_mtime_ns) != (ep['size'], ep['mtime_ns']):
                raise ValueError(f'Changed audited source: {source}')
            data_files.append((source, str(Path('OpenWAM/data/clean50') / source.relative_to(raw_root))))
        # Human-readable source metadata, useful for later re-audits.
        task_dir = Path(task['data_root']).parent
        for source in sorted((task_dir / 'instructions').glob('*.json')):
            data_files.append((source, str(Path('OpenWAM/data/clean50') / source.relative_to(raw_root))))
        source = task_dir / 'scene_info.json'
        if source.is_file():
            data_files.append((source, str(Path('OpenWAM/data/clean50') / source.relative_to(raw_root))))
    # Retain existing EA labels with explicit provenance even when switching
    # baseline to a different official source. Never pair them with new RGB.
    ea_root = ROOT / 'outputs/ea_lite/data'
    same_ea_source = manifest['source_root'] == json.loads((ea_root / 'manifest.ready.json').read_text())['source_root']
    ea_manifest = manifest if same_ea_source else json.loads((ea_root / 'manifest.ready.json').read_text())
    ea_destination = 'ea_lite' if same_ea_source else 'legacy_ea_lite'
    ea_expected = {}
    for task in ea_manifest['tasks']:
        for ep in task['episodes']:
            name = str(Path('OpenWAM/data') / ea_destination / Path(ep['labels']).relative_to(ea_root))
            data_files.append((Path(ep['labels']), name))
            ea_expected[name] = ep['labels_sha256']
    for name in ['manifest.json', 'manifest.ready.json', 'normalization_stats.npy']:
        data_files.append((ea_root / name, f'OpenWAM/data/{ea_destination}/{name}'))
    ea_expected[f'OpenWAM/data/{ea_destination}/normalization_stats.npy'] = ea_manifest['normalization_stats_sha256']
    code_files = []
    for directory in ['openwam', 'configs', 'scripts', 'assets/ea_robotwin']:
        for source in sorted((ROOT / directory).rglob('*')):
            if not source.is_file() or '__pycache__' in source.parts or source.suffix in {'.pyc', '.log'}:
                continue
            if source.stat().st_size > 5_000_000:
                raise ValueError(f'Unexpected large code asset: {source}')
            code_files.append((source, str(Path('OpenWAM') / source.relative_to(ROOT))))
    for name in ['AGENTS.md', 'LICENSE', 'EA_RSIG_DIT4DIT_LICENSE', 'README.md', 'pyproject.toml',
                 'requirements-ea-lock.txt', 'MODIFICATION_LOG.md', 'docs/思考与隐患.md',
                 'docs/BASELINE_CLUSTER_TRANSFER_20261015.md']:
        code_files.append((ROOT / name, f'OpenWAM/{name}'))
    code_files.append((Path(manifest['normalization_stats_path']), 'OpenWAM/data/baseline/normalization_stats.npy'))
    baseline = json.loads(manifest_path.read_text())
    for task in baseline['tasks']:
        for ep in task['episodes']:
            ep.pop('labels', None)
            ep.pop('labels_sha256', None)
    generated = [('OpenWAM/data/baseline/manifest.source.json', (json.dumps(baseline, ensure_ascii=False, indent=2) + '\n').encode())]
    foundation = ROOT / 'assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model'
    verification = json.loads((foundation / 'download_verification.json').read_text())
    ckpt_files = [(p, str(Path(foundation.name) / p.relative_to(foundation)))
                  for p in sorted(foundation.rglob('*')) if p.is_file() and '.partial' not in p.name]
    ckpt_expected = {str(Path(foundation.name) / name): item['sha256']
                     for name, item in verification['files'].items()}
    names = ['openwam_training_code.tar.gz', 'openwam_official_clean50_with_legacy_ea_labels.tar.gz' if not same_ea_source else 'openwam_clean50_with_ea_data.tar.gz', 'openwam_alpha_foundation.tar.gz']
    setup = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                 source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                 checkpoint_revision=verification['revision'], episodes=manifest['total_episodes'],
                 raw_bytes=sum(ep['size'] for t in manifest['tasks'] for ep in t['episodes']),
                 baseline_source_root=manifest['source_root'],
                 ea_label_source_root=ea_manifest['source_root'],
                 ea_labels_match_baseline=same_ea_source,
                 destination='/home/mingfeihangroup/xiaoming', partition='mhan0035',
                 scope='Training code, Clean50 HDF5 + instructions/scene metadata, all EA labels/stats, Foundation; no logs/venv/git/history checkpoints/videos/replays')
    (output / 'TRANSFER_INFO.json').write_text(json.dumps(setup, indent=2) + '\n')
    with ThreadPoolExecutor(max_workers=2) as pool:
        code = pool.submit(write_archive, output / names[0], code_files, generated)
        data = pool.submit(write_archive, output / names[1], data_files, expected=ea_expected)
        ckpt = (pool.submit(reuse_archive, reuse_foundation, output / names[2]) if reuse_foundation
                else pool.submit(write_archive, output / names[2], ckpt_files, expected=ckpt_expected))
        reports = [future.result() for future in [code, data, ckpt]]
    (output / 'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['archive']}\n" for r in reports))
    (output / 'README_TRANSFER.md').write_bytes((ROOT / 'docs/BASELINE_CLUSTER_TRANSFER_20261015.md').read_bytes())
    print(json.dumps(dict(output=str(output), archives=[{k: r[k] for k in ['archive', 'size_bytes', 'file_count', 'sha256']} for r in reports]), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--reuse-foundation', type=Path)
    package(**vars(parser.parse_args()))
