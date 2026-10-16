#!/usr/bin/env python3
"""Relocate baseline-only metadata once after unpacking; do not regenerate data."""
import argparse
import hashlib
import json
import os
import importlib.util
from pathlib import Path


def prepare(project, raw_root=None):
    project = Path(project).resolve()
    raw_root = Path(raw_root or project / 'data/clean50').resolve()
    metadata = project / 'data/baseline'
    manifest = json.loads((metadata / 'manifest.source.json').read_text())
    old_root = Path(manifest['source_root'])
    stats = metadata / 'normalization_stats.npy'
    assert hashlib.sha256(stats.read_bytes()).hexdigest() == manifest['normalization_stats_sha256']
    count = 0
    for task in manifest['tasks']:
        task['data_root'] = str(raw_root / Path(task['data_root']).relative_to(old_root))
        if task.get('startup_metadata', {}).get('version') != 1:
            raise ValueError('Missing prepared metadata')
        for episode in task['episodes']:
            path = raw_root / Path(episode['path']).relative_to(old_root)
            st = path.stat()
            if (st.st_size, st.st_mtime_ns) != (episode['size'], episode['mtime_ns']):
                raise ValueError(f'Source size/mtime changed: {path}; extract with GNU tar preserving PAX times')
            if path.is_symlink():
                raise ValueError(f'Dataset must contain real files: {path}')
            episode['path'] = str(path)
            episode.pop('labels', None)
            episode.pop('labels_sha256', None)
            count += 1
    manifest['source_root'] = str(raw_root)
    manifest['normalization_stats_path'] = str(stats)
    output = metadata / 'manifest.ready.json'
    content = json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'
    if output.exists() and output.read_text() != content:
        raise FileExistsError(f'Refusing to overwrite different metadata: {output}')
    if not output.exists():
        temporary = output.with_suffix('.tmp')
        temporary.write_text(content)
        os.replace(temporary, output)
    print(json.dumps(dict(manifest=str(output), episodes=count,
                          windows=sum(e['frames'] - 1 for t in manifest['tasks'] for e in t['episodes']),
                          auxiliary=False), indent=2))
    # Keep the existing EA relocation contract, including label existence and
    # source nanosecond timestamps. Baseline never reads these sidecars.
    ea_data = project / 'data/ea_lite'
    ea_source = ea_data / 'manifest.ready.json'
    ea_output = ea_data / 'manifest.cluster.json'
    if ea_source.exists():
        if ea_output.exists():
            current = json.loads(ea_output.read_text())
            if current['source_root'] != str(raw_root) or current['normalization_stats_path'] != str(ea_data / 'normalization_stats.npy'):
                raise FileExistsError(f'Different EA relocation exists: {ea_output}')
            print(f'EA manifest already relocated: {ea_output}')
        else:
            module_spec = importlib.util.spec_from_file_location(
                'relocate_ea', project / 'scripts/relocate_ea_lite_manifest.py')
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            print(json.dumps(module.relocate(ea_source, raw_root, ea_data, ea_output), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--raw-root', type=Path)
    prepare(**vars(parser.parse_args()))
