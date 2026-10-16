#!/usr/bin/env python3
"""Inspect one launch record; signal its verified torchrun only with explicit --stop."""
import argparse
import json
import os
from pathlib import Path
import signal


def inspect(record, stop=False):
    record = Path(record).resolve()
    data = json.loads(record.read_text())
    pid = int(data['torchrun_pid'])
    proc = Path(f'/proc/{pid}')
    if not (proc / 'cmdline').exists():
        print(f'PID {pid} is no longer running. No signal sent.')
        return
    args = (proc / 'cmdline').read_bytes().decode().replace('\0', ' ')
    # Both the launcher and unique output root must match. Never use pkill -f.
    if 'torchrun' not in args or str(record.parent) not in args:
        raise RuntimeError(f'PID {pid} no longer matches this run; refusing to signal it')
    if 'train.py' not in args and 'benchmark_ea_batch.py' not in args:
        raise RuntimeError('Unrecognized training entrypoint')
    print(f'Verified torchrun PID={pid}; run={record.parent}')
    if stop:
        os.kill(pid, signal.SIGTERM)
        print('SIGTERM sent to this torchrun; it will stop its own workers. No other job was selected.')
    else:
        print('Inspect only: no signal sent. Add --stop only when you want to end this run.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--stop', action='store_true')
    inspect(**vars(parser.parse_args()))
