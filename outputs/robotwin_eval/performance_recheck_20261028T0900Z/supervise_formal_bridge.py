"""Supervise CPU-only bridges; restart transport, never replay requests or reset models."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path('/remote-home/jinminghao/OpenWAM');HERE=Path(__file__).resolve().parent;OUT=HERE/'formal_bridge'
p=argparse.ArgumentParser();p.add_argument('role',choices=['model','sim']);p.add_argument('--replace-staged',action='store_true');a=p.parse_args()
lease=(OUT/f'{a.role}_supervisor.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
rows=json.loads((OUT/'model_host_processes.json').read_text());running={};restarts={};stop=False

def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def identity(pid):
    try:
        fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0] in ('Z','X') else fields[19]
    except FileNotFoundError:return None
def terminate(record,needle):
    pid=record['pid'];current=identity(pid)
    if current is None:return
    if current!=record['start_ticks']:raise RuntimeError('PID was reused; refusing to signal')
    cmd=Path('/proc',str(pid),'cmdline').read_bytes()
    if needle.encode() not in cmd or str(HERE).encode() not in cmd:raise RuntimeError('Owned bridge identity mismatch')
    os.kill(pid,signal.SIGTERM)

if a.replace_staged:
    assert not json.loads((HERE/'formal_transport_registry.json').read_text())['enabled']
    for row in rows:
        stats=OUT/f"gpu{row['gpu']}/gateway_stats.json"
        assert not stats.exists() or json.loads(stats.read_text())['requests']==0
    original=rows if a.role=='model' else json.loads((OUT/'sim_host_processes.json').read_text())
    save(OUT/f'{a.role}_before_supervision.json',original)
    for record in original:terminate(record,'batch_bridge.py')
    deadline=time.monotonic()+15
    while any(identity(rec['pid'])==rec['start_ticks'] for rec in original):
        if time.monotonic()>deadline:raise RuntimeError('Staged bridge did not exit')
        time.sleep(.2)

def launch(row):
    gpu=str(row['gpu']);d=OUT/f'gpu{gpu}';python=ROOT/('.venv/bin/python' if a.role=='model' else '.conda/robotwin-official/bin/python')
    if a.role=='model':
        command=[str(python),'-u',str(HERE/'batch_bridge.py'),'relay','--port',str(row['relay_port']),
                 '--backend-ports',*map(str,row['backend_ports']),'--compat32','--reply-mode','stream','--stats',str(d/'relay_stats.json')]
    else:
        command=[str(python),'-u',str(HERE/'batch_bridge.py'),'gateway','--port',str(row['relay_port']),
                 '--listen-ports',*map(str,row['gateway_ports']),'--window-ms','2','--stats',str(d/'gateway_stats.json')]
    log=(d/f'{a.role}_supervised.log').open('a')
    child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close();running[gpu]=child;restarts[gpu]=restarts.get(gpu,-1)+1

def stopping(*_):
    global stop
    stop=True
signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
try:
    while not stop:
        for row in rows:
            gpu=str(row['gpu'])
            if gpu not in running or running[gpu].poll() is not None:launch(row)
        tunnel=None
        if a.role=='model':
            tunnel=json.loads((OUT/'tunnel_process.json').read_text())
            if identity(tunnel['pid'])!=tunnel['start_ticks']:
                command=['bash',str(ROOT/'benchmarks/robotwin/remote_reverse_tunnel.sh'),'root@210.45.70.21','26199',
                         *[f"{row['relay_port']}:{row['relay_port']}" for row in rows]]
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=(OUT/'tunnel.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
                tunnel=dict(pid=child.pid,start_ticks=identity(child.pid));save(OUT/'tunnel_process.json',tunnel)
        save(OUT/f'{a.role}_supervisor_status.json',dict(wall=time.time(),
             bridge_source_sha256=hashlib.sha256((HERE/'batch_bridge.py').read_bytes()).hexdigest(),
             children={g:dict(pid=child.pid,start_ticks=identity(child.pid),restarts=restarts[g]) for g,child in running.items()},
             tunnel=tunnel))
        time.sleep(3)
finally:
    # Only children created by this supervisor are signalled on explicit stop.
    for child in running.values():
        if child.poll() is None:child.terminate()
    for child in running.values():
        try:child.wait(timeout=15)
        except subprocess.TimeoutExpired:pass
