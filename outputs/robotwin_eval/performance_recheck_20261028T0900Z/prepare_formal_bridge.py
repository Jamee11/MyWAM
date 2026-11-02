"""Prepare inactive six-model compatibility bridges. Does not reset live sessions."""
from pathlib import Path
import json
import os
import socket
import subprocess
import time

ROOT=Path('/remote-home/jinminghao/OpenWAM')
HERE=Path(__file__).resolve().parent
DEP=ROOT/'outputs/robotwin_eval/fullbackward32000_complete100_deploy_20261028T035728Z'
OUT=HERE/'formal_bridge';OUT.mkdir(exist_ok=False)
gpus=(1,2,4,5,6,7)
records=[];registry=dict(enabled=False,checkpoint_sha256=json.loads((DEP/'deployment.json').read_text())['checkpoint_sha256'],ports={})
for gpu in gpus:
    directory=OUT/f'gpu{gpu}';directory.mkdir()
    relay_port=35400+gpu*10
    backend=[32900+gpu*10+i for i in range(3)]
    s=socket.socket();s.bind(('127.0.0.1',relay_port));s.close()
    command=[str(ROOT/'.venv/bin/python'),'-u',str(HERE/'batch_bridge.py'),'relay',
             '--port',str(relay_port),'--backend-ports',*map(str,backend),'--compat32',
             '--reply-mode','stream','--stats',str(directory/'relay_stats.json')]
    child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=(directory/'relay.log').open('w'),
                           stderr=subprocess.STDOUT,start_new_session=True)
    row=dict(gpu=gpu,pid=child.pid,relay_port=relay_port,backend_ports=backend,
             gateway_ports=[port+1000 for port in backend],start_ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
    records.append(row)
    for index,port in enumerate(backend):
        registry['ports'][str(port)]=dict(gateway_port=port+1000,session=str(DEP/f'gpu{gpu}/session_slot{index}.json'))
(OUT/'model_host_processes.json').write_text(json.dumps(records,indent=2)+'\n')
command=['bash',str(ROOT/'benchmarks/robotwin/remote_reverse_tunnel.sh'),'root@210.45.70.21','26199',
         *[f"{row['relay_port']}:{row['relay_port']}" for row in records]]
child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=(OUT/'tunnel.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
(OUT/'tunnel_process.json').write_text(json.dumps(dict(pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19])))
(HERE/'formal_transport_registry.json').write_text(json.dumps(registry,indent=2)+'\n')
print('Prepared inactive compatibility relays for six GPUs; no model reset')
