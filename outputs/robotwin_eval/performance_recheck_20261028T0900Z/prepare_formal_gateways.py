"""Simulator-host counterpart, still inactive until registry is enabled."""
from pathlib import Path
import json
import socket
import subprocess

ROOT=Path('/remote-home/jinminghao/OpenWAM');HERE=Path(__file__).resolve().parent
OUT=HERE/'formal_bridge';rows=json.loads((OUT/'model_host_processes.json').read_text());processes=[]
for row in rows:
    directory=OUT/f"gpu{row['gpu']}"
    for port in row['gateway_ports']:
        s=socket.socket();s.bind(('127.0.0.1',port));s.close()
    command=[str(ROOT/'.conda/robotwin-official/bin/python'),'-u',str(HERE/'batch_bridge.py'),'gateway',
             '--port',str(row['relay_port']),'--listen-ports',*map(str,row['gateway_ports']),
             '--window-ms','2','--stats',str(directory/'gateway_stats.json')]
    child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=(directory/'gateway.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
    processes.append(dict(gpu=row['gpu'],pid=child.pid,ports=row['gateway_ports'],start_ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]))
(OUT/'sim_host_processes.json').write_text(json.dumps(processes,indent=2)+'\n')
print('Prepared inactive eighteen gateway endpoints')
