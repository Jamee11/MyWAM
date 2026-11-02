"""Retire only completed diagnostic services; retain all formal bridges and models."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('role',choices=['model','sim']);a=p.parse_args()
status=json.loads((HERE/'native_pilot_paired_status.json').read_text())
assert len(status)==5 and all(r['status']=='complete' for r in status)
accepted=json.loads((HERE/'acceptance_summary.json').read_text())
assert len(accepted['native'])==4 and all(r['matched_initial_input_gate_passed'] for r in accepted['native'])
records=[]
if a.role=='sim':
    records=[(r,'gateway',None) for r in json.loads((HERE/'gateway_processes.json').read_text()).values()]
    records.append((json.loads((HERE/'compat_gateway_process.json').read_text()),'gateway',None))
else:
    records.append((json.loads((HERE/'model_gpu3/process.json').read_text()),'model',None))
    for name,record in json.loads((HERE/'bridge_processes.json').read_text()).items():
        records.append((record,'tunnel' if name=='tunnel' else 'relay',None))
    raw=HERE/'compat_processes.json'
    for name,pid in json.loads(raw.read_text()).items():records.append((dict(pid=pid),name,raw.stat().st_mtime))
out=[]
for record,kind,birth in records:
    proc=Path('/proc')/str(record['pid'])
    try:fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:continue
    if fields[0] in ('Z','X'):continue
    tick=fields[19]
    if 'start_ticks' in record:assert tick==record['start_ticks']
    else:
        # Early compat launcher saved PID only: recover historical identity
        # from its private record mtime, kernel boot time and start ticks.
        boot=time.time()-float(Path('/proc/uptime').read_text().split()[0])
        assert birth is not None and abs(boot+int(tick)/os.sysconf('SC_CLK_TCK')-birth)<5
    command=(proc/'cmdline').read_bytes().split(b'\0')
    if kind=='model':assert str(HERE/'serve_chunk.py').encode() in command
    elif kind in ('relay','gateway'):
        assert str(HERE/'batch_bridge.py').encode() in command and kind.encode() in command
        assert b'--port' in command and command[command.index(b'--port')+1] in (b'34980',b'34981',b'34982')
    elif kind=='tunnel':
        assert Path(command[0].decode()).name=='ssh'
        assert any(b':34982:' in item or b':34910:' in item for item in command)
    else:raise ValueError(kind)
    assert (proc/'stat').read_text().rsplit(')',1)[1].split()[19]==tick
    os.kill(record['pid'],signal.SIGTERM)
    out.append(dict(pid=record['pid'],kind=kind,start_ticks=tick))
(HERE/f'diagnostic_cleanup_{a.role}.json').write_text(json.dumps(dict(wall=time.time(),stopped=out,
    preserved='six formal models, eighteen formal gateways/relays, formal supervisors/tunnel, all eval outputs'),indent=2)+'\n')
print('Retired only completed diagnostic services',a.role,len(out))
