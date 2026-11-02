"""Own one safe extra4090 slot for serial paired native timing trials."""
from pathlib import Path
import json
import os
import signal
import subprocess
import time
import yaml

ROOT=Path('/remote-home/jinminghao/OpenWAM')
HERE=Path(__file__).resolve().parent
MAIN=ROOT/'outputs/robotwin_eval/fullbackward32000_supplement70_4090_20261028T035728Z'
PYTHON=ROOT/'.conda/robotwin-official/bin/python'
state=json.loads((MAIN/'elastic_v3/state.json').read_text())
source=next(w for w in state['workers'].values() if w.get('status')=='running' and w.get('gpu')==4)
raw=Path('/proc',str(source['pid']),'environ').read_bytes().split(b'\0')
base={k.decode():v.decode() for field in raw if b'=' in field for k,v in [field.split(b'=',1)]}
base.update(ROBOTWIN_TEST_NUM='30',CUDA_VISIBLE_DEVICES='4',
            PILOT_REPLAY_INSTRUCTIONS=str(HERE/'native_pilots/legacy_direct/pilot_result.json'))
template=yaml.safe_load((MAIN/'policy_template.yml').read_text())
cases=[('legacy_direct','legacy',34910),('chunk_compat_batch_stream','chunk',35030),
       ('chunk_direct','chunk',34910),('chunk_batch_stream','chunk',35010),
       ('chunk_batch_grouped','chunk',35020)]

def gpu():
    raw=subprocess.check_output(['nvidia-smi','--id=4','--query-gpu=memory.free,pci.bus_id','--format=csv,noheader,nounits'],text=True)
    free,pci=raw.strip().split(',');return int(free),pci.strip()

results=[]
for name,variant,port in cases:
    directory=HERE/'native_pilots_paired'/name;directory.mkdir(parents=True,exist_ok=False)
    begin=time.monotonic()
    while gpu()[0]<10240:
        if time.monotonic()-begin>900:raise TimeoutError('GPU4 did not have safe pilot capacity')
        time.sleep(3)
    free,pci=gpu();cfg=dict(template,host='127.0.0.1',port=port,eval_video_log=False)
    policy=directory/'policy.yml';policy.write_text(yaml.safe_dump(cfg))
    env=dict(base,ROBOTWIN_RUNTIME_ROOT=str(directory/'runtime'),POLICY_CONFIG_PATH=str(policy))
    command=[str(PYTHON),'-u',str(HERE/'pilot_native.py'),'--variant',variant,'--pilot-episodes','2',
        '--policy',str(policy),'--task','click_alarmclock','--mode','demo_clean','--expected-pci',pci,
        '--evidence',str(directory/'render_affinity.json')]
    child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
        stdout=(directory/'client.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
    identity=Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19]
    row=dict(name=name,pid=child.pid,start_ticks=identity,free_mib_before=free,sim_gpu=4,
             started=time.time(),status='running',output=str(directory))
    results.append(row);(HERE/'native_pilot_paired_status.json').write_text(json.dumps(results,indent=2)+'\n')
    print('PILOT_START',name,child.pid,flush=True)
    stopped=False;start=time.monotonic()
    while child.poll() is None:
        current,_=gpu()
        if current<2048 or time.monotonic()-start>1800:
            p=Path('/proc',str(child.pid),'stat')
            if p.exists() and p.read_text().rsplit(')',1)[1].split()[19]==identity:
                os.killpg(child.pid,signal.SIGTERM)
            stopped=True;row['stop_reason']='capacity_or_time_guard';break
        time.sleep(2)
    try:code=child.wait(timeout=30)
    except subprocess.TimeoutExpired:raise RuntimeError('Owned pilot did not stop; no other job signalled')
    row.update(returncode=code,status='stopped' if stopped else 'complete' if code==0 else 'failed',wall_seconds=time.monotonic()-start)
    (HERE/'native_pilot_paired_status.json').write_text(json.dumps(results,indent=2)+'\n')
    if stopped or code:raise RuntimeError('Pilot failed: '+name)
    assert len(json.loads((directory/'pilot_result.json').read_text())['episodes'])==2
    print('PILOT_PASS',name,flush=True)
