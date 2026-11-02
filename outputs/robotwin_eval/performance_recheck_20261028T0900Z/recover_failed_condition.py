"""Independent complete70 retry of a terminal infrastructure failure.

Main queue state is never rewritten. A separate audited overlay supplies this
condition's complete recovery result to the reporter, preserving old attempts.
"""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import yaml

ROOT=Path('/remote-home/jinminghao/OpenWAM');HERE=Path(__file__).resolve().parent
MAIN=ROOT/'outputs/robotwin_eval/fullbackward32000_supplement70_4090_20261028T035728Z'
OUT=HERE/'recovery_stack_blocks_random_20261028T1240Z'
sys.path.insert(0,str(ROOT))
from benchmarks.robotwin.standard_eval import validate_result
from benchmarks.utils.transport import WSPolicyClient

def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
lease=(OUT/'recovery.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=json.loads((MAIN/'elastic_v3/state.json').read_text())
condition=next(c for c in state['conditions'] if c['task']=='stack_blocks_three' and c['mode']=='demo_randomized')
assert condition['status']=='failed'
model=json.loads((OUT/'model_gpu3/process.json').read_text())
assert model['checkpoint_sha256']==state['checkpoint_sha256']
row=dict(task=condition['task'],mode=condition['mode'],checkpoint_sha256=state['checkpoint_sha256'],
         continuation_seed=condition['continuation_seed'],status='waiting_model',attempts=[],reason='Recover terminal3 infrastructure interrupts; full70, no partial concatenation')
def persist():
    row['updated_wall']=time.time();save(MAIN/'recovery_overlay.json',dict(conditions=[row]));save(OUT/'state.json',row)
original_hook=sys.excepthook
def report_failure(kind,error,traceback):
    row.update(status='failed',supervisor_error=kind.__name__+': '+str(error));persist()
    original_hook(kind,error,traceback)
sys.excepthook=report_failure
persist();deadline=time.monotonic()+1800
validation=OUT/'model_gpu3/eager_validation/result.json'
while not validation.exists():
    # Model PID belongs to the L20 host, not this4090 PID namespace.
    # Shared canonical evidence plus the actual forwarded health API is authoritative.
    if time.monotonic()>deadline:raise TimeoutError('Recovery model canonical gate')
    time.sleep(2)
assert json.loads(validation.read_text())['passed']
while True:
    try:
        with WSPolicyClient('ws://127.0.0.1:35910',timeout=10,open_timeout=3) as client:health=client.ping()
        assert health.get('readiness_token')==model['token'];assert health.get('executable_action_chunk_v1')==dict(horizon=32)
        break
    except (OSError,TimeoutError):
        if time.monotonic()>deadline:raise
        time.sleep(2)
slot=Path('/tmp','openwam-recovery-'+hashlib.sha256(model['token'].encode()).hexdigest()[:24]+'.lock').open('a')
fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
def capacity():
    text=subprocess.check_output(['nvidia-smi','--id=4','--query-gpu=memory.free,pci.bus_id','--format=csv,noheader,nounits'],text=True)
    free,pci=text.strip().split(',');return int(free),pci.strip()
source=next(w for w in state['workers'].values() if w.get('status')=='running' and w.get('gpu')==4)
base=dict(field.split('=',1) for field in Path('/proc',str(source['pid']),'environ').read_bytes().decode().split('\0') if '=' in field)
base.update(CUDA_VISIBLE_DEVICES='4',ROBOTWIN_TEST_NUM='70')
template=yaml.safe_load((MAIN/'policy_template.yml').read_text())
for attempt in range(1,4):
    row['status']='waiting_capacity';persist();stable=0
    while stable<5:
        free,pci=capacity();stable=stable+1 if free>=9472 else 0;row['free_mib']=free;persist();time.sleep(2)
    directory=OUT/f'attempt{attempt}';directory.mkdir(exist_ok=False)
    policy=directory/'policy.yml';policy.write_text(yaml.safe_dump(dict(template,host='127.0.0.1',port=35910,eval_video_log=False)))
    env=dict(base,ROBOTWIN_RUNTIME_ROOT=str(directory/'runtime'),POLICY_CONFIG_PATH=str(policy))
    command=[str(ROOT/'.conda/robotwin-official/bin/python'),'-u',str(HERE/'run_chunk_native.py'),
        '--policy',str(policy),'--task',row['task'],'--mode',row['mode'],'--expected-pci',pci,
        '--evidence',str(directory/'render_affinity.json'),'--seed-start',str(row['continuation_seed'])]
    child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=(directory/'client.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
    ticks=Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19]
    record=dict(attempt=attempt,pid=child.pid,start_ticks=ticks,gpu=4,output=str(directory),started_wall=time.time())
    row['attempts'].append(record);row.update(status='running',pid=child.pid,gpu=4,output=str(directory));persist()
    start=time.monotonic();last=start;signature=None
    while child.poll() is None:
        free,_=capacity();log=(directory/'client.log').read_text(errors='replace')
        text=re.sub(r'\x1b\[[0-9;]*m','',log);counts=re.findall(r'Success rate:\s*(\d+)/(\d+)',text);steps=re.findall(r'step:\s*(\d+)\s*/\s*(\d+)',text)
        current=(counts[-1:] or None,steps[-1:] or None)
        if current!=signature:signature=current;last=time.monotonic()
        row.update(free_mib=free,observed_episodes=int(counts[-1][1]) if counts else 0,
                   observed_successes=int(counts[-1][0]) if counts else 0,step=list(steps[-1]) if steps else None);persist()
        if free<1536 or time.monotonic()-last>2700:
            p=Path('/proc',str(child.pid),'stat')
            if p.exists() and p.read_text().rsplit(')',1)[1].split()[19]==ticks:os.killpg(child.pid,signal.SIGTERM)
            record['guard']='free1536MiB or45min no progress';break
        time.sleep(2)
    code=child.wait(timeout=30)
    result=validate_result((directory/'client.log').read_text(errors='replace'),directory/'runtime',70,code)
    if result['complete']:
        seeds=result['accepted_seeds'];prefix=condition['prefix30_accepted_seeds']
        assert len(seeds)==len(set(seeds))==70 and seeds==sorted(seeds) and seeds[0]>=row['continuation_seed']
        assert not set(seeds).intersection(prefix) and len(set(seeds+prefix))==100
        result['continuation_seed_integrity_passed']=True
    record.update(returncode=code,result=result,wall_seconds=time.monotonic()-start)
    save(directory/'run.json',record)
    if result['complete']:
        row.update(status='complete',result=result);persist();break
    row['status']='failed' if attempt==3 else 'retrying';persist()
print('RECOVERY_TERMINAL',row['status'],flush=True)
