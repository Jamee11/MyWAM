"""Run a bounded independent4090 replay probe; preserve all formal jobs."""
from pathlib import Path
import json
import os
import signal
import subprocess
import time
import yaml
import argparse

ROOT=Path('/remote-home/jinminghao/OpenWAM');HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--name',default='sim_hotpath_replay_20261028T1020Z')
p.add_argument('--cases',default='baseline,parallel_validation,parallel,lazy_rgb,combined,baseline_repeat');p.add_argument('--profile',action='store_true');options=p.parse_args()
OUT=HERE/options.name;OUT.mkdir(exist_ok=False)
main=ROOT/'outputs/robotwin_eval/fullbackward32000_supplement70_4090_20261028T035728Z'
state=json.loads((main/'elastic_v3/state.json').read_text())
worker=next(w for w in state['workers'].values() if w.get('status')=='running' and w.get('gpu')==4)
env=dict(field.split('=',1) for field in Path('/proc',str(worker['pid']),'environ').read_bytes().decode().split('\0') if '=' in field)
env.update(ROBOTWIN_TEST_NUM='30',CUDA_VISIBLE_DEVICES='4',ROBOTWIN_RUNTIME_ROOT=str(OUT/'runtime'))
def gpu():
    text=subprocess.check_output(['nvidia-smi','--id=4','--query-gpu=memory.free,pci.bus_id','--format=csv,noheader,nounits'],text=True)
    free,pci=text.strip().split(',');return int(free),pci.strip()
begin=time.monotonic()
while gpu()[0]<9472:
    if time.monotonic()-begin>300:raise TimeoutError('No safe GPU4 slot for diagnostic replay')
    time.sleep(2)
free,pci=gpu();policy=OUT/'policy.yml';cfg=yaml.safe_load((main/'policy_template.yml').read_text());cfg.update(eval_video_log=False)
policy.write_text(yaml.safe_dump(cfg));env['POLICY_CONFIG_PATH']=str(policy)
command=[str(ROOT/'.conda/robotwin-official/bin/python'),'-u',str(HERE/'probe_sim_hotpath.py'),
    '--policy',str(policy),'--task','click_alarmclock','--mode','demo_clean','--expected-pci',pci,
    '--evidence',str(OUT/'render_affinity.json')]
command.extend(['--cases',options.cases])
if options.profile:command.append('--profile')
child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=(OUT/'client.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
tick=Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19]
record=dict(pid=child.pid,start_ticks=tick,free_before_mib=free,gpu=4,scope='Replay-only diagnostic; no model requests or formal result changes')
(OUT/'process.json').write_text(json.dumps(record,indent=2)+'\n');print('HOTPATH_PROBE_STARTED',child.pid,free,flush=True)
started=time.monotonic();minimum=free
while child.poll() is None:
    current,_=gpu();minimum=min(minimum,current)
    with (OUT/'memory.jsonl').open('a') as f:f.write(json.dumps(dict(wall=time.time(),free_mib=current))+'\n')
    if current<2048 or time.monotonic()-started>1500:
        p=Path('/proc',str(child.pid),'stat')
        if p.exists() and p.read_text().rsplit(')',1)[1].split()[19]==tick:os.killpg(child.pid,signal.SIGTERM)
        record['stopped_reason']='capacity_or_time_guard';break
    time.sleep(1)
try:code=child.wait(timeout=30)
except subprocess.TimeoutExpired:raise RuntimeError('Owned diagnostic did not stop')
record.update(returncode=code,wall_seconds=time.monotonic()-started,minimum_free_mib=minimum)
(OUT/'exit.json').write_text(json.dumps(record,indent=2)+'\n');print('HOTPATH_PROBE_EXIT',code,flush=True)
