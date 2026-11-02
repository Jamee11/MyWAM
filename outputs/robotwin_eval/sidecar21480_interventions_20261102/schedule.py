from pathlib import Path
from datetime import datetime,timezone
import copy,json,os,subprocess,threading,time,yaml,fcntl,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
R=Path('/remote-home/jinminghao/OpenWAM');O=Path(__file__).resolve().parent;PY=R/'.conda/robotwin-official/bin/python'
lease=(O/'scheduler.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
experiment=json.load(open(O/'experiment.json'));base=json.load(open(O/'base_env.json'));template=yaml.safe_load((O/'policy_template.yml').read_text())
lock=threading.RLock();reserved={};busyports=set();state=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),jobs=[],errors=[])
for task in experiment['tasks']:
 for domain in experiment['modes']:
  state['jobs'].append(dict(task=task,domain=domain,kind='manifest',status='queued',attempts=[]))
  for intervention in experiment['interventions']:state['jobs'].append(dict(task=task,domain=domain,kind=intervention,status='queued',attempts=[]))
def save():
 state['updated_utc']=datetime.now(timezone.utc).isoformat();p=O/'state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(O/'state.json')
def capacity():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,pci.bus_id','--format=csv,noheader,nounits'],text=True)
 return {int(x.split(',')[0]):dict(free=int(x.split(',')[1]),pci=x.split(',')[2].strip()) for x in raw.splitlines()}
def manifest_for(task,domain):
 job=next(j for j in state['jobs'] if j['task']==task and j['domain']==domain and j['kind']=='manifest')
 return job.get('result') if job['status']=='complete' else None
def ports_for(kind):
 d=O/'models'/kind
 if not (d/'eager_validation/result.json').exists() or not (d/'pool.json').exists():return []
 if not json.load(open(d/'eager_validation/result.json'))['passed']:return []
 return json.load(open(d/'pool.json'))['ports']
def select():
 for kind in ['eval','manifest']:
  for j in state['jobs']:
   if j['status']!='queued' or (kind=='manifest')!=(j['kind']=='manifest'):continue
   port=None
   if kind=='eval':
    if manifest_for(j['task'],j['domain']) is None:continue
    ports=[p for p in ports_for(j['kind']) if p not in busyports]
    if not ports:continue
    port=ports[0]
   caps=capacity();active=[x for x in state['jobs'] if x['status']=='running']
   counts={g:sum(x.get('gpu')==g for x in active) for g in caps}
   possible=[g for g in caps if counts[g]<4 and caps[g]['free']-reserved.get(g,0)>=6144]
   if not possible:continue
   gpu=min(possible,key=lambda g:(counts[g],-caps[g]['free']));reserved[gpu]=reserved.get(gpu,0)+6144
   if port is not None:busyports.add(port)
   j.update(status='running',gpu=gpu,port=port)
   return j,gpu,caps[gpu]['pci'],port
 return None

def run_worker(slot):
 while True:
  with lock:
   pending=[j for j in state['jobs'] if j['status'] in ('queued','running')]
   if not pending:return
   selected=select();save()
  if selected is None:time.sleep(3);continue
  j,gpu,pci,port=selected;attempt=len(j['attempts'])+1;directory=O/'jobs'/j['domain']/j['task']/j['kind']/f'attempt{attempt}';directory.mkdir(parents=True)
  policy=directory/'policy.yml';policy.write_text(yaml.safe_dump(dict(template,host='127.0.0.1',port=port or 1,eval_video_log=False)))
  command=[str(PY),'-u',str(O/'run_manifest.py'),'--policy',str(policy),'--task',j['task'],'--mode',j['domain'],'--expected-pci',pci,'--evidence',str(directory/'render_affinity.json'),'--operation','build_manifest' if j['kind']=='manifest' else 'run_eval','--result-file',str(directory/'result.json'),'--progress-file',str(directory/'progress.json'),'--episodes','10']
  if j['kind']!='manifest':
   manifest=manifest_for(j['task'],j['domain']);command+=['--manifest-file',manifest['path'],'--manifest-hash',manifest['manifest_hash']]
  env=dict(base,CUDA_VISIBLE_DEVICES=str(gpu),ROBOTWIN_RUNTIME_ROOT=str(directory/'runtime'),POLICY_CONFIG_PATH=str(policy),ROBOTWIN_TEST_NUM='10')
  child=subprocess.Popen(command,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=(directory/'client.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
  record=dict(pid=child.pid,start_ticks=Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19],gpu=gpu,port=port,output=str(directory),started_wall=time.time())
  with lock:j['attempts'].append(record);save()
  print('START',slot,j['task'],j['domain'],j['kind'],gpu,child.pid,flush=True)
  released=False;start=time.monotonic()
  while child.poll() is None:
   # Release only the launch reservation; do not terminate ongoing trials at arbitrary VRAM margins.
   if not released and time.monotonic()-start>180:
    with lock:reserved[gpu]-=6144;released=True
   time.sleep(3)
  code=child.returncode;valid=False;error=None;result=None
  try:
   assert code==0, f'process exited{code}'
   result=json.load(open(directory/'result.json'))
   if j['kind']=='manifest':
    from benchmarks.utils.eval_manifest import verify_manifest
    verify_manifest(result);assert len(result['entries'])==10
    seeds=[e['seed'] for e in result['entries']];assert len(set(seeds))==10 and seeds==sorted(seeds)
    result=dict(path=str(directory/'result.json'),manifest_hash=result['manifest_hash'],seeds=seeds)
   else:
    assert result['manifest_hash']==manifest['manifest_hash'] and len(result['episodes'])==10
    assert [(e['episode'],e['seed'],e['instruction']) for e in result['episodes']]==[(e['episode'],e['seed'],e['instructions']['unseen']) for e in json.load(open(manifest['path']))['entries']]
    assert result['successes']==sum(e['success'] for e in result['episodes'])
   valid=True
  except BaseException as e:error=repr(e)
  with lock:
   if not released:reserved[gpu]-=6144
   if port is not None:busyports.remove(port)
   record.update(returncode=code,ended_wall=time.time(),valid=valid,error=error)
   if valid:j.update(status='complete',result=result)
   else:
    j['status']='queued' if attempt<20 else 'failed';state['errors'].append(dict(task=j['task'],domain=j['domain'],kind=j['kind'],attempt=attempt,error=error))
   save()
  print('END',slot,j['task'],j['domain'],j['kind'],j['status'],error,flush=True)

def guarded(slot):
 try:run_worker(slot)
 except BaseException as e:
  import traceback;traceback.print_exc()
  with lock:state['errors'].append(dict(slot=slot,error=repr(e)));save()
threads=[threading.Thread(target=guarded,args=(i,)) for i in range(12)]
for t in threads:t.start()
for t in threads:t.join()
with lock:
 state['status']='complete' if all(j['status']=='complete' for j in state['jobs']) else 'finished_with_errors';save()
