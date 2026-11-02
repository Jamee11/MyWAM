from pathlib import Path
import os,json,subprocess,time,secrets,sys
R=Path('/remote-home/jinminghao/OpenWAM');O=Path(__file__).resolve().parent;sys.path.insert(0,str(O));sys.path.insert(0,str(R))
from intervention import MODES
m=json.load(open(O/'deployment.json'));token=secrets.token_hex(16);rows=[]
for (gpu,text,port,name) in [(1,4,40100,'replica0'),(2,6,40120,'replica1')]:
 d=O/name;d.mkdir()
 c=subprocess.Popen([str(R/'.venv/bin/python'),'-u',str(O/'serve_split.py'),'--ckpt-dir',m['checkpoint_ram'],'--port',str(port),'--slots','12','--manifest',str(d/'pool.json'),'--readiness-token',token,'--config',str(O/'deploy.yaml')],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES=f'{gpu},{text}',OMP_NUM_THREADS='4'),stdin=subprocess.DEVNULL,stdout=(d/'server.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
 rows.append(dict(pid=c.pid,start_ticks=Path(f'/proc/{c.pid}/stat').read_text().rsplit(')',1)[1].split()[19],gpu=gpu,text_gpu=text,directory=str(d),ports=list(range(port,port+12))))
(O/'model_processes.json').write_text(json.dumps(rows,indent=2)+'\n')
(O/'launch_status.json').write_text(json.dumps(dict(status='loading_models',replicas=2,sessions=24))+'\n')
from benchmarks.utils.transport import WSPolicyClient
import numpy as np
for row in rows:
 deadline=time.monotonic()+1800;d=Path(row['directory'])
 while not (d/'eager_validation/result.json').exists() or not (d/'pool.json').exists():
  f=Path(f"/proc/{row['pid']}/stat");assert f.exists() and f.read_text().rsplit(')',1)[1].split()[0]!='Z',str(d/'server.log')
  if time.monotonic()>deadline:raise TimeoutError('load')
  time.sleep(2)
fixtures=json.load(open(m['reference_fixtures']));checks=[];canonical={}
for row in rows:
 with WSPolicyClient(f"ws://127.0.0.1:{row['ports'][0]}",timeout=300) as client:
  assert client.ping()['readiness_token']==token
  def predict(mode):
   client.reset();out=client.predict_once(dict(fixtures[0],sidecar_intervention=mode,executable_action_chunk_v1=dict(horizon=32)))
   a=np.asarray(out['executable_action_chunk_v1']['actions']);assert a.shape==(32,20) and np.isfinite(a).all();return a
  full=predict('full')
  for mode in MODES:
   a=predict(mode);assert np.array_equal(full,predict('full')),'request intervention leaked into Full'
   if mode in canonical:assert np.array_equal(a,canonical[mode]),'replica action mismatch'
   canonical[mode]=a
   checks.append(dict(gpu=row['gpu'],mode=mode,max_abs_vs_full=float(np.max(np.abs(a-full))),full_restored_exact=True))
 for port in row['ports']:
  with WSPolicyClient(f'ws://127.0.0.1:{port}',timeout=60) as c:assert c.ping()['readiness_token']==token
ports=[p for pair in zip(rows[0]['ports'],rows[1]['ports']) for p in pair]
(O/'deployment_acceptance.json').write_text(json.dumps(dict(passed=True,checkpoint_sha256=m['checkpoint_sha256'],replicas=2,sessions=24,modes=list(MODES),checks=checks,cross_replica_exact=True),indent=2)+'\n')
(O/'available_ports.json').write_text(json.dumps(ports)+'\n')
c=subprocess.Popen(['bash',str(R/'benchmarks/robotwin/remote_reverse_tunnel.sh'),'root@210.45.70.21','26199',*[f'{p}:{p}' for p in ports]],stdin=subprocess.DEVNULL,stdout=(O/'tunnel.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
(O/'tunnel_process.json').write_text(json.dumps(dict(pid=c.pid,start_ticks=Path(f'/proc/{c.pid}/stat').read_text().rsplit(')',1)[1].split()[19]))+'\n');time.sleep(3);assert c.poll() is None
result=subprocess.run(['ssh','-n','-o','BatchMode=yes','-p','26199','root@210.45.70.21',f'{R}/.conda/robotwin-official/bin/python {O}/start_remote.py'],capture_output=True,text=True)
assert result.returncode==0,result.stderr
(O/'launch_status.json').write_text(json.dumps(dict(status='running_remote_evaluation',replicas=2,sessions=24,remote=result.stdout))+'\n')
