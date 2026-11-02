from pathlib import Path
import json,subprocess,os,secrets,time,sys
R=Path('/remote-home/jinminghao/OpenWAM');O=Path(__file__).resolve().parent;sys.path.insert(0,str(R));meta=json.load(open(O/'deployment.json'));token=secrets.token_hex(16);d=O/'replica0';d.mkdir();ports=list(range(40200,40204))
def status(stage,**kw):(O/'launch_status.json').write_text(json.dumps(dict(status=stage,**kw),indent=2)+'\n')
status('loading_sidecar_off_model',body_gpu=2,text_gpu=4)
c=subprocess.Popen([str(R/'.venv/bin/python'),'-u',str(O/'serve_split.py'),'--ckpt-dir',meta['checkpoint_ram'],'--port','40200','--slots','4','--manifest',str(d/'pool.json'),'--readiness-token',token,'--config',str(O/'deploy.yaml')],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,4',OMP_NUM_THREADS='4'),stdin=subprocess.DEVNULL,stdout=(d/'server.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
(O/'model_process.json').write_text(json.dumps(dict(pid=c.pid,start_ticks=Path(f'/proc/{c.pid}/stat').read_text().rsplit(')',1)[1].split()[19],body_gpu=2,text_gpu=4,ports=ports))+'\n')
start=time.monotonic()
while not (d/'eager_validation/result.json').exists() or not (d/'pool.json').exists():
 assert c.poll() is None,'Model exited; see replica0/server.log'
 if time.monotonic()-start>1800:raise TimeoutError('model loading')
 time.sleep(2)
validation=json.load(open(d/'eager_validation/result.json'));assert validation['passed']
from benchmarks.utils.transport import WSPolicyClient
import numpy as np
fixture=json.load(open(meta['reference_fixtures']))[0];reference=np.asarray(json.load(open(d/'eager_validation/actions.json'))[0]['sidecar_off']);checks=[]
for port in ports:
 with WSPolicyClient(f'ws://127.0.0.1:{port}',timeout=300) as client:
  assert client.ping()['readiness_token']==token
  a=[]
  for _ in range(2):
   client.reset();result=client.predict_once(dict(fixture,executable_action_chunk_v1=dict(horizon=32)));a.append(np.asarray(result['executable_action_chunk_v1']['actions']))
  assert np.array_equal(a[0],a[1]) and np.array_equal(a[0],reference)
  client.reset();checks.append(dict(port=port,passed=True))
(O/'deployment_acceptance.json').write_text(json.dumps(dict(passed=True,intervention='sidecar_off',checks=checks,engine_validation=validation),indent=2)+'\n');(O/'available_ports.json').write_text(json.dumps(ports)+'\n')
t=subprocess.Popen(['bash',str(R/'benchmarks/robotwin/remote_reverse_tunnel.sh'),'root@210.45.70.21','26199',*[f'{port}:{port}' for port in ports]],stdin=subprocess.DEVNULL,stdout=(O/'tunnel.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
(O/'tunnel_process.json').write_text(json.dumps(dict(pid=t.pid,start_ticks=Path(f'/proc/{t.pid}/stat').read_text().rsplit(')',1)[1].split()[19]))+'\n');time.sleep(3);assert t.poll() is None
cmd=f'{R}/.conda/robotwin-official/bin/python {O}/start_remote.py';done=subprocess.run(['ssh','-n','-o','BatchMode=yes','-p','26199','root@210.45.70.21',cmd],capture_output=True,text=True);assert done.returncode==0,done.stderr
status('running_remote_evaluation',sessions=4,episodes=100,checkpoint_sha256=meta['checkpoint_sha256'],remote=done.stdout)
