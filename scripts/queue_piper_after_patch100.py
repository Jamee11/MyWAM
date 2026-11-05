"""User-requested handover: audited Patch100 -> two Piper servers -> 4090 native100."""
from pathlib import Path
from datetime import datetime, timezone
import json, os, signal, socket, subprocess, sys, time, secrets

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from benchmarks.robotwin.piper_protocol import piper_tasks, DEFAULT_MANIFEST, DEFAULT_CHECKPOINT
from benchmarks.utils.transport import WSPolicyClient

O = Path(sys.argv[1]).absolute()
P = R/'outputs/robotwin_eval/patchwam_ramp100_final42960_supp70_20261103'
OLD = R/'outputs/robotwin_eval/patchwam_ramp100_final42960_all50_30ep_20261103'
SSH = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','26199','root@210.45.70.21']

def status(stage, **extra):
    value = dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(), **extra)
    temp=O/'status.tmp';temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(O/'status.json')

def remote(code):
    result=subprocess.run(SSH+['python3 -'],input=code,text=True,capture_output=True,check=True)
    return json.loads(result.stdout)

def start(command, log, env=None):
    with log.open('a') as stream:
        p=subprocess.Popen(command,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    return dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19])

def main():
    assert len(piper_tasks(DEFAULT_MANIFEST)) == 35
    while True:
        state=json.loads((P/'state.json').read_text());audit=P/'final_independent_audit.json'
        if state['status']=='complete' and audit.exists() and json.loads(audit.read_text())['passed']:
            break
        status('waiting_patch100_audit',remaining_patch_conditions=sum(c['status']!='complete' for c in state['conditions']))
        time.sleep(20)
    retired=[]
    for replica in ['replica2','replica3']:
        pool=json.loads((OLD/replica/'pool.json').read_text());pid=pool['pid'];proc=Path(f'/proc/{pid}')
        if proc.exists():
            command=proc.joinpath('cmdline').read_bytes()
            assert str(OLD/replica/'serve_split.py').encode() in command
            assert str(pool['checkpoint_dir']).encode() in command
            os.kill(pid,signal.SIGTERM);retired.append(pid)
    (O/'retired_patch_models.json').write_text(json.dumps(retired)+'\n')
    time.sleep(5)
    stage=R/'outputs/robotwin_eval/piper21952_smoke_20261104/checkpoint'
    models=[]
    for index,(devices,port) in enumerate([('3,4',43500),('2,6',43520)]):
        d=O/f'replica{index}';d.mkdir(exist_ok=True)
        command=[str(R/'.venv/bin/python'),'-u',str(R/'benchmarks/robotwin/serve_piper_sidecar.py'),'--ckpt-dir',str(stage),'--port',str(port),'--slots','16','--manifest',str(d/'pool.json'),'--readiness-token',secrets.token_hex(16),'--config',str(R/'configs/deploy_piper_eval.yaml')]
        models.append(dict(**start(command,d/'server.log',dict(os.environ,CUDA_VISIBLE_DEVICES=devices,PIPER_SPLIT_T5='1',OMP_NUM_THREADS='4',HF_ENDPOINT='https://hf-mirror.com',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')),directory=str(d),devices=devices,base_port=port))
    (O/'model_processes.json').write_text(json.dumps(models,indent=2)+'\n');status('loading_piper_models',models=models)
    fixtures=json.loads((R/'outputs/robotwin_eval/piper21952_smoke_multiview_20261104/native_fixtures.json').read_text())
    import numpy as np
    checks=[]
    for model in models:
        d=Path(model['directory']);deadline=time.monotonic()+1800
        while not (d/'pool.json').exists():
            proc=Path(f'/proc/{model["pid"]}')
            if not proc.exists() or proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]=='Z':raise RuntimeError(f'Model failed: {d}/server.log')
            if time.monotonic()>deadline:raise TimeoutError('Piper model loading timed out')
            time.sleep(5)
        time.sleep(2)
        with WSPolicyClient(f'ws://127.0.0.1:{model["base_port"]}',timeout=300) as client:
            assert client.ping()['executable_action_chunk_v1']=={'horizon':32}
            for i, fixture in enumerate(fixtures):
                client.reset();reply=client.predict_once({**fixture,'executable_action_chunk_v1':{'horizon':32}})
                actual=np.asarray(reply['executable_action_chunk_v1']['actions'],dtype=np.float32)
                reference=np.load(R/f'outputs/robotwin_eval/piper21952_smoke_multiview_20261104/fixture{i}_actions.npy')
                assert actual.shape==(32,20) and np.isfinite(actual).all()
                assert np.array_equal(actual,reference)
        checks.append(dict(replica=model['directory'],two_native_fixtures_exact=True))
    (O/'deployment_acceptance.json').write_text(json.dumps(dict(passed=True,checks=checks),indent=2)+'\n')
    ports=[p for pair in zip(range(43500,43516),range(43520,43536)) for p in pair]
    tunnel=start(['bash',str(R/'benchmarks/robotwin/remote_reverse_tunnel.sh'),'root@210.45.70.21','26199',*[f'{p}:{p}' for p in ports]],O/'tunnel.log')
    (O/'tunnel_process.json').write_text(json.dumps(tunnel)+'\n');time.sleep(2)
    assert Path(f'/proc/{tunnel["pid"]}').exists()
    env_updates=dict(SIM_GPUS=','.join(str(g) for _ in range(3) for g in range(8)),PORTS=','.join(map(str,ports[:24])),EPISODES='100',TASKS='all',OUTPUT=str(O/'eval'),CHECKPOINT=str(DEFAULT_CHECKPOINT))
    code=f'''from pathlib import Path
import json,os,subprocess
R=Path({str(R)!r}); O=Path({str(O)!r})
env=dict(os.environ,**{env_updates!r})
with (O/'simulator_launcher.log').open('a') as log:
 p=subprocess.Popen(['bash',str(R/'scripts/eval_robotwin_piper.sh'),'--min-free-mib','8192','--resume-episodes','--attempts','20'],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,start_ticks=Path(f'/proc/{{p.pid}}/stat').read_text().rsplit(')',1)[1].split()[19])))
'''
    controller=remote(code);(O/'remote_process.json').write_text(json.dumps(controller)+'\n')
    status('piper_full100_running',tasks=35,conditions=70,episodes=7000,target_workers=24,models=2,remote_controller=controller)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        status('handover_error',error=repr(exc));raise
