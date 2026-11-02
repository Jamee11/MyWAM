from pathlib import Path
import json,os,subprocess,time
R=Path('/remote-home/jinminghao/OpenWAM');P=Path(__file__).resolve().parent
placements=[('full','6',39700,'serve_chunk.py'),('write_off','1',39710,'serve_chunk.py'),('relation_reset','2,5',39720,'serve_split.py')]
record=[]
for mode,gpu,port,script in placements:
 O=P/'models'/mode;meta=json.load(open(O/'deployment.json'))
 cmd=[str(R/'.venv/bin/python'),'-u',str(O/script),'--ckpt-dir',meta['checkpoint_ram'],'--port',str(port),'--slots','4','--manifest',str(O/'pool.json'),'--readiness-token','intervention-'+mode,'--config',str(O/'deploy.yaml')]
 c=subprocess.Popen(cmd,cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdin=subprocess.DEVNULL,stdout=(O/'server.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
 record.append(dict(mode=mode,visible_gpus=gpu,ports=list(range(port,port+4)),pid=c.pid,start_ticks=Path('/proc',str(c.pid),'stat').read_text().rsplit(')',1)[1].split()[19]))
(P/'model_processes.json').write_text(json.dumps(record,indent=2)+'\n');print('INTERVENTION_MODELS_LAUNCHED',record,flush=True)
