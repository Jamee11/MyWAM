import copy,json,os,sys,time
from pathlib import Path
ROOT=Path('/remote-home/jinminghao/OpenWAM');sys.path.insert(0,str(ROOT))
from benchmarks.robotwin.sidecar_fast_runtime import install
install()
import openwam.deploy.server as sm
original=sm.build_server_from_config
OUT=Path(__file__).resolve().parent

def checked(cfg,*args,**kwargs):
 import numpy as np,torch
 assert cfg.optimization.compile.enabled is False and cfg.optimization.dit_cache.enabled
 server=original(cfg,*args,**kwargs);engine=server.engine;arch=engine.architecture
 assert arch.sidecar_enabled and tuple(arch.tap_blocks)==(7,15,23) and not arch._evaluation_compile_enabled and not arch.training
 assert engine._dit_cache is not None

 from intervention import install
 install(arch.sidecar)
 server._init_policy()
 metadata=json.loads((OUT/'deployment.json').read_text());wire=Path(metadata['reference_fixtures']).read_text();fixtures=json.loads(wire)
 values=[];checks=[]
 for i,payload in enumerate(fixtures):
  expected=None
  for repeat in range(2):
   server.reset();obs=server._obs_preprocessor.preprocess(copy.deepcopy(payload));conditions=server._policy._build_conditions(obs)
   torch.cuda.synchronize();start=time.perf_counter()
   with torch.inference_mode():result=engine.generate(conditions)
   torch.cuda.synchronize();value=result['actions'];value=value.detach().float().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
   if expected is None:expected=value.copy();values.append(value.tolist())
   error=float(np.max(np.abs(value-expected)));equal=bool(np.array_equal(value,expected));assert equal and np.isfinite(value).all()
   checks.append(dict(fixture=i,repeat=repeat,exact_equal=equal,max_abs_error=error,seconds=time.perf_counter()-start))
 reference=OUT/'reference_checkpoint_actions.json'
 import fcntl
 with (OUT/'reference_checkpoint_actions.lock').open('a') as lease:
  fcntl.flock(lease,fcntl.LOCK_EX)
  if not reference.exists():
   tmp=reference.with_suffix('.tmp');tmp.write_text(json.dumps(dict(checkpoint_sha256=metadata['checkpoint_sha256'],actions=values))+'\n');tmp.replace(reference)
 deadline=time.monotonic()+1800
 while not reference.exists():
  if time.monotonic()>deadline:raise TimeoutError('CanonicalCheckpoint reference not ready')
  time.sleep(2)
 canonical=json.loads(reference.read_text());assert canonical['checkpoint_sha256']==metadata['checkpoint_sha256']
 for a,b in zip(values,canonical['actions']):assert np.array_equal(np.asarray(a),np.asarray(b))
 d=Path(sys.argv[sys.argv.index('--manifest')+1]).parent/'eager_validation';d.mkdir()
 (d/'inputs.json').write_text(wire)
 report=dict(passed=True,selected_strategy='eager_cached',compile=False,cache=True,tap_blocks=[7,15,23],checks=checks,cross_replica_exact=True,checkpoint_sha256=metadata['checkpoint_sha256'],compile_evidence=dict(generated_kernel_count=0))
 (d/'result.json').write_text(json.dumps(report,indent=2)+'\n');server.reset();print('CHECKPOINT_EAGER_CACHED_VALIDATED',flush=True)
 return server
sm.build_server_from_config=checked
from serve_modes import main
main()
