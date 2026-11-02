"""Load identical trained weights, validate Full, then use existing full bypass."""
from pathlib import Path
import sys,copy,json,time
ROOT=Path('/remote-home/jinminghao/OpenWAM');OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from benchmarks.robotwin.sidecar_fast_runtime import install
install()
import openwam.deploy.server as sm
original=sm.build_server_from_config

def checked(cfg,*args,**kwargs):
 import numpy as np,torch
 assert not cfg.optimization.compile.enabled and cfg.optimization.dit_cache.enabled
 server=original(cfg,*args,**kwargs);engine=server.engine;arch=engine.architecture
 assert arch.sidecar_enabled and not arch.training and engine._dit_cache is not None
 server._init_policy();meta=json.load(open(OUT/'deployment.json'));fixtures=json.load(open(meta['reference_fixtures']));canonical=json.load(open(OUT/'reference_checkpoint_actions.json'))
 assert meta['checkpoint_sha256']==canonical['checkpoint_sha256']
 versions={n:p._version for n,p in arch.named_parameters()};calls={};handles=[]
 def hook(name):
  def count(*a,**kw):calls[name]=calls.get(name,0)+1
  return count
 for name,module in arch.sidecar.named_modules():
  if name:handles.append(module.register_forward_pre_hook(hook(name)))
 init=arch.sidecar.init_tokens
 def counted_init(*a,**kw):calls['init_tokens']=calls.get('init_tokens',0)+1;return init(*a,**kw)
 arch.sidecar.init_tokens=counted_init;write=arch.sidecar.write_action
 def predict(payload):
  server.reset();obs=server._obs_preprocessor.preprocess(copy.deepcopy(payload));conditions=server._policy._build_conditions(obs)
  with torch.inference_mode():value=engine.generate(conditions)['actions']
  value=value.detach().float().cpu().numpy() if torch.is_tensor(value) else np.asarray(value);assert np.isfinite(value).all();return value
 checks=[];arrays=[]
 for index,payload in enumerate(fixtures):
  arch.sidecar_enabled=True;full=predict(payload);assert np.array_equal(full,np.asarray(canonical['actions'][index])),'Full canonical mismatch'
  arch.sidecar.write_action=lambda index,actions,tokens,valid:actions
  try:write_off=predict(payload)
  finally:arch.sidecar.write_action=write
  arch.sidecar_enabled=False;calls.clear();off=predict(payload);off_repeat=predict(payload)
  assert not calls,'Sidecar executed despite full bypass';assert np.array_equal(off,off_repeat)
  arch.sidecar_enabled=True;assert np.array_equal(full,predict(payload)),'Full was not restored'
  checks.append(dict(fixture=index,full_canonical_exact=True,sidecar_calls_when_off=0,repeat_exact=True,full_restored_exact=True,off_vs_write_off_exact=bool(np.array_equal(off,write_off)),off_vs_write_off_max_abs=float(np.max(np.abs(off-write_off))),off_vs_full_max_abs=float(np.max(np.abs(off-full)))))
  arrays.append(dict(full=full.tolist(),write_off=write_off.tolist(),sidecar_off=off.tolist()))
 arch.sidecar_enabled=False;calls.clear();arch.sidecar.init_tokens=init
 for handle in handles:handle.remove()
 assert all(p._version==versions[n] for n,p in arch.named_parameters())
 server.reset();d=Path(sys.argv[sys.argv.index('--manifest')+1]).parent/'eager_validation';d.mkdir()
 (d/'result.json').write_text(json.dumps(dict(passed=True,intervention='sidecar_off',checkpoint_sha256=meta['checkpoint_sha256'],cache=True,compile=False,checks=checks,parameter_versions_unchanged=True),indent=2)+'\n')
 (d/'actions.json').write_text(json.dumps(arrays)+'\n');print('SIDECAR_COMPLETE_BYPASS_VALIDATED',flush=True);return server
sm.build_server_from_config=checked
from benchmarks.robotwin.serve_parallel import main
main()
