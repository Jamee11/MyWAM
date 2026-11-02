from pathlib import Path
import json,types,torch
O=Path(__file__).resolve().parent;ns={};exec((O/'models/full/intervention.py').read_text(),ns)
checks=[]
for mode in ['full','write_off','relation_reset']:
 z=torch.arange(2*15*4,dtype=torch.float32).reshape(2,15,4);initial=z.clone();actions=torch.ones(2,3,4)
 obj=types.SimpleNamespace(read_refine=lambda *a,**kw:(z,{}),write_action=lambda i,a,z,v:a+1)
 ns['apply_intervention'](obj,mode);out,_=obj.read_refine()
 assert torch.equal(z,initial)
 if mode=='relation_reset':assert torch.count_nonzero(out[:,4:8])==0 and torch.equal(out[:,:4],z[:,:4]) and torch.equal(out[:,8:],z[:,8:])
 else:assert torch.equal(out,z)
 assert torch.equal(obj.write_action(0,actions,z,None),actions if mode=='write_off' else actions+1)
 checks.append(dict(mode=mode,passed=True))
(O/'intervention_scope_checks.json').write_text(json.dumps(dict(passed=True,checks=checks),indent=2)+'\n');print('INTERVENTION_SCOPE_CHECKS_PASSED',flush=True)
