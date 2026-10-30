"""CPU-only deterministic target-entropy sample; does not evaluate model predictions."""
from pathlib import Path
import json
import numpy as np
ROOT=Path('/remote-home/jinminghao/OpenWAM');OUT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'outputs/sidecar/aloha_4350/manifest.ready.json').read_text())
samples=[]
for task in manifest['tasks']:
 eps=task['episodes']
 for i in [0,len(eps)//2,len(eps)-1]:
  with np.load(eps[i]['labels']) as data:
   indices=np.unique(np.linspace(0,len(data['targets'])-1,20,dtype=int));t=data['targets'][indices].copy();arms=data['arm_present'][indices]
  t[:,:2]*=arms;t[:,118:130]*=np.repeat(np.repeat(arms,2,axis=1),3,axis=1)
  t[:,121:124]*=t[:,14:15];t[:,127:130]*=t[:,15:16]
  samples.append(t)
t=np.concatenate(samples);boxes=t[:,70:118].reshape(-1,4,3,4);valid=t[:,118:130].reshape(-1,4,3)
# WanVideoVAE38 factor16 and DiT patch2 -> 384/32 x 320/32 = 12x10.
h,w=12,10;y,x=np.meshgrid((np.arange(h)+.5)/h,(np.arange(w)+.5)/w,indexing='ij');view=np.where(y<2/3,0,np.where(x<.5,1,2))
offset=np.array([[0,0],[0,2/3],[.5,2/3]]);scale=np.array([[1,2/3],[.5,1/3],[.5,1/3]])
lo=boxes[...,:2]*scale+offset;hi=boxes[...,2:]*scale+offset;centre=(lo+hi)/2;sigma=np.maximum((hi-lo)/2,np.array([1/w,1/h]))
distance=((x-centre[...,0,None,None])/sigma[...,0,None,None])**2+((y-centre[...,1,None,None])/sigma[...,1,None,None])**2
region=view[None,None]==np.arange(3)[None,None,:,None,None]
density=(np.exp(-.5*distance)*region*valid[...,None,None]).sum(axis=2);q=density/np.maximum(density.sum(axis=(-2,-1),keepdims=True),1e-8)
entropy=-(q*np.log(np.maximum(q,1e-30))).sum(axis=(-2,-1));rolevalid=valid.sum(axis=-1)>0;v=entropy[rolevalid]
report={'scope':'150 episodes: first/middle/last per task; up to20 evenly spaced frames each; deterministic, not exact training-batch weighted','frames':len(t),'valid_roles':int(rolevalid.sum()),'grid':[h,w],'target_entropy_mean':float(v.mean()),'target_entropy_std':float(v.std()),'target_entropy_p05':float(np.quantile(v,.05)),'target_entropy_p95':float(np.quantile(v,.95)),'interpretation':'Soft-target CE lower bound for this sample; no model outputs evaluated; subtracting from separately logged batch means is not an exact KL estimate.'}
(OUT/'target_entropy_sample.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
