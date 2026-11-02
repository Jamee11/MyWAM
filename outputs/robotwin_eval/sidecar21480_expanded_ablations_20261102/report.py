from pathlib import Path
from datetime import datetime,timezone
import json,time
O=Path(__file__).resolve().parent;E=json.load(open(O/'experiment.json'));modes=E['interventions']
while True:
 if not (O/'state.json').exists():time.sleep(5);continue
 s=json.load(open(O/'state.json'));jobs=s['jobs'];summary=[]
 qfile=O.parent/'sidecar21480_19arms_complete10_20261102/quarantined_conditions.json'
 q={r['key'] for r in json.load(open(qfile))['findings']} if qfile.exists() else set()
 def qkey(j):return '|'.join((j['task'],j['domain'],j['kind']))
 for mode in modes:
  for domain in ['all','demo_clean','demo_randomized']:
   paired=[]
   for j in jobs:
    if j['kind']!=mode or j['status']!='complete' or qkey(j) in q or '|'.join((j['task'],j['domain'],'full')) in q or (domain!='all' and domain!=j['domain']):continue
    full=next(x for x in jobs if x['kind']=='full' and x['task']==j['task'] and x['domain']==j['domain'])
    a=j['result']['episodes'];b=full['result']['episodes']
    assert [(v['episode'],v['seed'],v['instruction']) for v in a]==[(v['episode'],v['seed'],v['instruction']) for v in b]
    paired.extend(zip(a,b))
   if not paired:continue
   good=sum(a['success'] for a,b in paired);base=sum(b['success'] for a,b in paired)
   summary.append(dict(mode=mode,domain=domain,episodes=len(paired),successes=good,rate=100*good/len(paired),paired_full_rate=100*base/len(paired),delta_pp=100*(good-base)/len(paired),full_only=sum(bool(b['success'] and not a['success']) for a,b in paired),intervention_only=sum(bool(a['success'] and not b['success']) for a,b in paired)))
 result=dict(utc=datetime.now(timezone.utc).isoformat(),status=s['status'],complete_conditions=sum(j['kind']!='manifest' and j['status']=='complete' and qkey(j) not in q for j in jobs),quarantined_completed_conditions=sum(j['kind']!='manifest' and j['status']=='complete' and qkey(j) in q for j in jobs),total_conditions=360,active=sum(j['status']=='running' for j in jobs),summary=summary)
 p=O/'comparison.tmp';p.write_text(json.dumps(result,indent=2)+'\n');p.replace(O/'comparison.json')
 if s['status'] in ('complete','finished_with_errors'):(O/'final_results.json').write_text(json.dumps(result,indent=2)+'\n');break
 time.sleep(20)
