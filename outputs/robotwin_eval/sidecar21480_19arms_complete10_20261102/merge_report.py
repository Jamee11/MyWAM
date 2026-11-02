from pathlib import Path
from datetime import datetime,timezone
import json,time
O=Path(__file__).resolve().parent;E=O.parent;A=E/'sidecar21480_expanded_ablations_20261102';B=E/'sidecar21480_all_off_20261102';spec=json.load(open(O/'experiment.json'));refs=json.load(open(O/'sealed_reference.json'))
first={}
for root in [A,B]:
 s=json.load(open(root/'state.json'))
 for j in s['jobs']:
  if j['kind']!='manifest':first[j['task'],j['domain'],j['kind']]=j

def native_episodes(job,start):
 if job.get('imported_reference'):
  d=json.load(open(job['result']['source_result']));episodes=d['episodes'][start:start+5]
 else:
  assert job['attempts'][-1]['returncode']==0
  d=json.load(open(Path(job['attempts'][-1]['output'])/'result.json'));episodes=d['episodes']
 assert len(episodes)==5 and episodes==job['result']['episodes']
 log=Path(job['result']['source_result']).parent/'client.log' if job.get('imported_reference') else Path(job['attempts'][-1]['output'])/'client.log'
 data=log.read_bytes();assert b'OIDN Error:' not in data and b'ErrorDeviceLost' not in data
 return episodes,d['manifest_hash']

while True:
 if not (O/'state.json').exists():time.sleep(5);continue
 state=json.load(open(O/'state.json'));rows=[]
 for tail in state['jobs']:
  if tail['kind']=='manifest' or tail['status']!='complete':continue
  key=tail['task'],tail['domain'],tail['kind'];head=first[key];assert head['status'] in ('complete','failed')
  if tail.get('repair_full10'):
   assert tail['attempts'][-1]['returncode']==0
   directory=Path(tail['attempts'][-1]['output']);raw=json.load(open(directory/'result.json'));log=(directory/'client.log').read_bytes()
   assert b'OIDN Error:' not in log and b'ErrorDeviceLost' not in log
   episodes=raw['episodes'];assert len(episodes)==10 and episodes==tail['result']['episodes'];early=episodes[:5];late=episodes[5:];h1=h2=raw['manifest_hash']
  else:
   assert head['status']=='complete'
   early,h1=native_episodes(head,0);late,h2=native_episodes(tail,5);episodes=early+late
  reference=refs[tail['task']+'|'+tail['domain']]['manifest'];m=json.load(open(reference['path']))
  assert h1==h2==reference['manifest_hash']==m['manifest_hash']
  assert len(episodes)==10 and len({e['seed'] for e in episodes})==10
  assert [(e['episode'],e['seed'],e['instruction']) for e in episodes]==[(e['episode'],e['seed'],e['instructions']['unseen']) for e in m['entries']]
  rows.append(dict(task=tail['task'],domain=tail['domain'],mode=tail['kind'],episodes=episodes,successes=sum(e['success'] for e in episodes),manifest_hash=h1,first5_successes=sum(e['success'] for e in early),last5_successes=sum(e['success'] for e in late),seed_instruction_integrity_passed=True,render_log_clean=True,repaired_full10=bool(tail.get('repair_full10'))))
 full={(r['task'],r['domain']):r for r in rows if r['mode']=='full'};summary=[]
 for mode in spec['interventions']:
  for domain in ['all','demo_clean','demo_randomized']:
   subset=[r for r in rows if r['mode']==mode and (domain=='all' or r['domain']==domain)]
   subset=[r for r in subset if (r['task'],r['domain']) in full]
   if not subset:continue
   n=len(subset)*10;good=sum(r['successes'] for r in subset);base=sum(full[r['task'],r['domain']]['successes'] for r in subset)
   summary.append(dict(mode=mode,domain=domain,complete_conditions=len(subset),episodes=n,successes=good,rate=100*good/n,paired_full_rate=100*base/n,delta_pp=100*(good-base)/n))
 complete=state['status']=='complete' and len(rows)==380
 report=dict(utc=datetime.now(timezone.utc).isoformat(),status='complete' if complete else state['status'],completed_10ep_conditions=len(rows),total_conditions=380,episodes=10*len(rows),target_episodes=3800,summary=summary,rows=rows,scope='19 arms x10 tasks x2 domains x10 fixed-manifest episodes; preserves original first5 then appends second5. Historical3 arms reuse complete original10.')
 tmp=O/'merged10.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(O/'merged10_results.json')
 if complete:
  (O/'final_independent_audit.json').write_text(json.dumps(dict(passed=True,utc=report['utc'],conditions=380,episodes=3800,arms=19,episodes_per_condition=10,all_seed_instruction_manifests_exact=True,all_included_logs_render_error_free=True,all_results_native_returncode0_or_audited_original10=True),indent=2)+'\n');break
 if state['status']=='finished_with_errors':break
 time.sleep(20)
