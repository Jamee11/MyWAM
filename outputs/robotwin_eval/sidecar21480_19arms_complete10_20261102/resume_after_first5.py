from pathlib import Path
from datetime import datetime,timezone
import sys,json,time,subprocess
R=Path('/remote-home/jinminghao/OpenWAM');O=Path(__file__).resolve().parent;E=O.parent;A=E/'sidecar21480_expanded_ablations_20261102';B=E/'sidecar21480_all_off_20261102';sys.path.insert(0,str(R))
def status(stage,**kw):
 p=O/'queue_status.tmp';p.write_text(json.dumps(dict(status=stage,utc=datetime.now(timezone.utc).isoformat(),**kw),indent=2)+'\n');p.replace(O/'queue_status.json')
from render_integrity import failed_renderer
status('waiting_for_first5_to_finish',target_arms=19,target_episodes=3800)
while True:
 sa=json.load(open(A/'state.json'));sb=json.load(open(B/'state.json'))
 ja=[j for j in sa['jobs'] if j['kind']!='manifest'];jb=[j for j in sb['jobs'] if j['kind']!='manifest']
 if all(j['status']=='complete' or failed_renderer(j) for j in ja+jb):break
 qfile=O/'quarantined_conditions.json';nq=json.load(open(qfile))['quarantined_conditions'] if qfile.exists() else 0
 status('waiting_for_first5_to_finish',quarantined_completed_conditions=nq,first5_completed_conditions=sum(j['status']=='complete' for j in ja+jb),first5_total_conditions=380,first5_active=sum(j['status']=='running' for j in ja+jb),target_arms=19,target_episodes=3800,new_tail_episodes_to_run=1600)
 time.sleep(30)
from render_integrity import scan_completed,condition_key
render_audit=scan_completed();quarantine={r['key'] for r in render_audit['findings']}
legacy_repairs=sum(r['kind'] in ('full','write_off','relation_reset') for r in render_audit['findings'])
new_tail_episodes=1600+5*len(quarantine)+5*legacy_repairs
(O/'render_repair_plan.json').write_text(json.dumps(dict(quarantined_conditions=len(quarantine),legacy_conditions_to_replace_full10=legacy_repairs,new_tail_episodes_to_run=new_tail_episodes,reused_clean_legacy_tail_episodes=300-5*legacy_repairs,final_valid_episodes_target=3800,findings=render_audit['findings']),indent=2)+'\n')
# First5 scene/language remains auditable; flagged results will be replaced by full10 reruns.
reference=json.load(open(O/'sealed_reference.json'));checks=[]
for j in ja+jb:
 if failed_renderer(j):
  assert condition_key(j) in quarantine
  checks.append(dict(task=j['task'],domain=j['domain'],arm=j['kind'],passed=None,deferred_full10_repair=True,render_clean=False));continue
 ref=reference[j['task']+'|'+j['domain']]['manifest'];m=json.load(open(ref['path']))
 if j.get('imported_reference'):
  raw=json.load(open(j['result']['source_result']));episodes=raw['episodes'][:5]
 else:
  assert j['attempts'][-1]['returncode']==0
  raw=json.load(open(Path(j['attempts'][-1]['output'])/'result.json'));episodes=raw['episodes']
 assert raw['manifest_hash']==ref['manifest_hash']==m['manifest_hash']
 assert [(x['episode'],x['seed'],x['instruction']) for x in episodes]==[(x['episode'],x['seed'],x['instructions']['unseen']) for x in m['entries'][:5]]
 assert episodes==j['result']['episodes'] and len(episodes)==5
 checks.append(dict(task=j['task'],domain=j['domain'],arm=j['kind'],passed=True,render_clean=condition_key(j) not in quarantine))
assert len(checks)==380
(O/'first5_independent_audit.json').write_text(json.dumps(dict(passed=not quarantine,observed_seed_instruction_passed=True,conditions=380,episodes=sum(j['status']=='complete' for j in ja+jb)*5,quarantined_conditions=len(quarantine),repair_policy='Rerun full10 for flagged conditions, exclude original prefix',checks=checks),indent=2)+'\n')
# Reuse already validated idle models/endpoints; no reload or configuration change.
from benchmarks.utils.transport import WSPolicyClient
portmap={'shared18':json.load(open(A/'available_ports.json')),'sidecar_off':json.load(open(B/'available_ports.json'))}
for ports in portmap.values():
 for port in ports:
  with WSPolicyClient(f'ws://127.0.0.1:{port}',timeout=60) as c:assert c.ping().get('readiness_token');c.reset()
(O/'available_ports.json').write_text(json.dumps(portmap)+'\n');(O/'deployment_acceptance.json').write_text(json.dumps(dict(passed=True,reuses_validated_models=True,idle_ports=sum(len(v) for v in portmap.values()),first5_seed_instruction_audit_passed=True,quarantined_conditions_for_full10_repair=len(quarantine)))+'\n')
status('starting_second5_and_render_repairs',new_tail_episodes_to_run=new_tail_episodes,quarantined_conditions_for_full10_repair=len(quarantine))
cmd=f'{R}/.conda/robotwin-official/bin/python {O}/start_remote.py'
r=subprocess.run(['ssh','-n','-o','BatchMode=yes','-p','26199','root@210.45.70.21',cmd],capture_output=True,text=True);assert r.returncode==0,r.stderr
c=subprocess.Popen([str(R/'.venv/bin/python'),'-u',str(O/'merge_report.py')],cwd=R,stdin=subprocess.DEVNULL,stdout=(O/'merge_report.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
(O/'merge_report_process.json').write_text(json.dumps(dict(pid=c.pid,start_ticks=Path(f'/proc/{c.pid}/stat').read_text().rsplit(')',1)[1].split()[19]))+'\n')
status('running_second5_and_render_repairs',new_tail_episodes_to_run=new_tail_episodes,quarantined_conditions_for_full10_repair=len(quarantine),remote=r.stdout)
