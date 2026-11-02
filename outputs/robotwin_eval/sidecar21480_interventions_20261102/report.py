from pathlib import Path
from datetime import datetime,timezone
import json,time
O=Path(__file__).resolve().parent;R=O.parents[2]
while True:
 if not (O/'state.json').exists():time.sleep(5);continue
 s=json.load(open(O/'state.json'));jobs=s['jobs'];keys={(j['task'],j['domain']) for j in jobs};paired=[]
 for task,domain in sorted(keys):
  group={j['kind']:j for j in jobs if j['task']==task and j['domain']==domain and j['kind']!='manifest'}
  if len(group)==3 and all(j['status']=='complete' for j in group.values()):
   paired.append(dict(task=task,domain=domain,manifest_hash=group['full']['result']['manifest_hash'],successes={k:v['result']['successes'] for k,v in group.items()},episodes={k:v['result']['episodes'] for k,v in group.items()}))
 summary={}
 for domain in ['all','demo_clean','demo_randomized']:
  rows=[r for r in paired if domain=='all' or r['domain']==domain]
  if not rows:continue
  entry=dict(conditions=len(rows),episodes=len(rows)*10)
  for mode in ['full','write_off','relation_reset']:
   good=sum(r['successes'][mode] for r in rows);entry[mode]=dict(successes=good,rate=100*good/entry['episodes'])
   if mode!='full':
    worse=better=0
    for r in rows:
     assert [(e['episode'],e['seed'],e['instruction']) for e in r['episodes']['full']]==[(e['episode'],e['seed'],e['instruction']) for e in r['episodes'][mode]]
     for f,a in zip(r['episodes']['full'],r['episodes'][mode]):
      worse+=bool(f['success'] and not a['success']);better+=bool(a['success'] and not f['success'])
    entry[mode].update(delta_vs_full_pp=entry[mode]['rate']-entry['full']['rate'],full_only_success=worse,intervention_only_success=better)
  summary[domain]=entry
 result=dict(utc=datetime.now(timezone.utc).isoformat(),status=s['status'],paired_complete_conditions=len(paired),complete_evaluation_conditions=sum(j['status']=='complete' and j['kind']!='manifest' for j in jobs),active=sum(j['status']=='running' for j in jobs),summary=summary,rows=paired,scope='10 tasks x2 domains x3 inference interventions x10 sealed-manifest paired episodes; no video; original21480 checkpoint. Low-N screening, not whole-suite SR or proof of relation semantics.')
 p=O/'comparison.tmp';p.write_text(json.dumps(result,indent=2)+'\n');p.replace(O/'comparison.json')
 lines=['# 原21480 Sidecar推理干预：配对闭环评估','',f"更新：{result['utc']}；三组齐全条件：{len(paired)}/20；完成组条件：{result['complete_evaluation_conditions']}/60；运行：{result['active']}。",'', '|范围|配对条件|每组条数|Full|Write-Off|Relation-Reset|','|---|---:|---:|---:|---:|---:|']
 for k in ['all','demo_clean','demo_randomized']:
  if k in summary:
   a=summary[k];lines.append(f"|{k}|{a['conditions']}|{a['episodes']}|{a['full']['rate']:.2f}%|{a['write_off']['rate']:.2f}%|{a['relation_reset']['rate']:.2f}%|")
 lines+=['','只比较三组相同seed/语言清单全部完成的条件。Write-Off为去除全部三tap的直接Action写入；Relation-Reset为每tap后关系token索引4:8置零并进入后续tap与Write。不是只打乱head预测，token置零造成的分布变化不能自动解释为已证明语义因果。原21480历史74.83%来自全50任务/30条，不与此选择集直接排名。']
 (R/'docs/ROBOTWIN_21480_INTERVENTIONS_LIVE_20261102.md').write_text('\n'.join(lines)+'\n')
 if s['status']!='running':
  (O/'final_results.json').write_text(json.dumps(result,indent=2)+'\n');break
 time.sleep(20)
