from pathlib import Path
from datetime import datetime,timezone
import json,time
O=Path(__file__).resolve().parent
while True:
 if not (O/'state.json').exists():time.sleep(5);continue
 s=json.load(open(O/'state.json'));jobs=[j for j in s['jobs'] if j['kind']!='manifest'];done=[j for j in jobs if j['status']=='complete']
 d=dict(utc=datetime.now(timezone.utc).isoformat(),status=s['status'],completed_conditions=len(done),total_conditions=380,completed_native_episodes=sum(len(j['result']['episodes']) for j in done),active=sum(j['status']=='running' for j in jobs),full10_repairs=sum(bool(j.get('repair_full10')) for j in jobs),scope='Tail uses5 episodes except flagged conditions rerun full10. Use merged10_results.json for uniformly10-episode paired success rates.')
 f=O/'comparison.tmp';f.write_text(json.dumps(d,indent=2)+'\n');f.replace(O/'comparison.json')
 if s['status'] in ('complete','finished_with_errors'):break
 time.sleep(20)
