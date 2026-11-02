"""Recover failed render workers promptly; leave controllers and valid workers running."""
from pathlib import Path
from datetime import datetime,timezone
import json,signal,os,time
from render_integrity import O,ROOTS,MARKERS,scan_completed
actions=json.load(open(O/'render_guard_actions.json')) if (O/'render_guard_actions.json').exists() else []
signaled=set();pending={}
while True:
 for ident,when in list(pending.items()):
  if time.time()-when<15:continue
  pid,ticks=ident;d=Path('/proc',str(pid))
  try:
   fields=(d/'stat').read_text().rsplit(')',1)[1].split()
   if fields[19]==ticks and fields[0]!='Z' and os.getpgid(pid)==pid:
    os.killpg(pid,signal.SIGKILL);actions.append(dict(utc=datetime.now(timezone.utc).isoformat(),pid=pid,start_ticks=ticks,action='Escalated owned rendering-error worker to SIGKILL after SIGTERM grace'))
  except (FileNotFoundError,ProcessLookupError):pass
  del pending[ident]
 for root in ROOTS+[O]:
  if not (root/'state.json').exists():continue
  state=json.load(open(root/'state.json'))
  for j in state['jobs']:
   if j['status']!='running' or not j.get('attempts'):continue
   a=j['attempts'][-1];ident=(a['pid'],a['start_ticks'])
   if ident in signaled:continue
   log=Path(a['output'])/'client.log'
   if not log.exists():continue
   data=log.read_bytes()
   if not any(m in data for m in MARKERS):continue
   d=Path('/proc',str(a['pid']))
   try:
    if (d/'stat').read_text().rsplit(')',1)[1].split()[19]!=a['start_ticks']:continue
    argv=[x.decode() for x in (d/'cmdline').read_bytes().split(b'\0') if x]
    assert str(root/'run_manifest.py') in argv and os.getpgid(a['pid'])==a['pid']
    os.killpg(a['pid'],signal.SIGTERM);signaled.add(ident);pending[ident]=time.time()
    actions.append(dict(utc=datetime.now(timezone.utc).isoformat(),root=str(root),task=j['task'],domain=j['domain'],kind=j['kind'],pid=a['pid'],start_ticks=a['start_ticks'],log=str(log),action='SIGTERM own worker with explicit rendering error; native scheduler requeues; partial excluded'))
   except ProcessLookupError:pass
 scan_completed()
 p=O/'render_guard_actions.tmp';p.write_text(json.dumps(actions,indent=2)+'\n');p.replace(O/'render_guard_actions.json')
 time.sleep(30)
