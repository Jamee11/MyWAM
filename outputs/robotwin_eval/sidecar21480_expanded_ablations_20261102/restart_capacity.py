from pathlib import Path
from datetime import datetime,timezone
import json,signal,os,time,subprocess
O=Path(__file__).resolve().parent
rows=json.load(open(O/'remote_processes.json'))
def valid(r):
 try:return Path(f"/proc/{r['pid']}/stat").read_text().rsplit(')',1)[1].split()[19]==r['start_ticks']
 except FileNotFoundError:return False
for r in rows:
 if valid(r):os.kill(r['pid'],signal.SIGSTOP)
s=json.load(open(O/'state.json'));(O/f'state_before_capacity_expansion_{time.time_ns()}.json').write_text(json.dumps(s,indent=2)+'\n')
children=[]
for d in Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:
  argv=[a.decode() for a in (d/'cmdline').read_bytes().split(b'\0') if a]
  if str(O/'run_manifest.py') in argv:
   children.append(dict(pid=int(d.name),start_ticks=(d/'stat').read_text().rsplit(')',1)[1].split()[19]))
 except (OSError,ValueError):pass
for r in children:
 if valid(r):
  assert str(O) in Path(f"/proc/{r['pid']}/cmdline").read_bytes().decode();os.killpg(r['pid'],signal.SIGTERM)
time.sleep(3)
for r in children:
 if valid(r):
  try:os.killpg(r['pid'],signal.SIGKILL)
  except ProcessLookupError:pass
for r in rows:
 if valid(r):os.kill(r['pid'],signal.SIGKILL)
for j in s['jobs']:
 if j['status']=='running':
  j['status']='queued'
  if j['attempts']:j['attempts'][-1].update(valid=False,error='User-requested capacity expansion; startup attempt preserved and excluded',ended_wall=time.time(),interrupted_for_capacity=True)
s['status']='paused_for_capacity_expansion';(O/'state.json').write_text(json.dumps(s,indent=2)+'\n');(O/'capacity_restart_ready.json').write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),preserved_startup_attempts=len(children),admission_mib=4608,max_per_gpu=6))+'\n')
subprocess.run([str(O.parents[2]/'.conda/robotwin-official/bin/python'),str(O/'start_remote.py')],check=True)
