"""After current workers drain, retry terminal infrastructure failures with more VRAM headroom."""
from pathlib import Path
from datetime import datetime,timezone
import json,time,subprocess,os,signal
O=Path(__file__).resolve().parent;R=O.parents[2];PY=R/'.conda/robotwin-official/bin/python'
def status(stage,**kw):
 f=O/'retry_status.tmp';f.write_text(json.dumps(dict(status=stage,utc=datetime.now(timezone.utc).isoformat(),**kw),indent=2)+'\n');f.replace(O/'retry_status.json')
while True:
 s=json.load(open(O/'state.json'));jobs=[j for j in s['jobs'] if j['kind']!='manifest'];failed=[j for j in jobs if j['status']=='failed'];pending=[j for j in jobs if j['status'] in ('running','queued')]
 if not failed:
  if s['status']=='complete':status('complete');break
  status('monitoring',pending=len(pending));time.sleep(15);continue
 if pending:
  status('waiting_for_current_workers_to_drain',terminal_conditions=[dict(task=j['task'],domain=j['domain'],kind=j['kind'],attempts=len(j['attempts'])) for j in failed],pending=len(pending),next_admission_mib=8192);time.sleep(15);continue
 if s['status']!='finished_with_errors':time.sleep(3);continue
 for j in failed:
  assert len(j['attempts'])<80,'Repeated infrastructure failure needs investigation; retain logs'
  text=(Path(j['attempts'][-1]['output'])/'client.log').read_text(errors='replace')
  assert any(x in text for x in ['ErrorDeviceLost','cannot create buffer','out of memory','OIDN Error:']), 'Non-render failure requires diagnosis, not automatic relaunch'
 # Only retire drained CPU controllers/reporters, preserving model and valid results.
 for row in json.load(open(O/'remote_processes.json')):
  d=Path('/proc',str(row['pid']))
  try:
   if (d/'stat').read_text().rsplit(')',1)[1].split()[19]==row['start_ticks']:
    argv=[x.decode() for x in (d/'cmdline').read_bytes().split(b'\0') if x];assert str(O/row['script']) in argv;os.kill(row['pid'],signal.SIGTERM)
  except (FileNotFoundError,ProcessLookupError):pass
 (O/f'state_before_infrastructure_retry_{time.time_ns()}.json').write_text(json.dumps(s,indent=2)+'\n')
 for j in failed:j['status']='queued'
 s['status']='paused_for_infrastructure_retry';(O/'state.json').write_text(json.dumps(s,indent=2)+'\n');(O/'capacity_restart_ready.json').write_text(json.dumps(dict(reason='Renderer buffer/VRAM failures reached initial20 retry limit; preserve successful conditions and logs, retry only failed conditions with8192MiB admission',utc=datetime.now(timezone.utc).isoformat()))+'\n')
 status('relaunching_failed_conditions_only',conditions=len(failed),admission_mib=8192)
 subprocess.run([str(PY),str(O/'start_remote.py')],cwd=R,check=True);time.sleep(5)
