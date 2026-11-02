from pathlib import Path
import json,os,signal,subprocess
O=Path(__file__).resolve().parent;R=O.parents[2];PY=R/'.conda/robotwin-official/bin/python'
def retire(row,script):
 d=Path('/proc',str(row['pid']))
 try:
  if (d/'stat').read_text().rsplit(')',1)[1].split()[19]!=row['start_ticks']:return
  args=[a.decode() for a in (d/'cmdline').read_bytes().split(b'\0') if a];assert str(script) in args
  os.kill(row['pid'],signal.SIGTERM)
 except FileNotFoundError:pass

def spawn(script):
 c=subprocess.Popen([str(PY),'-u',str(script)],cwd=R,stdin=subprocess.DEVNULL,stdout=(script.parent/(script.name+'.log')).open('a'),stderr=subprocess.STDOUT,start_new_session=True)
 return dict(script=script.name,pid=c.pid,start_ticks=Path('/proc',str(c.pid),'stat').read_text().rsplit(')',1)[1].split()[19])
for name in ['sidecar21480_expanded_ablations_20261102','sidecar21480_all_off_20261102']:
 root=O.parent/name;f=root/'remote_processes.json';rows=json.load(open(f))
 for row in rows:
  if row['script']=='report.py':retire(row,root/'report.py')
 rows=[r for r in rows if r['script']!='report.py']+[spawn(root/'report.py')];f.write_text(json.dumps(rows,indent=2)+'\n')
f=O/'render_guard_process.json';old=json.load(open(f));retire(old,O/'render_guard.py');row=spawn(O/'render_guard.py');f.write_text(json.dumps(row)+'\n');print('Restarted reporters and render guard only; simulators/models/controllers retained',row)
