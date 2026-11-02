from pathlib import Path
import json,subprocess
O=Path(__file__).resolve().parent;R=O.parents[2];rows=[]
for name in ['schedule.py','report.py']:
 c=subprocess.Popen([str(R/'.conda/robotwin-official/bin/python'),'-u',str(O/name)],cwd=R,stdin=subprocess.DEVNULL,stdout=(O/(name+'.log')).open('a'),stderr=subprocess.STDOUT,start_new_session=True)
 rows.append(dict(script=name,pid=c.pid,start_ticks=Path(f'/proc/{c.pid}/stat').read_text().rsplit(')',1)[1].split()[19]))
(O/'remote_processes.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
