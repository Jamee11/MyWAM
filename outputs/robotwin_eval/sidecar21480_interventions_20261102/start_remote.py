from pathlib import Path
import subprocess,json
R=Path('/remote-home/jinminghao/OpenWAM');O=Path(__file__).resolve().parent
for script in ['schedule.py','report.py']:
 c=subprocess.Popen([str(R/'.conda/robotwin-official/bin/python'),'-u',str(O/script)],cwd=R,stdin=subprocess.DEVNULL,stdout=(O/(script+'.log')).open('a'),stderr=subprocess.STDOUT,start_new_session=True)
 (O/(script+'.process.json')).write_text(json.dumps(dict(pid=c.pid,start_ticks=Path('/proc',str(c.pid),'stat').read_text().rsplit(')',1)[1].split()[19]))+'\n')
 print(script,c.pid,flush=True)
