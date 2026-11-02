from pathlib import Path
import json,os,signal,time
O=Path(__file__).resolve().parent;s=json.load(open(O/'state.json'));jobs=[j for j in s['jobs'] if j['status']=='running'];assert len(jobs)==1
j=jobs[0];a=j['attempts'][-1];f=Path('/proc',str(a['pid']),'stat');assert f.read_text().rsplit(')',1)[1].split()[19]==a['start_ticks'];assert str(O/'run_manifest.py').encode() in Path('/proc',str(a['pid']),'cmdline').read_bytes();assert os.getpgid(a['pid'])==a['pid']
(Path(a['output'])/'stalled_recovery.json').write_text(json.dumps(dict(reason='No progress file/episode/action for nearly3hours; preserve log, retry same manifest',observed_wall=time.time()),indent=2)+'\n')
os.killpg(a['pid'],signal.SIGTERM);time.sleep(3)
if f.exists() and f.read_text().rsplit(')',1)[1].split()[19]==a['start_ticks'] and f.read_text().rsplit(')',1)[1].split()[0]!='Z':os.killpg(a['pid'],signal.SIGKILL)
print('STALLED_ATTEMPT_TERMINATED_FOR_RETRY',a['pid'],flush=True)
