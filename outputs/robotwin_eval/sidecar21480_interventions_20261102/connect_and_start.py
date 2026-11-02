from pathlib import Path
import subprocess,json,time
R=Path('/remote-home/jinminghao/OpenWAM');P=Path(__file__).resolve().parent
ports=[39600]+[39700+i for i in range(4)]+[39710+i for i in range(4)]+[39720+i for i in range(4)]
c=subprocess.Popen(['bash',str(R/'benchmarks/robotwin/remote_reverse_tunnel.sh'),'root@210.45.70.21','26199',*[str(p)+':'+str(p) for p in ports]],cwd=R,stdin=subprocess.DEVNULL,stdout=(P/'tunnel.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
(P/'tunnel_process.json').write_text(json.dumps(dict(pid=c.pid,start_ticks=Path('/proc',str(c.pid),'stat').read_text().rsplit(')',1)[1].split()[19]))+'\n');time.sleep(3);assert c.poll() is None
cmd=['ssh','-n','-o','BatchMode=yes','-p','26199','root@210.45.70.21',str(R/'.conda/robotwin-official/bin/python')+' '+str(P/'start_remote.py')]
r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0,r.stderr;print(r.stdout,flush=True)
cmd=['ssh','-n','-o','BatchMode=yes','-p','26199','root@210.45.70.21',"nohup "+str(R/'.conda/robotwin-official/bin/python')+' -u '+str(R/'outputs/robotwin_eval/xa_single_split_parity_20261102/run_live_parity.py')+' > '+str(R/'outputs/robotwin_eval/xa_single_split_parity_20261102/live_controller.log')+' 2>&1 < /dev/null &']
r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0,r.stderr;print('PARITY_ROLLOUT_AND_INTERVENTION_MANIFEST_QUEUE_STARTED',flush=True)
