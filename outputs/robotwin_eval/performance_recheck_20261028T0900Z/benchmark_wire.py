"""GPU fixture parity and transport timing, separate from physical rollout throughput."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sys
import threading
import time
import numpy as np

ROOT=Path('/remote-home/jinminghao/OpenWAM')
sys.path.insert(0,str(ROOT))
from benchmarks.utils.transport import WSPolicyClient
KEY='executable_action_chunk_v1'
p=argparse.ArgumentParser()
p.add_argument('--variant',choices=['legacy','chunk'],required=True)
p.add_argument('--ports',type=int,nargs='+',required=True)
p.add_argument('--rounds',type=int,default=4)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--carry-state',action='store_true')
p.add_argument('--vary-buffered-observations',action='store_true')
a=p.parse_args()
dep=ROOT/'outputs/robotwin_eval/fullbackward32000_complete100_deploy_20261028T035728Z'
metadata=json.loads((dep/'deployment.json').read_text())
fixtures=json.loads(Path(metadata['reference_fixtures']).read_text())
canonical=json.loads((dep/'reference_checkpoint_actions.json').read_text())
assert canonical['checkpoint_sha256']==metadata['checkpoint_sha256']
barrier=threading.Barrier(len(a.ports))
def worker(slot):
    records=[]
    with WSPolicyClient(f'ws://127.0.0.1:{a.ports[slot]}',timeout=120) as client:
        if a.carry_state:client.reset()
        for repeat in range(a.rounds):
            index=(slot+repeat)%len(fixtures);payload=copy.deepcopy(fixtures[index])
            if not a.carry_state:client.reset()
            barrier.wait(timeout=120);begin=time.monotonic()
            if a.variant=='legacy':
                actions=[client.predict_once(payload if step==0 or not a.vary_buffered_observations else
                    fixtures[(index+step)%len(fixtures)])['action'] for step in range(32)]
            else:
                result=client.predict_once({**payload,KEY:dict(horizon=32)})
                assert result[KEY]['start_step']==(repeat*32+1 if a.carry_state else 1)
                actions=result[KEY]['actions']
            seconds=time.monotonic()-begin
            actual=np.asarray(actions,dtype=np.float32);expected=np.asarray(canonical['actions'][index],dtype=np.float32)
            equal=np.array_equal(actual,expected)
            if not equal:barrier.abort()
            assert equal and actual.shape==expected.shape
            records.append(dict(slot=slot,repeat=repeat,fixture=index,seconds=seconds,
                                exact_equal=equal,max_abs_error=float(np.max(np.abs(actual-expected)))))
            barrier.wait(timeout=120)
    return records
begin=time.monotonic()
with ThreadPoolExecutor(max_workers=len(a.ports)) as pool:records=list(pool.map(worker,range(len(a.ports))))
report=dict(scope='Real GPU checkpoint/static-fixture transport benchmark; no simulator physics',
            checkpoint_sha256=metadata['checkpoint_sha256'],variant=a.variant,ports=a.ports,
            rounds=a.rounds,wall_seconds=time.monotonic()-begin,passed=True,records=records)
report.update(carry_state=a.carry_state,vary_buffered_observations=a.vary_buffered_observations)
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='records'}))
