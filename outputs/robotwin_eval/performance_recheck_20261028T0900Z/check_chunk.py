"""Real WAMPolicy/executor parity; synthetic deterministic engine, no GPU allocation."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import numpy as np

ROOT=Path('/remote-home/jinminghao/OpenWAM')
sys.path.insert(0,str(ROOT))
# Avoid importing model-loader/architecture packages for an engine-free CPU test.
# Policy and executor source files themselves are the unmodified production files.
package=ModuleType('openwam.deploy')
package.__path__=[str(ROOT/'openwam/deploy')]
sys.modules['openwam.deploy']=package
engine_interface=ModuleType('openwam.deploy.engine')
engine_interface.BaseInferenceEngine=object
sys.modules['openwam.deploy.engine']=engine_interface
from openwam.deploy.policy import WAMPolicy
from openwam.deploy.executors import ExecutionConfig
from chunk_transport import chunk_predict

class Engine:
    architecture=SimpleNamespace(binary_command_dims=(9,19))
    def __init__(self):self.calls=[]
    def generate(self,conditions):
        observation=copy.deepcopy(conditions['observation'])
        self.calls.append(observation)
        values=np.arange(32*20,dtype=np.float32).reshape(32,20)/1000+observation['marker']
        return dict(actions=values)

def make():
    engine=Engine()
    policy=WAMPolicy(engine,None,ExecutionConfig(mode='sync',inference_horizon=32))
    return SimpleNamespace(_policy=policy,_init_policy=lambda:None,
        _obs_preprocessor=SimpleNamespace(preprocess=lambda obs:dict(obs)),
        _request_count=0,_total_latency=0),engine

old,oe=make();new,ne=make();checks=[]
for episode,length in enumerate((97,7,65)):
    old._policy.reset();new._policy.reset();old._request_count=0;new._request_count=0
    local=[];a=[];b=[]
    for step in range(length):
        obs=dict(marker=float(step+episode*100),prompt='same instruction')
        a.append(old._policy.predict_action(obs))
        if not local:
            reply=chunk_predict(new,obs)
            assert reply['step']==step+1
            local=reply['executable_action_chunk_v1']['actions']
        b.append(np.asarray(local.pop(0),dtype=np.float32))
    assert np.array_equal(a,b)
    assert oe.calls==ne.calls
    checks.append(dict(episode=episode,executed_actions=length,exact_equal=True,
                       cumulative_engine_calls=len(oe.calls),engine_observations_equal=True))
assert len(oe.calls)==8
for server in (old,new):server._policy.reset()
old._policy.predict_action(dict(marker=0.0))
try:chunk_predict(old,dict(marker=0.0))
except ValueError:checks.append(dict(mixed_transport_rejected=True))
else:raise AssertionError('Mixed mode accepted')

# Exercise the actual thin ModelClient, including PNG call counts, indices,
# prompt changes in the middle of a chunk and early episode termination.
sys.path.insert(0,str(ROOT/'benchmarks/robotwin'))
import openwam2robotwin_interface as interface
from chunk_transport import install_client,KEY
from types import MethodType
class Wire:
    def __init__(self,*args,**kwargs):
        self.server,self.engine=make();self.requests=0
        self.server._obs_preprocessor=SimpleNamespace(preprocess=lambda obs:{**obs,'marker':float(obs['state'][0])})
    def ping(self):return dict(type='pong',**{KEY:dict(horizon=32)})
    def reset(self):
        self.server._policy.reset();self.server._request_count=0
        return dict(type='reset_ack')
    def predict(self,payload):
        self.requests+=1
        action=self.server._policy.predict_action(self.server._obs_preprocessor.preprocess(payload))
        self.server._request_count+=1
        return dict(action=action.tolist(),step=self.server._request_count)
    def predict_once(self,payload):
        self.requests+=1
        return chunk_predict(self.server,payload)
interface.WSPolicyClient=Wire
interface.ModelClient._wait_until_healthy=lambda self:None
old_step=interface.ModelClient.step
old_reset=interface.ModelClient.reset
old_client=interface.ModelClient(state_dim=20,action_indices=[3,9,19])
old_client.reset=MethodType(old_reset,old_client)
install_client()
new_client=interface.ModelClient(state_dim=20,action_indices=[3,9,19])
encode_calls=[0]
encode=interface.client.encode_numpy_b64
def counted(image):encode_calls[0]+=1;return encode(image)
interface.client.encode_numpy_b64=counted
for episode,length in enumerate((97,7,65)):
    old_client.reset();new_client.reset();left=[];right=[]
    for step in range(length):
        example=dict(cams={k:np.full((8,8,3),step%255,dtype=np.uint8) for k in ('head','left','right')},
                     lang='instruction A' if step<50 else 'instruction B',
                     state=np.full(20,episode*100+step,dtype=np.float32))
        left.append(old_step(old_client,example));right.append(new_client.step(example))
    assert np.array_equal(left,right)
    assert old_client._client.engine.calls==new_client._client.engine.calls
assert old_client._client.requests==169
assert new_client._client.requests==8
assert encode_calls[0]==3*(169+8)
checks.append(dict(thin_client_exact=True,mid_chunk_instruction_reset_equal=True,
                   old_requests=169,chunk_requests=8,png_calls=encode_calls[0],
                   engine_observations_equal=True,action_indices_equal=True))
out=Path(__file__).with_name('chunk_cpu_acceptance.json')
out.write_text(json.dumps(dict(passed=True,scope='Actual WAMPolicy and SyncInferenceExecutor, synthetic CPU engine; not GPU/native rollout acceptance',checks=checks),indent=2)+'\n')
print(out.read_text())
