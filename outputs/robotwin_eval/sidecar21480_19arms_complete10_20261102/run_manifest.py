"""Native evaluator with process-local CUDA/render affinity; no task/protocol changes."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path('/remote-home/jinminghao/OpenWAM')
sys.path[:0]=[str(ROOT),str(ROOT/'outputs/robotwin_eval/performance_recheck_20261028T0900Z')]
sys.path.insert(0,str(ROOT/'benchmarks/robotwin'))
from benchmarks.robotwin import eval_policy_wrapper as wrapper

p=argparse.ArgumentParser()
p.add_argument('--policy',required=True)
p.add_argument('--task',required=True)
p.add_argument('--mode',required=True)
p.add_argument('--expected-pci',required=True)
p.add_argument('--evidence',required=True)
p.add_argument('--operation',choices=['build_manifest','run_eval'],required=True)
p.add_argument('--result-file',required=True)
p.add_argument('--progress-file',required=True)
p.add_argument('--manifest-file')
p.add_argument('--manifest-hash')
p.add_argument('--episodes',type=int,default=5)
p.add_argument('--episode-start',type=int,default=5)
args=p.parse_args()
os.chdir(wrapper._prepare_runtime_root(os.environ['ROBOTWIN_PATH']))
wrapper._prewarm_cuda_for_curobo()
import torch
assert torch.cuda.is_available()
import sapien
from sapien.wrapper.scene import Scene

def pci_tail(s):return ':'.join(str(s).lower().split(':')[-2:])
device=sapien.Device('cuda:0')
assert pci_tail(device.pci_string)==pci_tail(args.expected_pci), (device.pci_string,args.expected_pci)
sapien.render.set_global_config(max_num_materials=50000,max_num_textures=50000)
pin=sapien.render.RenderSystem(device=device)
original=Scene.__init__

def scene_init(self,systems=None):
    if systems is None:
        systems=[sapien.physx.PhysxCpuSystem(),sapien.render.RenderSystem(device=device)]
    original(self,systems)
    assert pci_tail(self.render_system.device.pci_string)==pci_tail(args.expected_pci)

Scene.__init__=scene_init
evidence=dict(cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],cuda_name=torch.cuda.get_device_name(0),
    expected_pci=args.expected_pci,render_pci=device.pci_string,render_name=device.name,
    protocol='native scene CPU physics; explicit GPU affinity only; upstream RT/cameras/success/budgets unchanged')
Path(args.evidence).write_text(json.dumps(evidence,indent=2)+'\n')
print('RENDER_AFFINITY',json.dumps(evidence),flush=True)
from chunk_transport import install_client
install_client()
from skip_chunk_capture import install_production
install_production(Path(args.evidence).parent)
import observe_queue as observe
observe.install(Path(args.evidence).parent)
if os.environ.get('PARITY_SPLIT_URL'):
    import hashlib,numpy as np
    from benchmarks.utils.transport import WSPolicyClient
    mirror=WSPolicyClient(os.environ['PARITY_SPLIT_URL'],timeout=300)
    original_predict_once=WSPolicyClient.predict_once
    original_reset=WSPolicyClient.reset
    parity_path=Path(args.evidence).parent/'paired_action_chunks.jsonl'
    def paired_predict(self,payload):
        primary=original_predict_once(self,payload)
        reference=original_predict_once(mirror,payload)
        a=np.asarray(primary['executable_action_chunk_v1']['actions'])
        b=np.asarray(reference['executable_action_chunk_v1']['actions'])
        same=bool(np.array_equal(a,b));diff=float(np.max(np.abs(a-b)))
        record=dict(input_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),shape=list(a.shape),exact_equal=same,max_abs_error=diff)
        with parity_path.open('a') as stream:stream.write(json.dumps(record)+'\n')
        if not same:
            np.savez(Path(args.evidence).parent/'different_actions.npz',single=a,split=b)
            (Path(args.evidence).parent/'different_input.json').write_text(json.dumps(payload))
        return primary
    def paired_reset(self):
        primary=original_reset(self);original_reset(mirror);return primary
    WSPolicyClient.predict_once=paired_predict
    WSPolicyClient.reset=paired_reset
from benchmarks.utils.transport import WSPolicyClient
_predict=WSPolicyClient.predict_once
def mode_predict(self,payload):
 mode=os.environ['SIDECAR_INTERVENTION']
 return _predict(self,payload if mode=='sidecar_off' else dict(payload,sidecar_intervention=mode))
WSPolicyClient.predict_once=mode_predict
command=['labtasker','--operation',args.operation,'--task',args.task,'--mode',args.mode,'--policy-config',args.policy,'--result-file',args.result_file,'--progress-file',args.progress_file,'--total-episodes',str(args.episode_start+args.episodes)]
if args.operation=='run_eval':
 import yaml
 policy=yaml.safe_load(Path(args.policy).read_text())
 command+=['--host',policy['host'],'--port',str(policy['port']),'--episode-start',str(args.episode_start),'--num-episodes',str(args.episodes),'--manifest-file',args.manifest_file,'--manifest-hash',args.manifest_hash,'--instruction-type','unseen']
raise SystemExit(wrapper.main(command))
