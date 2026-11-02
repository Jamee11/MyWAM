"""Native evaluator with process-local CUDA/render affinity; no task/protocol changes."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path('/remote-home/jinminghao/OpenWAM')
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'benchmarks/robotwin'))
from benchmarks.robotwin import eval_policy_wrapper as wrapper

p=argparse.ArgumentParser()
p.add_argument('--policy',required=True)
p.add_argument('--task',required=True)
p.add_argument('--mode',required=True)
p.add_argument('--expected-pci',required=True)
p.add_argument('--evidence',required=True)
p.add_argument('--seed-start',type=int,required=True)
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
# Process-local seed continuation only: retain upstream expert loop and all action/success logic.
from continuation_adapter import install as _install_continuation
_original_bootstrap=wrapper.bootstrap_robotwin_module
def _continuation_bootstrap(*args_,**kwargs_):
    module=_original_bootstrap(*args_,**kwargs_)
    _install_continuation(module,args.seed_start)
    Path(args.evidence).with_name('continuation_seed.json').write_text(json.dumps(dict(seed_group=0,start_seed=args.seed_start,requested_episodes=70,task=args.task,mode=args.mode))+'\n')
    return module
wrapper.bootstrap_robotwin_module=_continuation_bootstrap

raise SystemExit(wrapper.main(['--config',args.policy,'--overrides','--task_name',args.task,
    '--task_config',args.mode,'--ckpt_setting','fullbackward32000_supplement70','--seed','0',
    '--policy_name','openwam2robotwin_interface']))
