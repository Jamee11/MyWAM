"""Exercise the real upstream get_obs body without creating GPU scenes."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from skip_chunk_capture import install_observation_gate, MARKER
ROOT=Path('/remote-home/jinminghao/OpenWAM')
tree=ast.parse((ROOT/'.external/robotwin-official-20261018/RoboTwin/envs/_base_task.py').read_text())
base=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Base_Task')
method=next(n for n in base.body if isinstance(n,ast.FunctionDef) and n.name=='get_obs')
namespace={'np':np,'deepcopy':deepcopy}
exec(compile(ast.Module(body=[method],type_ignores=[]),'upstream_get_obs','exec'),namespace)
class Camera:
    def __init__(self):self.captures=0;self.readbacks=0;self.fail=False
    def update_picture(self):self.captures+=1
    def get_config(self):return {key:{} for key in ['head_camera','left_camera','right_camera']}
    def get_rgb(self):
        self.readbacks+=1
        return {key:{'rgb':np.full((2,2,3),17,dtype=np.uint8)} for key in self.get_config()}
class Robot:
    def get_left_gripper_val(self):return .2
    def get_right_gripper_val(self):return .7
    def get_left_arm_jointState(self):return [.1,.2,.3]
    def get_right_arm_jointState(self):return [.4,.5,.6]
class Env:
    get_obs=namespace['get_obs']
    def __init__(self):
        self.cameras=Camera();self.robot=Robot();self.eval_video_path=None
        self.data_type=dict(rgb=True,endpose=True,qpos=True);self.updates=0;self.step=0
    def _update_render(self):self.updates+=1;np.random.rand(9)
    def get_arm_pose(self,side):return np.arange(7)+self.step

def run(skip):
    class Local(Env):pass
    stats=dict(get_obs=0,full_observations=0,skipped_observations=0)
    install_observation_gate(Local,lambda e:skip and e.step%32!=0,stats)
    np.random.seed(2718);env=Local();rows=[]
    for i in range(65):
        env.step=i;o=env.get_obs();rows.append(o)
        if i%32==0:assert isinstance(o['observation']['head_camera']['rgb'],np.ndarray)
        elif skip:assert o[MARKER] and o['observation']['head_camera']['rgb'] is None
    return env,rows,np.random.get_state(),stats
full,fa,fr,fs=run(False);skip,sa,sr,ss=run(True)
assert full.updates==skip.updates==65
assert full.cameras.captures==65 and skip.cameras.captures==3
assert skip.cameras.readbacks==3
assert np.array_equal(fr[1],sr[1]) and fr[2:]==sr[2:]
for a,b in zip(fa,sa):
    assert np.array_equal(a['joint_action']['vector'],b['joint_action']['vector'])
    assert np.array_equal(a['endpose']['left_endpose'],b['endpose']['left_endpose'])
    assert np.array_equal(a['endpose']['right_endpose'],b['endpose']['right_endpose'])
assert 'update_picture' not in skip.cameras.__dict__ and 'get_rgb' not in skip.cameras.__dict__
print(json.dumps(dict(passed=True,scene_updates=65,full_capture=65,skipped_capture=3,rng_exact=True,actual_proprio_exact=True,counts=ss)))

# Exercise production constructor -> bootstrap -> get_obs -> eval wrappers.
import sys,types,tempfile
from collections import deque
from skip_chunk_capture import install_production
interface=types.ModuleType('openwam2robotwin_interface')
class Model:
    def __init__(self):self._local_actions=deque();self._task_description='task';self.generations=0
    def step(self,example):
        if example['lang']!=self._task_description:
            self._task_description=example['lang'];self._local_actions.clear()
        if not self._local_actions:
            assert isinstance(example['cams']['head'],np.ndarray)
            self.generations+=1;self._local_actions.extend(range(31));return 0
        return self._local_actions.popleft()
interface.ModelClient=Model
interface.eval=lambda env,model,observation:model.step(dict(lang=env.get_instruction(),cams={'head':observation['observation']['head_camera']['rgb']}))
sys.modules['openwam2robotwin_interface']=interface
sys.path.insert(0,str(ROOT))
import benchmarks.robotwin
wrapper=types.ModuleType('benchmarks.robotwin.eval_policy_wrapper');wrapper.bootstrap_robotwin_module=lambda:object()
sys.modules[wrapper.__name__]=wrapper;benchmarks.robotwin.eval_policy_wrapper=wrapper
class ProductionEnv(Env):
    def __init__(self):super().__init__();self.eval_success=False;self.instruction='task'
    def get_instruction(self):return self.instruction
module=types.ModuleType('envs._base_task');module.Base_Task=ProductionEnv
sys.modules['envs']=types.ModuleType('envs');sys.modules['envs._base_task']=module
with tempfile.TemporaryDirectory() as output:
    install_production(output);wrapper.bootstrap_robotwin_module();model=interface.ModelClient();env=ProductionEnv()
    for i in range(65):
        env.step=i;observation=env.get_obs();interface.eval(env,model,observation)
    assert model.generations==3 and env.cameras.captures==3 and env.updates==65
    env.instruction='new instruction'
    observation=env.get_obs();assert MARKER not in observation
    interface.eval(env,model,observation)
    assert model.generations==4 and env.cameras.captures==4
    print('PRODUCTION_WRAPPER_PASS buffered32, fresh generation, instruction-change fresh image')
    # Explicitly execute the registered stats write while temp output exists.
    import atexit
    atexit._run_exitfuncs()
