"""Paired timing pilot with native30 instruction pool, stopping after N full rollouts.

Diagnostic output only: never a partial formal success-rate result.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT=Path('/remote-home/jinminghao/OpenWAM')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks/robotwin'),str(HERE)]
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--variant',choices=['legacy','chunk'],required=True)
p.add_argument('--pilot-episodes',type=int,default=2)
args,rest=p.parse_known_args();sys.argv=[sys.argv[0],*rest]
assert os.environ['ROBOTWIN_TEST_NUM']=='30'
if args.variant=='chunk':
    from chunk_transport import install_client
    install_client()
from benchmarks.robotwin import eval_policy_wrapper as wrapper
bootstrap=wrapper.bootstrap_robotwin_module
directory=Path(rest[rest.index('--evidence')+1]).parent
started=time.monotonic();context=dict(active=False,rows=[],actions=[],generations=[])

def observed_bootstrap(*a,**kw):
    module=bootstrap(*a,**kw)
    import numpy as np
    import openwam2robotwin_interface as interface
    from envs._base_task import Base_Task
    import observe
    observe.install(directory)
    original_reset=interface.reset_model
    original_step=interface.ModelClient.step
    original_close=Base_Task.close_env
    original_scene_init=Base_Task._init_task_env_
    original_instruction=Base_Task.set_instruction
    replay_path=os.environ.get('PILOT_REPLAY_INSTRUCTIONS')
    replay=json.loads(Path(replay_path).read_text())['episodes'] if replay_path else None
    def scene_init(env,*a,**kw):
        result=original_scene_init(env,*a,**kw)
        env._pilot_scene_seed=int(kw.get('seed',0))
        return result
    def instruction(env,*a,**kw):
        if replay is None:return original_instruction(env,*a,**kw)
        selected=replay[len(context['rows'])]['generations'][0]['instruction']
        return original_instruction(env,instruction=selected)
    def reset(model):
        result=original_reset(model)
        context.update(active=True,actions=[],generations=[],begin=time.monotonic())
        return result
    def step(model,example):
        index=len(context['actions'])
        if index%32==0:
            context['generations'].append(dict(step=index,
                camera_sha256={k:hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k,v in example['cams'].items() if v is not None},
                state=np.asarray(example['state']).tolist(),instruction=str(example['lang'])))
        value=original_step(model,example)
        context['actions'].append(np.asarray(value).copy())
        return value
    def close(env,*a,**kw):
        active=context['active']
        if active:
            index=len(context['rows'])
            actions=np.asarray(context['actions'])
            np.save(directory/f'episode{index}_actions.npy',actions)
            context['rows'].append(dict(episode=index,success=bool(env.eval_success),
                steps=int(env.take_action_cnt),scene_seed=getattr(env,'_pilot_scene_seed',None),
                rollout_seconds=time.monotonic()-context['begin'],
                generations=context['generations'],action_sha256=hashlib.sha256(actions.tobytes()).hexdigest()))
            context['active']=False
        result=original_close(env,*a,**kw)
        if active:
            report=dict(scope='Diagnostic pilot, native test_num30 instruction pool; not complete formal30/100 SR',
                        variant=args.variant,instruction_replay_source=replay_path,
                        episodes=context['rows'],wall_seconds=time.monotonic()-started)
            (directory/'pilot_result.json').write_text(json.dumps(report,indent=2)+'\n')
            if len(context['rows'])>=args.pilot_episodes:
                print('PAIRED_PILOT_COMPLETE',json.dumps({k:v for k,v in report.items() if k!='episodes'}),flush=True)
                raise SystemExit(0)
        return result
    interface.reset_model=reset;interface.ModelClient.step=step;Base_Task.close_env=close
    Base_Task._init_task_env_=scene_init;Base_Task.set_instruction=instruction
    return module
wrapper.bootstrap_robotwin_module=observed_bootstrap
runpy.run_path(str(ROOT/'outputs/robotwin_eval/full_backward32000_4090_all100_30ep_20261025T073709Z/run_native.py'),run_name='__main__')
