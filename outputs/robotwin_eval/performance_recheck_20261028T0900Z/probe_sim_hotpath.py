"""Fixed-action replay profiling with unchanged official physics and camera capture.

This diagnostic stops after the recorded action prefix; it is not formal SR.
One native process reuses the same official task/seed for all timing variants.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time

ROOT=Path('/remote-home/jinminghao/OpenWAM');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks/robotwin'),str(HERE)]
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--cases',default='baseline,parallel_validation,parallel,lazy_rgb,combined,baseline_repeat')
p.add_argument('--profile',action='store_true')
options,rest=p.parse_known_args();sys.argv=[sys.argv[0],*rest]
directory=Path(rest[rest.index('--evidence')+1]).parent
import numpy as np
actions=np.load(HERE/'native_pilots_paired/legacy_direct/episode0_actions.npy')
description=json.loads((HERE/'native_pilots_paired/legacy_direct/pilot_result.json').read_text())['episodes'][0]['generations'][0]['instruction']
assert actions.shape==(62,20) and os.environ['ROBOTWIN_TEST_NUM']=='30'
ctx=dict(active=False,case='startup',stats={},rows=[],model=None,env=None,parallel_checks=[])
results=[];guard=threading.Lock()

class CaseDone(BaseException):pass

def timed(owner,name,event):
    original=getattr(owner,name)
    def call(*a,**kw):
        begin=time.perf_counter()
        try:return original(*a,**kw)
        finally:
            if ctx['active']:
                seconds=time.perf_counter()-begin
                with guard:
                    row=ctx['stats'].setdefault(event,dict(calls=0,seconds=0.0,max_seconds=0.0))
                    row['calls']+=1;row['seconds']+=seconds;row['max_seconds']=max(row['max_seconds'],seconds)
    setattr(owner,name,call)

import openwam2robotwin_interface as interface
def initialize(self,**kw):
    self._action_type=kw.get('action_type','ee');self._send_state=True
    self._source_quaternion_order=kw.get('source_quaternion_order','wxyz')
    self._step=0;self._episode=-1;self._task_description='';ctx['model']=self
def reset(self,task_description=''):
    self._step=0;self._episode+=1;self._task_description=description
    ctx['env'].set_instruction(instruction=description)
    ctx.update(active=True,stats={},rows=[],generations=[],begin=time.perf_counter())
    if options.profile:
        import cProfile
        ctx['profiler']=cProfile.Profile();ctx['profiler'].enable()
def step(self,example):
    index=self._step;assert index<len(actions)
    if index%32==0:
        assert all(isinstance(v,np.ndarray) for v in example['cams'].values() if v is not None)
        ctx['generations'].append(dict(step=index,
            rgb_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in example['cams'].items() if v is not None},
            state=np.asarray(example['state']).tolist()))
    ctx['rows'].append(np.concatenate([ctx['env'].robot.left_entity.get_qpos(),ctx['env'].robot.right_entity.get_qpos()]))
    self._step+=1
    return actions[index].copy()
interface.ModelClient.__init__=initialize;interface.ModelClient.reset=reset;interface.ModelClient.step=step
evaluate=interface.eval
def eval_replay(env,model,observation):
    result=evaluate(env,model,observation)
    if env.eval_success or model._step>=len(actions):env.close_env()
    return result
interface.eval=eval_replay
from benchmarks.robotwin import eval_policy_wrapper as wrapper
bootstrap=wrapper.bootstrap_robotwin_module

def install(*a,**kw):
    module=bootstrap(*a,**kw)
    from envs._base_task import Base_Task
    from envs.robot.robot import Robot
    from envs.camera.camera import Camera
    from sapien.wrapper.scene import Scene
    from sim_hotpath import install_deferred_rgb,install_parallel_plans
    original_init=Base_Task._init_task_env_;original_close=Base_Task.close_env
    checked_classes=set()
    def init(env,*a,**kw):
        result=original_init(env,*a,**kw);ctx['env']=env
        if type(env) not in checked_classes:
            timed(type(env),'check_success','success_check');checked_classes.add(type(env))
        return result
    def close(env,*a,**kw):
        if not ctx['active']:return original_close(env,*a,**kw)
        if options.profile:
            import pstats
            ctx['profiler'].disable()
            ctx['profiler'].dump_stats(str(directory/(ctx['case']+'.pstats')))
            with (directory/(ctx['case']+'_profile.txt')).open('w') as stream:
                pstats.Stats(ctx['profiler'],stream=stream).sort_stats('cumulative').print_stats(45)
        elapsed=time.perf_counter()-ctx['begin'];ctx['active']=False
        row=dict(case=ctx['case'],steps=int(env.take_action_cnt),success=bool(env.eval_success),seconds=elapsed,
                 stats=ctx['stats'],generation_inputs=ctx['generations'],
                 communication_flag=bool(env.robot.communication_flag),parallel_checks=list(ctx['parallel_checks']))
        np.save(directory/(ctx['case']+'_joint_states.npy'),np.asarray(ctx['rows']))
        results.append(row)
        (directory/'hotpath_results.json').write_text(json.dumps(dict(scope='Fixed62-action prefix replay; no model inference; not formal SR',results=results),indent=2)+'\n')
        original_close(env,*a,**kw)
        print('HOTPATH_CASE',ctx['case'],row['steps'],round(elapsed,3),row['success'],flush=True)
        raise CaseDone
    Base_Task._init_task_env_=init;Base_Task.close_env=close
    timed(Robot,'left_plan_path','left_plan');timed(Robot,'right_plan_path','right_plan')
    pool=install_parallel_plans(ctx,lambda:ctx['active'] and ctx['case'] in ('parallel_validation','parallel','combined'),
                                validate=lambda:ctx['case']=='parallel_validation')
    install_deferred_rgb(lambda:ctx['active'] and ctx['case'] in ('lazy_rgb','combined') and
        ctx['model']._step%32!=0 and not ctx['env'].eval_success and ctx['env'].eval_video_path is None)
    for owner,name,event in [(Base_Task,'get_obs','get_obs'),(Base_Task,'take_action','take_action'),
        (Base_Task,'_update_render','render_update'),(Scene,'step','physics_step'),
        (Camera,'update_picture','camera_capture'),(Camera,'get_rgb','rgb_readback'),(Camera,'get_config','camera_config')]:
        timed(owner,name,event)
    original_eval=module.eval_policy
    def experiment(*a,**kw):
        import torch
        from threadpoolctl import threadpool_limits
        original_threads=torch.get_num_threads()
        try:
            for case in options.cases.split(','):
                if case in ('parallel','combined') and not (ctx['parallel_checks'] and all(x['exact'] for x in ctx['parallel_checks'])):
                    print('SKIP_UNVALIDATED_PARALLEL',case,flush=True);continue
                ctx.update(case=case,active=False)
                if case=='parallel_validation':ctx['parallel_checks']=[]
                threads=1 if case.startswith('threads1') else original_threads
                torch.set_num_threads(threads)
                with threadpool_limits(limits=threads):
                    try:original_eval(*a,**kw)
                    except CaseDone:continue
                raise RuntimeError('Replay did not stop at its diagnostic boundary')
        except BaseException as error:
            if not isinstance(error,SystemExit):
                (directory/'probe_error.json').write_text(json.dumps(dict(type=type(error).__name__,message=str(error),case=ctx['case'])))
            raise
        finally:
            pool.shutdown(wait=True);torch.set_num_threads(original_threads)
        raise SystemExit(0)
    module.eval_policy=experiment
    return module
wrapper.bootstrap_robotwin_module=install
runpy.run_path(str(ROOT/'outputs/robotwin_eval/full_backward32000_4090_all100_30ep_20261025T073709Z/run_native.py'),run_name='__main__')
