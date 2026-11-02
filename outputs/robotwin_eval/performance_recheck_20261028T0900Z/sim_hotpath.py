"""Opt-in simulator hot paths. Original physics and planner algorithms are retained."""
import copy
import threading
from concurrent.futures import ThreadPoolExecutor
import time


class DeferredRGB:
    """Resolve the current capture only on demand; refuse expired-frame reads."""
    def __init__(self,camera,owner,frame):
        self.camera=camera;self.owner=owner;self.frame=frame;self.value=None
    def resolve(self):
        if self.value is None:
            if self.owner._deferred_frame!=self.frame:
                raise RuntimeError('Deferred camera frame expired before consumption')
            self.value=(self.camera.get_picture('Color')*255).clip(0,255).astype('uint8')[:,:,:3]
        return self.value
    def __array__(self,dtype=None,copy=None):
        result=self.resolve()
        if dtype is not None:result=result.astype(dtype,copy=False)
        return result.copy() if copy else result
    def __deepcopy__(self,memo):return self
    def __getitem__(self,key):return self.resolve()[key]
    def tobytes(self,*a,**kw):return self.resolve().tobytes(*a,**kw)
    @property
    def __array_interface__(self):return self.resolve().__array_interface__


def install_deferred_rgb(should_defer):
    from envs.camera.camera import Camera
    picture=Camera.update_picture;rgb=Camera.get_rgb
    def update(self,*a,**kw):
        result=picture(self,*a,**kw)
        self._deferred_frame=getattr(self,'_deferred_frame',0)+1
        return result
    def get(self,*a,**kw):
        if not should_defer():return rgb(self,*a,**kw)
        result={}
        if self.collect_wrist_camera:
            result['left_camera']=dict(rgb=DeferredRGB(self.left_camera,self,self._deferred_frame))
            result['right_camera']=dict(rgb=DeferredRGB(self.right_camera,self,self._deferred_frame))
        for camera,name in zip(self.static_camera_list,self.static_camera_name):
            if name!='head_camera' or self.collect_head_camera:
                result[name]=dict(rgb=DeferredRGB(camera,self,self._deferred_frame))
        return result
    Camera.update_picture=update;Camera.get_rgb=get


def install_parallel_plans(context,enabled,validate=False):
    """Schedule independent arms together, at the original left-plan call site.

    This is experimental until the real trajectories pass exact reference checks.
    Both original Robot methods receive the actual current qpos. No future
    action is planned, and the original take_action control loop executes paths.
    """
    import numpy as np
    import torch
    from envs.robot.robot import Robot
    from envs._base_task import Base_Task
    left=Robot.left_plan_path;right=Robot.right_plan_path;take=Base_Task.take_action
    pool=ThreadPoolExecutor(max_workers=2);streams={};pending={}
    def worker(side,method,robot,target,qpos):
        if side not in streams:streams[side]=torch.cuda.Stream()
        with torch.cuda.stream(streams[side]):
            return method(robot,target,last_qpos=qpos)
    def equal(a,b):
        if a.get('status')!=b.get('status'):return False
        return all(np.array_equal(a.get(k),b.get(k)) for k in ('position','velocity'))
    def plan_left(robot,target,*a,**kw):
        if not context.get('inside_action') or not enabled():return left(robot,target,*a,**kw)
        if a or kw:raise ValueError('Parallel eval accepts only the native unconstrained EEF call')
        ql=robot.left_entity.get_qpos().copy();qr=robot.right_entity.get_qpos().copy()
        rt=context['right_target'].copy()
        reference=None
        checking=validate() if callable(validate) else validate
        if checking:
            begin=time.perf_counter();reference=(left(robot,target,last_qpos=ql),right(robot,rt,last_qpos=qr))
            serial=time.perf_counter()-begin
        begin=time.perf_counter()
        fl=pool.submit(worker,'left',left,robot,copy.deepcopy(target),ql)
        fr=pool.submit(worker,'right',right,robot,rt,qr)
        try:values=(fl.result(),fr.result())
        finally:
            # A failed future must not leave the other arm running into reset.
            for f in (fl,fr):
                try:f.result()
                except Exception:pass
        parallel=time.perf_counter()-begin
        if checking:
            exact=equal(reference[0],values[0]) and equal(reference[1],values[1])
            context.setdefault('parallel_checks',[]).append(dict(exact=bool(exact),serial_seconds=serial,parallel_seconds=parallel))
            # Validation replay always executes the original reference paths.
            values=reference
        pending[id(robot)]=values[1]
        return values[0]
    def plan_right(robot,target,*a,**kw):
        if id(robot) in pending:return pending.pop(id(robot))
        return right(robot,target,*a,**kw)
    def action(env,value,action_type='qpos'):
        if action_type!='ee' or not enabled():return take(env,value,action_type=action_type)
        assert len(value)==16 and not pending
        context.update(inside_action=True,right_target=np.asarray(value)[8:15].copy())
        try:return take(env,value,action_type=action_type)
        finally:context['inside_action']=False;pending.clear()
    Robot.left_plan_path=plan_left;Robot.right_plan_path=plan_right;Base_Task.take_action=action
    return pool


def install_formal_acceleration(deferred_rgb=False,parallel_plans=False):
    """Only call after the experiment's gates pass; one native policy per process."""
    import atexit
    import openwam2robotwin_interface as interface
    from envs._base_task import Base_Task
    state=dict(active=False,model=None,env=None)
    initialize=interface.ModelClient.__init__;reset=interface.reset_model
    get_obs=Base_Task.get_obs;close=Base_Task.close_env
    def model_init(self,*a,**kw):
        result=initialize(self,*a,**kw);state['model']=self;return result
    def model_reset(model):
        state['active']=False
        result=reset(model)
        state.update(active=True,model=model)
        return result
    def observation(env,*a,**kw):
        previous=state['env'];state['env']=env
        try:return get_obs(env,*a,**kw)
        finally:state['env']=previous
    def closing(env,*a,**kw):
        state['active']=False
        return close(env,*a,**kw)
    interface.ModelClient.__init__=model_init;interface.reset_model=model_reset
    Base_Task.get_obs=observation;Base_Task.close_env=closing
    def defer():
        env=state['env'];model=state['model']
        return bool(state['active'] and env is not None and not env.eval_success and
                    env.eval_video_path is None and model is not None and getattr(model,'_local_actions',()))
    if deferred_rgb:install_deferred_rgb(defer)
    if parallel_plans:
        pool=install_parallel_plans(state,lambda:state['active'])
        atexit.register(pool.shutdown,wait=True)
