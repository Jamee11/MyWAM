"""Record fresh expert trajectories in sealed eval scenes, using the existing v6 oracle."""
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def install_capture(module):
    import cv2
    import h5py
    import numpy as np
    from envs._base_task import Base_Task
    from benchmarks.robotwin import eval_policy_wrapper as wrapper
    from benchmarks.utils.rng_domain import GlobalRngDomain
    spec=importlib.util.spec_from_file_location('diagnostic_v6_oracle',os.environ['DIAGNOSTIC_ORACLE'])
    oracle=importlib.util.module_from_spec(spec);sys.modules[spec.name]=oracle;spec.loader.exec_module(oracle)
    assert oracle.ORACLE_SCHEMA_VERSION==6
    oracle.install_oracle_hook(Base_Task)
    original=module.eval_function_decorator
    module.eval_function_decorator=lambda policy,name:(lambda args:None) if name=='get_model' else original(policy,name)
    manifest=json.loads(Path(os.environ['ROBOTWIN_EPISODE_MANIFEST']).read_text())
    start,count=(int(os.environ[k]) for k in ('ROBOTWIN_EPISODE_START','ROBOTWIN_EPISODE_COUNT'))
    entries=manifest['entries'][start:start+count]

    def collect(task_name,task_env,args,model,st_seed,test_num=100,video_size=None,instruction_type=None):
        output=Path(os.environ['DIAGNOSTIC_EXPERT_OUTPUT']);output.mkdir(parents=True,exist_ok=False)
        results=[]
        for entry in entries:
            frames=[];domain=GlobalRngDomain.from_seed(entry['seed'])
            args.update(eval_mode=True,need_plan=True,save_data=False)
            with domain.activate():
                task_env.setup_demo(now_ep_num=entry['episode'],seed=entry['seed'],is_test=True,**args)
                wrapper._initialize_eval_task_state(task_name,task_env)
                task_env.set_instruction(instruction=entry['instructions']['unseen'])
                wrapper._canonicalize_torch_cuda(entry['seed'])
            original_picture=task_env._take_picture
            def picture():
                observation=task_env.get_obs()
                encoded={}
                for camera in ('head_camera','left_camera','right_camera'):
                    rgb=observation['observation'][camera]['rgb']
                    assert rgb.shape==(240,320,3)
                    ok,value=cv2.imencode('.jpg',rgb)
                    assert ok
                    encoded[camera]=value
                frames.append(dict(images=encoded,endpose=observation['endpose'],oracle=observation['ea_rsig_oracle_v3']))
            task_env._take_picture=picture
            try:
                with domain.activate():
                    task_env.play_once()
                    success=bool(task_env.plan_success and task_env.check_success())
                if not success or len(frames)<33:
                    raise RuntimeError(f'Expert failed or insufficient frames: {task_name} seed={entry["seed"]} frames={len(frames)}')
                file_path=output/f'episode{entry["episode"]}.hdf5'
                with h5py.File(file_path,'w') as file:
                    file.attrs.update(oracle_contract=oracle.ORACLE_CONTRACT,seed=entry['seed'],
                        manifest_hash=manifest['manifest_hash'],generation='fresh_expert_plan_in_sealed_eval_scene',
                        rgb_contract='legacy_cv2_imencode_RGB_no_decoder_swap',save_freq=task_env.save_freq)
                    for camera in ('head_camera','left_camera','right_camera'):
                        images=file.create_dataset(f'observation/{camera}/rgb',(len(frames),),dtype=h5py.vlen_dtype(np.dtype('uint8')))
                        for i,frame in enumerate(frames):images[i]=frame['images'][camera]
                    for key in frames[0]['endpose']:
                        file.create_dataset(f'endpose/{key}',data=np.asarray([f['endpose'][key] for f in frames]))
                    for key in frames[0]['oracle']:
                        value=np.asarray([f['oracle'][key] for f in frames]);assert np.isfinite(value).all()
                        file.create_dataset(f'ea_rsig_oracle_v3/{key}',data=value)
                results.append(dict(task=task_name,domain=args['task_config'],episode=entry['episode'],seed=entry['seed'],
                    instruction=entry['instructions']['unseen'],frames=len(frames),path=str(file_path),success=success))
                print('EXPERT_GT',results[-1],flush=True)
            finally:
                task_env._take_picture=original_picture
                with domain.activate():task_env.close_env()
        wrapper._write_labtasker_result(dict(operation='capture_expert_gt',rows=results,manifest_hash=manifest['manifest_hash']))
        return entries[-1]['seed']+1,len(results)
    module.eval_policy=collect


if __name__=='__main__':
    from benchmarks.robotwin import eval_policy_wrapper as wrapper
    wrapper._install_eval=install_capture
    raise SystemExit(wrapper.main())
