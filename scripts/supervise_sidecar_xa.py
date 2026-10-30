#!/usr/bin/env python3
"""Validate ten real eight-GPU updates, then start the authorized 10epoch XA run."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time


def save(path, data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def last_metrics(root):
    result=[]
    for rank in range(8):
        files=list(root.glob(f'*/metrics_rank{rank}.jsonl'))
        if len(files)!=1:return []
        lines=files[0].read_text().splitlines()
        try:row=json.loads(lines[-1])
        except (IndexError,json.JSONDecodeError):return []
        result.append(row)
    return result


def validate_preflight(rows, updates=10):
    if len(rows)!=8 or any(r['optimizer_step']!=updates for r in rows):
        raise RuntimeError('Missing common eight-rank optimizer-update acceptance')
    for row in rows:
        m=row['metrics']
        if not all(math.isfinite(float(v)) for v in m.values()) or m['loss_xa_raw']<=0 or m['loss_xa']<=0:
            raise RuntimeError('Nonfinite or absent XA/EA/main loss')
        if row['peak_allocated_bytes']>78*1024**3:
            raise RuntimeError('XA preflight has insufficient H100 memory headroom')
    return dict(ranks=8,updates=updates,finite=True,peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),
                metrics=rows[0]['metrics'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--node',default='aigc14')
    p.add_argument('--wait-for-stage',type=Path)
    args=p.parse_args()
    project=args.project.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if socket.gethostname().split('.')[0]!=args.node:raise RuntimeError('Wrong training node')
    lock=socket.socket();lock.bind(('127.0.0.1',29544));lock.listen(1)
    if (out/'production_launch.json').exists():raise RuntimeError('A production run was already launched here')
    state=dict(node=args.node,project=str(project),output=str(out),started=datetime.now(timezone.utc).isoformat())
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OPENWAM_VENV=str(project/'.venv'))
    if args.wait_for_stage:
        import sys
        sys.path.insert(0,str(project))
        from scripts.supervise_sidecar_c2r_two_stage import completed
        while True:
            try:
                predecessor=completed(args.wait_for_stage,10740)
                save(out/'predecessor_completed.json',dict(checkpoint=str(predecessor),
                    note='Launch dependency only; XA still initializes from original warmup2K'))
                break
            except (RuntimeError,FileNotFoundError) as error:
                save(out/'state.json',dict(state,status='waiting_for_c2r_final_checkpoint',detail=str(error)))
                time.sleep(30)

    def wait_capacity():
        while True:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            gpus=[list(map(int,line.split(','))) for line in raw.splitlines()]
            if len(gpus)==8 and all(free>75000 and utilization<5 for _,free,utilization in gpus):return
            save(out/'state.json',dict(state,status='waiting_for_eight_idle_gpus',gpus=gpus));time.sleep(30)

    def run(phase,extra):
        root=out/phase;root.mkdir()
        phase_env=dict(env,OPENWAM_OUTPUT=str(root/'runs'))
        command=['bash',str(project/'scripts/train_sidecar_xa_warmup2k_8gpu.sh'),*extra]
        with (root/'launch.log').open('x') as log:
            proc=subprocess.Popen(command,cwd=project,env=phase_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        info=dict(pid=proc.pid,command=command,output=str(root),started=datetime.now(timezone.utc).isoformat())
        save(root/'launch.json',info)
        if phase=='production':save(out/'production_launch.json',info)
        while proc.poll() is None:
            rows=last_metrics(root/'runs')
            save(out/'state.json',dict(state,status=phase,pid=proc.pid,
                optimizer_step=min((r['optimizer_step'] for r in rows),default=0),
                ranks_reporting=len(rows),last_update=datetime.now(timezone.utc).isoformat()))
            time.sleep(15)
        return proc.returncode,root

    for batch,accum in ((16,2),(8,4)):
        wait_capacity()
        overrides=[f'training.batch_size={batch}',f'training.gradient_accumulation_steps={accum}']
        code,preflight=run(f'preflight_batch{batch}',[*overrides,'training.stop_after_optimizer_updates=10',
                            'training.save_steps=null','training.save_optimizer_steps=null'])
        tail=(preflight/'launch.log').read_text(errors='replace')[-200000:]
        if code and batch==16 and ('CUDA out of memory' in tail or 'OutOfMemoryError' in tail):
            save(out/'memory_fallback.json',dict(from_batch=16,to_batch=8,global_batch=256))
            continue
        if code:raise RuntimeError(f'Preflight process exit={code}; inspect preserved logs')
        acceptance=validate_preflight(last_metrics(preflight/'runs'))
        acceptance.update(batch_size=batch,gradient_accumulation_steps=accum,global_batch=256)
        break
    else:raise RuntimeError('No XA batch profile passed')
    save(out/'preflight_acceptance.json',acceptance)
    wait_capacity()
    code,production=run('production',overrides)
    rows=last_metrics(production/'runs')
    finals=list((production/'runs').glob('*/checkpoint_step_21480.safetensors'))
    if code or len(rows)!=8 or any(r['optimizer_step']!=21480 for r in rows) or len(finals)!=1:
        raise RuntimeError(f'Production incomplete, exit={code}')
    save(out/'complete.json',dict(state,status='complete',optimizer_updates=21480,checkpoint=str(finals[0])))
    save(out/'state.json',dict(state,status='complete',optimizer_step=21480))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        import sys
        if '--output' in sys.argv:
            output=Path(sys.argv[sys.argv.index('--output')+1]);output.mkdir(parents=True,exist_ok=True)
            save(output/'failed.json',dict(error=repr(error),time=datetime.now(timezone.utc).isoformat()))
            save(output/'state.json',dict(status='failed',error=repr(error)))
        raise
