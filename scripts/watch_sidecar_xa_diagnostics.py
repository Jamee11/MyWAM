#!/usr/bin/env python3
"""Fetch XA checkpoints resumably and diagnose on local A100s, using no extra H100."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shlex
import sys
import time


def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def report(directory):
    rows=json.loads((directory/'results.json').read_text())['rows']
    summary=json.loads((directory/'summary.json').read_text())
    assert len(rows)==120
    groups=[]
    for g in summary['groups']:
        rr=[r for r in rows if r['mode']==g['mode'] and r['condition']==g['condition']]
        groups.append(dict(condition=g['condition'],mode=g['mode'],drift_mm=g['xyz_drift_mm']['mean'],
                           gt_mm=sum(r['gt_metrics']['position_difference_mm']['mean'] for r in rr)/len(rr)))
    nominal=[r for r in rows if r['mode']=='full' and r['condition']=='clean']
    result=dict(clean_gt_mm=sum(r['gt_metrics']['position_difference_mm']['mean'] for r in nominal)/len(nominal),groups=groups)
    write(directory/'checkpoint_metrics.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--training',required=True);p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--ssh-control',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--initial',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];args.output.mkdir(parents=True,exist_ok=True)
    if not args.training.startswith('/mnt/100T/lixiangyu/cuigm/bitdistill/OpenWAM/'):
        raise ValueError('Remote training must stay in the project directory')
    ssh=['ssh','-S',str(args.ssh_control),'-o','BatchMode=yes','-o','ProxyCommand=false','lixiangyu@127.0.0.1']
    def remote(command):
        if not args.ssh_control.exists():raise RuntimeError('Reconnect the SSH master; no credentials are stored here')
        return subprocess.check_output([*ssh,command],text=True,stderr=subprocess.PIPE)

    def fetch(step,source,previous):
        destination=args.output/'checkpoints'/f'optimizer_{step:05d}';destination.mkdir(parents=True,exist_ok=True)
        checkpoint=Path(source);expected=remote('sha256sum '+shlex.quote(source)).split()[0]
        if len(expected)!=64:raise RuntimeError('Invalid upstream checkpoint hash')
        rsh=shlex.join(ssh[:-1]);remote_dir='lixiangyu@127.0.0.1:'+str(checkpoint.parent)+'/'
        subprocess.run(['rsync','-a','-e',rsh,'--include=/config.yaml','--include=/normalization_stats.npy',
            '--include=/relational_sidecar_contract.json','--include=/tokenizer/***','--exclude=*',remote_dir,str(destination)+'/'],check=True)
        partial=destination/(checkpoint.name+'.partial');final=destination/checkpoint.name
        if not partial.exists() and not final.exists():
            # A private CoW/copy seed; never hardlink a file that rsync will modify.
            subprocess.run(['cp','--reflink=auto',str(previous),str(partial)],check=True)
            assert (partial.stat().st_dev,partial.stat().st_ino)!=(previous.stat().st_dev,previous.stat().st_ino)
        if not final.exists():
            with (destination/'rsync.log').open('a') as log:
                subprocess.run(['rsync','-az','--inplace','--partial','--ignore-times','-e',rsh,
                    'lixiangyu@127.0.0.1:'+source,str(partial)],stdout=log,stderr=subprocess.STDOUT,check=True)
            h=hashlib.sha256()
            with partial.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
            if h.hexdigest()!=expected:raise RuntimeError('Checkpoint transfer SHA256 mismatch; preserve partial')
            partial.rename(final)
        else:
            h=hashlib.sha256()
            with final.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
            if h.hexdigest()!=expected:raise RuntimeError('Existing local checkpoint hash changed')
        write(destination/'transfer.json',dict(status='complete',sha256=expected,source=source,checkpoint=str(final)))
        return final,expected

    previous_report=args.output/'comparison.json'
    results=json.loads(previous_report.read_text())['checkpoints'] if previous_report.exists() else []
    finished={r['optimizer_updates']:r for r in results}
    previous=args.initial
    for step in (0,5000,10000,15000,20000,21480):
        if step in finished:
            previous=Path(finished[step]['checkpoint'])
            continue
        if step:
            while True:
                code=('import json,glob,os; p='+repr(args.training)+'; '
                      'files=[f for f in glob.glob(p+"/production/runs/*/checkpoint_step_'+str(step)+'.safetensors") '
                      'if os.path.exists(f.replace(".safetensors",".complete.json"))]; '
                      'print(json.dumps(dict(files=files,failed=os.path.exists(p+"/failed.json"))))')
                try:
                    state=json.loads(remote('python3 -c '+shlex.quote(code)))
                    if len(state['files'])==1:
                        write(args.output/'state.json',dict(status='transferring',step=step,remote=state['files'][0]))
                        checkpoint,expected=fetch(step,state['files'][0],previous);previous=checkpoint;break
                    if state['failed']:
                        write(args.output/'stopped.json',dict(reason='training failed before next checkpoint',next_step=step));return
                    write(args.output/'state.json',dict(status='waiting_for_checkpoint',next_step=step))
                except (subprocess.CalledProcessError,RuntimeError) as error:
                    write(args.output/'state.json',dict(status='waiting_for_transfer_or_ssh',step=step,error=str(error)))
                time.sleep(30)
        else:
            checkpoint=args.initial;h=hashlib.sha256()
            with checkpoint.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
            expected=h.hexdigest()
            if expected!='49b6c013eb910b9d0bc3071ec40a38915631d8a54dd6b73b1942d6968383835b':
                raise RuntimeError('Initial checkpoint is not the originally transferred warmup2K')
        from safetensors import safe_open
        with safe_open(str(checkpoint),framework='pt',device='cpu') as f:assert len(f.keys())>2000
        directory=args.output/'diagnostics'/f'optimizer_{step:05d}';directory.mkdir(parents=True)
        for name in ('images','features'):(directory/name).mkdir()
        plan=json.loads((args.fixtures/'plan.json').read_text())
        plan.update(checkpoint=str(checkpoint.parent),checkpoint_name=checkpoint.name,checkpoint_sha256=expected,
                    scope='Same fixed windows across XA optimizer checkpoints, not a matched augmentation-only training control')
        write(directory/'inputs_ready.json',json.loads((args.fixtures/'inputs_ready.json').read_text()))
        while True:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            eligible=[i for i,free,util in (map(int,line.split(',')) for line in raw.splitlines()) if i in (4,5,6,7) and free>40000 and util<10]
            if eligible:gpu=eligible[0];break
            write(args.output/'state.json',dict(status='waiting_for_local_a100',step=step));time.sleep(30)
        plan['gpu']=gpu;write(directory/'plan.json',plan)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),HF_ENDPOINT='https://hf-mirror.com',HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',
            TOKENIZERS_PARALLELISM='false',HDF5_USE_FILE_LOCKING='FALSE')
        env.pop('PYTHONHOME',None);env.pop('PYTHONPATH',None)
        with (directory/'launch.log').open('x') as log:
            proc=subprocess.Popen([sys.executable,'-m','benchmarks.robotwin.sidecar_appearance_diagnostics',str(directory.resolve())],
                                  cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(args.output/'state.json',dict(status='diagnosing',step=step,pid=proc.pid,checkpoint=str(checkpoint)))
            if proc.wait()!=0 or not (directory/'complete.json').exists():raise RuntimeError('Paired checkpoint diagnostic failed')
        results.append(dict(optimizer_updates=step,checkpoint=str(checkpoint),sha256=expected,metrics=report(directory)))
        write(args.output/'comparison.json',dict(scope=plan['scope'],checkpoints=results))
    write(args.output/'complete.json',dict(status='complete',checkpoints=len(results)))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        import sys
        if '--output' in sys.argv:
            out=Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
            write(out/'failed.json',dict(status='failed',error=repr(error)))
            write(out/'state.json',dict(status='failed',error=repr(error)))
        raise
