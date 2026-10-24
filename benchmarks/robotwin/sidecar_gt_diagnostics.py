"""GT-scored inference diagnostics. No optimizer, training update or production edits."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from benchmarks.robotwin.sidecar_write_diagnostics import WriteProbe, action_metrics, distribution


def binary_counts(probability, target, valid):
    probability, target, valid = np.asarray(probability), np.asarray(target), np.asarray(valid, dtype=bool)
    prediction, truth = probability[valid] >= .5, target[valid] >= .5
    return dict(tp=int((prediction & truth).sum()), fp=int((prediction & ~truth).sum()),
                fn=int((~prediction & truth).sum()), tn=int((~prediction & ~truth).sum()))


def binary_report(counts):
    tp, fp, fn, tn = (counts[k] for k in ('tp', 'fp', 'fn', 'tn'))
    return dict(**counts, n=tp+fp+fn+tn, positives=tp+fn,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                balanced_accuracy=((tp/(tp+fn))+(tn/(tn+fp)))/2 if tp+fn and tn+fp else None,
                accuracy=(tp+tn)/(tp+fp+fn+tn) if tp+fp+fn+tn else None)


def representation_metrics(prediction, target, arms):
    """Use the same 130D masks/local-frame targets as EAHeads.losses."""
    target, arms = np.asarray(target).reshape(130), np.asarray(arms).reshape(2)
    valid = target[:2] * arms
    boxes, box_valid = target[70:118].reshape(4,3,4), target[118:130].reshape(4,3).copy()
    box_valid *= np.repeat(arms,2)[:,None]
    box_valid[1] *= target[14]; box_valid[3] *= target[15]
    h,w = prediction['grid']
    y,x = np.meshgrid((np.arange(h)+.5)/h,(np.arange(w)+.5)/w,indexing='ij')
    attention = np.asarray(prediction['attention']).reshape(4,h,w)
    masses,hits = [],[]
    for role in range(4):
        if not box_valid[role].any():
            continue
        inside = np.zeros((h,w),dtype=bool)
        for view,(ox,oy,sx,sy) in enumerate(((0,0,1,2/3),(0,2/3,.5,1/3),(.5,2/3,.5,1/3))):
            if box_valid[role,view] <= .5:
                continue
            a,b,c,d = boxes[role,view]
            inside |= (x>=ox+sx*a)&(x<=ox+sx*c)&(y>=oy+sy*b)&(y<=oy+sy*d)
        masses.append(float(attention[role][inside].sum()))
        hits.append(bool(inside.flat[attention[role].argmax()]))
    future = target[64:70].reshape(3,2)[:,:,None] * valid[None,:,None]
    future = np.broadcast_to(future,(3,2,2))
    moving = target[52:64].reshape(3,2,2)
    directions = target[16:52].reshape(3,2,2,3)
    cosine = (np.asarray(prediction['plan_directions']).reshape(3,2,2,3)*directions).sum(-1).clip(-1,1)
    sigmoid = lambda value: 1/(1+np.exp(-np.clip(np.asarray(value),-80,80)))
    return dict(ground_mass=masses,ground_hits=hits,
        presence=binary_counts(sigmoid(prediction['presence_logits']),box_valid.any(-1),np.ones(4,bool)),
        track=binary_counts(sigmoid(prediction['track_valid_logits']),target[:2],arms>.5),
        contact=binary_counts(sigmoid(prediction['contact_logits']),target[8:10],(valid*target[10:12])>.5),
        near=binary_counts(sigmoid(prediction['near_logits']),np.linalg.norm(target[2:8].reshape(2,3),axis=-1)<.10,valid>.5),
        goal_errors=np.abs(np.asarray(prediction['goal_proximity']).reshape(2)-target[12:14])[(valid*target[14:16])>.5].tolist(),
        moving=binary_counts(sigmoid(prediction['plan_moving_logits']).reshape(3,2,2),moving,future>.5),
        direction_cosine=cosine[(future*moving)>.5].tolist(),
        plan_by_horizon=[dict(horizon=step,
            moving=binary_counts(sigmoid(prediction['plan_moving_logits']).reshape(3,2,2)[i],moving[i],future[i]>.5),
            direction_cosine=cosine[i][(future[i]*moving[i])>.5].tolist()) for i,step in enumerate((8,16,32))])


def score_actions(prediction, target, normalizer):
    result = action_metrics(prediction,target,normalizer)
    for horizon in (8,32):
        a,b = np.asarray(prediction)[:horizon],np.asarray(target)[:horizon]
        result[f'gripper_mae_first{horizon}'] = float(np.abs(a[:,[9,19]]-b[:,[9,19]]).mean())
        result[f'gripper_counts_first{horizon}'] = binary_counts(a[:,[9,19]],b[:,[9,19]],np.ones((horizon,2),bool))
        result[f'xyz_errors_mm_first{horizon}'] = [np.linalg.norm(a[:,j:j+3]-b[:,j:j+3],axis=-1).tolist() for j in (0,10)]
        result[f'xyz_errors_mm_first{horizon}'] = (1000*np.asarray(result[f'xyz_errors_mm_first{horizon}'])).tolist()
    return result


def prepare(out):
    import cv2
    import h5py
    from openwam.dataloader.ea_rsig_lite import LiteRoboTwinSource
    from benchmarks.robotwin.prompt_template import format_prompt_for_inference
    from benchmarks.utils.client import build_payload, encode_numpy_b64

    root=Path(__file__).resolve().parents[2]
    manifest=json.loads((root/'outputs/sidecar/aloha_4350/manifest.ready.json').read_text())
    rows=[]
    for task in json.loads((out/'plan.json').read_text())['tasks']:
        entry=next(t for t in manifest['tasks'] if t['name']==task)
        for episode in entry['episodes'][:2]:
            path=Path(episode['path']); stat=path.stat()
            assert (stat.st_size,stat.st_mtime_ns)==(episode['size'],episode['mtime_ns'])
            label_path=Path(episode['labels'])
            assert hashlib.sha256(label_path.read_bytes()).hexdigest()==episode['labels_sha256']
            with np.load(label_path) as labels:
                targets,arms=labels['targets'].copy(),labels['arm_present'].copy()
            instructions=entry['startup_metadata']['instructions'][path.stem+'.json']
            prompt=format_prompt_for_inference((instructions.get('unseen') or instructions['seen'])[0])
            with h5py.File(path,'r') as file:
                n=len(file['endpose/left_endpose']);assert n>=35
                for anchor in (0,(n-33)//2):
                    current=LiteRoboTwinSource._read_eef_actions(None,file,anchor,anchor+1)[0]
                    gt=LiteRoboTwinSource._read_eef_actions(None,file,anchor+1,anchor+33)
                    assert gt.shape==(32,20) and targets.shape==(n,130)
                    cameras=[cv2.imdecode(np.frombuffer(bytes(file[f'observation/{camera}/rgb'][anchor]),np.uint8),cv2.IMREAD_COLOR)
                             for camera in ('head_camera','left_camera','right_camera')]
                    assert all(c is not None and c.shape==(240,320,3) for c in cameras)
                    sample=out/'inputs'/f'{task}_{path.stem}_t{anchor}';sample.mkdir(parents=True)
                    payload=build_payload(head=encode_numpy_b64(cameras[0]),left_wrist=encode_numpy_b64(cameras[1]),
                        right_wrist=encode_numpy_b64(cameras[2]),state=current.tolist(),prompt=prompt)
                    (sample/'payload.json').write_text(json.dumps(payload))
                    np.savez_compressed(sample/'gt.npz',actions=gt,targets=targets[anchor],arms=arms[anchor])
                    rows.append(dict(task=task,domain='Clean',episode=path.stem,anchor=anchor,
                        source=str(path),payload=str(sample/'payload.json'),gt=str(sample/'gt.npz')))
    assert len(rows)==12
    (out/'inputs_ready.json').write_text(json.dumps(dict(samples=rows,scope='Training-seen Clean diagnostic, not held-out performance'),indent=2))
    print('INPUTS_READY',len(rows),flush=True)


def summarize(out):
    raw=json.loads((out/'results.json').read_text())
    results=raw['results'];summaries=[]
    for domain in sorted({r['sample']['domain'] for r in results}):
        selected=[r for r in results if r['sample']['domain']==domain]
        reference={(r['sample']['source'],r['sample']['anchor']):r for r in selected if r['scale']==1}
        for scale in sorted({r['scale'] for r in selected}):
            rows=[r for r in selected if r['scale']==scale]
            assert len(rows)==len(reference)
            differences=[r['position_difference_mm']['mean']-reference[(r['sample']['source'],r['sample']['anchor'])]['position_difference_mm']['mean'] for r in rows]
            summaries.append(dict(domain=domain,scale=scale,windows=len(rows),
                xyz_error_mean_mm=float(np.mean([r['position_difference_mm']['mean'] for r in rows])),
                rotation_error_mean_degrees=float(np.mean([r['rotation_difference_degrees']['mean'] for r in rows])),
                gripper_mae=float(np.mean([r['gripper_mae_first32'] for r in rows])),
                paired_xyz_delta_vs_scale1_mm=distribution(differences),xyz_improved_windows=sum(d<0 for d in differences)))
    representations=[]
    for domain in sorted({r['sample']['domain'] for r in raw['representations']}):
        for tap in (7,15,23):
            rows=[r['metrics'] for r in raw['representations'] if r['tap']==tap and r['sample']['domain']==domain]
            metrics={}
            for name in ('presence','track','contact','near','moving'):
                metrics[name]=binary_report({key:sum(r[name][key] for r in rows) for key in ('tp','fp','fn','tn')})
            for name in ('ground_mass','ground_hits','goal_errors','direction_cosine'):
                values=[v for r in rows for v in r[name]]
                metrics[name]=distribution(values) if values else None
            metrics['plan_by_horizon']=[]
            for index,horizon in enumerate((8,16,32)):
                values=[v for r in rows for v in r['plan_by_horizon'][index]['direction_cosine']]
                counts={key:sum(r['plan_by_horizon'][index]['moving'][key] for r in rows) for key in ('tp','fp','fn','tn')}
                metrics['plan_by_horizon'].append(dict(horizon=horizon,moving=binary_report(counts),direction_cosine=distribution(values) if values else None))
            representations.append(dict(domain=domain,tap=tap,windows=len(rows),**metrics))
    strength=[]
    for domain in sorted({r['sample']['domain'] for r in results}):
        for scale in sorted({r['scale'] for r in results if r['sample']['domain']==domain}):
            for tap in (7,15,23):
                values=[v['ratio'] for r in results if r['sample']['domain']==domain and r['scale']==scale
                        for v in r['write_measurements'] if v['tap']==tap]
                strength.append(dict(domain=domain,scale=scale,tap=tap,actual_ratio=distribution(values)))
    report=dict(scope=raw['scope'],action_sweep=summaries,representations=representations,write_strength=strength)
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    return report


def diagnostic_devices(plan):
    import torch
    devices = tuple(torch.device(plan.get(key, default)) for key, default in
                    (('encoder_device', 'cuda:0'), ('model_device', 'cuda:1')))
    if any(device.type != 'cuda' or device.index is None for device in devices):
        raise ValueError('Diagnostic devices must be explicit CUDA devices')
    return devices


def run_loaded_diagnostics(server,out):
    import torch
    from openwam.model.video_backbone.wan.encode import encode_text
    out=Path(out)
    try:
        inputs=json.loads((out/'inputs_ready.json').read_text());rows=inputs['samples']
        plan=json.loads((out/'plan.json').read_text())
        encoder_device,model_device=diagnostic_devices(plan)
        scales=plan.get('scales',[0,.5,1,2,5])
        if 1 not in scales or len(scales)!=len(set(scales)) or any(not np.isfinite(s) or s<0 for s in scales):
            raise ValueError('Diagnostic scales must be unique, finite, nonnegative and include Full=1')
        arch,engine=server.engine.architecture,server.engine
        def capacity(gpu,minimum):
            while True:
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
                free={int(a):int(b) for a,b in (s.split(',') for s in raw.splitlines())}
                if free[gpu]>=minimum:return
                print('WAIT_MEMORY',gpu,free[gpu],minimum,flush=True);time.sleep(5)
        prompts=sorted({json.loads(Path(row['payload']).read_text())['prompt'] for row in rows})
        vb=arch.video_backbone;capacity(plan.get('encoder_gpu',4),14000);torch.cuda.set_device(encoder_device)
        vb.text_encoder.to(device=encoder_device,dtype=arch.dtype)
        encoded={}
        with torch.inference_mode():
            for prompt in prompts:
                encoded[prompt]=tuple(v.cpu() for v in encode_text([prompt],tokenizer=vb._tokenizer,text_encoder=vb.text_encoder,device=encoder_device))
        vb.text_encoder.to('cpu');torch.cuda.empty_cache();capacity(plan.get('model_gpu',7),20000)
        encoder=vb.text_encoder;vb.text_encoder=None
        try:
            torch.cuda.set_device(model_device);arch.set_dtype_device(arch.dtype,model_device)
        finally:vb.text_encoder=encoder
        arch.eval();engine._dit_cache=None
        engine._prompt_embed_cache={k:tuple(v.to(model_device) for v in values) for k,values in encoded.items()}
        def no_missed_prompt(*args,**kwargs):raise RuntimeError('Unexpected uncached diagnostic prompt')
        vb.text_encoder.forward=no_missed_prompt
        server._init_policy()
        if (out/'gradient_32.flag').exists():
            from benchmarks.robotwin.sidecar_gradient32 import run
            run(server,out,rows)
            return
        if (out/'gradient_only.flag').exists():
            import importlib
            import traceback
            from benchmarks.robotwin import sidecar_gradient_probe
            for attempt in range(3):
                try:
                    importlib.reload(sidecar_gradient_probe).run_smoke(server,out,rows)
                    return
                except Exception:
                    (out/f'gradient_attempt{attempt}_failed.txt').write_text(traceback.format_exc())
                    print('GRADIENT_ATTEMPT_FAILED',attempt,traceback.format_exc(),flush=True)
                    deadline=time.monotonic()+300
                    while not (out/f'RETRY_GRADIENT_{attempt+1}').exists():
                        if time.monotonic()>deadline or (out/'STOP').exists():return
                        time.sleep(2)
            return
        probe=WriteProbe(arch)
        saved=[stage.write.beta.detach().clone() for stage in arch.sidecar.stages]
        predictions=[];representations=[];captured=[]
        def capture(module,args,value):
            if len(captured)>=3:return
            entry={key:value[key].detach().float().cpu().numpy()[0] for key in
                ('track_valid_logits','near_logits','contact_logits','goal_proximity','plan_directions','plan_moving_logits')}
            entry.update(grid=list(value['grid']),attention=value['grounding']['attention'].detach().float().cpu().numpy()[0],
                presence_logits=value['grounding']['presence_logits'].detach().float().cpu().numpy()[0])
            captured.append(entry)
        hook=arch.sidecar.heads.register_forward_hook(capture)
        start=time.monotonic()
        with torch.inference_mode():
            for index,row in enumerate(rows):
                payload=json.loads(Path(row['payload']).read_text());obs=server._obs_preprocessor.preprocess(payload)
                conditions=server._policy._build_conditions(obs);conditions['seed']=42
                with np.load(row['gt']) as gt:
                    target,labels,arms=gt['actions'],gt['targets'],gt['arms']
                reference_actions=None
                for scale in scales:
                    captured.clear();probe.rows=[];server.reset()
                    with probe.use('full',scale=scale,collect=True):
                        actions=engine.generate(conditions)['actions']
                    predictions.append(dict(sample=row,scale=scale,**score_actions(actions,target,arch.normalizer),
                        actions=np.asarray(actions).tolist(),write_measurements=list(probe.rows)))
                    if scale==1:
                        reference_actions=np.asarray(actions).copy()
                        if 'reference_full_actions' in row:
                            prior=np.asarray(row['reference_full_actions'])
                            (out/'cross_run_reference_check.json').write_text(json.dumps(dict(
                                exact=bool(np.array_equal(actions,prior)),max_abs_error=float(np.max(np.abs(actions-prior))))))
                        assert len(captured)==3
                        for tap,prediction in zip((7,15,23),captured):
                            representations.append(dict(sample=row,tap=tap,metrics=representation_metrics(prediction,labels,arms)))
                    for stage,beta in zip(arch.sidecar.stages,saved):
                        torch.testing.assert_close(stage.write.beta,beta,rtol=0,atol=0)
                    (out/'progress.json').write_text(json.dumps(dict(status='complete' if len(predictions)==len(scales)*len(rows) else 'running',completed=len(predictions),total=len(scales)*len(rows),
                        elapsed_seconds=time.monotonic()-start,results=predictions,representations=representations)))
                    print('GT_SWEEP',index,row['task'],row['episode'],row['anchor'],scale,
                          predictions[-1]['position_difference_mm']['mean'],flush=True)
                if index==0:
                    server.reset()
                    with probe.use('full'):
                        repeated=engine.generate(conditions)['actions']
                    np.testing.assert_array_equal(repeated,reference_actions)
                    (out/'full_repeat_check.json').write_text(json.dumps(dict(exact=True,scale=1,seed=42)))
        hook.remove()
        result=dict(status='complete',scope=inputs['scope']+'; fixed seed42/10 steps/eager/no-cache; no optimizer updates',
            results=predictions,representations=representations,elapsed_seconds=time.monotonic()-start)
        (out/'results.json').write_text(json.dumps(result,indent=2))
        summarize(out)
        if (out/'run_gradient_smoke.flag').exists():
            try:
                from benchmarks.robotwin.sidecar_gradient_probe import run_smoke
                run_smoke(server,out,rows)
            except Exception as exc:
                (out/'gradient_smoke_failed.json').write_text(json.dumps(dict(error=repr(exc))))
                print('GRADIENT_SMOKE_FAILED',repr(exc),flush=True)
        print('COMPLETE',flush=True)
    except BaseException as exc:
        (out/'model_failed.json').write_text(json.dumps(dict(error=repr(exc))))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',type=Path,required=True)
    prepare(parser.parse_args().prepare)
