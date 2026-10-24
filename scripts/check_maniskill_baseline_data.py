#!/usr/bin/env python3
"""Exercise real first/middle/terminal windows and 64-sample collation."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import h5py
    import numpy as np
    from PIL import Image
    from openwam.dataloader.maniskill_absolute import ManiSkillAbsoluteDataset
    from openwam.dataloader.transforms.multiview import assemble_multiview_layout
    from openwam.dataloader.utils.normalization import apply_normalization
    import torch
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--wait-for-manifest',action='store_true')
    args=parser.parse_args()
    if args.wait_for_manifest:
        while not Path(args.manifest).exists():
            time.sleep(3)
    ds=ManiSkillAbsoluteDataset(args.manifest)
    checks=[];started=time.monotonic();samples=[]
    for task in sorted({r['task'] for r in ds.records}):
        episodes=[i for i,r in enumerate(ds.records) if r['task']==task]
        for e in (episodes[0],episodes[500],episodes[-1]):
            record=ds.records[e];length=record['steps'];first=ds.ends[e-1] if e else 0
            for start in (0,length//2,length-1):
                before=time.monotonic();sample=ds[first+start];samples.append(sample)
                assert sample['action'].shape==(32,80) and sample['proprio'].shape==(1,80)
                assert sample['action_mask'].sum().item()==min(32,length-start)*10
                assert sample['proprio_mask'].sum().item()==10
                assert not sample['action'][:,10:].any() and not sample['proprio'][:,10:].any()
                assert len(sample['video'])==9 and sample['video'][0].size==(320,384)
                assert not np.asarray(sample['video'][0])[256:,160:].any()
                assert sample['video_mask'].sum().item()==int((start+np.arange(0,33,4)<=length).sum())
                with np.load(record['numeric']) as numeric:
                    expected=apply_normalization(numeric['action'][start],ds.stats,'min-max')
                    np.testing.assert_allclose(sample['action'][0,:10],expected,atol=1e-6)
                with h5py.File(record['source_h5'],'r') as f:
                    tr=f[record['trajectory']]
                    front=np.asarray(tr['obs/sensor_data/front_camera/rgb'][start])
                    wrist=np.asarray(tr['obs/sensor_data/hand_camera/rgb'][start])
                expected_image=assemble_multiview_layout({'front_camera':Image.fromarray(front),'hand_camera':Image.fromarray(wrist)},['front_camera','hand_camera',None],384,320)
                np.testing.assert_array_equal(np.asarray(sample['video'][0]),np.asarray(expected_image))
                checks.append(dict(task=task,episode=e,start=start,seconds=time.monotonic()-before))
    # Use precisely the trainer's batching contract, including PIL video lists.
    batch=list((samples*2)[:64])
    actions=torch.stack([s['action'] for s in batch])
    proprio=torch.stack([s['proprio'] for s in batch])
    assert actions.shape==(64,32,80)
    assert proprio.shape==(64,1,80)
    result=dict(total_episodes=len(ds.records),total_windows=len(ds),window_checks=checks,
                batch_size=64,batch_action_shape=list(actions.shape),elapsed_seconds=time.monotonic()-started)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='window_checks'}),flush=True)


if __name__=='__main__':
    main()
