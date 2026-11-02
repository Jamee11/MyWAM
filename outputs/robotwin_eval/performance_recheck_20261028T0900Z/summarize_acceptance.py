"""Report matched-input model parity separately from stochastic closed-loop replay."""
from pathlib import Path
import json
import time
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE/'native_pilots_paired/legacy_direct'
if not (BASE/'pilot_result.json').exists():raise SystemExit('Waiting for paired baseline')
baseline=json.loads((BASE/'pilot_result.json').read_text())
rows=[]
for directory in sorted((HERE/'native_pilots_paired').iterdir()):
    if directory==BASE or not (directory/'pilot_result.json').exists():continue
    candidate=json.loads((directory/'pilot_result.json').read_text());episodes=[]
    for index,(a,b) in enumerate(zip(baseline['episodes'],candidate['episodes'])):
        x=np.load(BASE/f'episode{index}_actions.npy');y=np.load(directory/f'episode{index}_actions.npy')
        u=a['generations'][0];v=b['generations'][0];n=min(len(x),len(y));prefix=min(32,n)
        same_input=u['camera_sha256']==v['camera_sha256'] and np.array_equal(u['state'],v['state']) and u['instruction']==v['instruction']
        prefix_equal=np.array_equal(x[:prefix],y[:prefix])
        episodes.append(dict(scene_seed=a['scene_seed'],same_seed=a['scene_seed']==b['scene_seed'],
            same_initial_model_input=bool(same_input),first_chunk_executed_prefix_exact=bool(prefix_equal),
            baseline_success=a['success'],candidate_success=b['success'],
            baseline_steps=a['steps'],candidate_steps=b['steps'],
            baseline_seconds=a['rollout_seconds'],candidate_seconds=b['rollout_seconds'],
            full_actions_exact=bool(np.array_equal(x,y)),
            max_abs_action_difference_on_common_steps=float(np.max(np.abs(x[:n]-y[:n])))))
    passed=len(episodes)==2 and all(e['same_seed'] and e['same_initial_model_input'] and
        e['first_chunk_executed_prefix_exact'] and e['baseline_success']==e['candidate_success'] for e in episodes)
    rows.append(dict(name=directory.name,matched_initial_input_gate_passed=passed,episodes=episodes,
        baseline_mean_rollout_seconds=float(np.mean([e['baseline_seconds'] for e in episodes])),
        candidate_mean_rollout_seconds=float(np.mean([e['candidate_seconds'] for e in episodes]))))
wire=[]
for file in sorted(HERE.glob('wire_*.json')):
    value=json.loads(file.read_text())
    if isinstance(value,dict) and 'records' in value:
        records=[r for lane in value['records'] for r in lane]
        wire.append(dict(name=file.stem,passed=value['passed'],wall_seconds=value['wall_seconds'],
                         mean_request_seconds=float(np.mean([r['seconds'] for r in records])),
                         generated_chunks=len(records),actions_per_chunk=32))
compat=next((r for r in rows if r['name']=='chunk_compat_batch_stream'),None)
report=dict(wall=time.time(),scope='Native task is click_alarmclock/Clean; diagnostic instruction replay only, not formal SR; fixed-input GPU and CPU gates establish transport equivalence; closed-loop repeat can drift even legacy-versus-legacy',
            wire=wire,native=rows,compat_ready_for_new_tasks=bool(compat and compat['matched_initial_input_gate_passed']))
(HERE/'acceptance_summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
