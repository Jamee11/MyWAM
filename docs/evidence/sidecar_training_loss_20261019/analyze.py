"""Read-only loss audit of the four-rank Sidecar training run."""
from pathlib import Path
import json, math, csv
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/remote-home/jinminghao/OpenWAM')
RUN=ROOT/'outputs/sidecar/aloha50_C_joint_10epochs_20261016T131709Z/runs/2026-10-16_13-50-52'
OUT=Path(__file__).resolve().parent
ranks=[]; metadata=[]
for rank in range(4):
 path=RUN/f'metrics_rank{rank}.jsonl';size=path.stat().st_size
 with path.open('rb') as f: data=f.read(size)
 lines=data.splitlines();rows=[];partial=0
 for i,line in enumerate(lines):
  try: row=json.loads(line)
  except json.JSONDecodeError:
   if i==len(lines)-1 and not data.endswith(b'\n'):partial+=1;continue
   raise
  assert all(math.isfinite(float(v)) for v in row['metrics'].values())
  assert math.isfinite(row['lr']) and math.isfinite(row['seconds_micro'])
  rows.append(row)
 steps=[r['micro_step'] for r in rows]
 assert steps==list(range(1,steps[-1]+1)),f'rank{rank} discontinuity/duplicate'
 metadata.append(dict(rank=rank,snapshot_bytes=size,rows=len(rows),last_step=steps[-1],partial_final_line=partial))
 ranks.append(rows)
cutoff=min(r[-1]['micro_step'] for r in ranks);rows=ranks[0][:cutoff];keys=list(rows[0]['metrics'])
rank_diffs={k:max(float(np.max(np.abs(np.array([r['metrics'][k] for r in ranks[i][:cutoff]])-np.array([r['metrics'][k] for r in rows])))) for i in range(1,4)) for k in keys}
step=np.arange(1,cutoff+1);values={k:np.array([r['metrics'][k] for r in rows]) for k in keys}
lr=np.array([r['lr'] for r in rows]);latest=rows[-1]
windows=[]
for end in [1000,4000,8000,12000,16000,20000,24000,28000,32000,36000]:
 if end>cutoff:continue
 start=end-999;mask=(step>=start)&(step<=end)
 item=dict(micro_start=start,micro_end=end,optimizer_end=rows[end-1]['optimizer_step'],epoch=rows[end-1]['epoch'],lr_mean=float(lr[mask].mean()))
 for key in keys:
  item[key]=float(values[key][mask].mean());item[key+'_std']=float(values[key][mask].std())
 windows.append(item)
with (OUT/'window_statistics.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(windows[0]));writer.writeheader();writer.writerows(windows)
by_end={r['micro_end']:r for r in windows}
comparisons={}
for a,b in [(16000,20000),(20000,24000),(24000,28000),(28000,32000),(32000,36000),(20000,32000)]:
 if a in by_end and b in by_end:
  comparisons[f'{a}_to_{b}']={key:100*(by_end[b][key]-by_end[a][key])/by_end[a][key] for key in keys if key.startswith('loss') and by_end[a][key]!=0}
residual=values['loss_total']-values['loss_video']-values['loss_action']-values['loss_ea']
nonzero=values['grad_norm'][(step%2)==0]
report={'created_utc':datetime.now(timezone.utc).isoformat(),'run':str(RUN),'ranks':metadata,'common_cutoff_micro':cutoff,'optimizer_steps':latest['optimizer_step'],'epoch_zero_based':latest['epoch'],'planned_micro':42960,'progress_pct':100*cutoff/42960,'latest_lr':latest['lr'],'max_logged_lr':float(lr.max()),'rank_max_abs_metric_differences':rank_diffs,'finite_metrics':True,'windows':windows,'relative_changes_percent':comparisons,'max_abs_loss_decomposition_error':float(np.abs(residual).max()),'update_grad_norm':dict(median=float(np.median(nonzero)),p99=float(np.quantile(nonzero,.99)),max=float(nonzero.max()),zero_updates=int((nonzero==0).sum()),above_one=int((nonzero>1).sum())),'last1000_update_grad_norm':dict(median=float(np.median(nonzero[-500:])),max=float(nonzero[-500:].max()))}
(OUT/'summary.json').write_text(json.dumps(report,indent=2))
# Bin 100 microsteps to show the trend without presenting single-batch noise as convergence.
ends=np.arange(100,cutoff+1,100);x=ends/1000
labels=[('loss_total','Total'),('loss_video','Video'),('loss_action','Action'),('loss_ea_ground','Ground (raw)'),('loss_ea_relation','Relation (raw)'),('loss_ea_plan','Plan (raw)')]
fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
for ax,(key,label) in zip(axes.flat,labels):
 means=np.array([values[key][end-100:end].mean() for end in ends]);ax.plot(x,means,color='#2471A3',lw=1.4,label='100-microstep mean')
 if key in ('loss_action','loss_ea_relation','loss_ea_plan'):ax.set_yscale('log')
 for checkpoint,color in [(20,'#999999'),(32,'#D35400'),(36,'#6C3483')]:
  if checkpoint*1000<=cutoff:ax.axvline(checkpoint,color=color,ls='--',lw=1,alpha=.8)
 ax.set_title(label);ax.set_xlabel('Microstep (thousands; 2 microsteps / optimizer update)');ax.set_ylabel('Training loss');ax.grid(alpha=.2)
fig.suptitle('Sidecar joint training: four-rank mean, 100-microstep bins\nDashed: 20k / 32k / 36k; no held-out validation curve',fontsize=13)
fig.savefig(OUT/'loss_curves.png',dpi=160);fig.savefig(OUT/'loss_curves.pdf');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(15,3.8),constrained_layout=True)
for i in range(3):axes[0].plot(step[::50]/1000,values[f'ea_read_gate_{i}'][::50],label=f'tap {i}')
for i in range(3):axes[1].plot(step[::50]/1000,values[f'ea_write_gate_{i}'][::50],label=f'tap {i}')
axes[0].set_title('Read alpha');axes[1].set_title('Write beta');axes[2].plot(step[::50]/1000,lr[::50]);axes[2].set_title('Learning rate')
for ax in axes:ax.set_xlabel('Microstep (thousands)');ax.grid(alpha=.2)
axes[0].legend();axes[1].legend();fig.savefig(OUT/'gates_and_lr.png',dpi=160);plt.close(fig)
print(json.dumps({k:v for k,v in report.items() if k not in ['windows','relative_changes_percent']},indent=2))
for w in windows:
 print('WINDOW',w['micro_end'],{k:round(w[k],8) for k in ['loss_total','loss_video','loss_action','loss_ea','loss_ea_ground','loss_ea_relation','loss_ea_plan']})
print('CHANGES',json.dumps(comparisons.get('32000_to_36000',{}),indent=2))
