"""Read training logs and scalar checkpoint gates; never load a GPU model."""
import csv
import hashlib
import json
import math
import statistics
import struct
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/remote-home/jinminghao/OpenWAM')
OUT = Path(__file__).resolve().parent
RUN = Path((ROOT / 'outputs/sidecar/current_full_backward_warmup2k_run.txt').read_text().strip())
ranks, metadata = [], []
for rank in range(4):
    path = RUN / f'metrics_rank{rank}.jsonl'
    with path.open('rb') as stream:
        data = stream.read(path.stat().st_size)
    lines, rows = data.splitlines(), []
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1 and not data.endswith(b'\n'):
                continue
            raise
        assert all(math.isfinite(float(v)) for v in row['metrics'].values())
        rows.append(row)
    assert [x['micro_step'] for x in rows] == list(range(1, len(rows) + 1))
    ranks.append(rows)
    metadata.append(dict(rank=rank, rows=len(rows), snapshot_bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
cutoff = min(len(rows) for rows in ranks)
rows = ranks[0][:cutoff]
max_difference = max(abs(row['metrics'][key] - ranks[rank][i]['metrics'][key])
                     for rank in range(1, 4) for i, row in enumerate(rows) for key in row['metrics'])
windows = []
for end in [4000, 8000, 12000, 16000, 20000, cutoff]:
    chunk = rows[end - 1000:end]
    result = dict(micro_start=end - 999, micro_end=end, optimizer_end=chunk[-1]['optimizer_step'])
    result.update({key: statistics.mean(x['metrics'][key] for x in chunk) for key in chunk[0]['metrics'] if key.startswith('loss')})
    result.update({key: chunk[-1]['metrics'][key] for key in chunk[0]['metrics'] if 'gate' in key})
    norms = [x['metrics']['grad_norm'] for x in chunk if x['micro_step'] % 2 == 0]
    result.update(grad_mean=statistics.mean(norms), grad_max=max(norms), clipped_updates=sum(x > 1 for x in norms), lr=chunk[-1]['lr'])
    result['weighted_ea_parts'] = dict(ground=.0125 * result['loss_ea_ground'], relation=.005 * result['loss_ea_relation'], plan=.025 * result['loss_ea_plan'], deep=.05 * result['loss_ea_deep'])
    assert abs(sum(result['weighted_ea_parts'].values()) - result['loss_ea']) < 1e-6
    windows.append(result)
gates = {}
for step in [16000, 20000]:
    path = RUN / f'checkpoint_step_{step}.safetensors'
    with path.open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(size))
        gates[str(step)] = {}
        for key, item in header.items():
            if key.endswith(('.alpha', '.write.beta')):
                stream.seek(8 + size + item['data_offsets'][0])
                value = stream.read(item['data_offsets'][1] - item['data_offsets'][0])
                assert item['shape'] == [] and item['dtype'] in ('BF16', 'F32')
                gates[str(step)][key] = struct.unpack('<f', b'\x00\x00' + value if item['dtype'] == 'BF16' else value)[0]
report = dict(captured_utc=datetime.now(timezone.utc).isoformat(), run=str(RUN), ranks=metadata, common_micro=cutoff,
              latest=rows[-1], windows=windows, checkpoint_gates=gates, max_rank_metric_difference=max_difference,
              max_loss_decomposition_error=max(abs(x['metrics']['loss_total'] - x['metrics']['loss_action'] - x['metrics']['loss_video'] - x['metrics']['loss_ea']) for x in rows),
              all_metrics_finite=True, weight_decay_only_max_beta_shrink_2k_to_10k=1 - (1 - .01 * 5e-5) ** 8000)
(OUT / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
flat = [{key: val for key, val in item.items() if key != 'weighted_ea_parts'} for item in windows]
with (OUT / 'window_statistics.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
    writer.writeheader()
    writer.writerows(flat)

bins = [rows[i:i + 100] for i in range(0, cutoff - 99, 100)]
x = [chunk[-1]['micro_step'] / 1000 for chunk in bins]
fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
for ax, key, title in [(axes[0, 0], 'loss_action', 'Action training loss'), (axes[0, 1], 'loss_video', 'Video training loss')]:
    ax.plot(x, [statistics.mean(row['metrics'][key] for row in chunk) for chunk in bins])
    ax.set_title(title)
    ax.set_yscale('log')
for key, scale, label in [('loss_ea', 1, 'EA total'), ('loss_ea_ground', .0125, 'Ground contribution'), ('loss_ea_plan', .025, 'Plan contribution'), ('loss_ea_relation', .005, 'Relation contribution'), ('loss_ea_deep', .05, 'Deep contribution')]:
    axes[0, 2].plot(x, [scale * statistics.mean(row['metrics'][key] for row in chunk) for chunk in bins], label=label)
axes[0, 2].set_yscale('log')
axes[0, 2].set_title('Weighted auxiliary loss contributions')
axes[0, 2].legend(fontsize=8)
sample = rows[::50]
for i, block in enumerate([7, 15, 23]):
    xx = [row['micro_step'] / 1000 for row in sample]
    axes[1, 0].plot(xx, [row['metrics'][f'ea_read_gate_{i}'] for row in sample], label=f'Block {block}')
    axes[1, 1].plot(xx, [abs(row['metrics'][f'ea_write_gate_{i}']) for row in sample], label=f'Block {block}')
axes[1, 0].set_title('Read alpha (signed)')
axes[1, 1].set_title('Write |beta| (not residual strength)')
axes[1, 1].set_yscale('log')
axes[1, 0].legend()
axes[1, 1].legend()
axes[1, 2].plot(x, [max(row['metrics']['grad_norm'] for row in chunk) for chunk in bins])
axes[1, 2].axhline(1, color='red', linestyle=':', label='Clip threshold')
axes[1, 2].set_yscale('log')
axes[1, 2].set_title('Preclip grad norm: max per 100 microsteps')
axes[1, 2].legend()
for ax in axes.flat:
    for v, style in [(4, '--'), (16, ':'), (20, ':')]:
        ax.axvline(v, color='gray', linestyle=style, linewidth=1)
    ax.grid(alpha=.2)
    ax.set_xlabel('Microsteps (thousands; optimizer updates = microsteps / 2)')
fig.suptitle('Full-backward warmup2K: training logs, not held-out validation\nDashed: WAM begins updating; dotted: 16k and 20k checkpoints')
fig.savefig(OUT / 'training_trends.png', dpi=160)
fig.savefig(OUT / 'training_trends.pdf')
print(json.dumps({key: report[key] for key in ['captured_utc', 'common_micro', 'max_rank_metric_difference', 'max_loss_decomposition_error', 'all_metrics_finite', 'weight_decay_only_max_beta_shrink_2k_to_10k']}, indent=2))
