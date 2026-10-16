"""Instrument the unchanged official training loop; use save_steps=0 for profiling."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import torch
import train as entry
from openwam.train.openwam_trainer import OpenWAMTrainer

# Preserve every parameter and AdamW hyperparameter; bound ZeRO flat buffers.
if os.environ.get('EA_BENCH_GROUP_ELEMENTS'):
    original_optimizer = OpenWAMTrainer.build_optimizer
    def bounded_optimizer(self):
        original = original_optimizer(self)
        limit = int(os.environ['EA_BENCH_GROUP_ELEMENTS'])
        groups = []
        for group in original.param_groups:
            chunk, elements = [], 0
            for parameter in group['params']:
                if chunk and elements + parameter.numel() > limit:
                    groups.append(dict(group, params=chunk))
                    chunk, elements = [], 0
                chunk.append(parameter)
                elements += parameter.numel()
            if chunk:
                groups.append(dict(group, params=chunk))
        before = [id(p) for group in original.param_groups for p in group['params']]
        after = [id(p) for group in groups for p in group['params']]
        assert before == after and len(after) == len(set(after))
        for group in groups:
            assert all(group[key] == original.param_groups[0][key]
                       for key in ('lr', 'betas', 'weight_decay', 'eps', 'fused', 'foreach'))
        defaults = dict(original.defaults)
        defaults.pop('decoupled_weight_decay', None)  # AdamW constructor sets it itself.
        optimizer = torch.optim.AdamW(groups, **defaults)
        if int(os.environ.get('RANK', '0')) == 0:
            print('Bounded AdamW groups:', len(groups), 'largest elements:',
                  max(sum(p.numel() for p in group['params']) for group in groups), flush=True)
        return optimizer
    OpenWAMTrainer.build_optimizer = bounded_optimizer

original_log = OpenWAMTrainer.log_step
report_dir = Path(os.environ['EA_BENCH_REPORT_DIR'])
report_dir.mkdir(parents=True, exist_ok=True)
rank = int(os.environ.get('RANK', '0'))


def measured_log(self, **kwargs):
    torch.cuda.synchronize()
    row = {
        'rank': rank,
        'step': kwargs['global_step'],
        'optimizer_step': kwargs['opt_step'],
        'batch_per_gpu': int(self.cfg.training.batch_size),
        'seconds': 1.0 / kwargs['steps_per_sec'],
        'loss': float(kwargs['metrics']['loss_total']),
        'metrics': kwargs['metrics'],
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'free_bytes': torch.cuda.mem_get_info()[0],
    }
    assert row['optimizer_step'] == row['step'], 'Benchmark requires accumulation=1'
    with (report_dir / f'rank{rank}.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    original_log(self, **kwargs)
    torch.cuda.reset_peak_memory_stats()


OpenWAMTrainer.log_step = measured_log
# A sweep reuses one prepared model/optimizer; only the real batch changes.
if os.environ.get('EA_BENCH_SWEEP') == '1':
    import random
    original_prepare = OpenWAMTrainer.prepare_accelerate
    schedule = [(1, 6), (2, 6), (4, 6), (8, 20), (16, 20), (32, 20), (64, 20)]
    if os.environ.get('EA_BENCH_BATCHES'):
        batches = [int(value) for value in os.environ['EA_BENCH_BATCHES'].split(',')]
        steps = int(os.environ.get('EA_BENCH_STEPS_PER_BATCH', '8'))
        if not batches or any(batch < 1 for batch in batches) or steps < 3:
            raise ValueError('Sweep needs positive batches and at least three updates per batch')
        schedule = [(batch, steps) for batch in batches]

    class SweepLoader:
        def __init__(self, trainer):
            self.trainer = trainer

        def __len__(self):
            return sum(count for _, count in schedule)

        def __iter__(self):
            rng = random.Random(42)
            trainer = self.trainer
            world = trainer.accelerator.num_processes
            for batch, count in schedule:
                trainer.architecture.set_train_micro_batch_size(batch)
                trainer.cfg.training.batch_size = batch
                assert trainer.architecture.gradient_accumulation_steps() == 1
                assert trainer.architecture.train_batch_size() == batch * world
                (report_dir / f'current_rank{rank}.json').write_text(
                    json.dumps({'batch_per_gpu': batch, 'steps_planned': count}))
                if os.environ.get('EA_BENCH_PRODUCTION_LOADER') == '1':
                    loader = trainer.accelerator.prepare_data_loader(trainer.build_dataloader(batch))
                    if len(loader) < count:
                        raise ValueError('Prepared loader has fewer batches than requested')
                    iterator = iter(loader)
                    try:
                        for _ in range(count):
                            samples = next(iterator)
                            if len(samples) != batch:
                                raise ValueError('Prepared loader returned the wrong micro-batch')
                            yield samples
                    finally:
                        del iterator
                    continue
                for _ in range(count):
                    indices = rng.sample(range(len(trainer.dataset)), batch * world)
                    local = indices[rank * batch:(rank + 1) * batch]
                    assert len(local) == batch
                    yield [trainer.dataset[index] for index in local]

    def sweep_prepare(self, optimizer, dataloader, scheduler):
        assert int(self.cfg.training.gradient_accumulation_steps) == 1
        optimizer, _, scheduler = original_prepare(self, optimizer, dataloader, scheduler)
        return optimizer, SweepLoader(self), scheduler

    OpenWAMTrainer.prepare_accelerate = sweep_prepare

original_train = entry._train


def guarded_train(cfg):
    try:
        original_train(cfg)
    except torch.cuda.OutOfMemoryError as error:
        # An OOM rank must exit promptly; collective teardown could hang its peer.
        import traceback
        traceback.print_exc()
        (report_dir / f'oom_rank{rank}.txt').write_text(str(error))
        sys.stderr.flush()
        os._exit(86)


entry._train = guarded_train
if __name__ == '__main__':
    entry.main()
