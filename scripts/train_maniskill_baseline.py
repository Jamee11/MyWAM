#!/usr/bin/env python3
"""Register the isolated ManiSkill reader, then use the original Alpha trainer."""
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault('TORCH_WARM_POOL','0')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    import train as entry
    from openwam.dataloader.registry import register_dataset
    from openwam.dataloader.maniskill_absolute import ManiSkillAbsoluteDataset
    from openwam.train.openwam_trainer import OpenWAMTrainer
    register_dataset('maniskill_absolute')(ManiSkillAbsoluteDataset)
    original=OpenWAMTrainer.log_step

    def log(self,**kwargs):
        metrics={k:float(v) for k,v in kwargs['metrics'].items()}
        if not all(math.isfinite(x) for x in metrics.values()):
            raise RuntimeError('Nonfinite baseline training metric')
        row=dict(micro_step=kwargs['global_step'],optimizer_step=kwargs['opt_step'],epoch=kwargs['epoch'],
                 seconds_micro=1/kwargs['steps_per_sec'],lr=kwargs['lr'],metrics=metrics,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved())
        with Path(kwargs['output_path'],f'metrics_rank{self._rank}.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        original(self,**kwargs)
        if self.accelerator.sync_gradients:
            torch.cuda.reset_peak_memory_stats()

    OpenWAMTrainer.log_step=log
    entry.main()


if __name__=='__main__':
    main()
