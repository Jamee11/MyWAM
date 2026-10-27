#!/usr/bin/env python3
"""Use the standard trainer with per-rank Sidecar metrics, without the local GPU gate."""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))


def main():
    import torch
    import train as entry

    from openwam.train.openwam_trainer import OpenWAMTrainer

    original = OpenWAMTrainer.log_step

    def log(self, **kwargs):
        metrics = {key: float(value) for key, value in kwargs["metrics"].items()}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError("Nonfinite Sidecar training metrics")
        row = dict(
            micro_step=kwargs["global_step"],
            optimizer_step=kwargs["opt_step"],
            epoch=kwargs["epoch"],
            seconds_micro=1 / kwargs["steps_per_sec"],
            lr=kwargs["lr"],
            metrics=metrics,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        with Path(kwargs["output_path"], f"metrics_rank{self._rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        original(self, **kwargs)
        if self.accelerator.sync_gradients:
            torch.cuda.reset_peak_memory_stats()

    OpenWAMTrainer.log_step = log
    entry.main()


if __name__ == "__main__":
    main()
