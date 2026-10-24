"""Standard training with a CPU-ready GPU handover gate and lightweight metrics."""

import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def selected_gpu_memory():
    selected = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if selected != ["0", "1", "2", "3"]:
        raise ValueError("This launch is allocated physical GPU0,1,2,3")
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    devices = {row[0]: row for line in output.splitlines() if (row := [x.strip() for x in line.split(",")])}
    for index in selected:
        row = devices[index]
        if "A100" not in row[1] or int(row[2]) < 75000:
            raise ValueError(f"Expected A100-80GB at GPU{index}")
    return {index: int(devices[index][3]) for index in selected}


def write_marker(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def wait_for_gpu_release(handover):
    memory = selected_gpu_memory()
    write_marker(handover / "cpu_ready.json", dict(
        utc=datetime.now(timezone.utc).isoformat(), status="waiting_for_old_gpu_job_release",
        gpu_free_mib=memory, notice_seconds=90,
    ))
    print("GPU_HANDOVER_READY: CPU weights ready; notify user now. Waiting at least 90s and for all GPU0-3 >=70000MiB free.", flush=True)
    ready_at = time.monotonic()
    while time.monotonic() - ready_at < 90 or min(memory.values()) < 70000:
        time.sleep(5)
        memory = selected_gpu_memory()
    write_marker(handover / "gpu_release.json", dict(utc=datetime.now(timezone.utc).isoformat(), gpu_free_mib=memory))
    print("GPU_HANDOVER_RELEASED: old allocation cleared; preparing DeepSpeed model.", flush=True)


def main():
    # CPU imports and weights may proceed while the old run still holds the GPUs.
    import torch
    import train as entry

    from openwam.train.openwam_trainer import OpenWAMTrainer

    handover = Path(os.environ["EA_HANDOVER_DIR"]).resolve()
    handover.mkdir(parents=True, exist_ok=True)
    release = handover / "gpu_release.json"
    if release.exists():
        raise RuntimeError("Use a fresh handover directory for every launch")
    original_prepare = OpenWAMTrainer.prepare_accelerate
    original_log = OpenWAMTrainer.log_step

    def prepare(self, optimizer, dataloader, scheduler):
        if not self.cfg.training.initialize_model_on_cpu:
            raise ValueError("Handover requires CPU model initialization")
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            wait_for_gpu_release(handover)
        else:
            while not release.exists():
                time.sleep(2)
        self.accelerator.wait_for_everyone()
        return original_prepare(self, optimizer, dataloader, scheduler)

    def log(self, **kwargs):
        metrics = {key: float(value) for key, value in kwargs["metrics"].items()}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError("Nonfinite training metrics")
        row = dict(
            micro_step=kwargs["global_step"], optimizer_step=kwargs["opt_step"], epoch=kwargs["epoch"],
            seconds_micro=1 / kwargs["steps_per_sec"], lr=kwargs["lr"], metrics=metrics,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        with Path(kwargs["output_path"], f"metrics_rank{self._rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        original_log(self, **kwargs)
        if self.accelerator.sync_gradients:
            torch.cuda.reset_peak_memory_stats()

    OpenWAMTrainer.prepare_accelerate = prepare
    OpenWAMTrainer.log_step = log
    entry.main()


if __name__ == "__main__":
    main()
