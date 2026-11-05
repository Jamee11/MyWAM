"""Locate a completed weights-only stage; reused by the two-stage Slurm launcher."""
import argparse
import json
import math
from pathlib import Path
import struct


def completed_checkpoint(stage, updates, ranks=4):
    stage = Path(stage)
    candidates = list(stage.glob(f"runs/*/checkpoint_step_{updates}.safetensors"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one checkpoint after {updates} updates in {stage}")
    checkpoint = candidates[0]
    for rank in range(ranks):
        with (checkpoint.parent / f"metrics_rank{rank}.jsonl").open("rb") as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size - 16000))
            row = json.loads(stream.read().splitlines()[-1])
        if row["optimizer_step"] != updates or not all(math.isfinite(v) for v in row["metrics"].values()):
            raise RuntimeError(f"Incomplete/nonfinite rank {rank} in {stage}")
    marker = json.loads(checkpoint.with_suffix(".complete.json").read_text())
    if marker != dict(step=updates, bytes=checkpoint.stat().st_size):
        raise RuntimeError("Checkpoint completion marker does not match")
    with checkpoint.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(length))
    tensors = [v for k, v in header.items() if k != "__metadata__"]
    if not tensors or 8 + length + max(v["data_offsets"][1] for v in tensors) != checkpoint.stat().st_size:
        raise RuntimeError("Truncated checkpoint data")
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("updates", type=int)
    args = parser.parse_args()
    print(completed_checkpoint(args.stage, args.updates))
