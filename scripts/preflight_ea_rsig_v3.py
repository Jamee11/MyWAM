#!/usr/bin/env python3
"""Audit schema-5 metadata and decode real samples; never start training."""
import argparse
import json
import sys
from bisect import bisect_right
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwam.dataloader.ea_rsig_v3 import EARSIGV3Dataset
from openwam.dataloader.transforms.normalize import Normalizer, load_mode_stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset_dirs", nargs="+")
    p.add_argument("--action-space", default="qpos")
    p.add_argument("--report", type=Path, default=Path("outputs/setup/ea_data_preflight.json"))
    a = p.parse_args()
    data = EARSIGV3Dataset(a.dataset_dirs, action_space=a.action_space)
    mode = "joint" if a.action_space == "qpos" else "eef"
    normalizer = Normalizer(mode="min_max", stats=load_mode_stats(data.normalization_stats_path, mode))
    checks = []
    for index in (0, min(15, len(data) - 1), len(data) - 1):
        sample = data[index]
        eidx = bisect_right(data.ends, index)
        ridx, ep, length = data.episodes[eidx]
        row = index - (data.ends[eidx - 1] if eidx else 0)
        state = np.asarray(data._table(ridx, ep)["observation.state"][row], dtype=np.float32)
        table = data._table(ridx, ep)
        np.testing.assert_array_equal(sample["ea_effectors"].numpy().reshape(-1), np.asarray(table["observation.ea_rsig_v3.effector_state"][row], dtype=np.float32))
        np.testing.assert_array_equal(sample["ea_targets"].numpy()[0], np.asarray(table["observation.ea_rsig_v3.targets"][row], dtype=np.float32))
        expected = normalizer.normalize(state)
        expected[[6, 13]] = state[[6, 13]] > 0.5
        np.testing.assert_allclose(expected, sample["proprio"].numpy()[0], atol=1e-6)
        assert sample["action"].shape == (40, 14)
        assert sample["ea_targets"].shape == (1, 130)
        assert sample["ea_effectors"].shape == (2, 10)
        assert sample["ea_arm_present"].shape == (2,)
        assert all(im.size == (320, 384) for im in sample["video"])
        raw = normalizer.unnormalize(sample["action"].numpy())
        np.testing.assert_allclose(data.normalize(raw, "action"), sample["action"].numpy(), atol=1e-5)
        checks.append({"index": index, "action": list(sample["action"].shape), "proprio": list(sample["proprio"].shape),
                       "targets": list(sample["ea_targets"].shape), "frames": len(sample["video"]), "roundtrip_passed": True})
    report = {"dataset_dirs": [str(x) for x in data.roots], "shards": len(data.roots), "episodes": len(data.episodes),
              "windows": len(data), "stats_path": data.normalization_stats_path, "checks": checks,
              "scope": "metadata for all selected shards; three decoded samples, not full trajectory audit"}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
