#!/usr/bin/env python3
"""Sample every Aloha task through the actual augmented EEF reader, without a model/GPU."""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    import h5py
    import numpy as np
    import torch

    from openwam.dataloader.ea_rsig_lite import EARSIGLiteDataset

    jitter = dict(enabled=True, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
    dataset = EARSIGLiteDataset(args.manifest, color_jitter=jitter, startup_mode="prepared")
    source = json.loads(args.manifest.read_text())
    assert source["embodiment"] == "aloha" and source["total_episodes"] == 2500 and len(dataset.readers) == 50
    checks = []
    begin = 0
    for task_index, end in enumerate(dataset.ends):
        reader = dataset.readers[task_index]
        for index in (begin, begin + (end - begin) // 2, end - 1):
            sample = dataset[index]
            assert sample["action"].shape == (32, 80) and sample["proprio"].shape == (1, 80)
            assert sample["proprio_mask"].shape == (1, 80) and sample["ea_targets"].shape == (1, 130)
            assert len(sample["video"]) == 9 and all(frame.size == (320, 384) for frame in sample["video"])
            assert sample["ea_plan_steps"] == [8, 16, 32] and sample["ea_video_anchors"] == list(range(0, 33, 4))
            assert all(torch.isfinite(sample[k]).all() for k in ("action", "proprio", "ea_targets"))
            inactive = list(range(10, 34)) + list(range(44, 80))
            assert not sample["proprio_mask"][:, inactive].any() and not sample["action_mask"][:, inactive].any()
            with h5py.File(sample["episode_path"], "r") as file:
                start, length = sample["start_frame"], sample["episode_length"]
                raw = reader._read_eef_actions(file, start, min(start + 33, length))
            future = raw[1:]
            expected = np.concatenate((future, np.repeat(future[-1:], 32 - len(future), axis=0)))
            np.testing.assert_allclose(reader.denormalize_action(sample["action"].numpy()), expected, atol=3e-6)
            np.testing.assert_allclose(reader.denormalize_action(sample["proprio"].numpy()), raw[:1], atol=3e-6)
            checks.append(dict(task=sample["task_name"], index=index, frame=start, episode=sample["episode_path"]))
        print(f"Reader audit: {task_index + 1}/50 tasks", flush=True)
        begin = end
    baseline = EARSIGLiteDataset(
        args.manifest, tasks=[dataset.task_names[0]], auxiliary=False, color_jitter=jitter, startup_mode="prepared"
    )
    random.seed(42)
    torch.manual_seed(42)
    a = baseline[0]
    random.seed(42)
    torch.manual_seed(42)
    c = dataset[0]
    for key in ("action", "proprio", "action_mask", "proprio_mask", "video_mask"):
        torch.testing.assert_close(a[key], c[key], rtol=0, atol=0)
    assert a["prompt"] == c["prompt"] and not any(k.startswith("ea_") for k in a)
    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a["video"], c["video"]))
    report = dict(
        status="passed",
        tasks=50,
        episodes=2500,
        windows=len(dataset),
        samples=checks,
        baseline_augmented_samples_equal=True,
        raw_source="read_only",
        gpu_used=False,
        scope="150 first/middle/last windows across 50 tasks; not every RGB pixel",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)


if __name__ == "__main__":
    main()
