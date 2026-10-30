#!/usr/bin/env python3
"""Real-data CPU contract audit and view/bbox preview; no model/GPU inference."""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def audit(manifest_path, output_dir):
    import h5py
    import numpy as np
    from PIL import ImageDraw

    from benchmarks.robotwin.openwam2robotwin_interface import _extract_eef_proprio
    from openwam.dataloader.ea_rsig_lite import EARSIGLiteDataset
    from openwam.ea_lite_contract import tcp_context_from_eef20

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = EARSIGLiteDataset(manifest_path)
    checks = []
    indices = [0, len(dataset) // 2, len(dataset) - 1]
    for index in indices:
        random.seed(42)
        sample = dataset[index]
        assert sample["action"].shape == (32, 80) and sample["proprio"].shape == (1, 80)
        assert sample["action_mask"].shape == (32, 80) and len(sample["video"]) == 9
        assert sample["ea_targets"].shape == (1, 130)
        inactive = list(range(10, 34)) + list(range(44, 80))
        assert not sample["action_mask"][:, inactive].any() and not sample["proprio_mask"][:, inactive].any()
        reader = dataset.readers[next(i for i, end in enumerate(dataset.ends) if index < end)]
        with h5py.File(sample["episode_path"], "r") as file:
            row, length = sample["start_frame"], sample["episode_length"]
            raw = reader._read_eef_actions(file, row, min(row + 33, length))
            future = raw[1:]
            expected = np.concatenate((future, np.repeat(future[-1:], 32 - len(future), axis=0)))
            decoded = reader.denormalize_action(sample["action"].numpy())
            np.testing.assert_allclose(decoded, expected, atol=3e-6)
            np.testing.assert_allclose(tcp_context_from_eef20(raw[0]), sample["ea_effectors"], atol=1e-6)
            source = {
                key: file[f"endpose/{key}"][row]
                for key in ("left_endpose", "right_endpose", "left_gripper", "right_gripper")
            }
            client = _extract_eef_proprio({"endpose": source}, "wxyz")
            np.testing.assert_allclose(client, raw[0], atol=1e-6)
            np.testing.assert_allclose(tcp_context_from_eef20(client), sample["ea_effectors"], atol=1e-6)
        checks.append(
            dict(
                index=index,
                task=sample["task_name"],
                episode=sample["episode_path"],
                row=sample["start_frame"],
                valid_actions=int(sample["action_mask"].any(dim=-1).sum()),
            )
        )
    # Match the A reader to the C reader exactly, including sampled seen instruction.
    baseline = EARSIGLiteDataset(manifest_path, tasks=[dataset.task_names[0]], auxiliary=False)
    random.seed(123)
    a = baseline[0]
    random.seed(123)
    c = dataset[0]
    for key in ("action", "proprio", "action_mask", "proprio_mask", "video_mask"):
        np.testing.assert_array_equal(a[key].numpy(), c[key].numpy())
    assert a["prompt"] == c["prompt"]
    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a["video"], c["video"]))
    assert not any(key.startswith("ea_") for key in a)
    # Visual audit: source-view-normalized bbox mapped onto the SAME official L canvas.
    canvas = c["video"][0].copy()
    draw = ImageDraw.Draw(canvas)
    boxes = c["ea_targets"][0, 70:118].numpy().reshape(4, 3, 4)
    valid = c["ea_targets"][0, 118:130].numpy().reshape(4, 3)
    slots = ((0, 0, 320, 256), (0, 256, 160, 128), (160, 256, 160, 128))
    for view, (left, top, width, height) in enumerate(slots):
        for cell in range(1, 14):
            draw.line((left + cell * width / 14, top, left + cell * width / 14, top + height), fill=(100, 100, 100))
            draw.line((left, top + cell * height / 14, left + width, top + cell * height / 14), fill=(100, 100, 100))
        for role, color in enumerate(("red", "yellow", "cyan", "magenta")):
            if valid[role, view] <= 0.5:
                continue
            box = boxes[role, view] * np.array([width, height, width, height]) + [left, top, left, top]
            draw.rectangle(tuple(box), outline=color, width=2)
            draw.text((box[0], box[1]), f"r{role}", fill=color)
    canvas.save(output_dir / "camera_bbox_grid.png")
    result = dict(
        status="passed",
        tasks=len(dataset.readers),
        windows=len(dataset),
        samples=checks,
        alpha_actions="32x80, absolute EEF t+1..t+32",
        video_frames=9,
        plan_steps=[8, 16, 32],
        train_client_context_equal=True,
        baseline_conditional_samples_equal=True,
        source_data="read_only",
        scope="all episode inventory + regenerated labels; decoded sample windows, not all RGB frames",
    )
    (output_dir / "data_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", default=ROOT / "outputs/ea_lite/data/manifest.json")
    parser.add_argument("--output-dir", default=ROOT / "outputs/ea_lite/readiness")
    args = parser.parse_args()
    audit(args.manifest_path, args.output_dir)
