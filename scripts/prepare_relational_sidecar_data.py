#!/usr/bin/env python3
"""Prepare the requested dataset's Aloha raw EEF/EA data once, outside training."""

import argparse
import json
from pathlib import Path

from cache_ea_lite_startup import prepare as cache_startup
from prepare_ea_lite_data import prepare, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory; prepared artifacts are not overwritten")
    raw = args.dataset_root.resolve() / "raw" / "aloha"
    if not raw.is_dir():
        raise FileNotFoundError(raw)
    import h5py
    import numpy as np

    # This reader follows RoboTwin's legacy RGB-JPEG convention; reject a mixed codec.
    paths = sorted(raw.glob("*/demo_clean/data/episode*.hdf5"))
    if not paths:
        raise ValueError("No Aloha source episodes")
    print(f"Checking codec markers in {len(paths)} Aloha HDF5s (first/last frames)", flush=True)
    for index, path in enumerate(paths):
        with h5py.File(path, "r") as file:
            for camera in ("head_camera", "left_camera", "right_camera"):
                images = file[f"observation/{camera}/rgb"]
                if any(b"XPL-RGB1" in bytes(images[i]) for i in (0, len(images) - 1)):
                    raise ValueError(f"Standard RGB-JPEG needs an explicit decoder contract: {path}")
        if (index + 1) % 250 == 0:
            print(f"Codec markers: {index + 1}/{len(paths)} episodes", flush=True)
    manifest = prepare(raw, args.output_dir)
    # A virtual/invalid goal cannot receive a visible-goal grounding target.
    for task in manifest["tasks"]:
        for episode in task["episodes"]:
            path = Path(episode["labels"])
            with np.load(path) as data:
                labels = {k: data[k] for k in data.files}
            targets = labels["targets"]
            valid = targets[:, 118:130].reshape(-1, 4, 3)
            valid[:, 1] *= targets[:, 14:15]
            valid[:, 3] *= targets[:, 15:16]
            np.savez_compressed(path, **labels)
            episode["labels_sha256"] = sha256(path)
    manifest.update(
        source_dataset_root=str(args.dataset_root.resolve()),
        embodiment="aloha",
        rgb_contract="RoboTwin legacy RGB passed to cv2.imencode; decode without BGR swap",
        rgb_codec_check="first and last frame, all three cameras, every episode; not every JPEG",
        goal_grounding_mask="ground_bbox_valid AND goal_valid",
    )
    source = args.output_dir / "manifest.json"
    source.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(cache_startup(source, args.output_dir / "manifest.ready.json")), flush=True)


if __name__ == "__main__":
    main()
