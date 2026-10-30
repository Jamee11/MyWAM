#!/usr/bin/env python3
"""Rebase a copied Lite manifest; preserve every label/stat hash and source binding."""
import argparse
import copy
import json
from pathlib import Path


def relocate(manifest, raw_root, data_dir, output):
    data = copy.deepcopy(json.loads(Path(manifest).read_text()))
    if data.get("contract") != "openwam_alpha_eef32_ea_lite_v1":
        raise ValueError("Need a Lite manifest, not a legacy V3 manifest")
    raw_root, data_dir, output = map(lambda p: Path(p).resolve(), (raw_root, data_dir, output))
    old_raw = Path(data["source_root"])
    old_data = Path(data["normalization_stats_path"]).parent
    data["source_root"] = str(raw_root)
    data["normalization_stats_path"] = str(data_dir / Path(data["normalization_stats_path"]).relative_to(old_data))
    if not Path(data["normalization_stats_path"]).is_file():
        raise FileNotFoundError(data["normalization_stats_path"])
    count = 0
    for task in data["tasks"]:
        task["data_root"] = str(raw_root / Path(task["data_root"]).relative_to(old_raw))
        for episode in task["episodes"]:
            episode["path"] = str(raw_root / Path(episode["path"]).relative_to(old_raw))
            episode["labels"] = str(data_dir / Path(episode["labels"]).relative_to(old_data))
            stat = Path(episode["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (episode["size"], episode["mtime_ns"]):
                raise ValueError(f"Copied source size/mtime differs; use rsync -a or regenerate: {episode['path']}")
            if not Path(episode["labels"]).is_file():
                raise FileNotFoundError(episode["labels"])
            count += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the source manifest as an immutable handoff record; never overwrite a prior output.
    with output.open("x") as stream:
        stream.write(json.dumps(data, indent=2) + "\n")
    return dict(manifest=str(output), episodes=count, hashes="preserved; run preflight to verify bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    print(json.dumps(relocate(**vars(parser.parse_args())), indent=2))
