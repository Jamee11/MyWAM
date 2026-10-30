#!/usr/bin/env python3
"""Cache reader metadata offline, reusing existing Lite labels/stats without rebuilding them."""
import argparse
import io
import json
from pathlib import Path


def prepare(manifest, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    import h5py
    from PIL import Image
    data = json.loads(Path(manifest).read_text())
    if data.get("contract") != "openwam_alpha_eef32_ea_lite_v1":
        raise ValueError("Need existing Lite labels/manifest")
    for task in data["tasks"]:
        root = Path(task["data_root"])
        if [e["path"] for e in task["episodes"]] != sorted(e["path"] for e in task["episodes"]):
            raise ValueError("Episode order must match the official reader")
        with h5py.File(task["episodes"][0]["path"], "r") as file:
            for camera in ("head_camera", "left_camera", "right_camera"):
                if f"observation/{camera}/rgb" not in file:
                    raise ValueError(f"Missing camera {camera}")
            jpeg = file["observation/head_camera/rgb"][0]
            with Image.open(io.BytesIO(bytes(jpeg))) as image:
                size = [image.height, image.width]
        scene = root.parent / "scene_info.json"
        if not scene.exists():
            scene = root / "scene_info.json"
        instructions = {}
        for path in (root.parent / "instructions").glob("*.json"):
            try:
                instructions[path.name] = json.loads(path.read_text())
            except Exception:
                pass  # Match the existing official reader's malformed-instruction fallback.
        task["startup_metadata"] = dict(version=1, obs_image_size=size, instructions=instructions,
                                        scene_info=json.loads(scene.read_text()) if scene.exists() else {})
        print(f"Cached startup metadata: {task['name']}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(data, indent=2) + "\n")
    return dict(manifest=str(output), tasks=len(data["tasks"]), labels_and_stats="reused, unchanged",
                scope="offline metadata cache; not a full source integrity audit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))
