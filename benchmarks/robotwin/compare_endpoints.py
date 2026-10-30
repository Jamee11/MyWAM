"""Compare real-observation action chunks and interleaved endpoint isolation."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True)
    p.add_argument("--pool-a", required=True)
    p.add_argument("--pool-b", required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import cv2
    import h5py
    import numpy as np

    from benchmarks.robotwin.prompt_template import format_prompt_for_inference
    from benchmarks.utils.action_conversion import robotwin_endpose_to_eef20d
    from benchmarks.utils.client import build_payload, encode_numpy_b64
    from benchmarks.utils.transport import WSPolicyClient

    manifest = json.loads(args.manifest.read_text())
    inputs = []
    for task_name, instruction in [("adjust_bottle", "Adjust the bottle to an upright position."),
                                   ("place_shoe", "Place the shoe on the target.")]:
        task = next(t for t in manifest["tasks"] if t["name"] == task_name)
        path = task["episodes"][0]["path"]
        with h5py.File(path) as f:
            images = {}
            for camera in ["head", "left", "right"]:
                raw = f[f"observation/{camera}_camera/rgb"][0]
                encoded = np.frombuffer(raw, dtype=np.uint8)
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Real observation RGB decode failed")
                images[camera] = encode_numpy_b64(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # Preserve the released checkpoint's original adapter convention.
            state = robotwin_endpose_to_eef20d(f["endpose/left_endpose"][0], f["endpose/right_endpose"][0],
                                             f["endpose/left_gripper"][0], f["endpose/right_gripper"][0])
        inputs.append(build_payload(head=images["head"], left_wrist=images["left"],
                                    right_wrist=images["right"], state=state.tolist(),
                                    prompt=format_prompt_for_inference(instruction)))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "inputs.json").write_text(json.dumps(inputs))
    urls = [args.reference, args.pool_a, args.pool_b]
    deadline = time.monotonic() + 1800
    for url in urls:
        while True:
            try:
                with WSPolicyClient(url, timeout=10, open_timeout=5) as ws:
                    ws.ping()
                break
            except (OSError, TimeoutError):
                if time.monotonic() > deadline:
                    raise
                time.sleep(2)
    with WSPolicyClient(urls[0], timeout=300) as reference, \
            WSPolicyClient(urls[1], timeout=300) as a, WSPolicyClient(urls[2], timeout=300) as b:
        expected = []
        for payload in inputs:
            reference.reset()
            expected.append(np.array([reference.predict(payload)["action"] for _ in range(32)]))
        observed = []
        for assignment in [(0, 1), (1, 0)]:
            a.reset()
            b.reset()
            actions = [[], []]
            for _ in range(32):
                actions[0].append(a.predict(inputs[assignment[0]])["action"])
                actions[1].append(b.predict(inputs[assignment[1]])["action"])
            for index, values in enumerate(actions):
                values = np.array(values)
                target = expected[assignment[index]]
                if not np.isfinite(values).all() or not np.isfinite(target).all():
                    raise ValueError("Nonfinite actions")
                observed.append({"endpoint": urls[index + 1], "fixture": assignment[index],
                                 "max_abs_error": float(np.max(np.abs(values - target))),
                                 "bitwise_equal": bool(np.array_equal(values, target)),
                                 "allclose_atol_1e5_rtol_1e4": bool(np.allclose(values, target, atol=1e-5, rtol=1e-4)),
                                 "actions": values.tolist()})
        report = {"scope": "two real HDF5 first observations, 32-action chunks, interleaving and reset/swap",
                  "reference_actions": [v.tolist() for v in expected], "comparisons": observed,
                  "allclose": all(v["allclose_atol_1e5_rtol_1e4"] for v in observed)}
        (args.output / "result.json").write_text(json.dumps(report, indent=2))
        a.reset()
        b.reset()
        reference.reset()
    print(json.dumps({"allclose": report["allclose"], "comparisons": [dict(v, actions="omitted") for v in observed]}), flush=True)
    return 0 if report["allclose"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
