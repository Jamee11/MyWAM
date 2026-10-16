#!/usr/bin/env python3
"""Future runtime gate: strict checkpoint reload and current-only chunk generation.

This is offline inference, never a robot command or a closed-loop success test.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(checkpoint, manifest_path, report, device="cpu"):
    import h5py
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from openwam.dataloader.ea_rsig_lite import LiteRoboTwinSource
    from openwam.dataloader.transforms.multiview import format_prompt_for_inference
    from openwam.deploy.engine import JointInferenceEngine
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from openwam.deploy.obs_preprocess import ObsPreprocessor
    from openwam.deploy.policy import WAMPolicy
    from openwam.deploy.server import merge_deploy_cfg
    from openwam.ea_lite_contract import tcp_context_from_eef20
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(4)
    cfg, model = load_from_checkpoint_dir(str(checkpoint), device=device)
    if model.action_dim != 80 or not hasattr(model, "ea_bridge"):
        raise ValueError("Need a trained Lite checkpoint, not the native14 legacy model")
    deploy = OmegaConf.load(ROOT / "configs/deploy_ea_rsig_lite.yaml")
    deploy.inference.denoise_steps = 2
    cfg = merge_deploy_cfg(cfg, deploy)
    engine = JointInferenceEngine(cfg, architecture=model)
    manifest = json.loads(Path(manifest_path).read_text())
    entry = manifest["tasks"][0]
    with h5py.File(entry["episodes"][0]["path"], "r") as file:
        state = LiteRoboTwinSource._read_eef_actions(None, file, 0, 1)[0]
        images = {name: LiteRoboTwinSource._decode_jpeg(None, file[f"observation/{cam}/rgb"][0])
                  for name, cam in (("head_camera", "head_camera"), ("left_wrist_camera", "left_camera"),
                                    ("right_wrist_camera", "right_camera"))}
    observation = dict(images=images, prompt=format_prompt_for_inference(entry["name"].replace("_", " ")),
                       state=state, ea_effectors=tcp_context_from_eef20(state), ea_arm_present=[1, 1])
    processed = ObsPreprocessor(multiview=True, camera_layout=["head", "left", "right"],
                               img_height=384, img_width=320, requires_proprio=True).preprocess(observation)
    conditions = WAMPolicy.__new__(WAMPolicy)._build_conditions(processed)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    started = time.monotonic()
    result = engine.generate(conditions)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    actions = np.asarray(result["actions"])
    if actions.shape != (32, 20) or not np.isfinite(actions).all():
        raise ValueError("Expected finite 32x20 physical EEF output after official gather/unnormalize")
    output = dict(strict_reload=True, device=device, actions=list(actions.shape), finite=True,
                  seconds_chunk=time.monotonic() - started, denoise_steps=2,
                  oracle_input=False, scope="offline chunk generation; no robot command or closed-loop claim")
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--manifest-path", default=ROOT / "outputs/ea_lite/data/manifest.json")
    parser.add_argument("--report", default=ROOT / "outputs/ea_lite/readiness/deploy_runtime.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    check(**vars(args))
