#!/usr/bin/env python3
"""Strictly reload a saved EA checkpoint and exercise the official deploy engine."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402 -- source-tree imports follow sys.path bootstrap
import torch  # noqa: E402 -- source-tree imports follow sys.path bootstrap
from omegaconf import OmegaConf  # noqa: E402 -- source-tree imports follow sys.path bootstrap

from openwam.dataloader import build_dataset  # noqa: E402 -- source-tree imports follow sys.path bootstrap
from openwam.deploy.engine import JointInferenceEngine  # noqa: E402 -- source-tree imports follow sys.path bootstrap
from openwam.deploy.model_loader import (  # noqa: E402 -- source-tree imports follow sys.path bootstrap
    load_from_checkpoint_dir,
)
from openwam.deploy.server import merge_deploy_cfg  # noqa: E402 -- source-tree imports follow sys.path bootstrap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(16)
    os.environ["EA_DATASET_DIRS"] = json.dumps([str(a.dataset.resolve())])
    cfg, model = load_from_checkpoint_dir(str(a.checkpoint), device=a.device)
    if not hasattr(model, "ea_bridge") or any(t.device.type == "meta" for t in model.parameters()):
        raise AssertionError("Expected fully materialized EA model")
    deploy = OmegaConf.load(ROOT / "configs/deploy.yaml")
    deploy.optimization.dit_cache.enabled = False
    deploy.optimization.compile.enabled = False
    deploy.inference.denoise_steps = 2
    cfg = merge_deploy_cfg(cfg, deploy)
    engine = JointInferenceEngine(cfg, architecture=model)
    data = build_dataset(cfg.dataloader)
    sample = data[0]
    ridx, ep, _ = data.episodes[0]
    print("Generating via official deploy engine from current observation, without oracle labels", flush=True)
    result = engine.generate({"prompt": sample["prompt"], "first_frame_image": sample["first_frame_image"],
        "proprio": data._table(ridx, ep)["observation.state"][0],
        "ea_effectors": sample["ea_effectors"], "ea_arm_present": sample["ea_arm_present"]})
    actions = result["actions"]
    if actions.shape != (40, 14) or not np.isfinite(actions).all() or not np.isin(actions[:, [6, 13]], [0, 1]).all():
        raise AssertionError("Invalid native deployment actions")
    report = {"checkpoint_dir": str(a.checkpoint.resolve()), "strict_reload": True,
        "device": a.device, "dtype": str(model.dtype), "engine": "JointInferenceEngine",
        "denoise_steps": 2, "action_shape": list(actions.shape), "finite": True, "grippers_binary_01": True,
        "video_alpha": float(model.ea_bridge.core.video_alpha.detach().float().cpu()),
        "scope": "Offline generation only; no robot command or success-rate claim"}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
