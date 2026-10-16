#!/usr/bin/env python3
"""CPU-only real Foundation warmstart gate; does not run full-model training."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(foundation, report):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Run this readiness gate with CUDA_VISIBLE_DEVICES='' (CPU only)")
    import torch
    from hydra import compose, initialize_config_dir
    from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir
    torch.set_num_threads(4)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="train_ea_lite_4gpu")
    _, model, _ = build_architecture_from_ckpt_dir(str(foundation), weights_required=True, override_cfg=cfg)
    model.freeze_modules(list(cfg.model.freeze))
    model.set_dtype_device(torch.bfloat16, torch.device("cpu"))
    assert not any(p.is_meta or p.device.type != "cpu" for p in model.parameters())
    assert model.action_backbone.action_encoder.weight.shape[1] == 80
    assert model.proprio_encoder.weight.shape[1] == 80
    assert model.foundation_load_report["reset_policy_heads"] == []
    assert model.ea_bridge.action_out.weight.count_nonzero() == 0
    assert all(not p.requires_grad for n, p in model.named_parameters() if not n.startswith("ea_bridge."))
    result = dict(status="passed", scope="real Foundation loaded on CPU, no training/inference/GPU claim",
                  warmstart=model.foundation_load_report, action_dim=80, proprio_dim=80,
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  warmup_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  zero_initialized_action_adapter=True, video_observation_dit_blocks=0,
                  optimizer_step="not_run", four_gpu_capacity="pending")
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation", type=Path, default=ROOT / "assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/ea_lite/readiness/foundation_cpu.json")
    args = parser.parse_args()
    check(args.foundation, args.report)
