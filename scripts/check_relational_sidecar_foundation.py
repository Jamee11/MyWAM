#!/usr/bin/env python3
"""Real Foundation CPU warmstart gate; no training or GPU claim."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/sidecar/foundation_cpu.json")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Set CUDA_VISIBLE_DEVICES='' for this CPU gate")
    import torch
    from hydra import compose, initialize_config_dir
    from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir
    torch.set_num_threads(4)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="train_relational_sidecar_4gpu")
    _, model, _ = build_architecture_from_ckpt_dir(str(args.foundation), weights_required=True, override_cfg=cfg)
    model.freeze_modules(list(cfg.model.freeze))
    model.set_dtype_device(torch.bfloat16, torch.device("cpu"))
    assert not any(p.is_meta or p.device.type != "cpu" for p in model.parameters())
    assert model.action_backbone.action_encoder.weight.shape[1] == 80
    assert model.proprio_encoder.weight.shape[1] == 80
    assert model.tap_blocks == (7, 15, 23)
    assert model.sidecar.role_tokens.shape == (15, 512)
    assert all(s.write.beta == 0 and s.write.projection.weight.count_nonzero() > 0 for s in model.sidecar.stages)
    assert all(not p.requires_grad for n, p in model.named_parameters() if not n.startswith("sidecar."))
    assert model.foundation_load_report["reset_policy_heads"] == []
    report = dict(status="passed", scope="real Foundation CPU warmstart only", warmstart=model.foundation_load_report,
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  sidecar_parameters=sum(p.numel() for p in model.sidecar.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  tap_blocks=model.tap_blocks, gpu_capacity="not_tested", closed_loop="not_tested")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
