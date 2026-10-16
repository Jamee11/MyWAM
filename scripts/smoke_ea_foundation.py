#!/usr/bin/env python3
"""Verify the real Alpha warmstart; optionally run one real-data optimizer update."""
import argparse
import json
import os
import sys
import shutil
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from hydra import compose, initialize_config_dir
from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir, copy_ckpt_artifacts
from openwam.dataloader import build_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--foundation", type=Path, default=ROOT / "assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--train-step", action="store_true")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--roundtrip", action="store_true", help="Save/reload the full EA model and run deploy-side generation")
    p.add_argument("--roundtrip-dir", type=Path, default=ROOT / "outputs/setup/full_model_roundtrip")
    p.add_argument("--report", type=Path, default=ROOT / "outputs/setup/foundation_smoke.json")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(42)
    proof = json.loads((a.foundation / "download_verification.json").read_text())
    weight = proof["files"].get("checkpoint_step_154000.safetensors")
    if not weight or not weight["verified_upstream"]:
        raise ValueError("Verified foundation weight is required")
    if weight["sha256"] != "180a02653118b0f96da28a9cae9ec7b4c1c1e6cd0e4608b56c7b4884f8001d3d":
        raise ValueError("Unexpected foundation checkpoint hash")
    os.environ["EA_DATASET_DIRS"] = json.dumps([str(a.dataset.resolve())])
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="train", overrides=["model=dual_system_ea_rsig_v3", "dataloader=ea_rsig_v3"])
    _, arch, _ = build_architecture_from_ckpt_dir(str(a.foundation), weights_required=True, override_cfg=cfg)
    if any(p.device.type == "meta" for p in arch.parameters()):
        raise AssertionError("Meta tensors remain")
    report = {"foundation_revision": proof["revision"], "foundation_sha256": weight["sha256"],
              "warmstart": arch.foundation_load_report, "ea_contract": arch.ea_bridge.contract(),
              "train_step": False, "scope": "Real foundation warmstart; no closed-loop claim"}
    print(json.dumps(report, indent=2), flush=True)
    if a.train_step:
        device = torch.device(a.device)
        if device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            if free < 34 * 1024 ** 3:
                raise RuntimeError("Need at least 34 GiB currently free for this short SGD smoke")
        elif device.type == "cpu":
            available_kib = int(next(line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")).split()[1])
            if available_kib < 160 * 1024 ** 2:
                raise RuntimeError("Need 160 GiB free host memory for full FP32 AdamW smoke")
        else:
            raise ValueError("Smoke device must be cpu or cuda")
        data = build_dataset(cfg.dataloader)
        arch.freeze_modules(list(cfg.model.freeze))
        arch.init_training_schedulers()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        arch.set_dtype_device(dtype, torch.device("cpu"))
        arch.train()
        # Frozen T5/VAE prepare a real sample on CPU; only denoising experts occupy GPU.
        prepared = arch.prepare_inputs([data[0]])
        vb = arch.video_backbone
        vb.dit.to(device=device, dtype=dtype)
        vb._device, vb._dtype = device, dtype
        arch.action_backbone.to(device=device, dtype=dtype)
        arch.proprio_encoder.to(device=device, dtype=dtype)
        arch.ea_bridge.to(device=device, dtype=dtype)
        prepared = {k: v.to(device=device) if torch.is_tensor(v) else v for k, v in prepared.items()}
        params = [p for p in arch.parameters() if p.requires_grad]
        if device.type == "cpu":
            optimizer = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01, foreach=False)
        else:
            optimizer = torch.optim.SGD(params, lr=1e-5)
        before = arch.ea_bridge.core.video_alpha.detach().float().cpu().clone()
        prepared["use_gradient_checkpointing"] = True
        result = arch.compute_loss(**prepared)
        if not all(torch.isfinite(v).all() for v in result.values()):
            raise AssertionError("Nonfinite losses")
        result["loss"].backward()
        for name, module in (("video", vb.dit), ("action", arch.action_backbone), ("EA", arch.ea_bridge)):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise AssertionError(f"Missing/nonfinite {name} gradients")
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        after = arch.ea_bridge.core.video_alpha.detach().float().cpu()
        report.update(train_step=True, losses={k: float(v.detach().float().cpu()) for k, v in result.items()},
                      video_alpha_before=float(before), video_alpha_after=float(after),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                      optimizer="FP32 AdamW on CPU; distributed ZeRO capacity unverified" if device.type == "cpu" else "SGD CUDA capacity smoke; production AdamW/ZeRO unverified")
        optimizer.zero_grad(set_to_none=True)
        del optimizer
    if a.roundtrip:
        from omegaconf import OmegaConf
        from openwam.deploy.model_loader import load_from_checkpoint_dir
        from openwam.deploy.denoise_schedule import make_schedule
        output = a.roundtrip_dir
        output.mkdir(parents=True, exist_ok=True)
        saved = output / "checkpoint_step_1.safetensors"
        if saved.exists():
            raise FileExistsError("Use a fresh roundtrip directory; do not overwrite prior evidence")
        if not a.train_step:
            data = build_dataset(cfg.dataloader)
            arch.set_dtype_device(torch.float32, torch.device("cpu"))
        print("Saving full EA checkpoint and deployment assets", flush=True)
        # Match the official deploy loader's BF16 policy dtype before serialization.
        arch.set_dtype_device(torch.bfloat16, torch.device("cpu"))
        arch.save_checkpoint(str(saved))
        arch.save_assets_for_deployment(str(output), cfg)
        copy_ckpt_artifacts(str(a.foundation), str(output))
        shutil.copyfile(data.normalization_stats_path, output / "normalization_stats.npy")
        OmegaConf.save(cfg, output / "config.yaml")
        expected_alpha = arch.ea_bridge.core.video_alpha.detach().cpu().clone()
        del arch
        print("Strict reloading full EA checkpoint via official deploy loader", flush=True)
        _, loaded = load_from_checkpoint_dir(str(output), device="cpu")
        torch.testing.assert_close(loaded.ea_bridge.core.video_alpha.detach().cpu(), expected_alpha, rtol=0, atol=0)
        sample = data[0]
        ridx, ep, _ = data.episodes[0]
        schedule = make_schedule("sync", video_scheduler=loaded.video_scheduler,
                                 action_scheduler=loaded.action_scheduler, num_steps=2)
        print("Running two-step deployment generation from current observation only", flush=True)
        result = loaded.generate(schedule=schedule, prompt=sample["prompt"],
                                 first_frame_image=sample["first_frame_image"], num_frames=5,
                                 action_num_frames=41, height=384, width=320, num_inference_steps=2,
                                 decode_video=False, proprio=loaded.normalize_deploy_proprio(data._table(ridx, ep)["observation.state"][0]),
                                 ea_effectors=sample["ea_effectors"][None], ea_arm_present=sample["ea_arm_present"][None])
        import numpy as np
        actions = result["actions"]
        if actions.shape != (40, 14) or not np.isfinite(actions).all():
            raise AssertionError("Invalid deployment output shape/values")
        if not np.isin(actions[:, [6, 13]], [0, 1]).all():
            raise AssertionError("Deployment gripper contract failed")
        report["roundtrip"] = {"checkpoint": str(saved.resolve()), "size": saved.stat().st_size,
                               "strict_deploy_reload": True, "actions_shape": list(actions.shape),
                               "actions_finite": True, "grippers_binary_01": True,
                               "scope": "Two-step offline generation; no robot command or success-rate claim"}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2))
    print("foundation smoke report: " + str(a.report), flush=True)


if __name__ == "__main__":
    main()
