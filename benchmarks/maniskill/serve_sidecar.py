"""Reuse the checked Sidecar FFN runtime with native single-arm ManiSkill fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def validation_inputs(manifest):
    import h5py

    from benchmarks.robotwin.prompt_template import format_prompt_for_inference
    from benchmarks.utils.client import build_payload, encode_numpy_b64
    from openwam.maniskill_eef_contract import achieved_states

    records = json.loads(Path(manifest).read_text())["episodes"]
    inputs = []
    for task in ("push_cube", "pick_cube"):
        row = next(r for r in records if r["task"] == task)
        step = min(20, row["steps"] - 1)
        with h5py.File(row["source_h5"], "r") as f:
            obs = f[row["trajectory"]]["obs"]
            state = achieved_states(
                obs["extra/tcp_pose"][step : step + 1],
                obs["extra/robot_base_pose"][step : step + 1],
                obs["agent/qpos"][step : step + 1],
            )[0]
            inputs.append(
                build_payload(
                    encode_numpy_b64(obs["sensor_data/front_camera/rgb"][step]),
                    encode_numpy_b64(obs["sensor_data/hand_camera/rgb"][step]),
                    prompt=format_prompt_for_inference(row["prompt"]),
                    state=state.tolist(),
                )
            )
    return inputs


def archive_validation_inputs(plan_path):
    """Use current archived cameras/text/TCP; never feed auxiliary labels to the policy."""
    import h5py
    from benchmarks.robotwin.prompt_template import format_prompt_for_inference
    from benchmarks.utils.client import build_payload, encode_numpy_b64
    from openwam.maniskill_eef_contract import achieved_states
    plan = json.loads(Path(plan_path).read_text())
    inputs = []
    for task in ('push_cube', 'pick_cube'):
        row = next(r for r in plan['cases'] if r['task'] == task and r['condition'] == 'Clean')
        with h5py.File(Path(plan['archive_root']) / row['media'], 'r') as f:
            obs = f['traj_0/obs']
            state = achieved_states(obs['extra/tcp_pose'][0:1], obs['extra/robot_base_pose'][0:1],
                                    obs['agent/qpos'][0:1])[0]
            inputs.append(build_payload(encode_numpy_b64(obs['sensor_data/front_camera/rgb'][0]),
                                        encode_numpy_b64(obs['sensor_data/hand_camera/rgb'][0]),
                                        prompt=format_prompt_for_inference(row['prompt']), state=state.tolist()))
    return inputs


def reuse_checked_runtime(root, reference, output):
    """Use the validated FFN/cache path, warming it with one native observation."""
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from openwam.deploy.optimizations import DiTVelocityCache
    engine = root.engine
    engine.architecture.apply_compile_optimizations(OmegaConf.create({'enabled': True, 'sidecar_strategy': 'ffn'}))
    engine._dit_cache = DiTVelocityCache(cosine_threshold=.99, max_consecutive_skips=3)
    root._init_policy()
    payload = json.loads((reference/'server_manifest/validation/inputs.json').read_text())[0]
    obs = root._obs_preprocessor.preprocess(payload)
    with torch.inference_mode():
        actions = engine.generate(root._policy._build_conditions(obs))['actions']
    actions = actions.detach().float().cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
    if actions.shape != (32, 10) or not np.isfinite(actions).all():
        raise ValueError('Checkpoint smoke check requires finite physical EEF10 actions')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(passed=True, selected_strategy='ffn', mode='reuse_with_one_smoke_inference',
                  reference=str(reference), action_shape=list(actions.shape))
    (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    root.reset()
    return report


def main():
    import os
    import time

    import torch
    from omegaconf import OmegaConf

    from benchmarks.robotwin.check_sidecar_fast_runtime import validate_server
    from benchmarks.robotwin.sidecar_fast_runtime import install

    install()
    from openwam.model.architectures.dual_system.relational_sidecar import RelationalSidecarArchitecture

    place = RelationalSidecarArchitecture.set_dtype_device
    minimum_free = float(os.environ.get('MANISKILL_MIN_FREE_GIB', '0'))

    def place_when_capacity_available(self, dtype, device):
        if str(device).startswith("cuda") and minimum_free > 0:
            while torch.cuda.mem_get_info()[0] < minimum_free * 2**30:
                print(f"Waiting for >={minimum_free:g}GiB free before Sidecar CUDA placement", flush=True)
                time.sleep(10)
        return place(self, dtype, device)

    RelationalSidecarArchitecture.set_dtype_device = place_when_capacity_available
    import openwam.deploy.server as server_module

    original = server_module.build_server_from_config
    output = Path(sys.argv[sys.argv.index("--manifest") + 1]).with_suffix("")

    def checked_builder(cfg, *args, **kwargs):
        root = original(cfg, *args, **kwargs)
        arch = root.engine.architecture
        if not getattr(arch, "sidecar_enabled", False) or root.cfg.dataloader.type not in ("maniskill_sidecar", "maniskill_archive"):
            raise ValueError("Need a full enabled ManiSkill Sidecar checkpoint")
        plan_path = os.environ.get('OPENWAM_MANISKILL_EVAL_PLAN')
        if root.cfg.dataloader.type == 'maniskill_archive' and not plan_path:
            raise ValueError('Archive policy requires OPENWAM_MANISKILL_EVAL_PLAN')
        reference = os.environ.get('OPENWAM_MANISKILL_VALIDATED_RUN')
        if reference:
            report = reuse_checked_runtime(root, Path(reference), output/'validation')
        else:
            fixtures = archive_validation_inputs(plan_path) if plan_path else validation_inputs(
                ROOT / "outputs/maniskill_baseline/data_absolute_eef_20261017/manifest.ready.json")
            report = validate_server(root, output / "validation", strategies=("ffn",), inputs=fixtures)
        if not report["passed"] or report["selected_strategy"] != "ffn":
            raise RuntimeError("Sidecar FFN validation failed; no silent fallback")
        if not reference and any(len(a[0]) != 10 for a in report["variants"]["ffn_cached"]["actions"]):
            raise ValueError("Expected physical EEF10 outputs")
        (output / "sidecar_contract.json").write_text(
            json.dumps(
                dict(
                    enabled=True,
                    tap_blocks=list(arch.tap_blocks),
                    write_gates=[float(s.write.beta.detach().float().cpu()) for s in arch.sidecar.stages],
                    inference_horizon=int(root.cfg.inference.inference_horizon),
                    compile_strategy="ffn",
                    cache_cosine_threshold=0.99,
                    cache_max_skips=3,
                ),
                indent=2,
            )
            + "\n"
        )
        # Preserve replan8; validation generates full 32-action chunks directly.
        for target in (cfg, root.cfg, root.engine.cfg):
            OmegaConf.update(target, "optimization.compile.sidecar_strategy", "ffn")
        root.reset()
        print("MANISKILL_SIDECAR_VALIDATED_AND_SERVING", flush=True)
        return root

    server_module.build_server_from_config = checked_builder
    from benchmarks.robotwin.serve_parallel import main as serve

    serve()


if __name__ == "__main__":
    main()
