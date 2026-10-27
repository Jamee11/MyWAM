"""Real-observation eager/compiled/cache comparisons before accelerated evaluation."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def validate_server(server, output, *, strategies=("conservative", "sidecar", "ffn"), inputs=None):
    if not strategies or len(set(strategies)) != len(strategies) or set(strategies) - {"conservative", "sidecar", "ffn"}:
        raise ValueError("Select unique supported Sidecar compile strategies")
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from torch._inductor import metrics

    from benchmarks.utils.action_conversion import quat_xyzw_to_rot6d, rot6d_to_quat_xyzw
    from openwam.deploy.optimizations import DiTVelocityCache

    output.mkdir(parents=True, exist_ok=False)
    server._init_policy()
    engine = server.engine
    # These fixtures were built with the released model's legacy xyzw reader.
    # Recover the raw quaternion up to sign, then apply Sidecar's wxyz contract.
    provided_inputs = inputs is not None
    if inputs is None:
        inputs = json.loads((ROOT / "outputs/robotwin_eval/rebuild_official_20261018/action_parity/inputs.json").read_text())
        for payload in inputs:
            for offset in (0, 10):
                q = rot6d_to_quat_xyzw(np.array(payload["state"][offset + 3:offset + 9]))
                payload["state"][offset + 3:offset + 9] = quat_xyzw_to_rot6d(q[[1, 2, 3, 0]]).tolist()
    if len(inputs) != 2:
        raise ValueError("Validation requires two observation fixtures")
    (output / "inputs.json").write_text(json.dumps(inputs))
    report = {"scope": "two supplied native observation fixtures, seed42 and 10 steps" if provided_inputs else
              "two real RGB fixtures, wxyz-trained state contract, identical seed42 and 10 steps",
              "variants": {}, "comparisons": []}
    outputs = {}
    variants = [("eager", None, False), ("eager_cached", None, True)]
    variants += [(name + ("_cached" if cached else ""), name, cached)
                 for name in strategies for cached in [False, True]]
    for name, strategy, cached in variants:
        kernels_before = metrics.generated_kernel_count
        print("VARIANT", name, flush=True)
        engine.architecture.apply_compile_optimizations(OmegaConf.create({"enabled": strategy is not None, "sidecar_strategy": strategy}))
        engine._dit_cache = DiTVelocityCache(cosine_threshold=0.99, max_consecutive_skips=3) if cached else None
        values, times, stats = [], [], []
        for payload in inputs:
            server.reset()
            obs = server._obs_preprocessor.preprocess(payload)
            conditions = server._policy._build_conditions(obs)
            # First call includes cold compilation/text/VAE setup; record separately.
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                result = engine.generate(conditions)
            torch.cuda.synchronize()
            cold = time.perf_counter() - start
            for _ in range(3):
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    result = engine.generate(conditions)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
            action = result["actions"]
            action = action.detach().float().cpu().numpy() if torch.is_tensor(action) else np.asarray(action)
            if not np.isfinite(action).all():
                raise ValueError("Nonfinite actions")
            values.append(action)
            stats.append({"cold_s": cold, "cache": engine._dit_cache.stats if cached else None})
        outputs[name] = values
        report["variants"][name] = {"warm_seconds": times, "mean_warm_s": float(np.mean(times)),
                                    "fixtures": stats, "actions": [a.tolist() for a in values],
                                    "generated_kernels": metrics.generated_kernel_count - kernels_before}
        for reference in (["eager_cached", "eager"] if strategy and cached else ["eager"] if strategy else []):
            for i, (a, b) in enumerate(zip(values, outputs[reference])):
                report["comparisons"].append({"variant": name, "reference": reference, "fixture": i,
                    "max_abs_error": float(np.max(np.abs(a - b))),
                    "allclose_atol_005_rtol_01": bool(np.allclose(a, b, atol=0.005, rtol=0.01))})
        (output / "progress.json").write_text(json.dumps(report, indent=2))
    from torch._dynamo.utils import counters

    report["compile_evidence"] = {"stats": dict(counters["stats"]), "generated_kernel_count": metrics.generated_kernel_count}
    report["strategy_results"] = {}
    for strategy in strategies:
        checks = [c for c in report["comparisons"] if
                  (c["variant"], c["reference"]) in [(strategy, "eager"), (strategy + "_cached", "eager_cached")]]
        generated = sum(report["variants"][strategy + suffix]["generated_kernels"] for suffix in ["", "_cached"])
        report["strategy_results"][strategy] = {
            "passed": len(checks) == 4 and all(c["allclose_atol_005_rtol_01"] for c in checks) and generated > 0,
            "generated_kernels": generated,
            "warm_cached_s": report["variants"][strategy + "_cached"]["mean_warm_s"],
        }
    candidates = [(v["warm_cached_s"], k) for k, v in report["strategy_results"].items() if v["passed"]]
    report["selected_strategy"] = min(candidates)[1] if candidates else None
    report["passed"] = bool(candidates)
    if candidates:
        engine.architecture.apply_compile_optimizations(OmegaConf.create(
            {"enabled": True, "sidecar_strategy": report["selected_strategy"]}))
        engine._dit_cache = DiTVelocityCache(cosine_threshold=0.99, max_consecutive_skips=3)
        server.reset()
    (output / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": report["passed"], "compile_evidence": report["compile_evidence"],
                      "comparisons": report["comparisons"]}), flush=True)
    return report


def main():
    import argparse

    from omegaconf import OmegaConf

    from benchmarks.robotwin.sidecar_fast_runtime import install

    install()
    from openwam.deploy.server import build_server_from_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(ROOT / "configs/deploy_relational_sidecar.yaml")
    checkpoint = ROOT / "outputs/robotwin_eval/sidecar20000_currentenv_20261018/checkpoint_step_20000"
    server = build_server_from_config(cfg, str(checkpoint), device="cuda:0")
    report = validate_server(server, args.output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
