"""Serve one fixed ablation per process, with the existing numerical gate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from omegaconf import OmegaConf

    from benchmarks.robotwin.check_sidecar_fast_runtime import validate_server
    from benchmarks.robotwin.sidecar_ablation import MODES, configure
    from benchmarks.robotwin.sidecar_fast_runtime import install

    install()
    import openwam.deploy.server as server_module

    original = server_module.build_server_from_config
    out = Path(sys.argv[sys.argv.index("--manifest") + 1]).parent

    def checked_builder(cfg, *args, **kwargs):
        mode = OmegaConf.select(cfg, "evaluation.sidecar_ablation")
        if mode not in MODES:
            raise ValueError(f"Select an explicit Sidecar ablation mode: {MODES}")
        root = original(cfg, *args, **kwargs)
        intervention = configure(root.engine.architecture, mode)
        report = validate_server(root, out / "fast_validation", strategies=("ffn",))
        checks = [c for c in report["comparisons"] if
                  (c["variant"], c["reference"]) in [("ffn", "eager"), ("ffn_cached", "eager_cached")]]
        passed = report["passed"]
        if mode == "off":
            # No FFN executes in the bypassed branch: require parity, not kernels.
            passed = (len(checks) == 4 and all(c["allclose_atol_005_rtol_01"] for c in checks)
                      and report["strategy_results"]["ffn"]["generated_kernels"] == 0)
        intervention.update(passed=passed, compilation_active=mode != "off",
                            numerical_gate="same atol=.005/rtol=.01 against each mode's own eager reference")
        (out / "ablation_validation.json").write_text(json.dumps(intervention, indent=2) + "\n")
        if not passed:
            raise RuntimeError(f"Ablation numerical gate failed: {mode}")
        if not OmegaConf.select(cfg, "optimization.dit_cache.enabled", default=False):
            root.engine._dit_cache = None
        root.reset()
        print(f"ABLATION_VALIDATED_AND_SERVING mode={mode}", flush=True)
        return root

    server_module.build_server_from_config = checked_builder
    from benchmarks.robotwin.serve_parallel import main as serve

    serve()


if __name__ == "__main__":
    main()
