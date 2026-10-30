"""Validate alternative Sidecar compile scopes, then serve the same loaded model."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from omegaconf import OmegaConf

    from benchmarks.robotwin.check_sidecar_fast_runtime import validate_server
    from benchmarks.robotwin.sidecar_fast_runtime import install

    install()
    import openwam.deploy.server as server_module

    original = server_module.build_server_from_config
    config_path = Path(sys.argv[sys.argv.index("--config") + 1])
    out = Path(sys.argv[sys.argv.index("--manifest") + 1]).parent

    def checked_builder(cfg, *args, **kwargs):
        root = original(cfg, *args, **kwargs)
        report = validate_server(root, out / "fast_validation")
        if not report["passed"]:
            raise RuntimeError("All compile scopes failed the original numerical gate; full results preserved")
        strategy = report["selected_strategy"]
        for target in [cfg, root.cfg, root.engine.cfg]:
            OmegaConf.update(target, "optimization.compile.enabled", True)
            OmegaConf.update(target, "optimization.compile.sidecar_strategy", strategy)
            OmegaConf.update(target, "optimization.dit_cache.enabled", True)
        OmegaConf.save(cfg, config_path)
        root.reset()
        print(f"VALIDATED_AND_SERVING strategy={strategy} config={config_path}", flush=True)
        return root

    server_module.build_server_from_config = checked_builder
    from benchmarks.robotwin.serve_parallel import main as serve

    serve()


if __name__ == "__main__":
    main()
