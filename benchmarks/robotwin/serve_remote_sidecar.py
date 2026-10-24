"""Use the already-validated Sidecar FFN/cache path for split-host evaluation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from benchmarks.robotwin.sidecar_fast_runtime import install
    install()
    import openwam.deploy.server as server_module
    original = server_module.build_server_from_config

    def checked(*args, **kwargs):
        server = original(*args, **kwargs)
        from benchmarks.robotwin.check_sidecar_fast_runtime import validate_server
        out = Path(sys.argv[sys.argv.index("--manifest") + 1]).parent
        report = validate_server(server, out / "fast_validation", strategies=("ffn",))
        if not report["passed"] or report["selected_strategy"] != "ffn":
            raise RuntimeError("Sidecar FFN numerical/kernel gate failed; no silent fallback")
        server.reset()
        return server

    server_module.build_server_from_config = checked
    from benchmarks.robotwin.serve_parallel import main as serve
    serve()


if __name__ == "__main__":
    main()
