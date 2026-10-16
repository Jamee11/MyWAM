"""Independent synchronous policy endpoints sharing one serialized OpenWAM engine."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import signal
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=19000)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--readiness-token", required=True)
    args = parser.parse_args()
    if args.slots < 1 or not 1 <= args.port <= args.port + args.slots - 1 <= 65535:
        parser.error("invalid port range")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    import torch
    from omegaconf import OmegaConf

    from openwam.deploy.server import PolicyServer, build_server_from_config

    config = ROOT / "configs/deploy.yaml"
    cfg = OmegaConf.load(config)
    if cfg.inference.inference_mode != "sync" or cfg.inference.denoise_mode != "sync":
        raise ValueError("Shared evaluation endpoints require the official synchronous configuration")
    root = build_server_from_config(cfg, str(args.ckpt_dir.absolute()), device="cuda:0")
    # Inactive allocator blocks are not model weights; release them before simulators start.
    torch.cuda.empty_cache()
    logging.info("GPU allocated=%.3fGiB reserved=%.3fGiB", torch.cuda.memory_allocated() / 2**30,
                 torch.cuda.memory_reserved() / 2**30)
    lock = threading.Lock()
    failed = threading.Event()

    class Endpoint(PolicyServer):
        def predict(self, observation):
            # Includes GPU→CPU action materialization. No model calls overlap,
            # and the engine resets its DiT cache at every generate().
            with lock:
                return super().predict(observation)

        def _pong_payload(self):
            return {**super()._pong_payload(), "pool_slot": self.slot,
                    "pool_size": args.slots, "shared_engine_serialized": True}

    def run_slot(index):
        try:
            endpoint = Endpoint(root.engine, root.cfg, readiness_token=args.readiness_token)
            endpoint.slot = index
            endpoint.run(host="127.0.0.1", port=args.port + index)
        except BaseException:
            logging.exception("Endpoint %d exited", index)
            failed.set()

    record = {"pid": os.getpid(), "checkpoint_dir": str(args.ckpt_dir.resolve()),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "ports": list(range(args.port, args.port + args.slots)),
              "readiness_token": args.readiness_token,
              "deploy_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "session_state": "one PolicyServer/WAMPolicy/action buffer per port",
              "model_execution": "serialized under one lock; original generate path and parameters",
              "gpu_allocated_bytes": torch.cuda.memory_allocated(),
              "gpu_reserved_bytes": torch.cuda.memory_reserved()}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(record, indent=2) + "\n")
    for index in range(args.slots):
        threading.Thread(target=run_slot, args=(index,), daemon=True, name=f"policy-slot-{index}").start()
    while not failed.wait(1):
        pass
    raise RuntimeError("An endpoint failed; stopping the shared model process")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main()
