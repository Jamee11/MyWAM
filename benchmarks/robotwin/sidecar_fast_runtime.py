"""Opt-in evaluation-only Sidecar acceleration; training/default deploy stay unchanged.

Compile the complete Sidecar forward, never the foundation-only MoT loop.
Reuse the existing joint video/action velocity cache, reset by the engine for
each generation. Caching is approximate and must be reported as such.
"""
from __future__ import annotations

import logging


def install():
    import torch

    from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
    from openwam.model.architectures.dual_system.relational_sidecar import RelationalSidecarArchitecture
    from openwam.model.compile_options import compile_enabled

    cls = RelationalSidecarArchitecture
    if getattr(cls, "_evaluation_fast_runtime_installed", False):
        return

    def compile_sidecar(self, compile_cfg):
        self._compiled_mot_run_joint_loop = None
        if not hasattr(self, "_evaluation_eager_forward"):
            self._evaluation_eager_forward = self.forward
            self._evaluation_eager_read_refine = self.sidecar.read_refine
            self._evaluation_eager_write_action = self.sidecar.write_action
            self._evaluation_eager_sidecar_ffns = [stage.ffn for stage in self.sidecar.stages]
        self.forward = self._evaluation_eager_forward
        self.sidecar.read_refine = self._evaluation_eager_read_refine
        self.sidecar.write_action = self._evaluation_eager_write_action
        for stage, ffn in zip(self.sidecar.stages, self._evaluation_eager_sidecar_ffns):
            stage.ffn = ffn
        self._evaluation_compile_enabled = compile_enabled(compile_cfg, default=False, strict=True)
        if self._evaluation_compile_enabled:
            options = {"emulate_precision_casts": True, "force_same_precision": True,
                       "pattern_matcher": False, "epilogue_fusion": False, "max_fusion_size": 1}

            def compile_fn(fn):
                return torch.compile(fn, dynamic=False, fullgraph=False, options=options)

            strategy = compile_cfg.get("sidecar_strategy", "conservative")
            if strategy == "conservative":
                self.forward = compile_fn(self._evaluation_eager_forward)
            elif strategy == "sidecar":
                self.sidecar.read_refine = compile_fn(self._evaluation_eager_read_refine)
                self.sidecar.write_action = compile_fn(self._evaluation_eager_write_action)
            elif strategy == "ffn":
                for stage, ffn in zip(self.sidecar.stages, self._evaluation_eager_sidecar_ffns):
                    stage.ffn = compile_fn(ffn)
            else:
                raise ValueError(f"Unknown Sidecar evaluation compile strategy: {strategy}")
            logging.info("Evaluation-only compile strategy=%s; all three Sidecar read/write taps retained", strategy)

    def generate(self, *, proprio=None, active_action_mask=None, **kwargs):
        if any(k in kwargs for k in ("ea_targets", "ea_effectors", "ea_arm_present")):
            raise ValueError("Privileged or legacy EA context is forbidden during Sidecar generation")
        if self.sidecar_enabled:
            if kwargs.get("cfg_scale", 1) != 1:
                raise ValueError("Sidecar evaluation requires CFG=1")
            inactive = self._resolve_inactive_action_dims(active_action_mask, self.device)
            if inactive is None:
                raise ValueError("Attach the EEF normalizer or provide active_action_mask")
            kwargs["_proprio_sample_mask"] = (~inactive)[None, None]
        return DualSystemSelfAttnArchitecture.generate(
            self, proprio=proprio, active_action_mask=active_action_mask, **kwargs,
        )

    cls.apply_compile_optimizations = compile_sidecar
    cls.generate = generate
    cls._evaluation_fast_runtime_installed = True


if __name__ == "__main__":
    import os
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    install()
    # CPU construction can overlap validation on this GPU. CUDA placement waits
    # for an explicit handover written after the validator exits successfully.
    if os.environ.get("OPENWAM_GPU_HANDOVER_FILE"):
        from openwam.model.architectures.dual_system.relational_sidecar import RelationalSidecarArchitecture

        handover = Path(os.environ["OPENWAM_GPU_HANDOVER_FILE"])
        original_place = RelationalSidecarArchitecture.set_dtype_device

        def place_after_handover(self, dtype, device):
            if str(device).startswith("cuda"):
                while not handover.is_file():
                    time.sleep(2)
            return original_place(self, dtype, device)

        RelationalSidecarArchitecture.set_dtype_device = place_after_handover
    from benchmarks.robotwin.serve_parallel import main

    main()
