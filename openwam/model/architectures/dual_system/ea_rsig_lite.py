"""Alpha-compatible EEF architecture with causal EA-RSIG-Lite conditioning."""
from __future__ import annotations

import torch

from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.architectures.registry import register_architecture
from openwam.model.modules.ea_rsig_lite import EARSIGLiteBridge


@register_architecture("dual_system_ea_rsig_lite", status="supported",
                       note="Causal current patch EA, zero-init Action residual; Alpha 80D EEF.",
                       framework="dual_system", variant="joint_self_attn_ea_rsig_lite")
class EARSIGLiteArchitecture(DualSystemSelfAttnArchitecture):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        if cfg is None:
            return
        if self.action_dim != 80 or self.proprio_dim != 80:
            raise ValueError("Lite preserves the Foundation 80D action/state heads")
        ea = dict(cfg.get("ea_lite", {}))
        self.foundation_warmstart = bool(ea.get("foundation_warmstart", False))
        self.adapter_layer = int(ea.get("adapter_layer", 0))
        if not 0 <= self.adapter_layer < self.video_backbone.num_layers:
            raise ValueError("Invalid Action block boundary")
        self.loss_weights = {k: float(ea.get(k + "_weight", v))
                             for k, v in (("ground", .025), ("relation", .01), ("plan", .05))}
        self.ea_bridge = EARSIGLiteBridge(
            self.video_backbone.dim, self.action_backbone.dim, self.context_dim,
            **{k: ea[k] for k in ("slot_dim", "num_heads", "grid_height", "view_grid_width",
                                  "action_conditioning") if k in ea})
        self.training_stage = ea.get("stage", "warmup")
        if self.training_stage not in ("warmup", "joint", "action_only"):
            raise ValueError("Lite stage must be warmup, joint, or action_only")
        if self.training_stage != "joint":
            for name, parameter in self.named_parameters():
                if not name.startswith("ea_bridge.") and (
                    self.training_stage == "warmup" or not name.startswith("action_backbone.")):
                    parameter.requires_grad_(False)
        self._ea_training_output = None
        self._capture_ea_output = False

    def set_dtype_device(self, dtype, device):
        super().set_dtype_device(dtype, device)
        if hasattr(self, "ea_bridge"):
            self.ea_bridge.to(device=device, dtype=dtype)

    def apply_compile_optimizations(self, compile_cfg):
        self._compiled_mot_run_joint_loop = None

    def prepare_inputs(self, batch):
        samples = [batch] if isinstance(batch, dict) else batch
        for sample in samples:
            if sample.get("ea_plan_steps") != [8, 16, 32] or sample.get("ea_video_anchors") != list(range(0, 33, 4)):
                raise ValueError("Lite needs 32 actions / 9 video frames / plan8,16,32")
            if sample["action"].shape != (32, 80) or sample["proprio"].shape != (1, 80):
                raise ValueError("Lite needs official 80D EEF samples")
            if len(sample["video"]) != 9:
                raise ValueError("Lite needs nine sampled frames")
        inputs = super().prepare_inputs(batch)
        for key in ("ea_effectors", "ea_arm_present", "ea_targets"):
            if any(key not in sample for sample in samples):
                raise ValueError(f"Missing training EA field {key}")
            inputs[key] = torch.stack([torch.as_tensor(s[key]) for s in samples]).to(self.device, self.dtype)
        if inputs["ea_targets"].shape != (len(samples), 1, 130):
            raise ValueError("Lite targets must remain a separate training-only field")
        return inputs

    def _observe(self, pipeline_inputs, effectors, arms):
        clean = pipeline_inputs.get("first_frame_latents")
        if clean is None or clean.shape[2] != 1 or effectors is None or arms is None:
            raise ValueError("Lite needs clean current RGB latents and current EEF context")
        # Exactly one patch embedding; no Video DiT block, diffusion timestep or action.
        patches = self.video_backbone.dit.patchify(clean)
        context = pipeline_inputs["context"]
        mask = pipeline_inputs.get("context_mask")
        if mask is None and pipeline_inputs.get("seq_lens") is not None:
            lengths = torch.as_tensor(pipeline_inputs["seq_lens"], device=context.device)
            mask = torch.arange(context.shape[1], device=context.device)[None] < lengths[:, None]
        return self.ea_bridge.observe(patches, context, mask, effectors, arms)

    def forward(self, noisy_actions, action_timestep, *, proprio=None,
                use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False,
                ea_effectors=None, ea_arm_present=None, **pipeline_inputs):
        if "ea_targets" in pipeline_inputs:
            raise ValueError("Privileged targets must never enter forward")
        output = self._observe(pipeline_inputs, ea_effectors, ea_arm_present)
        self._ea_training_output = output if self._capture_ea_output else None
        pipeline_inputs = self._append_proprio_context_token(dict(pipeline_inputs), proprio)
        context, mask = pipeline_inputs.get("context"), pipeline_inputs.get("context_mask")
        if mask is None and pipeline_inputs.get("seq_lens") is not None:
            lengths = torch.as_tensor(pipeline_inputs["seq_lens"], device=context.device)
            mask = torch.arange(context.shape[1], device=context.device)[None] < lengths[:, None]
        pipeline_inputs.setdefault("force_per_token_t_mod", True)
        pipeline_inputs.setdefault("zero_clean_prefix_t_mod", True)
        vb, ab = self.video_backbone, self.action_backbone
        vs = vb.prepare(use_gradient_checkpointing=use_gradient_checkpointing,
                        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload, **pipeline_inputs)
        if noisy_actions is None:
            for layer in range(vb.num_layers):
                vs = vb.run_block(layer, vs)
            return vb.finalize(vs), None
        ast = ab.prepare_state(noisy_actions, action_timestep, context=context, context_mask=mask,
                               use_gradient_checkpointing=use_gradient_checkpointing,
                               use_gradient_checkpointing_offload=use_gradient_checkpointing_offload)
        driver = self.mot_driver or self.build_mot_driver()
        joint_mask = driver._build_attention_mask(vs.hidden_states.shape[1], ast.payload.x_action.shape[1],
                                                 int(vs.grid_height) * int(vs.grid_width), device=vs.hidden_states.device)
        for layer in range(vb.num_layers):
            vs, ast = driver.step(layer, vs, ast, attn_mask=joint_mask,
                                  use_gradient_checkpointing=use_gradient_checkpointing,
                                  use_gradient_checkpointing_offload=use_gradient_checkpointing_offload)
            if layer == self.adapter_layer:
                ast.payload.x_action = self.ea_bridge.condition_action(ast.payload.x_action, output["action_tokens"])
        return vb.finalize(vs), ab.extract_prediction(ast)

    def compute_loss(self, *, ea_targets=None, **inputs):
        if ea_targets is None or ea_targets.shape[-2:] != (1, 130):
            raise ValueError("Training requires separate Lite targets")
        self._ea_training_output = None
        self._capture_ea_output = True
        try:
            result = super().compute_loss(**inputs)
            parts = self.ea_bridge.core.losses(self._ea_training_output, ea_targets)
            extra = sum(self.loss_weights[k] * value for k, value in parts.items())
            result["loss"] = result["loss"] + extra
            result.update(loss_ea=extra.detach(), **{f"loss_ea_{k}": v.detach() for k, v in parts.items()})
            labels = ea_targets[:, 0].detach()
            result.update(ea_valid_track_fraction=labels[:, :2].float().mean(),
                          ea_valid_ground_fraction=labels[:, 118:130].float().mean(),
                          ea_valid_plan_fraction=labels[:, 64:70].float().mean(),
                          ea_moving_fraction=labels[:, 52:64].float().mean(),
                          ea_track_gate=self._ea_training_output["track_gate"].detach().float().mean(),
                          ea_adapter_out_norm=self.ea_bridge.action_out.weight.detach().float().norm())
            return result
        finally:
            self._ea_training_output = None
            self._capture_ea_output = False

    def generate(self, *, ea_effectors=None, ea_arm_present=None, **kwargs):
        if "ea_targets" in kwargs:
            raise ValueError("Privileged targets are forbidden during generation")
        if kwargs.get("cfg_scale", 1) != 1 or kwargs.get("dit_cache") is not None:
            raise ValueError("Lite readiness path requires CFG=1 and DiT cache disabled")
        if ea_effectors is None or ea_arm_present is None:
            raise ValueError("Lite deployment needs current EEF/arms, never targets")
        return super().generate(ea_effectors=ea_effectors, ea_arm_present=ea_arm_present, **kwargs)

    def load_checkpoint(self, path, strict=True):
        from safetensors.torch import load_file
        state = load_file(path)
        if any(k.startswith("ea_bridge.") for k in state):
            return super().load_checkpoint(path, strict=strict)
        if not self.foundation_warmstart:
            raise ValueError("Explicit foundation_warmstart is required")
        own = self.state_dict()
        missing = set(own) - set(state)
        expected = {key for key in own if key.startswith("ea_bridge.")}
        if missing != expected or set(state) - set(own):
            raise RuntimeError("Foundation keys do not match unchanged Alpha heads/backbones")
        if any(own[k].shape != value.shape for k, value in state.items()):
            raise RuntimeError("Lite must not resize or reset Foundation policy heads")
        self.load_state_dict(state, strict=False, assign=any(p.is_meta for p in self.parameters()))
        if any(p.is_meta for p in self.parameters()):
            raise RuntimeError("Unmaterialized parameters")
        self.foundation_load_report = dict(loaded_tensors=len(state), reset_policy_heads=[],
                                           new_ea_tensors=len(expected))

    def save_assets_for_deployment(self, output_path, cfg):
        import json
        from pathlib import Path
        super().save_assets_for_deployment(output_path, cfg)
        contract = dict(version=1, action_dim=80, raw_eef_dim=20, action_steps=32,
                        video_frames=9, plan_steps=[8, 16, 32], current_feature="clean_vae_patchify",
                        action_adapter_boundary=self.adapter_layer, joint_sequence_extra_tokens=0,
                        source_quaternion_order="wxyz", ee_to_tcp_local_x_m=.12,
                        attention_mode="official_mutual", video_residual=False,
                        loss_weights=self.loss_weights)
        Path(output_path, "ea_lite_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
