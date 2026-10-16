"""OpenWAM-alpha joint denoising with a detached observation-only EA bottleneck."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.architectures.registry import register_architecture
from openwam.model.modules.ea_bridge import EARSIGBridge


@register_architecture(
    "dual_system_ea_rsig_v3", status="supported",
    note="Alpha joint self-attention plus detached current-observation EA-RSIGv3.",
    framework="dual_system", variant="joint_self_attn_ea_rsig_v3",
)
class EARSIGV3Architecture(DualSystemSelfAttnArchitecture):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        if cfg is None:
            return
        ea = dict(cfg.get("ea_rsig_v3", {}))
        self.foundation_warmstart = bool(ea.pop("foundation_warmstart", False))
        self.logical_loss_batch_size = ea.pop("logical_loss_batch_size", None)
        if self.video_backbone is None:
            raise ValueError("EA-RSIG requires a video backbone")
        self.ea_bridge = EARSIGBridge(self.video_backbone.dim, self.context_dim, **ea)
        if self.ea_bridge.layer_index >= self.video_backbone.num_layers:
            raise ValueError("EA observation layer exceeds video backbone depth")
        self._ea_training_output = None

    def set_dtype_device(self, dtype, device):
        super().set_dtype_device(dtype, device)
        if getattr(self, "ea_bridge", None) is not None:
            self.ea_bridge.to(device=device, dtype=dtype)

    def apply_compile_optimizations(self, compile_cfg):
        # The observation pass and training residual are explicit eager paths.
        self._compiled_mot_run_joint_loop = None

    def prepare_inputs(self, batch):
        samples = [batch] if isinstance(batch, dict) else batch
        for sample in samples:
            if sample.get("ea_action_space") != self.ea_bridge.action_space:
                raise ValueError("EA sample action-space mismatch")
            if sample.get("ea_video_anchors") != list(self.ea_bridge.video_anchors):
                raise ValueError("EA sample video/plan anchor mismatch")
            if sample["action"].shape != (self.ea_bridge.action_horizon, self.action_dim):
                raise ValueError("EA action shape/horizon mismatch")
            if len(sample["video"]) != len(self.ea_bridge.video_anchors):
                raise ValueError("EA video length mismatch")
        inputs = super().prepare_inputs(batch)
        for key in ("ea_effectors", "ea_arm_present", "ea_targets"):
            if any(key not in s for s in samples):
                raise ValueError(f"Missing EA field: {key}")
            inputs[key] = torch.stack([torch.as_tensor(s[key]) for s in samples]).to(self.device, self.dtype)
        if inputs["ea_targets"].shape != (len(samples), 1, self.ea_bridge.core.target_dim):
            raise ValueError("Invalid EA targets; privileged labels must be separate")
        return inputs

    def _observe(self, pipeline_inputs, effectors, arms):
        clean = pipeline_inputs.get("first_frame_latents")
        if clean is None:
            raise ValueError("EA needs clean first_frame_latents, never noisy future latents")
        if effectors is None or arms is None:
            raise ValueError("EA needs deployable current EEF state and arm mask")
        if effectors.shape != (clean.shape[0], 2, 10) or arms.shape != (clean.shape[0], 2):
            raise ValueError("Invalid deployable EA context")
        # Whitelist observation/text inputs: exclude action, future, and oracle fields.
        obs_inputs = {k: pipeline_inputs[k] for k in ("context", "context_mask", "seq_lens") if k in pipeline_inputs}
        obs_inputs.update(latents=clean[:, :, :1], first_frame_latents=clean[:, :, :1],
                          timestep=torch.zeros(clean.shape[0], device=clean.device, dtype=clean.dtype),
                          fuse_vae_embedding_in_latents=True, force_per_token_t_mod=True,
                          zero_clean_prefix_t_mod=True)
        with torch.no_grad():
            state = self.video_backbone.prepare(**obs_inputs)
            for layer in range(self.ea_bridge.layer_index + 1):
                state = self.video_backbone.run_block(layer, state)
            hidden = state.hidden_states.detach()
        grid = (state.grid_frames, state.grid_height, state.grid_width)
        return self.ea_bridge.observe(hidden, grid, effectors, arms)

    def forward(self, noisy_actions, action_timestep, *, proprio=None,
                use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False,
                ea_effectors=None, ea_arm_present=None, ea_observation=None, **pipeline_inputs):
        if "ea_targets" in pipeline_inputs:
            raise ValueError("Privileged EA targets must not enter forward")
        output = ea_observation if ea_observation is not None else self._observe(
            pipeline_inputs, ea_effectors, ea_arm_present)
        self._ea_training_output = output if self.training and torch.is_grad_enabled() else None
        pipeline_inputs = self._append_proprio_context_token(dict(pipeline_inputs), proprio)
        pipeline_inputs.setdefault("force_per_token_t_mod", True)
        pipeline_inputs.setdefault("zero_clean_prefix_t_mod", True)
        vb, ab = self.video_backbone, self.action_backbone
        vs = vb.prepare(use_gradient_checkpointing=use_gradient_checkpointing,
                        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload, **pipeline_inputs)
        context = pipeline_inputs.get("context")
        mask = pipeline_inputs.get("context_mask")
        if context is not None and mask is None and pipeline_inputs.get("seq_lens") is not None:
            mask = torch.arange(context.shape[1], device=context.device)[None] < pipeline_inputs["seq_lens"].to(context.device)[:, None]
        if self.ea_bridge.action_conditioning:
            extended = self.ea_bridge.append_context({"context": context, "mask": mask}, output["condition_tokens"])
            context, mask = extended["context"], extended["mask"]
        grid = (vs.grid_frames, vs.grid_height, vs.grid_width)
        if noisy_actions is None:
            for layer in range(vb.num_layers):
                vs = vb.run_block(layer, vs)
                if layer == self.ea_bridge.layer_index and self.training and torch.is_grad_enabled():
                    vs.hidden_states = self.ea_bridge.inject_future(vs.hidden_states, grid, output)
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
            if layer == self.ea_bridge.layer_index and self.training and torch.is_grad_enabled():
                vs.hidden_states = self.ea_bridge.inject_future(vs.hidden_states, grid, output)
        return vb.finalize(vs), ab.extract_prediction(ast)

    def compute_loss(self, *, ea_targets=None, **inputs):
        if ea_targets is None:
            raise ValueError("Training needs EA supervision separately from model inputs")
        self._ea_training_output = None
        try:
            result = super().compute_loss(**inputs)
            if self._ea_training_output is None:
                raise RuntimeError("EA forward did not produce auxiliary representation")
            parts = self.ea_bridge.core.losses(self._ea_training_output, ea_targets,
                                              logical_batch_size=self.logical_loss_batch_size)
            extra = self.ea_bridge.loss_scale * sum(self.ea_bridge.loss_weights[k] * v for k, v in parts.items())
            result["loss"] = result["loss"] + extra
            result.update(loss_ea=extra.detach(), **{f"loss_ea_{k}": v.detach() for k, v in parts.items()})
            return result
        finally:
            self._ea_training_output = None

    def generate(self, *, ea_effectors=None, ea_arm_present=None, **kwargs):
        if kwargs.get("cfg_scale", 1.0) != 1.0:
            raise ValueError("EA deploy currently requires cfg_scale=1; paired context routing is unverified")
        if kwargs.get("dit_cache") is not None:
            raise ValueError("EA deploy currently requires dit_cache=None")
        if ea_effectors is None or ea_arm_present is None:
            raise ValueError("EA generation requires deployable EEF/arm context")
        result = super().generate(ea_effectors=ea_effectors, ea_arm_present=ea_arm_present, **kwargs)
        if self.ea_bridge.action_space in ("qpos", "robot_base_delta_ee"):
            result["actions"][..., [6, 13]] = result["actions"][..., [6, 13]] > 0.5
        return result

    def normalize_deploy_proprio(self, proprio):
        value = super().normalize_deploy_proprio(proprio)
        value[..., [6, 13]] = (value[..., [6, 13]] > 0.5).to(value.dtype)
        return value

    def load_checkpoint(self, path, strict=True):
        from safetensors.torch import load_file
        state = load_file(path)
        if any(k.startswith("ea_bridge.") for k in state):
            return super().load_checkpoint(path, strict=strict)
        if not self.foundation_warmstart:
            raise ValueError("Foundation-to-EA load requires explicit foundation_warmstart=true")
        own = self.state_dict()
        # Native 14D policy heads replace Alpha's 80D heads; all experts stay inherited.
        resize = {"action_backbone.action_encoder.weight", "action_backbone.action_decoder.weight",
                  "action_backbone.action_decoder.bias", "proprio_encoder.weight"}
        skipped = {k for k in resize if k in state and k in own and state[k].shape != own[k].shape}
        expected = {k for k in own if k.startswith("ea_bridge.")} | skipped
        missing = set(own) - (set(state) - skipped)
        unexpected = set(state) - set(own)
        bad_shapes = [k for k in set(own) & set(state) - skipped if own[k].shape != state[k].shape]
        if missing != expected or unexpected or bad_shapes:
            raise RuntimeError(f"Invalid foundation warmstart: missing={missing - expected}, unexpected={unexpected}, shapes={bad_shapes}")
        has_meta = any(p.device.type == "meta" for p in self.parameters())
        self.load_state_dict({k: v for k, v in state.items() if k not in skipped}, strict=False, assign=has_meta)
        if any(p.device.type == "meta" for p in self.parameters()):
            raise RuntimeError("Unmaterialized parameters after warmstart")
        report = {"source": str(Path(path).resolve()), "loaded_tensors": len(state) - len(skipped),
                  "reset_policy_heads": sorted(skipped), "new_ea_tensors": len(expected - skipped)}
        print("EA foundation warmstart: " + json.dumps(report), flush=True)
        self.foundation_load_report = report
