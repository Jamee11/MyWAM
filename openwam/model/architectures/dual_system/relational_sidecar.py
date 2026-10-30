"""Multi-layer EA variant; original MoT blocks, masks, heads and FM objective are retained."""

import json
import math
from pathlib import Path

import torch

from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.architectures.registry import register_architecture
from openwam.model.modules.relational_sidecar import RelationalSidecar


@register_architecture(
    "dual_system_relational_sidecar",
    status="supported",
    framework="dual_system",
    variant="joint_self_attn_relational_sidecar",
    note="Persistent 3-stage observation-only EA.",
)
class RelationalSidecarArchitecture(DualSystemSelfAttnArchitecture):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        if cfg is None:
            return
        if self.action_dim != 80 or self.proprio_dim != 80:
            raise ValueError("Sidecar preserves the Foundation 80D action/state contract")
        if (
            self._mot_driver_kwargs["attention_mask_mode"] != "mutual"
            or self._mot_driver_kwargs["video_attention_mask_mode"] != "first_frame_causal"
        ):
            raise ValueError("Sidecar requires the unchanged mutual + first_frame_causal masks")
        ea = dict(cfg.get("relational_sidecar", {}))
        self.sidecar_enabled = bool(ea.get("enabled", True))
        self.tap_blocks = tuple(ea.get("tap_blocks", [7, 15, 23]))
        if (
            len(self.tap_blocks) != 3
            or tuple(sorted(set(self.tap_blocks))) != self.tap_blocks
            or not 0 <= self.tap_blocks[0] <= self.tap_blocks[-1] < self.video_backbone.num_layers
        ):
            raise ValueError("Need three increasing, valid zero-based tap blocks")
        self.foundation_warmstart = bool(ea.get("foundation_warmstart", False))
        self.deep_supervision = float(ea.get("deep_supervision", 0.1))
        self.aux_loss_scale = float(ea.get("aux_loss_scale", 1.0))
        self.isolation_warmup_updates = int(ea.get("isolation_warmup_updates", 0))
        self.backbone_warmup_updates = int(ea.get("backbone_warmup_updates", 0))
        self.aux_heads_only_after_warmup = bool(ea.get("aux_heads_only_after_warmup", False))
        self.aux_shared_gradient_scale = float(ea.get("aux_shared_gradient_scale", 1.0))
        self.aux_gradient_anneal_updates = ea.get("aux_gradient_anneal_updates", 0)
        self.main_task_gradient_gain = float(ea.get("main_task_gradient_gain", 1.0))
        if not math.isfinite(self.main_task_gradient_gain) or not 1 <= self.main_task_gradient_gain <= 4:
            raise ValueError("Main-task token gradient gain must be finite and in [1,4]")
        if not math.isfinite(self.aux_shared_gradient_scale) or not 0 <= self.aux_shared_gradient_scale <= 1:
            raise ValueError("Shared EA gradient scale must be finite and between zero and one")
        if type(self.aux_gradient_anneal_updates) is not int or self.aux_gradient_anneal_updates < 0:
            raise ValueError("EA gradient anneal updates must be a nonnegative integer")
        if self.aux_heads_only_after_warmup and (
            self.aux_shared_gradient_scale != 1.0 or self.aux_gradient_anneal_updates
        ):
            raise ValueError("Heads-only and shared EA gradient annealing cannot both be enabled")
        self.completed_optimizer_updates = 0
        if not math.isfinite(self.aux_loss_scale) or self.aux_loss_scale < 0 or self.isolation_warmup_updates < 0:
            raise ValueError("Sidecar loss scale and isolation warmup must be finite/nonnegative")
        if self.backbone_warmup_updates < 0 or (self.backbone_warmup_updates and self.isolation_warmup_updates):
            raise ValueError("Backbone warmup must be nonnegative and cannot overlap isolation warmup")
        self.loss_weights = {
            k: float(ea.get(k + "_weight", v)) for k, v in (("ground", 0.025), ("relation", 0.01), ("plan", 0.05))
        }
        self.sidecar = RelationalSidecar(
            self.video_backbone.dim,
            self.action_backbone.dim,
            self.context_dim,
            **{
                k: ea[k]
                for k in (
                    "latent_dim",
                    "heads",
                    "num_ea_tokens",
                    "read_gate_init",
                    "write_gate_init",
                    "action_conditioning",
                )
                if k in ea
            },
        )
        self.training_stage = ea.get("stage", "warmup")
        if self.training_stage not in ("warmup", "action_only", "joint"):
            raise ValueError("Sidecar stage must be warmup, action_only or joint")
        if self.isolation_warmup_updates and (self.training_stage != "joint" or not self.sidecar.action_conditioning):
            raise ValueError("Isolation warmup requires joint stage and action_conditioning=true")
        if self.backbone_warmup_updates and (
            self.training_stage != "joint" or not self.sidecar.action_conditioning or not self.sidecar_enabled
        ):
            raise ValueError("Backbone warmup requires enabled Sidecar, joint stage and action_conditioning=true")
        if not self.sidecar_enabled:
            self.sidecar.requires_grad_(False)
            self.training_stage = "joint"
        for name, parameter in self.named_parameters():
            if not name.startswith("sidecar.") and self.training_stage != "joint":
                if self.training_stage == "warmup" or not name.startswith("action_backbone."):
                    parameter.requires_grad_(False)
        from openwam.model.modules.cross_appearance import configure_xa
        configure_xa(self, ea.get('cross_appearance', {}))
        if self.xa_enabled and (self.training_stage != 'joint' or not self.sidecar_enabled
                               or self.backbone_warmup_updates or self.isolation_warmup_updates):
            raise ValueError('XA requires the complete joint model without repeated warmup')

    def set_training_progress(self, completed_optimizer_updates):
        # Trainer supplies its restored update counter before each accumulated microstep.
        self.completed_optimizer_updates = int(completed_optimizer_updates)
        if self.completed_optimizer_updates < 0:
            raise ValueError("Completed optimizer updates cannot be negative")

    @property
    def isolation_warmup_active(self):
        return self.training and self.completed_optimizer_updates < self.isolation_warmup_updates

    @property
    def backbone_warmup_active(self):
        return self.training and self.completed_optimizer_updates < self.backbone_warmup_updates

    @property
    def aux_heads_only_active(self):
        return (
            self.training
            and self.aux_heads_only_after_warmup
            and self.training_stage == "joint"
            and not self.isolation_warmup_active
            and not self.backbone_warmup_active
        )

    @property
    def aux_head_input_gradient_scale(self):
        if not self.training or self.training_stage != "joint":
            return 1.0
        if self.aux_heads_only_active:
            return 0.0
        start = max(self.isolation_warmup_updates, self.backbone_warmup_updates)
        if self.completed_optimizer_updates < start:
            return 1.0
        elapsed = self.completed_optimizer_updates - start
        fraction = min(elapsed / self.aux_gradient_anneal_updates, 1.0) if self.aux_gradient_anneal_updates else 1.0
        return 1.0 + (self.aux_shared_gradient_scale - 1.0) * fraction

    def set_dtype_device(self, dtype, device):
        super().set_dtype_device(dtype, device)
        if hasattr(self, "sidecar"):
            self.sidecar.to(device=device, dtype=dtype)
        if hasattr(self, 'xa_projector'):
            self.xa_projector.to(device=device,dtype=dtype)

    def apply_compile_optimizations(self, compile_cfg):
        # The official compiled loop contains no sidecar; never silently use it at deployment.
        self._compiled_mot_run_joint_loop = None

    def prepare_inputs(self, batch):
        samples = [batch] if isinstance(batch, dict) else batch
        for sample in samples:
            if sample.get("ea_plan_steps") != [8, 16, 32] or sample.get("ea_video_anchors") != list(range(0, 33, 4)):
                raise ValueError("Need regenerated 8/16/32 labels aligned with 32 actions / nine video frames")
            if sample["action"].shape != (32, 80) or sample["proprio"].shape != (1, 80) or len(sample["video"]) != 9:
                raise ValueError("Sidecar needs the official 80D EEF / 32-step contract")
            if "proprio_mask" not in sample:
                raise ValueError("Sidecar requires the observed proprio dimension mask")
        result = super().prepare_inputs(batch)
        result["ea_targets"] = torch.stack([torch.as_tensor(s["ea_targets"]) for s in samples]).to(
            self.device, self.dtype
        )
        if self.xa_enabled and self.training:
            clean_samples = []
            for sample in samples:
                if 'xa_clean_video' not in sample:
                    raise ValueError('XA requires retained nominal frames before augmentation')
                clean_samples.append(dict(sample,video=sample['xa_clean_video'],
                                          first_frame_image=[sample['xa_clean_video'][0]]))
            clean = super().prepare_inputs(clean_samples)
            result['xa_clean_inputs'] = {k:clean[k] for k in ('input_latents','first_frame_latents')}
        return result

    def forward(
        self,
        noisy_actions,
        action_timestep,
        *,
        proprio=None,
        return_aux=False,
        action_hidden_layer=None,
        action_hidden_only=False,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        **pipeline_inputs,
    ):
        if any(k in pipeline_inputs for k in ("ea_targets", "ea_effectors", "ea_arm_present")):
            raise ValueError("Sidecar inputs are current RGB/text/80D proprio+mask only; labels stay in loss")
        if not self.sidecar_enabled:
            result = super().forward(
                noisy_actions,
                action_timestep,
                proprio=proprio,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                **pipeline_inputs,
            )
            return (*result, ()) if return_aux else result
        vb, ab = self.video_backbone, self.action_backbone
        if noisy_actions is None or pipeline_inputs.get("first_frame_latents") is None or proprio is None:
            raise ValueError("Sidecar requires joint action generation and a clean current frame/proprio")
        if pipeline_inputs["first_frame_latents"].shape[2] != 1:
            raise ValueError("Exactly one clean current-frame latent is required")
        if (
            pipeline_inputs.get("force_per_token_t_mod", True) is False
            or pipeline_inputs.get("zero_clean_prefix_t_mod", True) is False
        ):
            raise ValueError("Current-frame time modulation must remain protected")
        text = pipeline_inputs["context"]
        text_mask = pipeline_inputs.get("context_mask")
        if text_mask is None and pipeline_inputs.get("seq_lens") is not None:
            text_mask = (
                torch.arange(text.shape[1], device=text.device)[None]
                < pipeline_inputs["seq_lens"].to(text.device)[:, None]
            )
        mask = pipeline_inputs.get("_proprio_sample_mask")
        if mask is None:
            raise ValueError("Sidecar forward requires an explicit current 80D proprio mask")
        isolated = self.isolation_warmup_active
        z, context, valid, arms = self.sidecar.init_tokens(
            text.detach() if isolated else text, text_mask, proprio.detach() if isolated else proprio, mask
        )
        inputs = dict(pipeline_inputs)
        inputs["_proprio_sample_mask"] = mask.reshape(text.shape[0], 1, 80)
        inputs = self._append_proprio_context_token(inputs, proprio)
        inputs.setdefault("force_per_token_t_mod", True)
        inputs.setdefault("zero_clean_prefix_t_mod", True)
        vs = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **inputs,
        )
        ast = ab.prepare_state(
            noisy_actions,
            action_timestep,
            context=inputs.get("context"),
            context_mask=inputs.get("context_mask"),
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        driver = self._mot_driver or self.build_mot_driver()
        attention_mask = driver._build_attention_mask(
            vs.hidden_states.shape[1],
            ast.payload.x_action.shape[1],
            driver._video_tokens_per_frame(vs),
            device=vs.hidden_states.device,
        )
        aux = []
        action_hidden = None
        checkpointed = use_gradient_checkpointing and torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
        for layer in range(vb.num_layers):
            vs, ast = driver.step(
                layer,
                vs,
                ast,
                attn_mask=attention_mask,
                use_gradient_checkpointing=checkpointed,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
            if layer in self.tap_blocks:
                index = self.tap_blocks.index(layer)
                h, w = vs.grid_height, vs.grid_width
                current = vs.hidden_states[:, : h * w]
                z, output = self.sidecar.read_refine(
                    index, z, current.detach() if isolated else current, context, valid, arms, h, w,
                    detach_head_inputs=self.aux_heads_only_active,
                    head_gradient_scale=self.aux_head_input_gradient_scale,
                )
                if not isolated:
                    write_tokens = z
                    if self.training and not self.backbone_warmup_active and self.main_task_gradient_gain != 1:
                        write_tokens = z.detach() + self.main_task_gradient_gain * (z - z.detach())
                    ast.payload.x_action = self.sidecar.write_action(index, ast.payload.x_action, write_tokens, valid)
                if return_aux:
                    output["read_hidden"] = current
                    aux.append(output)
            if layer == action_hidden_layer:
                action_hidden = ast.payload.x_action
                if action_hidden_only:
                    return action_hidden
        result = vb.finalize(vs), ab.extract_prediction(ast)
        if action_hidden_layer is not None:
            if action_hidden is None:
                raise ValueError('Action hidden layer outside the model')
            return (*result,tuple(aux),action_hidden) if return_aux else (*result,action_hidden)
        return (*result, tuple(aux)) if return_aux else result

    def forward_with_aux(self, *args, **kwargs):
        return self(*args, return_aux=True, **kwargs)

    def compute_loss(self, *, ea_targets=None, xa_clean_inputs=None, **inputs):
        if not self.sidecar_enabled:
            return super().compute_loss(**inputs)
        if ea_targets is None:
            raise ValueError("Training requires separate auxiliary labels")
        captured = []
        aligned = []
        noise_context = {} if self.xa_enabled and self.training else None
        if noise_context is not None:
            if xa_clean_inputs is None:
                raise ValueError('Missing nominal XA input latents')
            from openwam.model.modules.cross_appearance import CrossAppearanceState
            if self._xa_state is None:
                self._xa_state = CrossAppearanceState(self)

        def predict(*args, **kwargs):
            if noise_context is not None:
                teacher = self._xa_state.teacher(self,kwargs,xa_clean_inputs,noise_context)
                video,action,aux,hidden = self(*args,return_aux=True,
                    action_hidden_layer=self.xa_config['student_layer'],**kwargs)
                from openwam.model.modules.cross_appearance import alignment_loss
                aligned.append(alignment_loss(self.xa_projector(hidden),teacher,inputs.get('action_is_pad')))
            else:
                video, action, aux = self.forward_with_aux(*args, **kwargs)
            captured.extend(aux)
            return video, action

        result = super().compute_loss(forward_fn=predict, noise_context=noise_context, **inputs)
        parts = self.sidecar.heads.losses(captured[-1], ea_targets)
        deep = (
            self.loss_weights["ground"] * self.sidecar.heads.losses(captured[0], ea_targets)["ground"]
            + self.loss_weights["relation"] * self.sidecar.heads.losses(captured[1], ea_targets)["relation"]
        )
        extra = self.aux_loss_scale * (
            sum(self.loss_weights[k] * v for k, v in parts.items()) + self.deep_supervision * deep
        )
        result["loss"] = result["loss"] + extra
        result.update(
            loss_ea=extra.detach(), loss_ea_deep=deep.detach(), **{f"loss_ea_{k}": v.detach() for k, v in parts.items()}
        )
        if aligned:
            weighted,metrics = self._xa_state.objective(aligned[0],extra,self.completed_optimizer_updates)
            result['loss'] = result['loss']+weighted
            result.update(metrics)
        if self.isolation_warmup_updates:
            result["ea_isolation_warmup"] = extra.new_tensor(float(self.isolation_warmup_active))
        if self.backbone_warmup_updates:
            result["ea_backbone_frozen_warmup"] = extra.new_tensor(float(self.backbone_warmup_active))
        if self.aux_heads_only_after_warmup:
            result["ea_aux_heads_only"] = extra.new_tensor(float(self.aux_heads_only_active))
        if self.aux_shared_gradient_scale != 1.0 or self.aux_gradient_anneal_updates:
            result["ea_shared_gradient_scale"] = extra.new_tensor(self.aux_head_input_gradient_scale)
        if self.main_task_gradient_gain != 1:
            result["ea_main_task_gradient_gain"] = extra.new_tensor(
                self.main_task_gradient_gain if self.training and not self.backbone_warmup_active else 1.0
            )
        for i, stage in enumerate(self.sidecar.stages):
            result[f"ea_read_gate_{i}"] = stage.alpha.detach()
            result[f"ea_write_gate_{i}"] = stage.write.beta.detach()
        return result

    def on_optimizer_step(self):
        if self.xa_enabled and self._xa_state is not None:
            self._xa_state.update(self)

    def generate(self, *, proprio=None, active_action_mask=None, **kwargs):
        if any(k in kwargs for k in ("ea_targets", "ea_effectors", "ea_arm_present")):
            raise ValueError("Privileged or legacy EA context is forbidden during Sidecar generation")
        if not self.sidecar_enabled:
            return super().generate(proprio=proprio, active_action_mask=active_action_mask, **kwargs)
        if kwargs.get("cfg_scale", 1) != 1 or kwargs.get("dit_cache") is not None:
            raise ValueError("Initial Sidecar deployment requires CFG=1 and DiT cache disabled")
        inactive = self._resolve_inactive_action_dims(active_action_mask, self.device)
        if inactive is None:
            raise ValueError("Attach the EEF normalizer or provide active_action_mask for deployment")
        kwargs["_proprio_sample_mask"] = (~inactive)[None, None]
        return super().generate(proprio=proprio, active_action_mask=active_action_mask, **kwargs)

    def load_checkpoint(self, path, strict=True):
        from safetensors.torch import load_file

        state = load_file(path)
        if self.xa_enabled and not any(k.startswith('xa_projector.') for k in state):
            own = self.state_dict()
            missing = set(own)-set(state)
            if missing != {k for k in own if k.startswith('xa_projector.')} or set(state)-set(own):
                raise RuntimeError('XA warmstart permits only new projector tensors')
            self.load_state_dict(state,strict=False,assign=any(p.is_meta for p in self.parameters()))
            if any(p.is_meta for p in self.parameters()):
                raise RuntimeError('XA projector must be materialized')
            return
        if any(k.startswith("sidecar.") for k in state):
            return super().load_checkpoint(path, strict=strict)
        if not self.foundation_warmstart:
            raise ValueError("Explicit foundation_warmstart is required for Foundation-only weights")
        own = self.state_dict()
        expected = {k for k in own if k.startswith("sidecar.")}
        if (
            set(own) - set(state) != expected
            or set(state) - set(own)
            or any(own[k].shape != v.shape for k, v in state.items())
        ):
            raise RuntimeError("Foundation keys/shapes must match unchanged Alpha policy heads/backbones")
        self.load_state_dict(state, strict=False, assign=any(p.is_meta for p in self.parameters()))
        if any(p.is_meta for p in self.parameters()):
            raise RuntimeError("Unmaterialized Sidecar parameters")
        self.foundation_load_report = dict(
            loaded_tensors=len(state), reset_policy_heads=[], new_ea_tensors=len(expected)
        )

    def save_assets_for_deployment(self, output_path, cfg):
        super().save_assets_for_deployment(output_path, cfg)
        Path(output_path, "relational_sidecar_contract.json").write_text(
            json.dumps(
                dict(
                    version=1,
                    tap_blocks=self.tap_blocks,
                    tokens=15,
                    action_dim=80,
                    plan_steps=[8, 16, 32],
                    input_context="current normalized 80D proprio plus dimension mask",
                    source_quaternion_order="wxyz",
                    action_frame="official_robotwin_endpose_reference",
                    plan_frame="each_role_current_local_rotation",
                    spatial_geometry="runtime first-frame L canvas",
                    compile=False,
                ),
                indent=2,
            )
            + "\n"
        )
