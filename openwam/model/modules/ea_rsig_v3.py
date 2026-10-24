"""EA-RSIG-v3 task-grounded relational foresight bottleneck."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EA_RSIG_V3_MAX_ARMS = 2
EA_RSIG_V3_MAX_TRACKS = 2
EA_RSIG_V3_EFFECTOR_STATE_DIM = 10
EA_RSIG_V3_DEPLOYABLE_CONTEXT_DIM = 22
EA_RSIG_V3_LEGACY_TARGET_DIM = 95
EA_RSIG_V3_GROUND_TARGET_DIM = 30
EA_RSIG_V3_TARGET_DIM = 125
EA_RSIG_V3_DUAL_TRACK_TARGET_DIM = 130
EA_RSIG_V3_HORIZON_FRACTIONS = (0.25, 0.50, 1.00)
EA_RSIG_V3_HORIZON_NAMES = ("p25", "p50", "p100")
EA_RSIG_V3_HORIZONS = len(EA_RSIG_V3_HORIZON_FRACTIONS)


def ea_rsig_v3_plan_horizon_steps(action_horizon: int) -> tuple[int, int, int]:
    """Return the fixed three plan anchors at 25%, 50%, and 100% of a chunk."""
    action_horizon = int(action_horizon)
    if action_horizon < 4:
        raise ValueError(f"EA-RSIG-v3 action_horizon must be at least 4, got {action_horizon}")
    if action_horizon % 4:
        raise ValueError(
            f"EA-RSIG-v3 requires action_horizon divisible by 4 for exact 25/50/100% anchors, "
            f"got {action_horizon}"
        )
    steps = tuple(round(action_horizon * value) for value in EA_RSIG_V3_HORIZON_FRACTIONS)
    if len(set(steps)) != EA_RSIG_V3_HORIZONS or steps[-1] != action_horizon:
        raise ValueError(f"Invalid EA-RSIG-v3 plan anchors {steps} for chunk {action_horizon}")
    return steps


def validate_ea_rsig_v3_video_plan_alignment(
    action_horizon: int, video_indices: tuple[int, ...] | list[int],
) -> tuple[int, int, int]:
    """Fail closed unless every fixed plan anchor is a supervised video anchor."""
    video_indices = tuple(int(value) for value in video_indices)
    plan_steps = ea_rsig_v3_plan_horizon_steps(action_horizon)
    if not video_indices or video_indices[0] != 0 or video_indices[-1] != int(action_horizon):
        raise ValueError(
            f"Video anchors must start at 0 and end at {action_horizon}, got {video_indices}"
        )
    missing = tuple(value for value in plan_steps if value not in video_indices)
    if missing:
        raise ValueError(f"Plan anchors {missing} are absent from video anchors {video_indices}")
    return plan_steps


def split_ea_rsig_v3_state(
    state: torch.Tensor,
    policy_state_dim: int,
    target_dim: int = EA_RSIG_V3_TARGET_DIM,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Remove privileged V3 targets before the policy state encoder."""
    deployable_dim = int(policy_state_dim) + EA_RSIG_V3_DEPLOYABLE_CONTEXT_DIM
    valid_dims = (deployable_dim, deployable_dim + int(target_dim))
    if state.shape[-1] not in valid_dims:
        raise ValueError(
            f"EA-RSIG-v3 expects {valid_dims[0]}D inference or {valid_dims[1]}D "
            f"training state for a {policy_state_dim}D policy, got {tuple(state.shape)}"
        )
    start = int(policy_state_dim)
    stop = start + EA_RSIG_V3_MAX_ARMS * EA_RSIG_V3_EFFECTOR_STATE_DIM
    effectors = state[..., start:stop].reshape(
        *state.shape[:-1], EA_RSIG_V3_MAX_ARMS, EA_RSIG_V3_EFFECTOR_STATE_DIM
    )
    arm_present = state[..., stop : stop + EA_RSIG_V3_MAX_ARMS]
    targets = state[..., deployable_dim:] if state.shape[-1] > deployable_dim else None
    return state[..., :policy_state_dim], effectors, arm_present, targets


class EARSIGV3Module(nn.Module):
    """Ground Object/Goal, form typed relations, and expose gated Action tokens."""

    _ACTION_TOKEN_MODES = frozenset({
        "plan_only",
        "progress_plan",
        "ground_progress_plan",
        "relation_progress_plan",
        "full",
    })

    def __init__(
        self,
        *,
        hidden_dim: int,
        slot_dim: int = 512,
        num_views: int,
        num_arms: int | None = None,
        num_tracks: int = 1,
        grid_height: int = 14,
        view_grid_width: int = 14,
        num_heads: int = 8,
        min_sigma_cells: float = 1.0,
        near_threshold: float = 0.10,
        near_loss_weight: float = 0.5,
        goal_loss_weight: float = 0.5,
        action_token_mode: str = "full",
        condition_gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        if num_views not in (2, 3):
            raise ValueError(f"EA-RSIG-v3 supports 2 or 3 views, got {num_views}")
        if num_tracks not in (1, 2):
            raise ValueError(f"EA-RSIG-v3 supports 1 or 2 interaction tracks, got {num_tracks}")
        if num_arms is None:
            num_arms = num_views - 1
        if num_arms not in (1, 2) or num_tracks > num_arms:
            raise ValueError(f"Invalid EA-RSIG-v3 num_arms={num_arms}, num_tracks={num_tracks}")
        if slot_dim % num_heads:
            raise ValueError(f"slot_dim={slot_dim} must be divisible by num_heads={num_heads}")
        if action_token_mode not in self._ACTION_TOKEN_MODES:
            raise ValueError(
                f"Unsupported EA-RSIG-v3 action_token_mode={action_token_mode!r}; "
                f"expected one of {sorted(self._ACTION_TOKEN_MODES)}"
            )
        if min_sigma_cells <= 0 or near_threshold <= 0:
            raise ValueError("min_sigma_cells and near_threshold must be positive")

        self.hidden_dim = int(hidden_dim)
        self.slot_dim = int(slot_dim)
        self.num_views = int(num_views)
        self.num_arms = int(num_arms)
        self.num_tracks = int(num_tracks)
        self.grid_height = int(grid_height)
        self.view_grid_width = int(view_grid_width)
        self.min_sigma_cells = float(min_sigma_cells)
        self.near_threshold = float(near_threshold)
        self.near_loss_weight = float(near_loss_weight)
        self.goal_loss_weight = float(goal_loss_weight)
        self.action_token_mode = action_token_mode
        self.condition_gate_init = float(condition_gate_init)

        self.ground_queries = nn.Parameter(torch.randn(2 * self.num_tracks, self.slot_dim) * 0.02)
        self.ground_key = nn.Linear(self.hidden_dim, self.slot_dim)
        self.ground_value = nn.Linear(self.hidden_dim, self.slot_dim)
        self.view_embeddings = nn.Parameter(torch.randn(self.num_views, self.slot_dim) * 0.02)
        self.presence_head = nn.Linear(self.slot_dim, 1)
        self.null_slots = nn.Parameter(torch.randn(2 * self.num_tracks, self.slot_dim) * 0.02)

        self.eef_type_embedding = nn.Parameter(torch.randn(1, self.slot_dim) * 0.02)
        self.effector_encoder = nn.Sequential(
            nn.Linear(EA_RSIG_V3_EFFECTOR_STATE_DIM, self.slot_dim),
            nn.SiLU(),
            nn.Linear(self.slot_dim, self.slot_dim),
        )
        self.track_valid_head = nn.Sequential(
            nn.Linear(2 * self.slot_dim, self.slot_dim),
            nn.SiLU(),
            nn.Linear(self.slot_dim, 1),
        )

        self.relation_type_embeddings = nn.Parameter(torch.randn(2, self.slot_dim) * 0.02)
        self.relation_encoder = nn.Sequential(
            nn.Linear(3 * self.slot_dim, 2 * self.slot_dim),
            nn.SiLU(),
            nn.Linear(2 * self.slot_dim, self.slot_dim),
            nn.LayerNorm(self.slot_dim),
        )
        self.contact_head = nn.Linear(self.slot_dim, 1)
        self.near_head = nn.Linear(self.slot_dim, 1)
        self.goal_proximity_head = nn.Linear(self.slot_dim, 1)

        self.progress_query = nn.Parameter(torch.randn(1, 1, self.slot_dim) * 0.02)
        self.progress_attention = nn.MultiheadAttention(
            self.slot_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.progress_norm_1 = nn.LayerNorm(self.slot_dim)
        self.progress_ffn = nn.Sequential(
            nn.Linear(self.slot_dim, 2 * self.slot_dim),
            nn.SiLU(),
            nn.Linear(2 * self.slot_dim, self.slot_dim),
        )
        self.progress_norm_2 = nn.LayerNorm(self.slot_dim)

        self.eef_plan_head = nn.Sequential(
            nn.Linear(3 * self.slot_dim, self.slot_dim),
            nn.SiLU(),
            nn.Linear(self.slot_dim, EA_RSIG_V3_HORIZONS * 4),
        )
        self.object_plan_head = nn.Sequential(
            nn.Linear(3 * self.slot_dim, self.slot_dim),
            nn.SiLU(),
            nn.Linear(self.slot_dim, EA_RSIG_V3_HORIZONS * 4),
        )
        self.eef_plan_to_hidden = nn.Linear(4, self.hidden_dim)
        self.object_plan_to_hidden = nn.Linear(4, self.hidden_dim)
        # These are learned V3 latents, never privileged bbox/pose targets.
        self.ground_to_hidden = nn.Linear(self.slot_dim, self.hidden_dim)
        self.relation_to_hidden = nn.Linear(self.slot_dim, self.hidden_dim)
        self.progress_to_hidden = nn.Linear(self.slot_dim, self.hidden_dim)
        self.ground_gate = nn.Parameter(torch.tensor(self.condition_gate_init))
        self.relation_gate = nn.Parameter(torch.tensor(self.condition_gate_init))
        self.progress_gate = nn.Parameter(torch.tensor(self.condition_gate_init))

        # Training-only H18 adapter. Its input is observation H18, never the
        # noisy future-video H18 or privileged supervision targets.
        self.video_context = nn.Linear((2 * self.num_tracks + 1) * self.slot_dim, 128)
        self.video_query = nn.Linear(self.hidden_dim, 128)
        self.video_out = nn.Linear(128, self.hidden_dim)
        self.video_alpha = nn.Parameter(torch.zeros(()))
        self._video_output: dict[str, torch.Tensor | dict[str, torch.Tensor]] | None = None

    @property
    def spatial_tokens(self) -> int:
        return self.num_views * self.grid_height * self.view_grid_width

    @property
    def target_dim(self) -> int:
        return EA_RSIG_V3_TARGET_DIM if self.num_tracks == 1 else EA_RSIG_V3_DUAL_TRACK_TARGET_DIM

    def _to_current_grid(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 3:
            if hidden.shape[1] < self.spatial_tokens or hidden.shape[2] != self.hidden_dim:
                raise ValueError(
                    f"EA-RSIG-v3 needs at least {self.spatial_tokens} tokens with "
                    f"hidden_dim={self.hidden_dim}, got {tuple(hidden.shape)}"
                )
            current = hidden[:, : self.spatial_tokens]
            return current.reshape(
                hidden.shape[0], self.grid_height, self.num_views,
                self.view_grid_width, self.hidden_dim,
            ).permute(0, 2, 1, 3, 4)
        if hidden.ndim == 5:
            b, c, _, h, w = hidden.shape
            if c != self.hidden_dim or h != self.grid_height or w != self.num_views * self.view_grid_width:
                raise ValueError(f"Invalid EA-RSIG-v3 hidden grid {tuple(hidden.shape)}")
            return hidden[:, :, 0].permute(0, 2, 3, 1).reshape(
                b, h, self.num_views, self.view_grid_width, c
            ).permute(0, 2, 1, 3, 4)
        raise ValueError(f"EA-RSIG-v3 hidden must be 3D or 5D, got {tuple(hidden.shape)}")

    def _ground(self, current: torch.Tensor) -> dict[str, torch.Tensor]:
        b, views, height, width, _ = current.shape
        keys = self.ground_key(current)
        keys = keys + self.view_embeddings[:views].to(keys.dtype)[None, :, None, None]
        values = self.ground_value(current)
        logits = torch.einsum(
            "rd,bvhwd->brvhw", self.ground_queries.to(keys.dtype), keys
        ) * (self.slot_dim**-0.5)
        role_count = 2 * self.num_tracks
        attention = logits.flatten(2).softmax(dim=-1).reshape(b, role_count, views, height, width)
        raw_slots = torch.einsum("brvhw,bvhwd->brd", attention, values)
        presence_logits = self.presence_head(raw_slots).squeeze(-1)
        presence = presence_logits.sigmoid().unsqueeze(-1)
        slots = presence * raw_slots + (1 - presence) * self.null_slots.to(raw_slots.dtype)
        return {
            "logits": logits,
            "attention": attention,
            "raw_slots": raw_slots,
            "slots": slots,
            "presence_logits": presence_logits,
        }

    def _relations(
        self,
        eef_slots: torch.Tensor,
        ground_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_tracks == 1:
            obj, goal = ground_slots[:, 0], ground_slots[:, 1]
        else:
            tracks = ground_slots.reshape(ground_slots.shape[0], self.num_tracks, 2, self.slot_dim)
            obj, goal = tracks[:, :, 0], tracks[:, :, 1]
        eo_type = self.relation_type_embeddings[0].to(obj.dtype)
        og_type = self.relation_type_embeddings[1].to(obj.dtype)
        if self.num_tracks == 2:
            active_eef = eef_slots[:, :self.num_tracks]
            eo = self.relation_encoder(torch.cat((
                active_eef, obj,
                eo_type[None, None].expand(eef_slots.shape[0], self.num_tracks, -1),
            ), dim=-1))
            og = self.relation_encoder(torch.cat((
                obj, goal, og_type[None, None].expand(eef_slots.shape[0], self.num_tracks, -1),
            ), dim=-1))
            return eo, og
        eo = self.relation_encoder(torch.cat((
            eef_slots,
            obj[:, None].expand(-1, EA_RSIG_V3_MAX_ARMS, -1),
            eo_type[None, None].expand(eef_slots.shape[0], EA_RSIG_V3_MAX_ARMS, -1),
        ), dim=-1))
        og = self.relation_encoder(torch.cat((obj, goal, og_type[None].expand_as(obj)), dim=-1))
        return eo, og[:, None]

    def _progress(
        self,
        eo: torch.Tensor,
        og: torch.Tensor,
        arm_present: torch.Tensor,
        track_gate: torch.Tensor,
    ) -> torch.Tensor:
        sequence = torch.cat((eo, og), dim=1)
        attention_mask = None
        if self.num_tracks == 1:
            valid = torch.cat((
                arm_present.bool(),
                torch.ones_like(arm_present[:, :1], dtype=torch.bool),
            ), dim=1)
        else:
            weights = torch.cat((track_gate, track_gate), dim=1)
            attention_mask = weights.clamp_min(1e-6).log()[:, None, :]
            attention_mask = attention_mask.repeat_interleave(
                self.progress_attention.num_heads, dim=0
            )
            valid = None
        query = self.progress_query.to(sequence.dtype).expand(sequence.shape[0], -1, -1)
        attended = self.progress_attention(
            query,
            sequence,
            sequence,
            key_padding_mask=None if valid is None else ~valid,
            attn_mask=attention_mask,
            need_weights=False,
        )[0]
        progress = self.progress_norm_1(query + attended)
        progress = self.progress_norm_2(progress + self.progress_ffn(progress))[:, 0]
        return progress

    def _active_arm_count(self, arm_present: torch.Tensor) -> int:
        """Fixed by the registered embodiment; avoid a GPU `.item()` each batch."""
        # Panda uses two views/one arm; Aloha uses three views/two arms.
        # The dataset preflight must verify the arm mask before training.
        return self.num_arms

    def _plan(
        self,
        eef_slots: torch.Tensor,
        ground_slots: torch.Tensor,
        progress: torch.Tensor,
        arm_present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.num_tracks == 2:
            tracks = ground_slots.reshape(ground_slots.shape[0], self.num_tracks, 2, self.slot_dim)
            obj, goal = tracks[:, :, 0], tracks[:, :, 1]
            active_eef = eef_slots[:, :self.num_tracks]
            eef_raw = self.eef_plan_head(torch.cat((
                active_eef, obj, progress[:, None].expand(-1, self.num_tracks, -1),
            ), dim=-1)).reshape(-1, self.num_tracks, EA_RSIG_V3_HORIZONS, 4)
            object_raw = self.object_plan_head(torch.cat((
                obj, goal, progress[:, None].expand(-1, self.num_tracks, -1),
            ), dim=-1)).reshape(-1, self.num_tracks, EA_RSIG_V3_HORIZONS, 4)
            raw = torch.stack((eef_raw, object_raw), dim=3).permute(0, 2, 1, 3, 4)
            directions = F.normalize(raw[..., :3], dim=-1)
            moving_logits = raw[..., 3]
            features = torch.cat((directions, moving_logits.sigmoid().unsqueeze(-1)), dim=-1)
            eef = self.eef_plan_to_hidden(features[:, :, :, 0])
            objects = self.object_plan_to_hidden(features[:, :, :, 1])
            return directions, moving_logits, features, (eef + objects).flatten(1, 2)

        obj, goal = ground_slots[:, 0], ground_slots[:, 1]
        eef_raw = self.eef_plan_head(torch.cat((
            eef_slots,
            obj[:, None].expand(-1, EA_RSIG_V3_MAX_ARMS, -1),
            progress[:, None].expand(-1, EA_RSIG_V3_MAX_ARMS, -1),
        ), dim=-1)).reshape(-1, EA_RSIG_V3_MAX_ARMS, EA_RSIG_V3_HORIZONS, 4)
        obj_raw = self.object_plan_head(torch.cat((obj, goal, progress), dim=-1)).reshape(
            -1, 1, EA_RSIG_V3_HORIZONS, 4
        )
        raw = torch.cat((eef_raw, obj_raw), dim=1).permute(0, 2, 1, 3)
        directions = F.normalize(raw[..., :3], dim=-1)
        moving_logits = raw[..., 3]
        features = torch.cat((directions, moving_logits.sigmoid().unsqueeze(-1)), dim=-1)
        role_mask = torch.cat((arm_present, torch.ones_like(arm_present[:, :1])), dim=1)
        features = features * role_mask[:, None, :, None].to(features.dtype)

        eef = self.eef_plan_to_hidden(features[:, :, :EA_RSIG_V3_MAX_ARMS])
        object_plan = self.object_plan_to_hidden(features[:, :, 2])[:, :, None]
        per_arm_tokens = eef + object_plan
        active_arm_count = self._active_arm_count(arm_present)
        if active_arm_count == 1:
            plan_tokens = per_arm_tokens[:, :, :1].squeeze(2)
        else:
            # Keep arm identity per horizon: [p25/arm0, p25/arm1, ..., p100/arm1].
            # This replaces the old active-EEF mean, which could cancel opposed motions.
            plan_tokens = per_arm_tokens.flatten(1, 2)
        return directions, moving_logits, features, plan_tokens

    def _gated_tokens(self, values: torch.Tensor, projection: nn.Linear, gate: torch.Tensor) -> torch.Tensor:
        return projection(values) * gate.to(dtype=values.dtype)

    def _compose_action_tokens(
        self,
        ground_slots: torch.Tensor,
        eef_object_relations: torch.Tensor,
        object_goal_relation: torch.Tensor,
        progress: torch.Tensor,
        plan_tokens: torch.Tensor,
        track_gate: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        active_arm_count = self.num_arms
        ground_tokens = self._gated_tokens(ground_slots, self.ground_to_hidden, self.ground_gate)
        ground_tokens = ground_tokens * track_gate.repeat_interleave(2, dim=1).unsqueeze(-1)
        relation_values = torch.cat((
            eef_object_relations[:, :active_arm_count], object_goal_relation,
        ), dim=1)
        relation_tokens = self._gated_tokens(
            relation_values, self.relation_to_hidden, self.relation_gate
        )
        relation_mask = (
            torch.cat((track_gate, track_gate), dim=1)
            if self.num_tracks == 2
            else torch.cat((track_gate.expand(-1, active_arm_count), track_gate), dim=1)
        )
        relation_tokens = relation_tokens * relation_mask.unsqueeze(-1)
        progress_tokens = self._gated_tokens(
            progress[:, None], self.progress_to_hidden, self.progress_gate
        )
        any_track = 1 - (1 - track_gate).prod(dim=1, keepdim=True)
        progress_tokens = progress_tokens * any_track.unsqueeze(-1)
        if self.num_tracks == 2:
            plan_mask = track_gate[:, None].expand(-1, EA_RSIG_V3_HORIZONS, -1)
            plan_tokens = plan_tokens * plan_mask.reshape(track_gate.shape[0], -1).unsqueeze(-1)
        else:
            plan_tokens = plan_tokens * any_track.unsqueeze(-1)
        pieces: list[torch.Tensor] = []
        if self.action_token_mode in {"ground_progress_plan", "full"}:
            pieces.append(ground_tokens)
        if self.action_token_mode in {"relation_progress_plan", "full"}:
            pieces.append(relation_tokens)
        if self.action_token_mode in {
            "progress_plan", "ground_progress_plan", "relation_progress_plan", "full",
        }:
            pieces.append(progress_tokens)
        pieces.append(plan_tokens)
        return {
            "action_tokens": torch.cat(pieces, dim=1),
            "ground_tokens": ground_tokens,
            "relation_tokens": relation_tokens,
            "progress_tokens": progress_tokens,
            "plan_tokens": plan_tokens,
        }

    def forward(
        self,
        hidden: torch.Tensor,
        effector_state: torch.Tensor,
        arm_present: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if effector_state.shape[-2:] != (EA_RSIG_V3_MAX_ARMS, EA_RSIG_V3_EFFECTOR_STATE_DIM):
            raise ValueError(f"Invalid effector_state shape {tuple(effector_state.shape)}")
        if arm_present.shape[-1] != EA_RSIG_V3_MAX_ARMS:
            raise ValueError(f"Invalid arm_present shape {tuple(arm_present.shape)}")
        current = self._to_current_grid(hidden)
        ground = self._ground(current)
        eef_slots = self.effector_encoder(effector_state.to(current.dtype))
        eef_slots = eef_slots + self.eef_type_embedding.to(eef_slots.dtype)
        eef_slots = eef_slots * arm_present.to(eef_slots.dtype).unsqueeze(-1)
        # Presence supervision only changes policy behavior if downstream heads
        # consume the null-interpolated slots rather than arbitrary raw attention.
        semantic_slots = ground["slots"]
        if self.num_tracks == 1:
            object_slots = semantic_slots[:, :1]
            track_eef = eef_slots[:, :1]
        else:
            object_slots = semantic_slots.reshape(
                semantic_slots.shape[0], self.num_tracks, 2, self.slot_dim
            )[:, :, 0]
            track_eef = eef_slots[:, :self.num_tracks]
        track_valid_logits = self.track_valid_head(
            torch.cat((track_eef, object_slots), dim=-1)
        ).squeeze(-1)
        track_gate = track_valid_logits.sigmoid() * arm_present[:, :self.num_tracks].to(
            track_valid_logits.dtype
        )
        eo, og = self._relations(eef_slots, semantic_slots)
        progress = self._progress(eo, og, arm_present, track_gate)
        directions, moving_logits, plan_features, plan_tokens = self._plan(
            eef_slots, semantic_slots, progress, arm_present
        )
        action_condition = self._compose_action_tokens(
            semantic_slots, eo, og, progress, plan_tokens, track_gate
        )
        return {
            "grounding": ground,
            "arm_present": arm_present,
            "track_valid_logits": track_valid_logits,
            "track_gate": track_gate,
            "eef_slots": eef_slots,
            "eef_object_relations": eo,
            "object_goal_relation": og,
            "progress": progress,
            "contact_logits": self.contact_head(eo).squeeze(-1),
            "near_logits": self.near_head(eo).squeeze(-1),
            "goal_proximity": self.goal_proximity_head(og).squeeze(-1).sigmoid(),
            "plan_directions": directions,
            "plan_moving_logits": moving_logits,
            "plan_features": plan_features,
            "condition_gates": torch.stack((
                self.ground_gate, self.relation_gate, self.progress_gate,
            )),
            **action_condition,
        }

    def inject_video_hidden(
        self,
        hidden: torch.Tensor,
        context: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Add a v3 relation-conditioned residual at training's second H18 pass."""
        observation_h18, effector_state, arm_present = context
        representation = self(observation_h18, effector_state, arm_present)
        self._video_output = representation
        track_gate = representation["track_gate"]
        any_track = 1 - (1 - track_gate).prod(dim=1, keepdim=True)
        if self.num_tracks == 2:
            relation = torch.cat((
                (representation["eef_object_relations"] * track_gate.unsqueeze(-1)).flatten(1),
                (representation["object_goal_relation"] * track_gate.unsqueeze(-1)).flatten(1),
                representation["progress"] * any_track,
            ), dim=-1)
        else:
            relation = torch.cat((
                representation["eef_object_relations"][:, 0] * track_gate[:, :1],
                representation["object_goal_relation"][:, 0] * track_gate[:, :1],
                representation["progress"] * any_track,
            ), dim=-1)
        condition = self.video_context(relation).to(hidden.dtype)
        if hidden.ndim == 5:
            tokens = hidden.permute(0, 2, 3, 4, 1)
            delta = self.video_out(F.silu(self.video_query(tokens) + condition[:, None, None, None]))
            delta = delta.permute(0, 4, 1, 2, 3)
        elif hidden.ndim == 3:
            delta = self.video_out(F.silu(self.video_query(hidden) + condition[:, None]))
        else:
            raise ValueError(f"Expected 3D or 5D H18, got {tuple(hidden.shape)}")
        return hidden + self.video_alpha.to(hidden.dtype) * delta

    def take_video_output(self):
        """Reuse the training residual's v3 graph for action and auxiliary losses."""
        output, self._video_output = self._video_output, None
        return output

    def training_diagnostics(
        self,
        output: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return GPU-side scalars; the trainer synchronizes only on logging steps."""
        target = targets[:, 0].detach().float()
        track_target = target[:, :2] if self.num_tracks == 2 else target[:, 38:39]
        logits = output["track_valid_logits"].detach().float()
        gate = output["track_gate"].detach().float()
        valid = track_target > 0.5
        predicted = gate > 0.5

        def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            weight = mask.to(value.dtype)
            return (value * weight).sum() / weight.sum().clamp_min(1)

        diagnostics = {
            "track_valid_loss": F.binary_cross_entropy_with_logits(logits, track_target),
            "track_gate_valid_mean": masked_mean(gate, valid),
            "track_gate_invalid_mean": masked_mean(gate, ~valid),
            "track_gate_precision": masked_mean(track_target, predicted),
            "track_gate_recall": masked_mean(predicted.float(), valid),
            "track_valid_count": valid.float().sum(),
            "track_invalid_count": (~valid).float().sum(),
            "ground_token_rms": output["ground_tokens"].detach().float().square().mean().sqrt(),
            "relation_token_rms": output["relation_tokens"].detach().float().square().mean().sqrt(),
            "progress_token_rms": output["progress_tokens"].detach().float().square().mean().sqrt(),
            "plan_token_rms": output["plan_tokens"].detach().float().square().mean().sqrt(),
            "action_token_rms": output["action_tokens"].detach().float().square().mean().sqrt(),
            "video_alpha": self.video_alpha.detach().float(),
        }
        for index in range(gate.shape[1]):
            diagnostics[f"track{index}_gate_mean"] = gate[:, index].mean()
        for name, value in zip(("ground", "relation", "progress"), output["condition_gates"]):
            diagnostics[f"condition_gate_{name}"] = value.detach().float()
        return diagnostics

    def bbox_gaussian_targets(
        self,
        boxes: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Convert normalized xyxy boxes into globally normalized VHW distributions."""
        role_count = 2 * self.num_tracks
        if boxes.shape[-3:] != (role_count, 3, 4) or valid.shape[-2:] != (role_count, 3):
            raise ValueError(f"Invalid grounding targets boxes={boxes.shape}, valid={valid.shape}")
        boxes = boxes[:, :, : self.num_views].float()
        valid = valid[:, :, : self.num_views].float()
        y = torch.arange(self.grid_height, device=boxes.device, dtype=torch.float32)
        x = torch.arange(self.view_grid_width, device=boxes.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        cx = 0.5 * (x1 + x2) * self.view_grid_width - 0.5
        cy = 0.5 * (y1 + y2) * self.grid_height - 0.5
        sx = (0.5 * (x2 - x1) * self.view_grid_width).clamp_min(self.min_sigma_cells)
        sy = (0.5 * (y2 - y1) * self.grid_height).clamp_min(self.min_sigma_cells)
        distance = (
            ((xx - cx[..., None, None]) / sx[..., None, None]) ** 2
            + ((yy - cy[..., None, None]) / sy[..., None, None]) ** 2
        )
        targets = torch.exp(-0.5 * distance) * valid[..., None, None]
        return targets / targets.sum(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-8)

    def losses(
        self,
        output: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        targets: torch.Tensor,
        logical_batch_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if logical_batch_size is not None:
            from openwam.model.modules.training_memory import slice_batch_tree
            batch = targets.shape[0]
            if logical_batch_size < 1 or batch % logical_batch_size:
                raise ValueError("EA batch must be divisible by logical loss batch size")
            if batch > logical_batch_size:
                groups = [self.losses(slice_batch_tree(output, start, start + logical_batch_size, batch),
                                      targets[start:start + logical_batch_size])
                          for start in range(0, batch, logical_batch_size)]
                return {name: torch.stack([group[name] for group in groups]).mean()
                        for name in groups[0]}
        if self.num_tracks == 2:
            return self._dual_track_losses(output, targets)
        if targets.shape[-1] != EA_RSIG_V3_TARGET_DIM:
            raise ValueError(
                f"EA-RSIG-v3 targets must be {EA_RSIG_V3_TARGET_DIM}D, got {tuple(targets.shape)}"
            )
        target = targets[:, 0].float()
        legacy = target[:, :EA_RSIG_V3_LEGACY_TARGET_DIM]
        boxes = target[:, 95:119].reshape(-1, 2, 3, 4)
        ground_valid = target[:, 119:125].reshape(-1, 2, 3)
        soft_targets = self.bbox_gaussian_targets(boxes, ground_valid)
        logits = output["grounding"]["logits"].float()
        log_probability = logits.flatten(2).log_softmax(dim=-1).reshape_as(logits)
        role_valid = (ground_valid[:, :, : self.num_views].sum(dim=-1) > 0).float()
        ground_ce = -(soft_targets * log_probability).sum(dim=(-3, -2, -1))
        ground_ce = (ground_ce * role_valid).sum() / role_valid.sum().clamp_min(1)
        presence_loss = F.binary_cross_entropy_with_logits(
            output["grounding"]["presence_logits"].float(), role_valid
        )
        object_present = legacy[:, 38:39]
        track_loss = F.binary_cross_entropy_with_logits(
            output["track_valid_logits"].float(), object_present
        )
        ground_loss = ground_ce + presence_loss + track_loss

        arm_present = output["arm_present"].float()
        relation_xyz = legacy[:, 40:46].reshape(-1, 2, 3)
        relation_mask = arm_present * object_present
        near_target = (relation_xyz.norm(dim=-1) < self.near_threshold).float()
        near_error = F.binary_cross_entropy_with_logits(
            output["near_logits"].float(), near_target, reduction="none"
        )
        near_loss = (near_error * relation_mask).sum() / relation_mask.sum().clamp_min(1)
        contact = legacy[:, 46:48]
        contact_valid = legacy[:, 48:50] * arm_present
        contact_error = F.binary_cross_entropy_with_logits(
            output["contact_logits"].float(), contact, reduction="none"
        )
        contact_loss = (contact_error * contact_valid).sum() / contact_valid.sum().clamp_min(1)
        goal_proximity = legacy[:, 54:55]
        # The single-arm policy sees only the current RGB views. An absent goal
        # has no visual location to support an oracle proximity target.
        goal_visible = (ground_valid[:, 1, : self.num_views].sum(dim=-1) > 0).float()
        goal_valid = legacy[:, 55:56] * goal_visible
        goal_error = F.smooth_l1_loss(
            output["goal_proximity"].float(), goal_proximity, reduction="none"
        )
        goal_loss = (goal_error * goal_valid).sum() / goal_valid.sum().clamp_min(1)
        relation_loss = contact_loss + self.near_loss_weight * near_loss + self.goal_loss_weight * goal_loss

        plan_direction = legacy[:, 56:83].reshape(-1, 3, 3, 3)
        plan_moving = legacy[:, 83:92].reshape(-1, 3, 3)
        future_valid = legacy[:, 92:95]
        plan_role_mask = torch.cat((arm_present, object_present), dim=1)
        future_mask = future_valid.unsqueeze(-1) * plan_role_mask.unsqueeze(1)
        moving_error = F.binary_cross_entropy_with_logits(
            output["plan_moving_logits"].float(), plan_moving, reduction="none"
        )
        moving_loss = (moving_error * future_mask).sum() / future_mask.sum().clamp_min(1)
        direction_mask = future_mask * plan_moving
        direction_error = 1 - (
            output["plan_directions"].float() * plan_direction
        ).sum(dim=-1).clamp(-1, 1)
        direction_loss = (direction_error * direction_mask).sum() / direction_mask.sum().clamp_min(1)
        plan_loss = moving_loss + direction_loss
        return {"ground": ground_loss, "relation": relation_loss, "plan": plan_loss}

    def _dual_track_losses(
        self,
        output: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Loss contract for two arm-aligned Object/Goal interaction tracks."""
        if targets.shape[-1] != EA_RSIG_V3_DUAL_TRACK_TARGET_DIM:
            raise ValueError(
                f"Dual-track EA-RSIG-v3 targets must be {EA_RSIG_V3_DUAL_TRACK_TARGET_DIM}D, "
                f"got {tuple(targets.shape)}"
            )
        target = targets[:, 0].float()
        track_valid = target[:, 0:2]
        relation_xyz = target[:, 2:8].reshape(-1, 2, 3)
        contact = target[:, 8:10]
        contact_valid = target[:, 10:12]
        goal_proximity = target[:, 12:14]
        goal_valid = target[:, 14:16]
        plan_direction = target[:, 16:52].reshape(-1, 3, 2, 2, 3)
        plan_moving = target[:, 52:64].reshape(-1, 3, 2, 2)
        future_valid = target[:, 64:70].reshape(-1, 3, 2)
        boxes = target[:, 70:118].reshape(-1, 4, 3, 4)
        ground_valid = target[:, 118:130].reshape(-1, 4, 3)

        soft_targets = self.bbox_gaussian_targets(boxes, ground_valid)
        logits = output["grounding"]["logits"].float()
        log_probability = logits.flatten(2).log_softmax(dim=-1).reshape_as(logits)
        role_valid = (ground_valid[:, :, :self.num_views].sum(dim=-1) > 0).float()
        ground_ce = -(soft_targets * log_probability).sum(dim=(-3, -2, -1))
        ground_ce = (ground_ce * role_valid).sum() / role_valid.sum().clamp_min(1)
        presence_loss = F.binary_cross_entropy_with_logits(
            output["grounding"]["presence_logits"].float(), role_valid
        )
        track_loss = F.binary_cross_entropy_with_logits(
            output["track_valid_logits"].float(), track_valid
        )
        ground_loss = ground_ce + presence_loss + track_loss

        arm_valid = output["arm_present"][:, :2].float() * track_valid
        near_target = (relation_xyz.norm(dim=-1) < self.near_threshold).float()
        near_error = F.binary_cross_entropy_with_logits(
            output["near_logits"].float(), near_target, reduction="none"
        )
        near_loss = (near_error * arm_valid).sum() / arm_valid.sum().clamp_min(1)
        contact_error = F.binary_cross_entropy_with_logits(
            output["contact_logits"].float(), contact, reduction="none"
        )
        contact_mask = contact_valid * arm_valid
        contact_loss = (contact_error * contact_mask).sum() / contact_mask.sum().clamp_min(1)
        goal_error = F.smooth_l1_loss(
            output["goal_proximity"].float(), goal_proximity, reduction="none"
        )
        goal_mask = goal_valid * track_valid
        goal_loss = (goal_error * goal_mask).sum() / goal_mask.sum().clamp_min(1)
        relation_loss = contact_loss + self.near_loss_weight * near_loss + self.goal_loss_weight * goal_loss

        future_mask = (
            future_valid[:, :, :, None] * track_valid[:, None, :, None]
        ).expand(-1, -1, -1, 2)
        moving_error = F.binary_cross_entropy_with_logits(
            output["plan_moving_logits"].float(), plan_moving, reduction="none"
        )
        moving_loss = (moving_error * future_mask).sum() / future_mask.sum().clamp_min(1)
        direction_mask = future_mask * plan_moving
        direction_error = 1 - (
            output["plan_directions"].float() * plan_direction
        ).sum(dim=-1).clamp(-1, 1)
        direction_loss = (direction_error * direction_mask).sum() / direction_mask.sum().clamp_min(1)
        return {
            "ground": ground_loss,
            "relation": relation_loss,
            "plan": moving_loss + direction_loss,
        }

    @staticmethod
    def diagnostics(output: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        attention = output["grounding"]["attention"]
        flat = attention.flatten(2)
        peak = flat.argmax(dim=-1)
        width = attention.shape[-1]
        cells_per_view = attention.shape[-2] * width
        view = torch.div(peak, cells_per_view, rounding_mode="floor")
        cell = peak % cells_per_view
        uv = torch.stack(((cell % width).float(), torch.div(cell, width, rounding_mode="floor").float()), dim=-1)
        entropy = -(flat.float() * flat.float().clamp_min(1e-8).log()).sum(dim=-1)
        return {
            "ground_peak_view": view,
            "ground_peak_uv": uv,
            "ground_entropy": entropy,
            "ground_presence_probability": output["grounding"]["presence_logits"].sigmoid(),
            "track_valid_probability": output["track_valid_logits"].sigmoid(),
            "contact_probability": output["contact_logits"].sigmoid(),
            "near_probability": output["near_logits"].sigmoid(),
            "goal_proximity": output["goal_proximity"],
            "plan_directions": output["plan_directions"],
            "plan_moving_probability": output["plan_moving_logits"].sigmoid(),
        }
