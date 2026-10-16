import pytest
import torch

from openwam.model.modules.ea_rsig_v3 import (
    EARSIGV3Module,
    ea_rsig_v3_plan_horizon_steps,
    split_ea_rsig_v3_state,
    validate_ea_rsig_v3_video_plan_alignment,
)


def test_plan_horizons_are_fixed_quarter_half_full_video_anchors():
    assert ea_rsig_v3_plan_horizon_steps(8) == (2, 4, 8)
    assert ea_rsig_v3_plan_horizon_steps(40) == (10, 20, 40)
    assert validate_ea_rsig_v3_video_plan_alignment(8, [0, 2, 4, 6, 8]) == (2, 4, 8)
    assert validate_ea_rsig_v3_video_plan_alignment(40, [0, 10, 20, 30, 40]) == (10, 20, 40)
    with pytest.raises(ValueError, match="absent from video anchors"):
        validate_ea_rsig_v3_video_plan_alignment(40, [0, 10, 30, 40])
    with pytest.raises(ValueError, match="divisible by 4"):
        ea_rsig_v3_plan_horizon_steps(10)


def test_v3_state_contract_and_training_only_video_gradient():
    module = EARSIGV3Module(hidden_dim=32, slot_dim=16, num_views=2,
                          grid_height=2, view_grid_width=2, num_heads=4)
    state = torch.zeros(2, 1, 16 + 22 + 125)
    state[..., 16 + 20] = 1  # Panda arm0 only.
    policy, effectors, arms, labels = split_ea_rsig_v3_state(state, 16)
    assert policy.shape == (2, 1, 16)
    assert labels.shape == (2, 1, 125)
    assert split_ea_rsig_v3_state(state[..., :38], 16)[-1] is None

    observation_h18 = torch.randn(2, 8, 32)
    output = module(observation_h18, effectors[:, 0], arms[:, 0])
    assert output["action_tokens"].shape == (2, 8, 32)
    fm_h18 = torch.randn(2, 32, 2, 2, 4, requires_grad=True)
    adapted = module.inject_video_hidden(fm_h18, (observation_h18, effectors[:, 0], arms[:, 0]))
    torch.testing.assert_close(adapted, fm_h18)
    assert module.take_video_output() is not None
    assert module.take_video_output() is None
    adapted.square().mean().backward()
    assert module.video_alpha.grad is not None
    assert torch.isfinite(module.video_alpha.grad)

    dual = EARSIGV3Module(hidden_dim=32, slot_dim=16, num_views=3,
                         grid_height=2, view_grid_width=2, num_heads=4)
    dual_output = dual(torch.randn(2, 12, 32), torch.zeros(2, 2, 10), torch.ones(2, 2))
    assert dual_output["action_tokens"].shape == (2, 12, 32)

    tracks = EARSIGV3Module(hidden_dim=32, slot_dim=16, num_views=3, num_arms=2,
                            num_tracks=2, grid_height=2, view_grid_width=2, num_heads=4)
    track_state = torch.zeros(2, 1, 14 + 22 + 130)
    track_state[..., 34:36] = 1
    _, track_effectors, track_arms, track_labels = split_ea_rsig_v3_state(
        track_state, 14, tracks.target_dim
    )
    track_labels[..., :2] = 1
    track_output = tracks(
        torch.randn(2, 12, 32), track_effectors[:, 0], track_arms[:, 0]
    )
    assert track_output["action_tokens"].shape == (2, 15, 32)
    assert track_output["track_gate"].shape == (2, 2)
    assert track_output["plan_directions"].shape == (2, 3, 2, 2, 3)
    assert all(torch.isfinite(value) for value in tracks.losses(track_output, track_labels).values())
    track_video = torch.randn(2, 32, 2, 2, 6, requires_grad=True)
    track_adapted = tracks.inject_video_hidden(
        track_video, (torch.randn(2, 12, 32), track_effectors[:, 0], track_arms[:, 0])
    )
    assert track_adapted.shape == track_video.shape
    assert tracks.take_video_output() is not None
    track_adapted.square().mean().backward()
    assert torch.isfinite(tracks.video_alpha.grad)

    gated = tracks._compose_action_tokens(
        torch.randn(1, 4, 16), torch.randn(1, 2, 16), torch.randn(1, 2, 16),
        torch.randn(1, 16), torch.randn(1, 6, 32), torch.tensor([[1.0, 0.0]]),
    )["action_tokens"]
    torch.testing.assert_close(gated[:, [2, 3, 5, 7, 10, 12, 14]], torch.zeros(1, 7, 32))

    # Track validity is applied once at each Action-token boundary. Internal
    # relation/plan features remain available to their GT-masked auxiliary heads.
    ground = torch.randn(1, 4, 16)
    eo = torch.randn(1, 2, 16)
    og = torch.randn(1, 2, 16)
    progress = torch.randn(1, 16)
    plan = torch.randn(1, 6, 32)
    full = tracks._compose_action_tokens(ground, eo, og, progress, plan, torch.ones(1, 2))
    half = tracks._compose_action_tokens(
        ground, eo, og, progress, plan, torch.full((1, 2), 0.5)
    )
    per_track = [*range(8), *range(9, 15)]
    torch.testing.assert_close(
        half["action_tokens"][:, per_track],
        full["action_tokens"][:, per_track] * 0.5,
    )
    torch.testing.assert_close(
        half["progress_tokens"], full["progress_tokens"] * 0.75
    )

    diagnostics = tracks.training_diagnostics(track_output, track_labels)
    assert {
        "track_valid_loss", "track_gate_valid_mean", "track_gate_invalid_mean",
        "track_gate_precision", "track_gate_recall", "action_token_rms",
        "condition_gate_ground", "condition_gate_relation", "condition_gate_progress",
    } <= diagnostics.keys()
    assert all(value.ndim == 0 and not value.requires_grad for value in diagnostics.values())


def test_single_arm_goal_proximity_needs_a_visible_goal():
    module = EARSIGV3Module(hidden_dim=32, slot_dim=16, num_views=2,
                            grid_height=2, view_grid_width=2, num_heads=4)
    output = module(torch.randn(1, 8, 32), torch.zeros(1, 2, 10),
                    torch.tensor([[1.0, 0.0]]))
    targets = torch.zeros(1, 1, 125)
    targets[..., 55] = 1.0  # Simulator knows the goal position.
    targets[..., 54] = 1.0
    near = {**output, "goal_proximity": torch.zeros(1, 1)}
    far = {**output, "goal_proximity": torch.ones(1, 1)}
    torch.testing.assert_close(module.losses(near, targets)["relation"],
                               module.losses(far, targets)["relation"])
    targets[..., 122] = 1.0  # Goal is visible in the front RGB view.
    assert not torch.allclose(module.losses(near, targets)["relation"],
                              module.losses(far, targets)["relation"])

