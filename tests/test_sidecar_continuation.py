"""A/C continuation must use their own complete state and preserve the prior screen."""
import json

import pytest

from scripts.train_sidecar_four_arm import screen_source


def test_default_screen_contract_is_unchanged(tmp_path):
    common = tmp_path / 'common'
    state = common / 'accel_state_step_4000'
    state.mkdir(parents=True)
    expected = dict(global_step=4000, opt_step=2000, epoch=0)
    (state / 'trainer_state.json').write_text(json.dumps(expected))
    result = screen_source(tmp_path / 'A_control', dict(common=str(common)))
    assert result == (False, 4000, 6000, common, state, expected)


def test_continuation_resumes_each_own_3k_state_and_stops_at_4k(tmp_path):
    sources = {}
    for name in ('A_control', 'C_wam_ramp'):
        state = tmp_path / name / 'old/accel_state_step_6000'
        state.mkdir(parents=True)
        (state / 'trainer_state.json').write_text(json.dumps(dict(global_step=6000, opt_step=3000, epoch=1)))
        sources[name] = dict(weights_dir=str(tmp_path / name / 'pinned'), full_state=str(state))
    plan = dict(continuation=True, resume_micro=6000, target_micro=8000, sources=sources)
    a, c = (screen_source(tmp_path / name, plan) for name in ('A_control', 'C_wam_ramp'))
    assert a[0:3] == c[0:3] == (True, 6000, 8000)
    assert a[3] != c[3] and a[4] != c[4]
    assert a[5] == c[5] == dict(global_step=6000, opt_step=3000, epoch=1)
    with pytest.raises(ValueError, match='exactly 1000'):
        screen_source(tmp_path / 'C_wam_ramp', dict(plan, target_micro=10000))
    (c[4] / 'trainer_state.json').write_text(json.dumps(dict(global_step=6000, opt_step=2000, epoch=1)))
    with pytest.raises(ValueError, match='complete source'):
        screen_source(tmp_path / 'C_wam_ramp', plan)
