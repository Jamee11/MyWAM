import numpy as np
import pytest

from benchmarks.robotwin.sidecar_gate_sweep import summarize, xyz_distance


def test_physical_xyz_excludes_rotation_and_gripper():
    a = np.zeros((32,20)); b = a.copy()
    b[:,0] = .003; b[:,1] = .004; b[:,10] = .005
    b[:,3:10] = 100
    assert xyz_distance(a,b) == pytest.approx(5.)
    assert xyz_distance(a,a) == 0
    b[0,0] = np.nan
    with pytest.raises(ValueError):
        xyz_distance(a,b)


def test_summary_pairs_with_same_scale_and_sample():
    rows = []
    for scale in [1, 100]:
        for condition in ['clean','light_strong']:
            a = np.zeros((32,20)); a[:,0] = scale
            if condition != 'clean': a[:,0] += .002
            rows.append(dict(sample_id=0, scale=scale, condition=condition, actions=a.tolist(),
                gt_metrics=dict(position_difference_mm=dict(mean=3.),rotation_difference_degrees=dict(mean=4.)),
                write_measurements=[dict(ratio=.01*scale)]))
    groups = summarize(rows)['groups']
    assert len(groups) == 4
    for r in groups:
        assert r['drift_xyz_mm'] == pytest.approx(0 if r['condition']=='clean' else 1)
    with pytest.raises(AssertionError):
        summarize(rows + rows[:1])
