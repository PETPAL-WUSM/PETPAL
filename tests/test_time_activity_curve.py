import pytest
import numpy as np
from petpal.utils.time_activity_curve import TimeActivityCurve

def test_post_init_sets_uncertainty_when_missing_and_converts_activity_to_float():
    times = np.array([0.0, 10.0, 20.0])
    activity = [1, 2, 3]  # list input should be converted to 1D float array
    tac = TimeActivityCurve(times=times, activity=activity)
    assert isinstance(tac.uncertainty, np.ndarray)
    assert tac.uncertainty.shape == times.shape
    assert np.all(np.isnan(tac.uncertainty))
    assert tac.activity.dtype == float
    assert tac.activity.shape == times.shape

def test_post_init_raises_when_activity_is_none():
    times = np.array([0.0, 5.0])
    with pytest.raises(ValueError):
        TimeActivityCurve(times=times, activity=None)

def test_post_init_asserts_on_shape_mismatch_between_fields():
    times = np.array([0.0, 10.0, 20.0])
    activity = np.array([1.0, 2.0])  # different shape -> should trigger assertion
    with pytest.raises(AssertionError):
        TimeActivityCurve(times=times, activity=activity)

def test_post_init_flattens_multidimensional_activity_to_1d_float():
    times = np.array([0.0, 10.0, 20.0])
    activity = np.array([[1, 2, 3]], dtype=int)  # 2D input should be ravelled and cast to float
    tac = TimeActivityCurve(times=times, activity=activity)
    assert tac.activity.ndim == 1
    assert tac.activity.shape == times.shape
    assert tac.activity.dtype == float
    assert np.allclose(tac.activity, np.array([1.0, 2.0, 3.0]))