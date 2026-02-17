import numpy as np

from semanticnn.ibp import IntervalBounds, drift_inf_bound
from semanticnn.invariants import margin_lower_bound
from semanticnn.regions import BoxRegion


def test_region_validation_and_conversion():
    region = BoxRegion(lower=[-1, 0], upper=[1, 2])
    assert region.lower.dtype == np.float64
    assert np.all(region.lower <= region.upper)


def test_margin_lower_bound():
    lower = np.array([2.0, 0.0, -1.0])
    upper = np.array([3.0, 1.5, 0.2])
    assert margin_lower_bound(lower, upper, label=0) == 0.5


def test_drift_inf_bound():
    ref = IntervalBounds(lower=np.array([0.0, 1.0]), upper=np.array([1.0, 2.0]))
    cand = IntervalBounds(lower=np.array([0.4, 1.3]), upper=np.array([1.4, 2.6]))
    eta = drift_inf_bound(ref, cand)
    assert eta == 1.6
