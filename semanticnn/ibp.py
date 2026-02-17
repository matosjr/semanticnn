from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semanticnn.onnx_ir import FeedForwardIR


@dataclass(frozen=True)
class IntervalBounds:
    lower: np.ndarray
    upper: np.ndarray


def _affine_interval(weight: np.ndarray, bias: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> IntervalBounds:
    w_pos = np.maximum(weight, 0.0)
    w_neg = np.minimum(weight, 0.0)
    out_lower = w_pos @ lower + w_neg @ upper + bias
    out_upper = w_pos @ upper + w_neg @ lower + bias
    return IntervalBounds(out_lower, out_upper)


def bound_network(model: FeedForwardIR, lower: np.ndarray, upper: np.ndarray) -> IntervalBounds:
    lo, hi = lower.astype(np.float64), upper.astype(np.float64)
    for layer in model.layers:
        b = _affine_interval(layer.weight, layer.bias, lo, hi)
        lo, hi = b.lower, b.upper
        if layer.relu_after:
            lo = np.maximum(lo, 0.0)
            hi = np.maximum(hi, 0.0)
    return IntervalBounds(lo, hi)


def drift_inf_bound(ref_bounds: IntervalBounds, cand_bounds: IntervalBounds) -> float:
    # drift = cand - ref
    d_lower = cand_bounds.lower - ref_bounds.upper
    d_upper = cand_bounds.upper - ref_bounds.lower
    return float(np.max(np.maximum(np.abs(d_lower), np.abs(d_upper))))
