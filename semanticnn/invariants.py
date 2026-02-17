from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarginInvariant:
    label: int
    kappa: float = 0.0


def margin(logits: np.ndarray, label: int) -> float:
    if logits.ndim != 1:
        raise ValueError("Expected 1D logits.")
    correct = logits[label]
    others = np.delete(logits, label)
    return float(correct - np.max(others))


def margin_lower_bound(lower: np.ndarray, upper: np.ndarray, label: int) -> float:
    """Sound lower bound for margin over interval logits.

    m(x) = y_label - max_{j!=label} y_j
    lower bound = lower[label] - max_{j!=label} upper[j]
    """
    correct_lb = lower[label]
    max_other_ub = np.max(np.delete(upper, label))
    return float(correct_lb - max_other_ub)
