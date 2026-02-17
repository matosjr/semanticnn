from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BoxRegion:
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        if lower.shape != upper.shape:
            raise ValueError("Region lower/upper shapes must match.")
        if np.any(lower > upper):
            raise ValueError("Region lower bounds must be <= upper bounds.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @staticmethod
    def from_json(path: str | Path) -> "BoxRegion":
        raw = json.loads(Path(path).read_text())
        return BoxRegion(lower=raw["lower"], upper=raw["upper"])
