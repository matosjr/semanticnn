from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Certificate:
    ref_hash: str
    cand_hash: str
    invariant: dict[str, Any]
    region: dict[str, Any]
    method: dict[str, Any]
    bounds: dict[str, Any]
    theorem: str
    verdict: str
    assumptions: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
