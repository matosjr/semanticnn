from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from semanticnn.certificates import Certificate, file_sha256
from semanticnn.ibp import bound_network, drift_inf_bound
from semanticnn.invariants import MarginInvariant, margin, margin_lower_bound
from semanticnn.onnx_ir import load_feedforward_onnx
from semanticnn.regions import BoxRegion


@dataclass(frozen=True)
class VerificationConfig:
    samples: int = 256
    seed: int = 0


@dataclass(frozen=True)
class VerificationResult:
    status: str
    certificate: Certificate
    counterexamples: list[dict]


def _sample_box(region: BoxRegion, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=(n, region.lower.size))
    return region.lower + (region.upper - region.lower) * u


def verify_models(
    ref_model_path: str,
    cand_model_path: str,
    region: BoxRegion,
    invariant: MarginInvariant,
    config: VerificationConfig = VerificationConfig(),
) -> VerificationResult:
    ref = load_feedforward_onnx(ref_model_path)
    cand = load_feedforward_onnx(cand_model_path)

    if ref.input_dim != cand.input_dim or ref.output_dim != cand.output_dim:
        raise ValueError("Reference and candidate model dimensions are incompatible.")
    if region.lower.size != ref.input_dim:
        raise ValueError("Region dimensionality must match model input dimensionality.")

    ref_b = bound_network(ref, region.lower, region.upper)
    cand_b = bound_network(cand, region.lower, region.upper)
    eta = drift_inf_bound(ref_b, cand_b)

    ref_margin_lb = margin_lower_bound(ref_b.lower, ref_b.upper, invariant.label)
    cand_margin_lb = margin_lower_bound(cand_b.lower, cand_b.upper, invariant.label)

    theorem = "Margin Preservation via Bounded Drift"
    assumptions = [
        "Models are affine/ReLU feed-forward ONNX graphs supported by the MVP parser.",
        "Interval bound propagation is sound for supported operators.",
        "Region is a closed box with independent dimensions.",
    ]

    counterexamples: list[dict] = []
    status = "INCONCLUSIVE"

    if ref_margin_lb >= invariant.kappa + 2.0 * eta:
        status = "PASS"
    else:
        pts = _sample_box(region, config.samples, config.seed)
        for p in pts:
            ref_logits = ref.eval(p)
            cand_logits = cand.eval(p)
            if margin(ref_logits, invariant.label) >= invariant.kappa and margin(cand_logits, invariant.label) < invariant.kappa:
                counterexamples.append(
                    {
                        "x": p.tolist(),
                        "ref_margin": margin(ref_logits, invariant.label),
                        "cand_margin": margin(cand_logits, invariant.label),
                    }
                )
                status = "FAIL"
                break

    cert = Certificate(
        ref_hash=file_sha256(ref_model_path),
        cand_hash=file_sha256(cand_model_path),
        invariant={"type": "margin", "label": invariant.label, "kappa": invariant.kappa},
        region={"type": "box", "lower": region.lower.tolist(), "upper": region.upper.tolist()},
        method={"abstract_domain": "IBP", "refinement": "none", "falsification_samples": config.samples},
        bounds={
            "drift_inf_eta": eta,
            "ref_margin_lower_bound": ref_margin_lb,
            "cand_margin_lower_bound": cand_margin_lb,
        },
        theorem=theorem,
        verdict=status,
        assumptions=assumptions,
    )
    return VerificationResult(status=status, certificate=cert, counterexamples=counterexamples)


def write_result(result: VerificationResult, output_path: str | Path) -> None:
    payload = {
        "status": result.status,
        "certificate": json.loads(result.certificate.to_json()),
        "counterexamples": result.counterexamples,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2))
