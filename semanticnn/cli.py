from __future__ import annotations

import json

import click

from semanticnn.invariants import MarginInvariant
from semanticnn.regions import BoxRegion
from semanticnn.verification import VerificationConfig, verify_models, write_result


@click.group()
def main() -> None:
    """SemanticNN CLI."""


@main.command("verify")
@click.option("--ref", "ref_path", required=True, type=click.Path(exists=True))
@click.option("--cand", "cand_path", required=True, type=click.Path(exists=True))
@click.option("--region", "region_path", required=True, type=click.Path(exists=True))
@click.option("--label", required=True, type=int)
@click.option("--kappa", default=0.0, type=float)
@click.option("--samples", default=256, type=int)
@click.option("--seed", default=0, type=int)
@click.option("--out", "out_path", required=True, type=click.Path())
def verify_cmd(ref_path: str, cand_path: str, region_path: str, label: int, kappa: float, samples: int, seed: int, out_path: str) -> None:
    region = BoxRegion.from_json(region_path)
    inv = MarginInvariant(label=label, kappa=kappa)
    res = verify_models(ref_path, cand_path, region, inv, VerificationConfig(samples=samples, seed=seed))
    write_result(res, out_path)
    click.echo(json.dumps({"status": res.status, "out": out_path}, indent=2))


@main.command("inspect-sdnet")
@click.option("--ref", "ref_path", required=True, type=click.Path(exists=True))
@click.option("--cand", "cand_path", required=True, type=click.Path(exists=True))
@click.option("--region", "region_path", required=True, type=click.Path(exists=True))
@click.option("--label", default=0, type=int)
def inspect_sdnet_cmd(ref_path: str, cand_path: str, region_path: str, label: int) -> None:
    region = BoxRegion.from_json(region_path)
    inv = MarginInvariant(label=label, kappa=0.0)
    res = verify_models(ref_path, cand_path, region, inv)
    click.echo(
        json.dumps(
            {
                "drift_inf_eta": res.certificate.bounds["drift_inf_eta"],
                "ref_margin_lower_bound": res.certificate.bounds["ref_margin_lower_bound"],
                "cand_margin_lower_bound": res.certificate.bounds["cand_margin_lower_bound"],
            },
            indent=2,
        )
    )
