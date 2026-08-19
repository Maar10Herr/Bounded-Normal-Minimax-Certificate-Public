#!/usr/bin/env python3
"""Run structural and recomputation-based certificate corruption checks."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from decimal import Decimal
from pathlib import Path

from src.verification.verify_certificate_arb import (
    validate_certificate_data,
    verify,
)


def run(certificate: Path) -> dict[str, object]:
    original = json.loads(certificate.read_text(encoding="utf-8"))
    structural_mutators = {
        "negative_weight": lambda d: d["prior"]["level_masses"].__setitem__(
            0, "-0.1"
        ),
        "invalid_normalization": lambda d: d["prior"][
            "level_masses"
        ].__setitem__(-1, "0.2"),
        "duplicate_support": lambda d: d["prior"]["levels"].__setitem__(
            1, d["prior"]["levels"][0]
        ),
        "outside_support": lambda d: d["prior"]["levels"].__setitem__(
            -1, "4.0001"
        ),
        "false_gap": lambda d: d["claims"].__setitem__(
            "minimax_gap_upper", "1e-30"
        ),
        "inverted_endpoint": lambda d: d["claims"].__setitem__(
            "bayes_risk_interval", ["0.9", "0.8"]
        ),
        "low_precision": lambda d: d["settings"].__setitem__(
            "precision_bits", 64
        ),
    }
    structural_rows: list[dict[str, object]] = []
    for name, mutate in structural_mutators.items():
        data = copy.deepcopy(original)
        mutate(data)
        rejected = False
        message = ""
        try:
            validate_certificate_data(data)
        except ValueError as exc:
            rejected = True
            message = str(exc)
        structural_rows.append(
            {"case": name, "rejected": rejected, "message": message}
        )

    # These corruptions remain syntactically valid.  They must be rejected by
    # a numerical recomputation, not by schema checks.  The coarser settings
    # are still outward-rounded and make this audit reasonably quick.
    def alter_weights(data: dict[str, object]) -> None:
        masses = data["prior"]["level_masses"]
        displacement = Decimal("0.000001")
        masses[0] = str(Decimal(masses[0]) + displacement)
        masses[1] = str(Decimal(masses[1]) - displacement)

    numerical_mutators = {
        "altered_weights": alter_weights,
        "altered_support": lambda d: d["prior"]["levels"].__setitem__(
            0, "0.5984799842028226"
        ),
        "understated_worst_risk": lambda d: (
            d["claims"]["worst_case_risk_interval"].__setitem__(
                1, d["claims"]["worst_case_risk_interval"][0]
            ),
            d["claims"]["minimax_risk_interval"].__setitem__(
                1, d["claims"]["worst_case_risk_interval"][0]
            ),
            d["claims"].__setitem__(
                "minimax_gap_upper",
                str(
                    __import__("decimal").Decimal(
                        d["claims"]["worst_case_risk_interval"][0]
                    )
                    - __import__("decimal").Decimal(
                        d["claims"]["bayes_risk_interval"][0]
                    )
                ),
            ),
        ),
    }
    numerical_rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="bounded-normal-corruption-") as tmp:
        temporary_root = Path(tmp)
        for name, mutate in numerical_mutators.items():
            data = copy.deepcopy(original)
            mutate(data)
            path = temporary_root / f"{name}.json"
            path.write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            try:
                result = verify(
                    path,
                    risk_panels=512,
                    curvature_cells=128,
                    max_nodes=2001,
                )
                numerical_rows.append(
                    {
                        "case": name,
                        "rejected": not result["passed"],
                        "passed": result["passed"],
                        "claim_checks": result["claim_checks"],
                        "recomputed_gap_upper": result["recomputed_enclosures"][
                            "minimax_gap_upper"
                        ],
                    }
                )
            except ValueError as exc:
                numerical_rows.append(
                    {
                        "case": name,
                        "rejected": True,
                        "passed": False,
                        "message": str(exc),
                    }
                )

    rows = structural_rows + numerical_rows
    return {
        "format": "bounded-normal-corruption-audit-v1",
        "certificate": str(certificate),
        "structural_cases": structural_rows,
        "numerical_recomputation_cases": numerical_rows,
        "serialized_partition_policy": (
            "the certificate contains no solver-generated branch list; the "
            "verifier constructs and covers [0,m] independently, so an omitted "
            "or malformed serialized branch cannot be supplied"
        ),
        "passed": all(bool(row["rejected"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--certificate",
        type=Path,
        default=Path("certificates/final/m4_epsilon_1e-8_arb.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.certificate)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
