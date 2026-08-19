#!/usr/bin/env python3
"""Fast structural and independence checks for the Arb verifier."""

from __future__ import annotations

import ast
import copy
import json
import sys
import unittest
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = ROOT / "certificates" / "final" / "m4_epsilon_1e-8_arb.json"
VERIFIER = ROOT / "src" / "verification" / "verify_certificate_arb.py"
if (ROOT / ".vendor").exists():
    sys.path.insert(0, str(ROOT / ".vendor"))

from src.verification.verify_certificate_arb import (
    validate_certificate_data,
    verify,
)


class VerifierCorruptionTests(unittest.TestCase):
    @staticmethod
    def certificate() -> dict[str, object]:
        return json.loads(CERTIFICATE.read_text(encoding="utf-8"))

    def assert_rejected(self, data: dict[str, object]) -> None:
        with self.assertRaises(ValueError):
            validate_certificate_data(data)

    def test_negative_mass_is_rejected(self) -> None:
        data = copy.deepcopy(self.certificate())
        data["prior"]["level_masses"][0] = "-0.1"
        self.assert_rejected(data)

    def test_duplicate_support_is_rejected(self) -> None:
        data = copy.deepcopy(self.certificate())
        data["prior"]["levels"][1] = data["prior"]["levels"][0]
        self.assert_rejected(data)

    def test_corrupted_claimed_gap_is_rejected(self) -> None:
        data = copy.deepcopy(self.certificate())
        data["claims"]["minimax_gap_upper"] = "1e-30"
        self.assert_rejected(data)

    def test_low_precision_is_rejected(self) -> None:
        data = copy.deepcopy(self.certificate())
        data["settings"]["precision_bits"] = 64
        self.assert_rejected(data)

    def test_support_outside_parameter_space_is_rejected(self) -> None:
        data = copy.deepcopy(self.certificate())
        data["prior"]["levels"][-1] = "4.0000000000000001"
        self.assert_rejected(data)

    def test_nonunit_normalization_is_rejected(self) -> None:
        data = copy.deepcopy(self.certificate())
        data["prior"]["level_masses"][-1] = "0.23357778864667602"
        self.assert_rejected(data)

    def test_zero_mass_is_rejected(self) -> None:
        data = copy.deepcopy(self.certificate())
        data["prior"]["level_masses"][0] = "0"
        self.assert_rejected(data)

    def test_nonfinite_input_is_rejected(self) -> None:
        for field in ("NaN", "Infinity", "-Infinity"):
            data = copy.deepcopy(self.certificate())
            data["m"] = field
            self.assert_rejected(data)

    def test_inverted_claim_interval_is_rejected(self) -> None:
        data = copy.deepcopy(self.certificate())
        data["claims"]["bayes_risk_interval"] = ["0.9", "0.8"]
        self.assert_rejected(data)

    def test_false_minimax_endpoints_are_rejected(self) -> None:
        data = copy.deepcopy(self.certificate())
        data["claims"]["minimax_risk_interval"][0] = "0.1"
        self.assert_rejected(data)

    def test_structurally_consistent_understatement_needs_recomputation(
        self,
    ) -> None:
        data = copy.deepcopy(self.certificate())
        false_worst = data["claims"]["worst_case_risk_interval"][0]
        data["claims"]["worst_case_risk_interval"][1] = false_worst
        data["claims"]["minimax_risk_interval"][1] = false_worst
        data["claims"]["minimax_gap_upper"] = str(
            Decimal(false_worst)
            - Decimal(data["claims"]["bayes_risk_interval"][0])
        )
        # This deliberately demonstrates the boundary of structural checks.
        # The full verifier must compare the claim with a recomputation.
        validate_certificate_data(data)

    def test_invalid_runtime_overrides_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            verify(
                CERTIFICATE,
                risk_panels=0,
                curvature_cells=1,
                max_nodes=1,
            )
        with self.assertRaises(ValueError):
            verify(
                CERTIFICATE,
                risk_panels=1,
                curvature_cells=0,
                max_nodes=1,
            )
        with self.assertRaises(ValueError):
            verify(
                CERTIFICATE,
                risk_panels=1,
                curvature_cells=1,
                max_nodes=0,
            )

    def test_no_discovery_solver_import(self) -> None:
        tree = ast.parse(VERIFIER.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        self.assertFalse(
            any(name.endswith("solver_fast") for name in imported)
        )
        self.assertFalse(
            any(name.endswith("solver_certified") for name in imported)
        )


if __name__ == "__main__":
    unittest.main()
