#!/usr/bin/env python3
"""Independent Arb reconstruction of bounded-normal certificates.

This verifier deliberately does not import ``solver_certified``.  It uses
python-flint/Arb ball arithmetic, direct Gaussian-mixture ratios, a separate
Simpson implementation, and direct interval boxes for the curvature
integral.  The resulting enclosure is normally looser and slower than the
posterior-flow certificate; its purpose is independent falsification and
verification, not certificate construction.

Install the pinned trusted arithmetic dependency with

    uv sync

and run inside the project-local environment with

    uv run python verify_certificate_arb.py CERTIFICATE.json
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

try:
    from flint import arb, ctx
except ImportError as exc:  # pragma: no cover - dependency diagnostic
    raise SystemExit(
        "python-flint is required; run `uv sync` in the project directory"
    ) from exc


ctx.prec = 192
_BOUND_SCALE = 10**40


def _point(value: str | int) -> arb:
    return arb(str(value))


def _interval(lower: arb, upper: arb) -> arb:
    return lower.union(upper)


def _lower_scalar(value: arb) -> arb:
    integer = (value.lower() * _BOUND_SCALE).floor()
    return arb(integer - 1) / _BOUND_SCALE


def _upper_scalar(value: arb) -> arb:
    integer = (value.upper() * _BOUND_SCALE).ceil()
    return arb(integer + 1) / _BOUND_SCALE


def _lower_text(value: arb) -> str:
    return _lower_scalar(value).str(50)


def _upper_text(value: arb) -> str:
    return _upper_scalar(value).str(50)


def _width_text(value: arb) -> str:
    return _width_scalar(value).str(50)


def _width_scalar(value: arb) -> arb:
    return _upper_scalar(value) - _lower_scalar(value)


def _square(value: arb) -> arb:
    if value.contains(0):
        return arb(0).union(value.lower() * value.lower()).union(
            value.upper() * value.upper()
        )
    return value * value


def _maximum_upper(*values: arb) -> arb:
    return max((_upper_scalar(value) for value in values), key=float)


def _minimum_lower(*values: arb) -> arb:
    return min((_lower_scalar(value) for value in values), key=float)


def _normal_density(value: arb) -> arb:
    return (-_square(value) / 2).exp() / (2 * arb.pi()).sqrt()


@dataclass(frozen=True)
class RiskComponents:
    total: arb
    simpson_sum: arb
    quadrature_remainder: arb
    tail: arb


def _finite_decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{label} must be a decimal string or integer")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError(f"{label} must be finite")
    return result


def validate_certificate_data(data: dict[str, Any]) -> dict[str, Any]:
    """Validate the compact certificate before any expensive computation."""

    if not isinstance(data, dict):
        raise ValueError("certificate must be a JSON object")
    if data.get("format") not in {
        "bounded-normal-mean-arb-certificate-v0",
        "bounded-normal-mean-arb-certificate-v1",
    }:
        raise ValueError("unsupported Arb certificate format")
    m = _finite_decimal(data.get("m"), "m")
    if m <= 0:
        raise ValueError("m must be positive")

    prior = data.get("prior")
    if not isinstance(prior, dict):
        raise ValueError("prior must be an object")
    raw_levels = prior.get("levels")
    raw_masses = prior.get("level_masses")
    if not isinstance(raw_levels, list) or not isinstance(raw_masses, list):
        raise ValueError("prior levels and masses must be lists")
    if not raw_levels or len(raw_levels) != len(raw_masses):
        raise ValueError("prior levels and masses must have equal positive length")
    levels = [
        _finite_decimal(value, f"prior.levels[{index}]")
        for index, value in enumerate(raw_levels)
    ]
    masses = [
        _finite_decimal(value, f"prior.level_masses[{index}]")
        for index, value in enumerate(raw_masses)
    ]
    if levels[0] < 0 or levels[-1] > m:
        raise ValueError("prior support lies outside [0,m]")
    if any(left >= right for left, right in zip(levels, levels[1:])):
        raise ValueError("prior levels must be strictly increasing")
    if any(mass <= 0 for mass in masses):
        raise ValueError("all retained level masses must be positive")
    if sum(masses) != Decimal(1):
        raise ValueError("level masses must sum exactly to one")

    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("settings must be an object")
    cutoff = _finite_decimal(settings.get("tail_cutoff"), "tail_cutoff")
    tolerance = _finite_decimal(
        settings.get("separation_tolerance"),
        "separation_tolerance",
    )
    if cutoff <= 0 or cutoff != cutoff.to_integral_value():
        raise ValueError("tail_cutoff must be a positive integer")
    if tolerance <= 0:
        raise ValueError("separation_tolerance must be positive")
    for key in ("risk_panels", "curvature_cells", "max_nodes"):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    precision_bits = settings.get("precision_bits")
    if (
        isinstance(precision_bits, bool)
        or not isinstance(precision_bits, int)
        or precision_bits < 128
    ):
        raise ValueError("precision_bits must be an integer at least 128")

    claims = data.get("claims")
    claim_values: dict[str, Any] | None = None
    if claims is not None:
        if not isinstance(claims, dict):
            raise ValueError("claims must be an object")

        def interval_claim(key: str) -> tuple[Decimal, Decimal]:
            raw = claims.get(key)
            if not isinstance(raw, list) or len(raw) != 2:
                raise ValueError(f"claims.{key} must have two endpoints")
            lower = _finite_decimal(raw[0], f"claims.{key}[0]")
            upper = _finite_decimal(raw[1], f"claims.{key}[1]")
            if lower > upper:
                raise ValueError(f"claims.{key} is inverted")
            return lower, upper

        bayes = interval_claim("bayes_risk_interval")
        worst = interval_claim("worst_case_risk_interval")
        minimax = interval_claim("minimax_risk_interval")
        gap = _finite_decimal(
            claims.get("minimax_gap_upper"),
            "claims.minimax_gap_upper",
        )
        if minimax != (bayes[0], worst[1]):
            raise ValueError("claimed minimax interval must use Bayes lower and worst-risk upper")
        if gap != worst[1] - bayes[0]:
            raise ValueError("claimed gap must equal worst-risk upper minus Bayes lower")
        if gap < 0 or gap >= tolerance:
            raise ValueError("claimed gap must be nonnegative and below tolerance")
        claim_values = {
            "bayes": bayes,
            "worst": worst,
            "minimax": minimax,
            "gap": gap,
        }

    return {
        "m": m,
        "levels": levels,
        "masses": masses,
        "tail_cutoff": int(cutoff),
        "tolerance": tolerance,
        "precision_bits": precision_bits,
        "claims": claim_values,
    }


@dataclass(frozen=True)
class DirectMixture:
    locations: tuple[arb, ...]
    weights: tuple[arb, ...]
    m: arb

    @classmethod
    def from_certificate(cls, data: dict[str, Any]) -> "DirectMixture":
        prior = data["prior"]
        actual_locations: list[arb] = []
        actual_weights: list[arb] = []
        for level_text, mass_text in zip(
            prior["levels"], prior["level_masses"]
        ):
            level = _point(level_text)
            mass = _point(mass_text)
            if level.is_zero():
                actual_locations.append(arb(0))
                actual_weights.append(mass)
            else:
                actual_locations.extend((-level, level))
                actual_weights.extend((mass / 2, mass / 2))
        return cls(
            tuple(actual_locations),
            tuple(actual_weights),
            _point(data["m"]),
        )

    def posterior_state(
        self, y: arb
    ) -> tuple[arb, arb, arb, arb, arb]:
        terms = tuple(
            weight * (theta * y - theta * theta / 2).exp()
            for theta, weight in zip(self.locations, self.weights)
        )
        denominator = sum(terms, arb(0))
        probabilities = tuple(term / denominator for term in terms)
        mean = sum(
            (probability * theta for probability, theta in zip(
                probabilities, self.locations
            )),
            arb(0),
        )
        second = arb(0)
        third = arb(0)
        fourth_moment = arb(0)
        fifth_moment = arb(0)
        for probability, theta in zip(probabilities, self.locations):
            centered = theta - mean
            centered_second = _square(centered)
            centered_third = centered_second * centered
            centered_fourth = _square(centered_second)
            centered_fifth = centered_fourth * centered
            second += probability * centered_second
            third += probability * centered_third
            fourth_moment += probability * centered_fourth
            fifth_moment += probability * centered_fifth
        fourth = fourth_moment - 3 * _square(second)
        fifth = fifth_moment - 10 * third * second
        return mean, second, third, fourth, fifth

    def _integrand_point(self, t: arb, z: arb) -> arb:
        mean = self.posterior_state(t + z)[0]
        return _square(mean - t) * _normal_density(z)

    def _integrand_fourth(self, t: arb, z: arb) -> arb:
        mean, variance, third, fourth, fifth = self.posterior_state(t + z)
        q = mean - t
        variance_square = _square(variance)
        third_square = _square(third)
        u0 = _square(q)
        u1 = 2 * q * variance
        u2 = 2 * (variance_square + q * third)
        u3 = 2 * (3 * variance * third + q * fourth)
        u4 = 2 * (
            3 * third_square + 4 * variance * fourth + q * fifth
        )
        z2 = _square(z)
        z3 = z2 * z
        z4 = _square(z2)
        polynomial = (
            u4
            - 4 * z * u3
            + 6 * (z2 - 1) * u2
            + 4 * (3 * z - z3) * u1
            + (z4 - 6 * z2 + 3) * u0
        )
        return polynomial * _normal_density(z)

    def risk_components(
        self,
        t: arb,
        *,
        tail_cutoff: int,
        panels: int,
    ) -> RiskComponents:
        cutoff = arb(tail_cutoff)
        base = cutoff / panels
        width = 2 * base
        simpson_total = arb(0)
        remainder_total = arb(0)
        width_square = width * width
        coefficient = width_square * width_square * width / 2880
        for panel in range(panels):
            lower = -cutoff + 2 * panel * base
            center = lower + base
            upper = lower + width
            simpson = width / 6 * (
                self._integrand_point(t, lower)
                + 4 * self._integrand_point(t, center)
                + self._integrand_point(t, upper)
            )
            z_box = lower.union(upper)
            fourth = self._integrand_fourth(t, z_box)
            simpson_total += simpson
            remainder_total -= coefficient * fourth
        tail_upper = (
            8
            * self.m
            * self.m
            * _normal_density(cutoff)
            / cutoff
        )
        tail = arb(0).union(tail_upper)
        return RiskComponents(
            simpson_total + remainder_total + tail,
            simpson_total,
            remainder_total,
            tail,
        )

    def risk(
        self,
        t: arb,
        *,
        tail_cutoff: int,
        panels: int,
    ) -> arb:
        return self.risk_components(
            t,
            tail_cutoff=tail_cutoff,
            panels=panels,
        ).total

    def curvature(
        self,
        lower_t: arb,
        upper_t: arb,
        *,
        tail_cutoff: int,
        cells: int,
    ) -> arb:
        t_box = lower_t.union(upper_t)
        cutoff = arb(tail_cutoff)
        width = 2 * cutoff / cells
        integral = arb(0)
        for cell in range(cells):
            lower_z = -cutoff + cell * width
            upper_z = lower_z + width
            z_box = lower_z.union(upper_z)
            mean, variance, third, _, _ = self.posterior_state(
                t_box + z_box
            )
            q = mean - t_box
            state = _square(variance - 1) + q * third
            integral += width * state * _normal_density(z_box)

        m2 = self.m * self.m
        variance_term = _maximum_upper(arb(1), _square(m2 - 1))
        absolute_state = variance_term + 4 * m2 * m2
        tail_probability = 2 * _normal_density(cutoff) / cutoff
        tail_magnitude = 2 * absolute_state * tail_probability
        return 2 * integral + _interval(-tail_magnitude, tail_magnitude)


@dataclass(order=True)
class _Cell:
    priority: float
    serial: int
    lower: arb = field(compare=False)
    upper: arb = field(compare=False)
    lower_risk: arb = field(compare=False)
    upper_risk: arb = field(compare=False)
    bound: arb = field(compare=False)


@dataclass(frozen=True)
class SeparationResult:
    worst: arb
    nodes: int
    exhausted: bool
    incumbent_lower: arb
    certified_upper: arb
    max_simpson_rounding_width: arb
    max_quadrature_remainder_width: arb
    max_tail_upper: arb


def _cell_bound(
    mixture: DirectMixture,
    lower: arb,
    upper: arb,
    lower_risk: arb,
    upper_risk: arb,
    *,
    tail_cutoff: int,
    curvature_cells: int,
) -> arb:
    curvature = mixture.curvature(
        lower,
        upper,
        tail_cutoff=tail_cutoff,
        cells=curvature_cells,
    )
    curvature_lower = _lower_scalar(curvature)
    k = _maximum_upper(arb(0), -curvature_lower)
    left = _upper_scalar(lower_risk)
    right = _upper_scalar(upper_risk)
    width = _upper_scalar(upper - lower)
    correction = k * width * width / 2
    if correction.contains(0):
        return _maximum_upper(left, right)
    difference = right - left
    if _lower_scalar(difference) >= _upper_scalar(correction):
        return right
    if _upper_scalar(difference) <= -_upper_scalar(correction):
        return left
    vertex = left + _square(difference + correction) / (4 * correction)
    return _maximum_upper(left, right, vertex)


def maximize_risk(
    mixture: DirectMixture,
    *,
    tail_cutoff: int,
    risk_panels: int,
    curvature_cells: int,
    tolerance: arb,
    max_nodes: int,
) -> SeparationResult:
    risk_cache: dict[str, arb] = {}
    work_tolerance = tolerance / 2
    max_simpson_rounding_width = arb(0)
    max_quadrature_remainder_width = arb(0)
    max_tail_upper = arb(0)

    def point_risk(t: arb) -> arb:
        nonlocal max_simpson_rounding_width
        nonlocal max_quadrature_remainder_width
        nonlocal max_tail_upper
        # Use the interval representation, not a display-rounded midpoint:
        # distinct dyadic branch points must never alias in the cache.
        key = t.str(max(80, int(ctx.prec * 0.32) + 20), radius=True)
        if key not in risk_cache:
            components = mixture.risk_components(
                t, tail_cutoff=tail_cutoff, panels=risk_panels
            )
            risk_cache[key] = components.total
            max_simpson_rounding_width = _maximum_upper(
                max_simpson_rounding_width,
                _width_scalar(components.simpson_sum),
            )
            max_quadrature_remainder_width = _maximum_upper(
                max_quadrature_remainder_width,
                _width_scalar(components.quadrature_remainder),
            )
            max_tail_upper = _maximum_upper(
                max_tail_upper,
                _upper_scalar(components.tail),
            )
        return risk_cache[key]

    lower = arb(0)
    upper = mixture.m
    lower_risk = point_risk(lower)
    upper_risk = point_risk(upper)
    incumbent = _maximum_upper(
        _lower_scalar(lower_risk), _lower_scalar(upper_risk)
    )
    root_bound = _cell_bound(
        mixture,
        lower,
        upper,
        lower_risk,
        upper_risk,
        tail_cutoff=tail_cutoff,
        curvature_cells=curvature_cells,
    )
    heap: list[_Cell] = [
        _Cell(
            -float(_upper_scalar(root_bound)),
            0,
            lower,
            upper,
            lower_risk,
            upper_risk,
            root_bound,
        )
    ]
    serial = 1
    nodes = 1

    while heap and nodes < max_nodes:
        global_upper = max(
            (_upper_scalar(cell.bound) for cell in heap), key=float
        )
        if _upper_scalar(global_upper - incumbent) < work_tolerance:
            certified_upper = _maximum_upper(
                global_upper, incumbent + work_tolerance
            )
            return SeparationResult(
                incumbent.union(certified_upper),
                nodes,
                False,
                incumbent,
                certified_upper,
                max_simpson_rounding_width,
                max_quadrature_remainder_width,
                max_tail_upper,
            )
        cell = heapq.heappop(heap)
        midpoint = (cell.lower + cell.upper) / 2
        midpoint_risk = point_risk(midpoint)
        incumbent = _maximum_upper(
            incumbent, _lower_scalar(midpoint_risk)
        )
        for child_lower, child_upper, left_risk, right_risk in (
            (
                cell.lower,
                midpoint,
                cell.lower_risk,
                midpoint_risk,
            ),
            (
                midpoint,
                cell.upper,
                midpoint_risk,
                cell.upper_risk,
            ),
        ):
            bound = _cell_bound(
                mixture,
                child_lower,
                child_upper,
                left_risk,
                right_risk,
                tail_cutoff=tail_cutoff,
                curvature_cells=curvature_cells,
            )
            if _upper_scalar(bound - incumbent) <= work_tolerance:
                continue
            heapq.heappush(
                heap,
                _Cell(
                    -float(_upper_scalar(bound)),
                    serial,
                    child_lower,
                    child_upper,
                    left_risk,
                    right_risk,
                    bound,
                ),
            )
            serial += 1
        nodes += 2

    if heap:
        global_upper = max(
            (_upper_scalar(cell.bound) for cell in heap), key=float
        )
    else:
        global_upper = incumbent
    certified_upper = _maximum_upper(
        global_upper, incumbent + work_tolerance
    )
    return SeparationResult(
        incumbent.union(certified_upper),
        nodes,
        bool(heap),
        incumbent,
        certified_upper,
        max_simpson_rounding_width,
        max_quadrature_remainder_width,
        max_tail_upper,
    )


def verify(
    certificate_path: Path,
    *,
    risk_panels: int,
    curvature_cells: int,
    max_nodes: int,
) -> dict[str, Any]:
    if risk_panels < 1:
        raise ValueError("risk_panels must be positive")
    if curvature_cells < 1:
        raise ValueError("curvature_cells must be positive")
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    data = json.loads(certificate_path.read_text(encoding="utf-8"))
    validated = validate_certificate_data(data)
    ctx.prec = int(validated["precision_bits"])
    mixture = DirectMixture.from_certificate(data)
    cutoff = int(validated["tail_cutoff"])
    tolerance = _point(str(validated["tolerance"]))

    level_components = [
        mixture.risk_components(
            _point(level),
            tail_cutoff=cutoff,
            panels=risk_panels,
        )
        for level in data["prior"]["levels"]
    ]
    level_risks = [components.total for components in level_components]
    bayes = sum(
        (
            _point(mass) * risk
            for mass, risk in zip(
                data["prior"]["level_masses"], level_risks
            )
        ),
        arb(0),
    )
    bayes_simpson = sum(
        (
            _point(mass) * components.simpson_sum
            for mass, components in zip(
                data["prior"]["level_masses"], level_components
            )
        ),
        arb(0),
    )
    bayes_remainder = sum(
        (
            _point(mass) * components.quadrature_remainder
            for mass, components in zip(
                data["prior"]["level_masses"], level_components
            )
        ),
        arb(0),
    )
    bayes_tail = sum(
        (
            _point(mass) * components.tail
            for mass, components in zip(
                data["prior"]["level_masses"], level_components
            )
        ),
        arb(0),
    )
    separation = maximize_risk(
        mixture,
        tail_cutoff=cutoff,
        risk_panels=risk_panels,
        curvature_cells=curvature_cells,
        tolerance=tolerance,
        max_nodes=max_nodes,
    )
    bayes_lower = _lower_scalar(bayes)
    bayes_upper = _upper_scalar(bayes)
    worst_lower = _lower_scalar(separation.worst)
    worst_upper = _upper_scalar(separation.worst)
    gap_upper = _upper_scalar(worst_upper - bayes_lower)
    recomputed_enclosures = {
        "bayes_risk_interval": [
            _lower_text(bayes),
            _upper_text(bayes),
        ],
        "worst_case_risk_interval": [
            _lower_text(separation.worst),
            _upper_text(separation.worst),
        ],
        "minimax_risk_interval": [
            _lower_text(bayes),
            _upper_text(separation.worst),
        ],
        "minimax_gap_upper": _upper_text(gap_upper),
    }

    supplied_claims = validated["claims"]
    claim_checks: dict[str, bool]
    if supplied_claims is None:
        claim_checks = {
            "claims_present": False,
            "bayes_claim_contains_recomputation": False,
            "worst_claim_contains_recomputation": False,
            "minimax_claim_contains_recomputation": False,
            "gap_claim_dominates_recomputation": False,
        }
    else:
        bayes_claim_lower = _point(str(supplied_claims["bayes"][0]))
        bayes_claim_upper = _point(str(supplied_claims["bayes"][1]))
        worst_claim_lower = _point(str(supplied_claims["worst"][0]))
        worst_claim_upper = _point(str(supplied_claims["worst"][1]))
        minimax_claim_lower = _point(str(supplied_claims["minimax"][0]))
        minimax_claim_upper = _point(str(supplied_claims["minimax"][1]))
        gap_claim = _point(str(supplied_claims["gap"]))
        claim_checks = {
            "claims_present": True,
            "bayes_claim_contains_recomputation": (
                bayes_claim_lower <= bayes_lower
                and bayes_claim_upper >= bayes_upper
            ),
            "worst_claim_contains_recomputation": (
                worst_claim_lower <= worst_lower
                and worst_claim_upper >= worst_upper
            ),
            "minimax_claim_contains_recomputation": (
                minimax_claim_lower <= bayes_lower
                and minimax_claim_upper >= worst_upper
            ),
            "gap_claim_dominates_recomputation": (
                gap_claim >= gap_upper
            ),
        }
    passed = (
        not separation.exhausted
        and gap_upper < tolerance
        and all(claim_checks.values())
    )
    return {
        "certificate": str(certificate_path.resolve()),
        "arithmetic": (
            f"python-flint Arb balls, {int(validated['precision_bits'])} bits"
        ),
        "formula_independence": (
            "direct mixture-ratio intervals; no solver_certified import; "
            "independent Simpson and curvature-box implementations"
        ),
        "risk_panels": risk_panels,
        "curvature_cells": curvature_cells,
        "nodes": separation.nodes,
        "exhausted": separation.exhausted,
        "bayes_risk": str(bayes),
        "worst_risk": str(separation.worst),
        "gap_upper": str(gap_upper),
        "requested_tolerance": str(tolerance),
        "recomputed_enclosures": recomputed_enclosures,
        "claim_checks": claim_checks,
        "error_budget": {
            "bayes_risk": {
                "enclosure_width": _width_text(bayes),
                "simpson_sum_rounding_width": _width_text(bayes_simpson),
                "quadrature_remainder_width": _width_text(
                    bayes_remainder
                ),
                "gaussian_tail_upper": _upper_text(bayes_tail),
            },
            "worst_case_risk": {
                "enclosure_width": _width_text(separation.worst),
                "maximum_point_simpson_rounding_width": _upper_text(
                    separation.max_simpson_rounding_width
                ),
                "maximum_point_quadrature_remainder_width": _upper_text(
                    separation.max_quadrature_remainder_width
                ),
                "maximum_point_gaussian_tail_upper": _upper_text(
                    separation.max_tail_upper
                ),
            },
            "branch_and_bound": {
                "incumbent_lower": _lower_text(
                    separation.incumbent_lower
                ),
                "certified_upper": _upper_text(
                    separation.certified_upper
                ),
                "envelope_width": _upper_text(
                    separation.certified_upper
                    - separation.incumbent_lower
                ),
                "exhausted": separation.exhausted,
            },
            "arithmetic": {
                "precision_bits": int(validated["precision_bits"]),
                "rounding_policy": (
                    "all elementary operations are enclosed by Arb balls; "
                    "their radii are included in the displayed widths"
                ),
            },
        },
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--risk-panels", type=int)
    parser.add_argument("--curvature-cells", type=int)
    parser.add_argument("--max-nodes", type=int)
    args = parser.parse_args()
    supplied = json.loads(args.certificate.read_text(encoding="utf-8"))
    if not isinstance(supplied, dict):
        raise SystemExit("certificate must be a JSON object")
    settings = supplied.get("settings")
    if not isinstance(settings, dict):
        raise SystemExit("settings must be an object")
    result = verify(
        args.certificate,
        risk_panels=(
            args.risk_panels
            if args.risk_panels is not None
            else int(settings.get("risk_panels", 1024))
        ),
        curvature_cells=(
            args.curvature_cells
            if args.curvature_cells is not None
            else int(settings.get("curvature_cells", 256))
        ),
        max_nodes=(
            args.max_nodes
            if args.max_nodes is not None
            else int(settings.get("max_nodes", 2001))
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
