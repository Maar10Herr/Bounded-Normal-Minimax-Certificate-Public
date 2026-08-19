#!/usr/bin/env python3
"""Deterministic float64 prototype for the bounded normal mean problem.

This is research code, not a certificate.  Its two deliberate design choices are:

1. integrate with respect to Z ~ N(0,1), after cancelling the common Gaussian
   factor in every posterior ratio; and
2. maximize risk over the continuous parameter interval with an adaptive
   Taylor branch-and-bound, rather than an ex ante parameter grid.

The Taylor upper bounds are mathematically valid in exact arithmetic.  This
float64 implementation does not round them outward, so its reported gaps must
be treated as numerical evidence only.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
from numpy.polynomial.hermite import hermgauss


@dataclass(frozen=True)
class SymmetricPrior:
    """A symmetric prior parameterized by masses on nonnegative levels.

    ``level_masses[k]`` is the total mass at ``+/- levels[k]``.  At level zero
    the mass is placed once; at a positive level it is split equally.
    """

    levels: np.ndarray
    level_masses: np.ndarray

    def __post_init__(self) -> None:
        levels = np.asarray(self.levels, dtype=float)
        masses = np.asarray(self.level_masses, dtype=float)
        if levels.ndim != 1 or masses.ndim != 1 or levels.size != masses.size:
            raise ValueError("levels and level_masses must be equal-length vectors")
        if levels.size == 0 or np.any(~np.isfinite(levels)) or np.any(levels < 0):
            raise ValueError("levels must be a nonempty finite nonnegative vector")
        if np.any(~np.isfinite(masses)) or np.any(masses < 0):
            raise ValueError("level masses must be finite and nonnegative")
        if not math.isclose(float(masses.sum()), 1.0, rel_tol=2e-12, abs_tol=2e-14):
            raise ValueError("level masses must sum to one")
        if np.any(np.diff(levels) <= 0):
            raise ValueError("levels must be strictly increasing")
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "level_masses", masses)

    def atoms(self) -> tuple[np.ndarray, np.ndarray]:
        locations: list[float] = []
        weights: list[float] = []
        for level, mass in zip(self.levels, self.level_masses):
            if level == 0.0:
                locations.append(0.0)
                weights.append(float(mass))
            else:
                locations.extend([-float(level), float(level)])
                weights.extend([float(mass) / 2.0, float(mass) / 2.0])
        return np.asarray(locations), np.asarray(weights)

    def as_jsonable(self) -> dict[str, list[float]]:
        return {
            "levels": self.levels.tolist(),
            "level_masses": self.level_masses.tolist(),
        }


@dataclass(frozen=True)
class RiskEvaluation:
    risk: float
    first_derivative: float
    second_derivative: float


@dataclass(frozen=True)
class MaximumResult:
    maximizer: float
    lower: float
    upper: float
    nodes: int
    exhausted: bool


@dataclass(frozen=True)
class BoundDiagnostic:
    lower: float
    upper: float
    width: float
    second_derivative_lower: float
    second_derivative_upper: float


@dataclass(frozen=True)
class CurvatureInterval:
    lower: float
    upper: float


@dataclass(frozen=True)
class ExchangeIteration:
    iteration: int
    levels: list[float]
    level_masses: list[float]
    bayes_risk: float
    worst_t: float
    worst_risk_lower: float
    worst_risk_upper: float
    numerical_gap_lower: float
    numerical_gap_upper: float
    separation_nodes: int


@lru_cache(maxsize=None)
def _normal_hermite_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the immutable standard-normal Gauss--Hermite rule."""

    x, w = hermgauss(order)
    z = math.sqrt(2.0) * x
    normal_weights = w / math.sqrt(math.pi)
    z.setflags(write=False)
    normal_weights.setflags(write=False)
    return z, normal_weights


class GaussianMixtureRisk:
    """Risk and derivatives evaluated by deterministic Gauss-Hermite rules."""

    def __init__(self, prior: SymmetricPrior, order: int = 128):
        if order < 16:
            raise ValueError("Gauss-Hermite order must be at least 16")
        self.prior = prior
        self.locations, self.weights = prior.atoms()
        if np.any(self.weights <= 0):
            raise ValueError("all retained atom weights must be positive")
        self.z, self.normal_weights = _normal_hermite_rule(order)
        self.log_weights = np.log(self.weights)

    def posterior_state(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return q=delta(t+z)-t, posterior variance, and third central moment."""

        d = self.locations - float(t)
        logits = (
            self.log_weights[None, :]
            + self.z[:, None] * d[None, :]
            - 0.5 * d[None, :] ** 2
        )
        logits -= np.max(logits, axis=1, keepdims=True)
        unnormalized = np.exp(logits)
        posterior = unnormalized / np.sum(unnormalized, axis=1, keepdims=True)
        q = posterior @ d
        centered = d[None, :] - q[:, None]
        variance = np.sum(posterior * centered**2, axis=1)
        third = np.sum(posterior * centered**3, axis=1)
        return q, variance, third

    def posterior_moments_y(self, y: float) -> tuple[float, float, float]:
        """Return posterior mean, variance, and third central moment at y."""

        logits = (
            self.log_weights
            + self.locations * float(y)
            - 0.5 * self.locations**2
        )
        logits -= np.max(logits)
        posterior = np.exp(logits)
        posterior /= posterior.sum()
        mean = float(posterior @ self.locations)
        centered = self.locations - mean
        variance = float(posterior @ (centered * centered))
        third = float(posterior @ (centered * centered * centered))
        return mean, variance, third

    def posterior_cumulants_y(
        self, y: float
    ) -> tuple[float, float, float, float, float]:
        """Return mean and posterior cumulants of orders two through five."""

        logits = (
            self.log_weights
            + self.locations * float(y)
            - 0.5 * self.locations**2
        )
        logits -= np.max(logits)
        posterior = np.exp(logits)
        posterior /= posterior.sum()
        mean = float(posterior @ self.locations)
        centered = self.locations - mean
        variance = float(posterior @ (centered**2))
        third = float(posterior @ (centered**3))
        fourth_moment = float(posterior @ (centered**4))
        fifth_moment = float(posterior @ (centered**5))
        fourth = fourth_moment - 3.0 * variance * variance
        fifth = fifth_moment - 10.0 * third * variance
        return mean, variance, third, fourth, fifth

    def evaluate(self, t: float) -> RiskEvaluation:
        q, variance, third = self.posterior_state(t)
        risk = float(self.normal_weights @ (q * q))
        first = float(self.normal_weights @ (2.0 * q * (variance - 1.0)))
        second = float(
            self.normal_weights
            @ (2.0 * (variance - 1.0) ** 2 + 2.0 * q * third)
        )
        return RiskEvaluation(risk, first, second)

    def risk(self, t: float) -> float:
        return self.evaluate(t).risk

    def level_risks(self) -> np.ndarray:
        result = []
        for level in self.prior.levels:
            if level == 0.0:
                result.append(self.risk(0.0))
            else:
                result.append(0.5 * (self.risk(-level) + self.risk(level)))
        return np.asarray(result)

    def bayes_risk(self) -> float:
        return float(self.prior.level_masses @ self.level_risks())


def global_second_derivative_bound(m: float) -> float:
    """A proved, intentionally crude bound for sup_t |r''(t)|.

    q is in [-2m,2m], posterior variance V is in [0,m^2], and the posterior
    third central moment satisfies |kappa_3| <= 2m V <= 2m^3.  Since

        r''(t) = 2 E[(V-1)^2 + q*kappa_3],

    the returned constant is valid for every prior on [-m,m].
    """

    if not math.isfinite(m) or m <= 0:
        raise ValueError("m must be finite and positive")
    variance_term = max(1.0, abs(m * m - 1.0))
    return 2.0 * variance_term**2 + 8.0 * m**4


def _sharp_third_support(
    mean_lower: np.ndarray | float,
    mean_upper: np.ndarray | float,
    variance_lower: np.ndarray | float,
    variance_upper: np.ndarray | float,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sharp common-state enclosure for a bounded third central moment."""

    mean_lower_array = np.asarray(mean_lower, dtype=float)
    mean_upper_array = np.asarray(mean_upper, dtype=float)
    variance_lower_array = np.asarray(variance_lower, dtype=float)
    variance_upper_array = np.asarray(variance_upper, dtype=float)
    left_radius = np.maximum(0.0, mean_upper_array - alpha)
    right_radius = np.maximum(0.0, beta - mean_lower_array)

    lower = -left_radius * variance_upper_array
    lower = np.maximum(lower, -0.25 * left_radius**3)
    lower = np.maximum(
        lower,
        np.where(
            left_radius > 0.0,
            -left_radius * variance_upper_array
            + variance_lower_array**2
            / np.where(left_radius > 0.0, left_radius, 1.0),
            0.0,
        ),
    )

    upper = right_radius * variance_upper_array
    upper = np.minimum(upper, 0.25 * right_radius**3)
    upper = np.minimum(
        upper,
        np.where(
            right_radius > 0.0,
            right_radius * variance_upper_array
            - variance_lower_array**2
            / np.where(right_radius > 0.0, right_radius, 1.0),
            0.0,
        ),
    )
    return lower, upper


def local_joint_second_derivative_interval(
    evaluator: GaussianMixtureRisk,
    lower: float,
    upper: float,
) -> CurvatureInterval:
    """Dependence-preserving numerical enclosure of ``r''`` on a cell.

    This routine tests the mathematical bound using the evaluator's
    Gauss-Hermite rule.  The state inequalities are rigorous in exact
    arithmetic, but Gauss-Hermite error and float64 rounding are not enclosed.

    Write ``mu=E[theta|Y]``, ``V=Var(theta|Y)``, and let ``[alpha,beta]`` be
    the convex hull of the retained atoms.  The bound retains the same ``mu``
    in

        q*kappa3 = V * (mu-t) * (nu-mu),  nu in [alpha,beta],

    instead of multiplying independent intervals for q and kappa3.
    """

    if not (0.0 <= lower <= upper):
        raise ValueError("invalid parameter interval")
    if lower == upper:
        state = evaluator.evaluate(lower)
        return CurvatureInterval(
            state.second_derivative, state.second_derivative
        )

    center = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    alpha = float(np.min(evaluator.locations))
    beta = float(np.max(evaluator.locations))

    q_lower, _, _ = evaluator.posterior_state(lower)
    q_center, variance_center, third_center = evaluator.posterior_state(center)
    q_upper, _, _ = evaluator.posterior_state(upper)
    mean_lower = q_lower + lower
    mean_center = q_center + center
    mean_upper = q_upper + upper

    # delta'(y)=V(y)>=0, so these are the exact endpoint bounds for the
    # posterior mean on Y in [lower+z, upper+z].
    mean_lo = np.minimum(mean_lower, mean_upper)
    mean_hi = np.maximum(mean_lower, mean_upper)

    # kappa3/V is a weighted average of theta-mu and hence lies in
    # [alpha-mu,beta-mu].  Thus |(log V)'| is bounded by the local C below.
    local_log_slope = np.maximum(mean_hi - alpha, beta - mean_lo)
    variance_tube_upper = variance_center * np.exp(
        local_log_slope * half_width
    )
    variance_tube_lower = variance_center * np.exp(
        -local_log_slope * half_width
    )

    # Bhatia-Davis retains the same posterior mean:
    # V <= (beta-mu)(mu-alpha).  Maximize this concave quadratic on mean_lo:hi.
    variance_vertex = 0.5 * (alpha + beta)
    variance_mean = np.clip(variance_vertex, mean_lo, mean_hi)
    variance_bhatia_davis = np.maximum(
        0.0, (beta - variance_mean) * (variance_mean - alpha)
    )
    variance_upper = np.minimum(
        variance_tube_upper, variance_bhatia_davis
    )
    variance_lower = np.minimum(variance_tube_lower, variance_upper)

    # Flow-tube refinement.  Since kappa3'=kappa4 and
    # |kappa4| <= mu4+3V^2 <= C^2 V+3V^2, kappa3 stays in a short interval
    # about its center value.  Intersect this with
    # (alpha-mu)V <= kappa3 <= (beta-mu)V.
    fourth_bound = (
        local_log_slope * local_log_slope * variance_upper
        + 3.0 * variance_upper * variance_upper
    )
    third_flow_lower = third_center - half_width * fourth_bound
    third_flow_upper = third_center + half_width * fourth_bound
    third_support_lower, third_support_upper = _sharp_third_support(
        mean_lo,
        mean_hi,
        variance_lower,
        variance_upper,
        alpha,
        beta,
    )
    third_lower = np.maximum(third_flow_lower, third_support_lower)
    third_upper = np.minimum(third_flow_upper, third_support_upper)
    # Cover harmless last-bit inversions from float64 intersections.
    third_lo_fixed = np.minimum(third_lower, third_upper)
    third_hi_fixed = np.maximum(third_lower, third_upper)
    third_lower, third_upper = third_lo_fixed, third_hi_fixed

    for _ in range(2):
        third_magnitude = np.maximum(
            np.abs(third_lower), np.abs(third_upper)
        )
        variance_flow_lower = np.maximum(
            0.0, variance_center - half_width * third_magnitude
        )
        variance_flow_upper = variance_center + half_width * third_magnitude
        variance_lower = np.maximum(variance_lower, variance_flow_lower)
        variance_upper = np.minimum(variance_upper, variance_flow_upper)
        variance_lo_fixed = np.minimum(variance_lower, variance_upper)
        variance_hi_fixed = np.maximum(variance_lower, variance_upper)
        variance_lower, variance_upper = (
            variance_lo_fixed,
            variance_hi_fixed,
        )
        third_support_lower, third_support_upper = _sharp_third_support(
            mean_lo,
            mean_hi,
            variance_lower,
            variance_upper,
            alpha,
            beta,
        )
        third_lower = np.maximum(third_lower, third_support_lower)
        third_upper = np.minimum(third_upper, third_support_upper)
        third_lo_fixed = np.minimum(third_lower, third_upper)
        third_hi_fixed = np.maximum(third_lower, third_upper)
        third_lower, third_upper = third_lo_fixed, third_hi_fixed

    variance_part = np.maximum(
        (variance_lower - 1.0) ** 2,
        (variance_upper - 1.0) ** 2,
    )
    variance_part_lower = np.where(
        (variance_lower <= 1.0) & (1.0 <= variance_upper),
        0.0,
        np.minimum(
            (variance_lower - 1.0) ** 2,
            (variance_upper - 1.0) ** 2,
        ),
    )

    # If mu-t >= 0, the largest positive (mu-t)(nu-mu) uses t=lower,
    # nu=beta.  If mu-t <= 0, it uses t=upper, nu=alpha.  Each remaining
    # maximization is a concave quadratic in the *same* mu.
    positive_lo = np.maximum(mean_lo, lower)
    positive_hi = np.minimum(mean_hi, beta)
    positive_mu = np.clip(
        0.5 * (lower + beta), positive_lo, positive_hi
    )
    positive_product = np.where(
        positive_lo <= positive_hi,
        np.maximum(0.0, (positive_mu - lower) * (beta - positive_mu)),
        0.0,
    )

    negative_lo = np.maximum(mean_lo, alpha)
    negative_hi = np.minimum(mean_hi, upper)
    negative_mu = np.clip(
        0.5 * (alpha + upper), negative_lo, negative_hi
    )
    negative_product = np.where(
        negative_lo <= negative_hi,
        np.maximum(0.0, (upper - negative_mu) * (negative_mu - alpha)),
        0.0,
    )

    shared_mean_product = variance_upper * np.maximum(
        positive_product, negative_product
    )

    # A second, locally convergent enclosure follows the coupled ODE
    # q_t=V-1, V_t=kappa3, kappa3_t=kappa4.  It is intersected at the product
    # level with the shared-mean bound above; neither feasible set is replaced
    # by an unchecked Cartesian product.
    q_slope = np.maximum(
        np.abs(variance_lower - 1.0),
        np.abs(variance_upper - 1.0),
    )
    q_flow_lower = q_center - half_width * q_slope
    q_flow_upper = q_center + half_width * q_slope
    q_support_lower = mean_lo - upper
    q_support_upper = mean_hi - lower
    q_interval_lower = np.maximum(q_flow_lower, q_support_lower)
    q_interval_upper = np.minimum(q_flow_upper, q_support_upper)
    q_lo_fixed = np.minimum(q_interval_lower, q_interval_upper)
    q_hi_fixed = np.maximum(q_interval_lower, q_interval_upper)
    q_interval_lower, q_interval_upper = q_lo_fixed, q_hi_fixed

    product_candidates = np.stack(
        [
            q_interval_lower * third_lower,
            q_interval_lower * third_upper,
            q_interval_upper * third_lower,
            q_interval_upper * third_upper,
        ],
        axis=0,
    )
    flow_product_lower = np.min(product_candidates, axis=0)
    flow_product = np.max(product_candidates, axis=0)
    q_third_part = np.minimum(shared_mean_product, flow_product)

    # Shared-mean lower bound.  For a negative product, the extremal nu is
    # alpha when mu-t>=0 and beta when mu-t<=0.  The remaining products are
    # monotone on the admissible half-intervals, so their maxima occur at the
    # displayed mean endpoints.
    positive_q_negative = np.maximum(
        0.0, (mean_hi - lower) * (mean_hi - alpha)
    )
    negative_q_negative = np.maximum(
        0.0, (upper - mean_lo) * (beta - mean_lo)
    )
    shared_mean_product_lower = -variance_upper * np.maximum(
        positive_q_negative, negative_q_negative
    )
    q_third_part_lower = np.maximum(
        shared_mean_product_lower, flow_product_lower
    )

    integrand_upper = 2.0 * (variance_part + q_third_part)
    integrand_lower = 2.0 * (
        variance_part_lower + q_third_part_lower
    )

    # Centered enclosure of the *whole* coupled curvature integrand.  At
    # fixed z, with y=t+z,
    #
    #   S(t)=2[(V-1)^2+q*kappa3],
    #   S'(t)=6(V-1)kappa3+2q*kappa4.
    #
    # Keeping the exact common center S(c) avoids paying the Cartesian
    # dependency loss between the two summands.  The support/flow tubes above
    # bound the derivative on the cell, so intersecting the two enclosures is
    # admissible and is locally first-order sharp.
    q_magnitude = np.maximum(
        np.abs(q_interval_lower), np.abs(q_interval_upper)
    )
    third_magnitude = np.maximum(
        np.abs(third_lower), np.abs(third_upper)
    )
    variance_minus_one_magnitude = np.maximum(
        np.abs(variance_lower - 1.0),
        np.abs(variance_upper - 1.0),
    )
    integrand_center = 2.0 * (
        (variance_center - 1.0) ** 2 + q_center * third_center
    )
    integrand_slope_magnitude = (
        6.0 * variance_minus_one_magnitude * third_magnitude
        + 2.0 * q_magnitude * fourth_bound
    )
    centered_lower = (
        integrand_center - half_width * integrand_slope_magnitude
    )
    centered_upper = (
        integrand_center + half_width * integrand_slope_magnitude
    )
    integrand_lower = np.maximum(integrand_lower, centered_lower)
    integrand_upper = np.minimum(integrand_upper, centered_upper)
    integrand_lo_fixed = np.minimum(integrand_lower, integrand_upper)
    integrand_hi_fixed = np.maximum(integrand_lower, integrand_upper)
    integrand_lower, integrand_upper = (
        integrand_lo_fixed,
        integrand_hi_fixed,
    )

    result_upper = float(evaluator.normal_weights @ integrand_upper)
    result_lower = float(evaluator.normal_weights @ integrand_lower)
    if not math.isfinite(result_lower) or not math.isfinite(result_upper):
        raise ArithmeticError("invalid local joint curvature bound")
    if result_lower > result_upper + 1e-12 * max(
        1.0, abs(result_lower), abs(result_upper)
    ):
        raise ArithmeticError("inverted local joint curvature enclosure")
    return CurvatureInterval(
        min(result_lower, result_upper),
        max(result_lower, result_upper),
    )


def _posterior_flow_enclosure(
    evaluator: GaussianMixtureRisk,
    y_lower: float,
    y_upper: float,
) -> tuple[float, float, float, float, float, float]:
    """Numerical flow tube for (mu,V,kappa3) on one y interval."""

    if y_lower > y_upper:
        raise ValueError("invalid observation interval")
    center = 0.5 * (y_lower + y_upper)
    half_width = 0.5 * (y_upper - y_lower)
    alpha = float(np.min(evaluator.locations))
    beta = float(np.max(evaluator.locations))
    mean_lower = evaluator.posterior_moments_y(y_lower)[0]
    mean_center, variance_center, third_center = (
        evaluator.posterior_moments_y(center)
    )
    mean_upper = evaluator.posterior_moments_y(y_upper)[0]
    mean_lo = min(mean_lower, mean_upper)
    mean_hi = max(mean_lower, mean_upper)
    local_log_slope = max(mean_hi - alpha, beta - mean_lo)

    variance_lower = variance_center * math.exp(
        -local_log_slope * half_width
    )
    variance_upper = variance_center * math.exp(
        local_log_slope * half_width
    )
    variance_vertex = min(
        max(0.5 * (alpha + beta), mean_lo), mean_hi
    )
    variance_upper = min(
        variance_upper,
        max(
            0.0,
            (beta - variance_vertex) * (variance_vertex - alpha),
        ),
    )
    variance_lower = min(variance_lower, variance_upper)

    fourth_bound = (
        local_log_slope * local_log_slope * variance_upper
        + 3.0 * variance_upper * variance_upper
    )
    third_support_lower, third_support_upper = _sharp_third_support(
        mean_lo,
        mean_hi,
        variance_lower,
        variance_upper,
        alpha,
        beta,
    )
    third_lower = max(
        third_center - half_width * fourth_bound,
        float(third_support_lower),
    )
    third_upper = min(
        third_center + half_width * fourth_bound,
        float(third_support_upper),
    )
    if third_lower > third_upper:
        third_lower, third_upper = third_upper, third_lower
    for _ in range(2):
        third_magnitude = max(abs(third_lower), abs(third_upper))
        variance_lower = max(
            variance_lower,
            max(0.0, variance_center - half_width * third_magnitude),
        )
        variance_upper = min(
            variance_upper,
            variance_center + half_width * third_magnitude,
        )
        if variance_lower > variance_upper:
            variance_lower, variance_upper = variance_upper, variance_lower
        third_support_lower, third_support_upper = _sharp_third_support(
            mean_lo,
            mean_hi,
            variance_lower,
            variance_upper,
            alpha,
            beta,
        )
        third_lower = max(third_lower, float(third_support_lower))
        third_upper = min(third_upper, float(third_support_upper))
        if third_lower > third_upper:
            third_lower, third_upper = third_upper, third_lower
    return (
        mean_lo,
        mean_hi,
        variance_lower,
        variance_upper,
        third_lower,
        third_upper,
    )


def _interval_product(
    left: tuple[float, float],
    right: tuple[float, float],
) -> tuple[float, float]:
    values = [
        left_value * right_value
        for left_value in left
        for right_value in right
    ]
    return min(values), max(values)


def _interval_square(interval: tuple[float, float]) -> tuple[float, float]:
    lower, upper = interval
    if lower <= 0.0 <= upper:
        return 0.0, max(lower * lower, upper * upper)
    values = (lower * lower, upper * upper)
    return min(values), max(values)


def _interval_add(
    *intervals: tuple[float, float],
) -> tuple[float, float]:
    return (
        sum(interval[0] for interval in intervals),
        sum(interval[1] for interval in intervals),
    )


def _interval_scale(
    scalar: float,
    interval: tuple[float, float],
) -> tuple[float, float]:
    values = (scalar * interval[0], scalar * interval[1])
    return min(values), max(values)


def _q_third_joint_interval(
    *,
    alpha: float,
    beta: float,
    t: float,
    mean_lower: float,
    mean_upper: float,
    variance_upper: float,
    third_lower: float,
    third_upper: float,
) -> tuple[float, float]:
    """Enclose q*kappa3 while retaining their shared posterior mean."""

    q_interval = (mean_lower - t, mean_upper - t)
    flow_lower, flow_upper = _interval_product(
        q_interval, (third_lower, third_upper)
    )

    positive_lo = max(mean_lower, t)
    positive_hi = min(mean_upper, beta)
    if positive_lo <= positive_hi:
        positive_mu = min(
            max(0.5 * (t + beta), positive_lo), positive_hi
        )
        positive = max(
            0.0, (positive_mu - t) * (beta - positive_mu)
        )
    else:
        positive = 0.0
    negative_lo = max(mean_lower, alpha)
    negative_hi = min(mean_upper, t)
    if negative_lo <= negative_hi:
        negative_mu = min(
            max(0.5 * (alpha + t), negative_lo), negative_hi
        )
        negative = max(
            0.0, (t - negative_mu) * (negative_mu - alpha)
        )
    else:
        negative = 0.0
    shared_upper = variance_upper * max(positive, negative)

    positive_negative = (
        max(0.0, (mean_upper - t) * (mean_upper - alpha))
        if mean_upper >= t
        else 0.0
    )
    negative_negative = (
        max(0.0, (t - mean_lower) * (beta - mean_lower))
        if mean_lower <= t
        else 0.0
    )
    shared_lower = -variance_upper * max(
        positive_negative, negative_negative
    )
    return max(flow_lower, shared_lower), min(flow_upper, shared_upper)


def risk_point_trapezoid_diagnostic(
    evaluator: GaussianMixtureRisk,
    t: float,
    *,
    tail_cutoff: float = 8.0,
    z_cells: int = 128,
) -> tuple[float, float]:
    """Numerically test a local-state trapezoid enclosure of point risk.

    Exact endpoint values are combined with a local enclosure of
    ``H''(z)`` for ``H=q^2*phi``.  The formulas are rigorous in exact
    arithmetic; this float64 diagnostic does not provide directed rounding.
    """

    if tail_cutoff <= 0.0 or z_cells < 2:
        raise ValueError("invalid quadrature settings")
    alpha = float(np.min(evaluator.locations))
    beta = float(np.max(evaluator.locations))
    edges = np.linspace(-tail_cutoff, tail_cutoff, z_cells + 1)
    width = float(edges[1] - edges[0])

    def density(z: float) -> float:
        return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

    endpoint_h = []
    for z in edges:
        mean = evaluator.posterior_moments_y(t + float(z))[0]
        endpoint_h.append((mean - t) ** 2 * density(float(z)))
    trapezoid = width * (
        0.5 * endpoint_h[0]
        + sum(endpoint_h[1:-1])
        + 0.5 * endpoint_h[-1]
    )
    lower_error = 0.0
    upper_error = 0.0
    for z_lower, z_upper in zip(edges[:-1], edges[1:]):
        (
            mean_lower,
            mean_upper,
            variance_lower,
            variance_upper,
            third_lower,
            third_upper,
        ) = _posterior_flow_enclosure(
            evaluator, t + float(z_lower), t + float(z_upper)
        )
        q_interval = (mean_lower - t, mean_upper - t)
        q_square = _interval_square(q_interval)
        variance_square = (
            variance_lower * variance_lower,
            variance_upper * variance_upper,
        )
        q_third = _q_third_joint_interval(
            alpha=alpha,
            beta=beta,
            t=t,
            mean_lower=mean_lower,
            mean_upper=mean_upper,
            variance_upper=variance_upper,
            third_lower=third_lower,
            third_upper=third_upper,
        )
        z_square = _interval_square((float(z_lower), float(z_upper)))
        z_square_minus_one = (z_square[0] - 1.0, z_square[1] - 1.0)
        density_lower = density(
            max(abs(float(z_lower)), abs(float(z_upper)))
        )
        nearest = (
            0.0
            if z_lower <= 0.0 <= z_upper
            else min(abs(float(z_lower)), abs(float(z_upper)))
        )
        density_upper = density(nearest)

        term_one = (2.0 * variance_square[0], 2.0 * variance_square[1])
        term_two = (2.0 * q_third[0], 2.0 * q_third[1])
        term_three = _interval_product(z_square_minus_one, q_square)
        z_q = _interval_product(
            (float(z_lower), float(z_upper)), q_interval
        )
        z_q_variance = _interval_product(
            z_q, (variance_lower, variance_upper)
        )
        term_four = (
            -4.0 * z_q_variance[1],
            -4.0 * z_q_variance[0],
        )
        bracket = (
            term_one[0] + term_two[0] + term_three[0] + term_four[0],
            term_one[1] + term_two[1] + term_three[1] + term_four[1],
        )
        second = _interval_product(
            bracket, (density_lower, density_upper)
        )
        lower_error += max(0.0, second[1]) * width**3 / 12.0
        upper_error += max(0.0, -second[0]) * width**3 / 12.0

    q_tail = max(abs(alpha - t), abs(beta - t))
    tail = (
        2.0
        * q_tail
        * q_tail
        * density(tail_cutoff)
        / tail_cutoff
    )
    return max(0.0, trapezoid - lower_error), (
        trapezoid + upper_error + tail
    )


def risk_point_simpson_diagnostic(
    evaluator: GaussianMixtureRisk,
    t: float,
    *,
    tail_cutoff: float = 8.0,
    z_panels: int = 64,
) -> tuple[float, float]:
    """Numerically test a fourth-order joint-state Simpson enclosure."""

    if tail_cutoff <= 0.0 or z_panels < 1:
        raise ValueError("invalid quadrature settings")
    alpha = float(np.min(evaluator.locations))
    beta = float(np.max(evaluator.locations))
    edges = np.linspace(-tail_cutoff, tail_cutoff, 2 * z_panels + 1)

    def density(z: float) -> float:
        return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

    lower_total = 0.0
    upper_total = 0.0
    for panel in range(z_panels):
        z_lower = float(edges[2 * panel])
        z_center = float(edges[2 * panel + 1])
        z_upper = float(edges[2 * panel + 2])
        panel_width = z_upper - z_lower
        endpoint_values = []
        for z in (z_lower, z_center, z_upper):
            mean = evaluator.posterior_moments_y(t + z)[0]
            endpoint_values.append((mean - t) ** 2 * density(z))
        simpson = panel_width * (
            endpoint_values[0]
            + 4.0 * endpoint_values[1]
            + endpoint_values[2]
        ) / 6.0

        (
            mean_lower,
            mean_upper,
            variance_lower,
            variance_upper,
            third_lower,
            third_upper,
        ) = _posterior_flow_enclosure(
            evaluator, t + z_lower, t + z_upper
        )
        q_interval = (mean_lower - t, mean_upper - t)
        variance_interval = (variance_lower, variance_upper)
        third_interval = (third_lower, third_upper)
        q_square = _interval_square(q_interval)
        variance_square = _interval_square(variance_interval)
        third_square = _interval_square(third_interval)

        local_c = max(mean_upper - alpha, beta - mean_lower)
        _, _, _, fourth_center, _ = evaluator.posterior_cumulants_y(
            t + z_center
        )
        fifth_magnitude = (
            local_c**3 * variance_upper
            + 10.0 * local_c * variance_upper * variance_upper
        )
        fourth_flow = (
            fourth_center - 0.5 * panel_width * fifth_magnitude,
            fourth_center + 0.5 * panel_width * fifth_magnitude,
        )
        fourth_generic = (
            -3.0 * variance_upper * variance_upper,
            local_c * local_c * variance_upper,
        )
        fourth_interval = (
            max(fourth_flow[0], fourth_generic[0]),
            min(fourth_flow[1], fourth_generic[1]),
        )
        if fourth_interval[0] > fourth_interval[1]:
            fourth_interval = (
                min(fourth_interval),
                max(fourth_interval),
            )
        fifth_interval = (-fifth_magnitude, fifth_magnitude)

        u0 = q_square
        u1 = _interval_scale(
            2.0, _interval_product(q_interval, variance_interval)
        )
        u2 = _interval_scale(
            2.0,
            _interval_add(
                variance_square,
                _q_third_joint_interval(
                    alpha=alpha,
                    beta=beta,
                    t=t,
                    mean_lower=mean_lower,
                    mean_upper=mean_upper,
                    variance_upper=variance_upper,
                    third_lower=third_lower,
                    third_upper=third_upper,
                ),
            ),
        )
        u3 = _interval_scale(
            2.0,
            _interval_add(
                _interval_scale(
                    3.0,
                    _interval_product(
                        variance_interval, third_interval
                    ),
                ),
                _interval_product(q_interval, fourth_interval),
            ),
        )
        u4 = _interval_scale(
            2.0,
            _interval_add(
                _interval_scale(3.0, third_square),
                _interval_scale(
                    4.0,
                    _interval_product(
                        variance_interval, fourth_interval
                    ),
                ),
                _interval_product(q_interval, fifth_interval),
            ),
        )

        z_interval = (z_lower, z_upper)
        z_square = _interval_square(z_interval)
        z_cube = _interval_product(z_square, z_interval)
        z_fourth = _interval_square(z_square)
        coefficient_two = (z_square[0] - 1.0, z_square[1] - 1.0)
        coefficient_one = _interval_add(
            _interval_scale(3.0, z_interval),
            _interval_scale(-1.0, z_cube),
        )
        coefficient_zero = _interval_add(
            z_fourth,
            _interval_scale(-6.0, z_square),
            (3.0, 3.0),
        )
        bracket = _interval_add(
            u4,
            _interval_scale(
                -4.0, _interval_product(z_interval, u3)
            ),
            _interval_scale(
                6.0, _interval_product(coefficient_two, u2)
            ),
            _interval_scale(
                4.0, _interval_product(coefficient_one, u1)
            ),
            _interval_product(coefficient_zero, u0),
        )
        density_lower = density(max(abs(z_lower), abs(z_upper)))
        nearest = (
            0.0
            if z_lower <= 0.0 <= z_upper
            else min(abs(z_lower), abs(z_upper))
        )
        density_upper = density(nearest)
        fourth_h = _interval_product(
            bracket, (density_lower, density_upper)
        )
        coefficient = panel_width**5 / 2880.0
        lower_total += simpson - fourth_h[1] * coefficient
        upper_total += simpson - fourth_h[0] * coefficient

    q_tail = max(abs(alpha - t), abs(beta - t))
    tail = (
        2.0
        * q_tail
        * q_tail
        * density(tail_cutoff)
        / tail_cutoff
    )
    return max(0.0, lower_total), upper_total + tail


def local_joint_second_derivative_upper(
    evaluator: GaussianMixtureRisk,
    lower: float,
    upper: float,
) -> float:
    """Compatibility wrapper returning the upper side of the joint bound."""

    return local_joint_second_derivative_interval(
        evaluator, lower, upper
    ).upper


def _semiconvex_secant_upper(
    lower_value: float,
    upper_value: float,
    curvature_magnitude: float,
    width: float,
) -> float:
    """Maximize the exact linear-plus-parabolic secant majorant."""

    if curvature_magnitude <= 0.0 or width <= 0.0:
        return max(lower_value, upper_value)
    coefficient = 0.5 * curvature_magnitude * width * width
    difference = upper_value - lower_value
    fraction = min(
        1.0, max(0.0, 0.5 + difference / (2.0 * coefficient))
    )
    return (
        (1.0 - fraction) * lower_value
        + fraction * upper_value
        + coefficient * fraction * (1.0 - fraction)
    )


def secant_cell_diagnostic(
    evaluator: GaussianMixtureRisk,
    lower: float,
    upper: float,
) -> BoundDiagnostic:
    """Return the joint secant upper bound on one parameter cell."""

    if not (0.0 <= lower < upper):
        raise ValueError("invalid parameter interval")
    lower_risk = evaluator.risk(lower)
    upper_risk = evaluator.risk(upper)
    curvature = local_joint_second_derivative_interval(
        evaluator, lower, upper
    )
    width = upper - lower
    cell_upper = _semiconvex_secant_upper(
        lower_risk,
        upper_risk,
        max(0.0, -curvature.lower),
        width,
    )
    cell_lower = max(lower_risk, upper_risk)
    return BoundDiagnostic(
        lower=cell_lower,
        upper=cell_upper,
        width=cell_upper - cell_lower,
        second_derivative_lower=curvature.lower,
        second_derivative_upper=curvature.upper,
    )


def maximize_risk_adaptive_joint(
    evaluator: GaussianMixtureRisk,
    m: float,
    tolerance: float = 1e-8,
    max_nodes: int = 250_000,
) -> MaximumResult:
    """Adaptive maximization using the local joint state and secant bound.

    If ``f'' >= -K`` on ``[a,b]``, semiconvex interpolation gives

        sup_[a,b] f <= max(f(a),f(b)) + K*(b-a)^2/8.

    Unlike the midpoint Taylor bound, this cancels the cell's linear trend.
    The result remains numerical evidence until quadrature and rounding are
    enclosed.
    """

    if tolerance <= 0 or max_nodes < 1:
        raise ValueError("tolerance and max_nodes must be positive")
    cache: dict[float, float] = {}

    def value(t: float) -> float:
        key = float(t)
        if key not in cache:
            cache[key] = evaluator.risk(key)
        return cache[key]

    left_value = value(0.0)
    right_value = value(m)
    if left_value >= right_value:
        best_t, best = 0.0, left_value
    else:
        best_t, best = m, right_value

    def record(lower: float, upper: float) -> tuple[float, float, float]:
        nonlocal best_t, best
        lower_risk = value(lower)
        upper_risk = value(upper)
        if lower_risk > best:
            best_t, best = lower, lower_risk
        if upper_risk > best:
            best_t, best = upper, upper_risk
        curvature = local_joint_second_derivative_interval(
            evaluator, lower, upper
        )
        width = upper - lower
        cell_upper = _semiconvex_secant_upper(
            lower_risk,
            upper_risk,
            max(0.0, -curvature.lower),
            width,
        )
        return -cell_upper, lower, upper

    queue: list[tuple[float, float, float]] = [record(0.0, m)]
    nodes = 1
    while queue:
        global_upper = -queue[0][0]
        if global_upper - best <= tolerance:
            return MaximumResult(best_t, best, global_upper, nodes, False)
        if nodes >= max_nodes:
            return MaximumResult(best_t, best, global_upper, nodes, True)
        _, lower, upper = heapq.heappop(queue)
        center = 0.5 * (lower + upper)
        center_risk = value(center)
        if center_risk > best:
            best_t, best = center, center_risk
        heapq.heappush(queue, record(lower, center))
        heapq.heappush(queue, record(center, upper))
        nodes += 2
    return MaximumResult(best_t, best, best, nodes, False)


def maximize_risk_adaptive(
    evaluator: GaussianMixtureRisk,
    m: float,
    tolerance: float = 1e-6,
    max_nodes: int = 250_000,
) -> MaximumResult:
    """Adaptive continuous maximization using midpoint Taylor enclosures.

    The enclosure for I=[c-h,c+h] is

        r(I) <= r(c) + |r'(c)| h + M_2 h^2/2.

    Exact arithmetic would make this a rigorous global certificate.  The
    float64 values returned here are not outward-rounded.
    """

    if tolerance <= 0 or max_nodes < 1:
        raise ValueError("tolerance and max_nodes must be positive")
    m2 = global_second_derivative_bound(m)
    cache: dict[float, RiskEvaluation] = {}

    def value(t: float) -> RiskEvaluation:
        key = float(t)
        if key not in cache:
            cache[key] = evaluator.evaluate(key)
        return cache[key]

    left = value(0.0)
    right = value(m)
    best_t = 0.0 if left.risk >= right.risk else m
    best = max(left.risk, right.risk)

    def interval_record(lo: float, hi: float) -> tuple[float, float, float]:
        nonlocal best, best_t
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        mid = value(center)
        if mid.risk > best:
            best = mid.risk
            best_t = center
        upper = mid.risk + abs(mid.first_derivative) * half + 0.5 * m2 * half**2
        return -upper, lo, hi

    queue: list[tuple[float, float, float]] = [interval_record(0.0, m)]
    nodes = 1
    while queue:
        global_upper = -queue[0][0]
        if global_upper - best <= tolerance:
            return MaximumResult(best_t, best, global_upper, nodes, False)
        if nodes >= max_nodes:
            return MaximumResult(best_t, best, global_upper, nodes, True)
        _, lo, hi = heapq.heappop(queue)
        center = 0.5 * (lo + hi)
        heapq.heappush(queue, interval_record(lo, center))
        heapq.heappush(queue, interval_record(center, hi))
        nodes += 2
    return MaximumResult(best_t, best, best, nodes, False)


def _softmax_reference(logits: np.ndarray) -> np.ndarray:
    full = np.concatenate([np.asarray(logits, dtype=float), np.zeros(1)])
    full -= np.max(full)
    values = np.exp(full)
    return values / values.sum()


def optimize_level_masses(
    levels: Iterable[float],
    initial_masses: Iterable[float] | None = None,
    *,
    quadrature_order: int = 128,
    gradient_tolerance: float = 2e-11,
    max_iterations: int = 300,
) -> SymmetricPrior:
    """Fully reoptimize positive level masses with a small BFGS routine."""

    levels_array = np.asarray(sorted(set(float(x) for x in levels)), dtype=float)
    count = levels_array.size
    if count == 0:
        raise ValueError("at least one support level is required")
    if initial_masses is None:
        masses = np.full(count, 1.0 / count)
    else:
        masses = np.asarray(list(initial_masses), dtype=float)
        if masses.size != count or np.any(masses <= 0):
            masses = np.full(count, 1.0 / count)
        else:
            masses /= masses.sum()
    if count == 1:
        return SymmetricPrior(levels_array, np.ones(1))
    if count == 2:
        lower = 1e-14
        upper = 1.0 - lower

        def directional_difference(first_mass: float) -> float:
            trial = SymmetricPrior(
                levels_array,
                np.asarray([first_mass, 1.0 - first_mass]),
            )
            risks = GaussianMixtureRisk(
                trial, order=quadrature_order
            ).level_risks()
            return float(risks[0] - risks[1])

        lower_difference = directional_difference(lower)
        upper_difference = directional_difference(upper)
        if lower_difference <= 0:
            return SymmetricPrior(
                levels_array, np.asarray([lower, 1.0 - lower])
            )
        if upper_difference >= 0:
            return SymmetricPrior(
                levels_array, np.asarray([upper, 1.0 - upper])
            )
        for _ in range(80):
            midpoint = 0.5 * (lower + upper)
            difference = directional_difference(midpoint)
            if difference > 0:
                lower = midpoint
            else:
                upper = midpoint
        first_mass = 0.5 * (lower + upper)
        return SymmetricPrior(
            levels_array,
            np.asarray([first_mass, 1.0 - first_mass]),
        )

    logits = np.log(masses[:-1]) - math.log(float(masses[-1]))
    inverse_hessian = np.eye(count - 1)

    def objective_and_gradient(u: np.ndarray) -> tuple[float, np.ndarray]:
        level_masses = _softmax_reference(u)
        prior = SymmetricPrior(levels_array, level_masses)
        evaluator = GaussianMixtureRisk(prior, order=quadrature_order)
        level_risks = evaluator.level_risks()
        bayes = float(level_masses @ level_risks)
        gradient = level_masses[:-1] * (level_risks[:-1] - bayes)
        return bayes, gradient

    bayes, gradient = objective_and_gradient(logits)
    for _ in range(max_iterations):
        if float(np.max(np.abs(gradient))) <= gradient_tolerance:
            break
        direction = inverse_hessian @ gradient
        directional_gain = float(gradient @ direction)
        if not math.isfinite(directional_gain) or directional_gain <= 0:
            inverse_hessian = np.eye(count - 1)
            direction = gradient.copy()
            directional_gain = float(gradient @ direction)
        step = 1.0
        accepted = False
        for _ in range(50):
            trial_logits = logits + step * direction
            trial_bayes, trial_gradient = objective_and_gradient(trial_logits)
            if trial_bayes >= bayes + 1e-4 * step * directional_gain:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        displacement = trial_logits - logits
        curvature = -(trial_gradient - gradient)
        denominator = float(curvature @ displacement)
        if denominator > 1e-14 * max(
            1.0, float(np.linalg.norm(curvature) * np.linalg.norm(displacement))
        ):
            rho = 1.0 / denominator
            identity = np.eye(count - 1)
            transform = identity - rho * np.outer(displacement, curvature)
            inverse_hessian = (
                transform @ inverse_hessian @ transform.T
                + rho * np.outer(displacement, displacement)
            )
        else:
            inverse_hessian = np.eye(count - 1)
        logits, bayes, gradient = trial_logits, trial_bayes, trial_gradient

    optimized = _softmax_reference(logits)
    keep = optimized > 5e-13
    optimized = optimized[keep]
    optimized /= optimized.sum()
    return SymmetricPrior(levels_array[keep], optimized)


def optimize_prior_joint(
    prior: SymmetricPrior,
    m: float,
    *,
    quadrature_order: int = 128,
    gradient_tolerance: float = 2e-10,
    max_iterations: int = 500,
    prune_threshold: float = 1e-10,
) -> SymmetricPrior:
    """Jointly reoptimize level masses and all nonboundary locations.

    Masses use reference-softmax coordinates.  Ordered interior locations
    use a softmax of the gaps between 0, the locations, and ``m``.  This
    keeps every trial prior feasible without projection or an ex ante
    parameter grid.  Existing atoms at 0 and m are treated as active
    boundaries; their first-order location directions are one-sided.

    The envelope theorem gives the location derivative

        partial b / partial a_k = W_k r'(a_k),

    where ``W_k`` is the total mass on the symmetric level.  This identity
    avoids finite-difference location gradients.
    """

    if not math.isfinite(m) or m <= 0:
        raise ValueError("m must be finite and positive")
    if prior.levels[-1] > m + 2e-13 * max(1.0, m):
        raise ValueError("prior support exceeds m")

    boundary_tolerance = 2e-13 * max(1.0, m)
    has_zero = bool(abs(float(prior.levels[0])) <= boundary_tolerance)
    has_m = bool(abs(float(prior.levels[-1]) - m) <= boundary_tolerance)
    interior_start = 1 if has_zero else 0
    interior_stop = prior.levels.size - (1 if has_m else 0)
    interior_initial = prior.levels[interior_start:interior_stop]
    interior_count = interior_initial.size
    support_count = prior.levels.size

    if support_count == 1:
        return prior
    if support_count == 2 and interior_count == 0:
        return optimize_level_masses(
            prior.levels,
            prior.level_masses,
            quadrature_order=quadrature_order,
            gradient_tolerance=gradient_tolerance,
            max_iterations=max_iterations,
        )

    mass_logits = (
        np.log(prior.level_masses[:-1])
        - math.log(float(prior.level_masses[-1]))
    )
    if interior_count:
        location_gaps = np.diff(
            np.concatenate(
                [
                    np.asarray([0.0]),
                    interior_initial,
                    np.asarray([m]),
                ]
            )
        )
        if np.any(location_gaps <= 0):
            raise ValueError("interior levels must lie strictly inside (0,m)")
        location_logits = (
            np.log(location_gaps[:-1])
            - math.log(float(location_gaps[-1]))
        )
    else:
        location_logits = np.empty(0)
    parameters = np.concatenate([mass_logits, location_logits])
    mass_dimension = support_count - 1

    def decode(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        masses = _softmax_reference(vector[:mass_dimension])
        if interior_count:
            gap_fractions = _softmax_reference(
                vector[mass_dimension:]
            )
            interiors = m * np.cumsum(gap_fractions[:-1])
        else:
            gap_fractions = np.ones(1)
            interiors = np.empty(0)
        pieces: list[np.ndarray] = []
        if has_zero:
            pieces.append(np.asarray([0.0]))
        pieces.append(interiors)
        if has_m:
            pieces.append(np.asarray([m]))
        levels = np.concatenate(pieces)
        return levels, masses, gap_fractions

    def objective_and_gradient(
        vector: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        levels, masses, gap_fractions = decode(vector)
        trial_prior = SymmetricPrior(levels, masses)
        evaluator = GaussianMixtureRisk(
            trial_prior, order=quadrature_order
        )
        evaluations = [
            evaluator.evaluate(float(level)) for level in levels
        ]
        level_risks = np.asarray(
            [evaluation.risk for evaluation in evaluations]
        )
        bayes = float(masses @ level_risks)
        mass_gradient = masses[:-1] * (level_risks[:-1] - bayes)

        if interior_count:
            interior_indices = np.arange(
                interior_start, interior_stop
            )
            location_gradient = np.asarray(
                [
                    masses[index] * evaluations[index].first_derivative
                    for index in interior_indices
                ]
            )
            interior_levels = levels[interior_start:interior_stop]
            jacobian = np.empty((interior_count, interior_count))
            for row in range(interior_count):
                for column in range(interior_count):
                    jacobian[row, column] = (
                        m
                        * gap_fractions[column]
                        * (
                            (1.0 if column <= row else 0.0)
                            - interior_levels[row] / m
                        )
                    )
            transformed_location_gradient = (
                jacobian.T @ location_gradient
            )
        else:
            transformed_location_gradient = np.empty(0)
        gradient = np.concatenate(
            [mass_gradient, transformed_location_gradient]
        )
        return bayes, gradient

    inverse_hessian = np.eye(parameters.size)
    bayes, gradient = objective_and_gradient(parameters)
    for _ in range(max_iterations):
        if float(np.max(np.abs(gradient))) <= gradient_tolerance:
            break
        direction = inverse_hessian @ gradient
        directional_gain = float(gradient @ direction)
        if not math.isfinite(directional_gain) or directional_gain <= 0:
            inverse_hessian = np.eye(parameters.size)
            direction = gradient.copy()
            directional_gain = float(gradient @ direction)
        step = 1.0
        accepted = False
        for _ in range(60):
            trial_parameters = parameters + step * direction
            trial_bayes, trial_gradient = objective_and_gradient(
                trial_parameters
            )
            if trial_bayes >= bayes + 1e-4 * step * directional_gain:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        displacement = trial_parameters - parameters
        curvature = -(trial_gradient - gradient)
        denominator = float(curvature @ displacement)
        scale = max(
            1.0,
            float(
                np.linalg.norm(curvature)
                * np.linalg.norm(displacement)
            ),
        )
        if denominator > 1e-14 * scale:
            rho = 1.0 / denominator
            identity = np.eye(parameters.size)
            transform = identity - rho * np.outer(
                displacement, curvature
            )
            inverse_hessian = (
                transform @ inverse_hessian @ transform.T
                + rho * np.outer(displacement, displacement)
            )
        else:
            inverse_hessian = np.eye(parameters.size)
        parameters = trial_parameters
        bayes, gradient = trial_bayes, trial_gradient

    levels, masses, _ = decode(parameters)
    keep = masses > prune_threshold
    levels = levels[keep]
    masses = masses[keep]
    masses /= masses.sum()
    return SymmetricPrior(levels, masses)


def exchange_solve(
    m: float,
    *,
    maximum_levels: int = 3,
    quadrature_order: int = 128,
    separation_tolerance: float = 2e-5,
) -> tuple[SymmetricPrior, list[ExchangeIteration]]:
    """Minimal adaptive support-exchange prototype.

    Locations are added by the continuous separation oracle.  At every outer
    iteration, positive weights and all nonboundary locations are jointly
    reoptimized by ``optimize_prior_joint``.
    """

    if maximum_levels < 1:
        raise ValueError("maximum_levels must be positive")
    prior = SymmetricPrior(np.asarray([m]), np.ones(1))
    history: list[ExchangeIteration] = []

    for iteration in range(maximum_levels):
        prior = optimize_prior_joint(
            prior,
            m,
            quadrature_order=quadrature_order,
        )
        evaluator = GaussianMixtureRisk(prior, order=quadrature_order)
        bayes = evaluator.bayes_risk()
        maximum = maximize_risk_adaptive_joint(
            evaluator,
            m,
            tolerance=separation_tolerance,
        )
        history.append(
            ExchangeIteration(
                iteration=iteration,
                levels=prior.levels.tolist(),
                level_masses=prior.level_masses.tolist(),
                bayes_risk=bayes,
                worst_t=maximum.maximizer,
                worst_risk_lower=maximum.lower,
                worst_risk_upper=maximum.upper,
                numerical_gap_lower=max(0.0, maximum.lower - bayes),
                numerical_gap_upper=max(0.0, maximum.upper - bayes),
                separation_nodes=maximum.nodes,
            )
        )
        if maximum.upper - bayes <= separation_tolerance:
            break
        if prior.levels.size >= maximum_levels:
            break
        distance = np.min(np.abs(prior.levels - maximum.maximizer))
        if distance <= 2e-8 * max(1.0, m):
            break
        new_levels = np.sort(np.append(prior.levels, maximum.maximizer))
        old_mass = 1.0 - min(0.05, 1.0 / (new_levels.size + 2.0))
        new_masses = np.full(new_levels.size, (1.0 - old_mass) / new_levels.size)
        for old_level, old_weight in zip(
            prior.levels, prior.level_masses
        ):
            index = int(np.argmin(np.abs(new_levels - old_level)))
            new_masses[index] += old_mass * old_weight
        new_masses /= new_masses.sum()
        prior = SymmetricPrior(new_levels, new_masses)
    return prior, history


def finite_difference_derivative_check(
    prior: SymmetricPrior,
    test_t: float,
    *,
    quadrature_order: int = 192,
    step: float = 2e-5,
) -> dict[str, float]:
    evaluator = GaussianMixtureRisk(prior, order=quadrature_order)
    center = evaluator.evaluate(test_t)
    plus = evaluator.risk(test_t + step)
    minus = evaluator.risk(test_t - step)
    numerical_first = (plus - minus) / (2.0 * step)
    numerical_second = (plus - 2.0 * center.risk + minus) / (step * step)
    return {
        "analytic_first": center.first_derivative,
        "finite_difference_first": numerical_first,
        "first_absolute_error": abs(center.first_derivative - numerical_first),
        "analytic_second": center.second_derivative,
        "finite_difference_second": numerical_second,
        "second_absolute_error": abs(center.second_derivative - numerical_second),
    }


def directional_derivative_check(
    prior: SymmetricPrior,
    t: float,
    *,
    quadrature_order: int = 192,
    epsilon: float = 1e-5,
) -> dict[str, float]:
    """Check d b((1-e)pi+e pair_t)/de = r(t)-b numerically."""

    base_eval = GaussianMixtureRisk(prior, order=quadrature_order)
    base_bayes = base_eval.bayes_risk()
    predicted = base_eval.risk(t) - base_bayes

    levels = prior.levels.copy()
    masses = (1.0 - epsilon) * prior.level_masses
    nearby = np.flatnonzero(np.abs(levels - t) <= 1e-13)
    if nearby.size:
        masses[int(nearby[0])] += epsilon
    else:
        levels = np.append(levels, t)
        masses = np.append(masses, epsilon)
        order = np.argsort(levels)
        levels, masses = levels[order], masses[order]
    perturbed = SymmetricPrior(levels, masses / masses.sum())
    perturbed_bayes = GaussianMixtureRisk(
        perturbed, order=quadrature_order
    ).bayes_risk()
    quotient = (perturbed_bayes - base_bayes) / epsilon
    return {
        "predicted_g": predicted,
        "forward_difference": quotient,
        "absolute_error": abs(predicted - quotient),
        "epsilon": epsilon,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=float, default=2.0)
    parser.add_argument("--maximum-levels", type=int, default=3)
    parser.add_argument("--quadrature-order", type=int, default=128)
    parser.add_argument("--separation-tolerance", type=float, default=2e-5)
    args = parser.parse_args()

    prior, history = exchange_solve(
        args.m,
        maximum_levels=args.maximum_levels,
        quadrature_order=args.quadrature_order,
        separation_tolerance=args.separation_tolerance,
    )
    probe = min(0.37 * args.m, args.m)
    output = {
        "status": "numerical prototype; not a certificate",
        "m": args.m,
        "prior": prior.as_jsonable(),
        "history": [asdict(row) for row in history],
        "risk_derivative_check": finite_difference_derivative_check(prior, probe),
        "directional_derivative_check": directional_derivative_check(prior, probe),
        "known_limitations": [
            "Gauss-Hermite truncation/roundoff is not enclosed",
            "joint posterior-flow bounds are not outward-rounded in fast mode",
            "joint mass/location BFGS is local and needs certificate replay",
            "outer exchange has no proved finite iteration rate",
        ],
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
