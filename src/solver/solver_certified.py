#!/usr/bin/env python3
"""Conservative high-precision prototype for a bounded-normal certificate.

The implementation uses Decimal interval arithmetic with explicit directed
rounding and one-ulp widening of transcendental results.  It does not use a
parameter grid: parameter intervals are selected adaptively by a global
branch-and-bound.

At a fixed parameter t it uses

    q(t,z) = delta_pi(t+z) - t,
    d q / d z = Var(theta | t+z) >= 0,

to enclose the coupled posterior ratio through endpoint evaluations of the
same Gaussian mixture.  A reachable posterior flow tube retains the common
mean, variance, and cumulants through order five.  Fourth-order composite
Simpson enclosures certify point risks, while a joint q*kappa3 bound and an
exact semiconvex secant majorant certify parameter intervals.  The obsolete
Cartesian rectangle and second-order midpoint rules are retained only as
explicit no-go/regression baselines.

The trusted computing base is Python's ``decimal`` implementation, including
its correctly rounded elementary operations.  ``exp`` and ``sqrt`` are
evaluated with guard digits, rounded to the requested precision, and widened
by one representable Decimal on each side.
"""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from decimal import (
    Decimal,
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    getcontext,
    localcontext,
)
from pathlib import Path
from typing import Callable


PI_LOWER = Decimal(
    "3.14159265358979323846264338327950288419716939937510582097494459230781640628620898"
)
PI_UPPER = Decimal(
    "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899"
)


def _precision() -> int:
    return getcontext().prec


def _binary(
    left: Decimal,
    right: Decimal,
    operation: Callable[[Decimal, Decimal], Decimal],
    rounding: str,
) -> Decimal:
    with localcontext() as context:
        context.prec = _precision()
        context.rounding = rounding
        return operation(left, right)


def _add(left: Decimal, right: Decimal, rounding: str) -> Decimal:
    return _binary(left, right, lambda x, y: x + y, rounding)


def _sub(left: Decimal, right: Decimal, rounding: str) -> Decimal:
    return _binary(left, right, lambda x, y: x - y, rounding)


def _mul(left: Decimal, right: Decimal, rounding: str) -> Decimal:
    return _binary(left, right, lambda x, y: x * y, rounding)


def _div(left: Decimal, right: Decimal, rounding: str) -> Decimal:
    return _binary(left, right, lambda x, y: x / y, rounding)


def _widened_unary(value: Decimal, name: str) -> tuple[Decimal, Decimal]:
    precision = _precision()
    with localcontext() as guard:
        guard.prec = precision + 16
        guard.rounding = ROUND_HALF_EVEN
        if name == "exp":
            result = value.exp()
        elif name == "sqrt":
            result = value.sqrt()
        else:
            raise ValueError(name)
    with localcontext() as lower_context:
        lower_context.prec = precision
        lower_context.rounding = ROUND_FLOOR
        lower = +result
        lower = lower.next_minus()
    with localcontext() as upper_context:
        upper_context.prec = precision
        upper_context.rounding = ROUND_CEILING
        upper = +result
        upper = upper.next_plus()
    return lower, upper


@dataclass(frozen=True)
class Interval:
    lower: Decimal
    upper: Decimal

    def __post_init__(self) -> None:
        if self.lower.is_nan() or self.upper.is_nan() or self.lower > self.upper:
            raise ValueError(f"invalid interval [{self.lower}, {self.upper}]")

    @staticmethod
    def point(value: Decimal | str | int) -> "Interval":
        number = value if isinstance(value, Decimal) else Decimal(value)
        return Interval(number, number)

    def __add__(self, other: "Interval") -> "Interval":
        return Interval(
            _add(self.lower, other.lower, ROUND_FLOOR),
            _add(self.upper, other.upper, ROUND_CEILING),
        )

    def __sub__(self, other: "Interval") -> "Interval":
        return Interval(
            _sub(self.lower, other.upper, ROUND_FLOOR),
            _sub(self.upper, other.lower, ROUND_CEILING),
        )

    def __mul__(self, other: "Interval") -> "Interval":
        lower_products = [
            _mul(x, y, ROUND_FLOOR)
            for x in (self.lower, self.upper)
            for y in (other.lower, other.upper)
        ]
        upper_products = [
            _mul(x, y, ROUND_CEILING)
            for x in (self.lower, self.upper)
            for y in (other.lower, other.upper)
        ]
        return Interval(min(lower_products), max(upper_products))

    def __truediv__(self, other: "Interval") -> "Interval":
        if other.lower <= 0 <= other.upper:
            raise ZeroDivisionError("interval denominator contains zero")
        lower_quotients = [
            _div(x, y, ROUND_FLOOR)
            for x in (self.lower, self.upper)
            for y in (other.lower, other.upper)
        ]
        upper_quotients = [
            _div(x, y, ROUND_CEILING)
            for x in (self.lower, self.upper)
            for y in (other.lower, other.upper)
        ]
        return Interval(min(lower_quotients), max(upper_quotients))

    def __neg__(self) -> "Interval":
        return Interval(-self.upper, -self.lower)

    def exp(self) -> "Interval":
        lower, _ = _widened_unary(self.lower, "exp")
        _, upper = _widened_unary(self.upper, "exp")
        return Interval(lower, upper)

    def sqrt(self) -> "Interval":
        if self.lower < 0:
            raise ValueError("cannot take the square root of a negative interval")
        lower, _ = _widened_unary(self.lower, "sqrt")
        _, upper = _widened_unary(self.upper, "sqrt")
        return Interval(max(Decimal(0), lower), upper)

    def square(self) -> "Interval":
        if self.lower <= 0 <= self.upper:
            upper = max(
                _mul(self.lower, self.lower, ROUND_CEILING),
                _mul(self.upper, self.upper, ROUND_CEILING),
            )
            return Interval(Decimal(0), upper)
        lower = min(
            _mul(self.lower, self.lower, ROUND_FLOOR),
            _mul(self.upper, self.upper, ROUND_FLOOR),
        )
        upper = max(
            _mul(self.lower, self.lower, ROUND_CEILING),
            _mul(self.upper, self.upper, ROUND_CEILING),
        )
        return Interval(lower, upper)

    def intersect(self, lower: Decimal, upper: Decimal) -> "Interval":
        new_lower = max(self.lower, lower)
        new_upper = min(self.upper, upper)
        if new_lower > new_upper:
            raise ArithmeticError("computed enclosure misses a proved range")
        return Interval(new_lower, new_upper)

    def intersection(self, other: "Interval") -> "Interval":
        return self.intersect(other.lower, other.upper)

    def as_strings(self) -> list[str]:
        return [str(self.lower), str(self.upper)]


@dataclass(frozen=True)
class DecimalSymmetricPrior:
    levels: tuple[Decimal, ...]
    level_masses: tuple[Decimal, ...]

    @staticmethod
    def from_mapping(data: dict[str, list[str]]) -> "DecimalSymmetricPrior":
        levels = tuple(Decimal(value) for value in data["levels"])
        masses = tuple(Decimal(value) for value in data["level_masses"])
        if not levels or len(levels) != len(masses):
            raise ValueError("invalid levels/masses")
        if levels[0] < 0 or any(b <= a for a, b in zip(levels, levels[1:])):
            raise ValueError("levels must be strictly increasing and nonnegative")
        if any(weight <= 0 for weight in masses) or sum(masses) != Decimal(1):
            raise ValueError("level masses must be positive and sum exactly to one")
        return DecimalSymmetricPrior(levels, masses)

    def atoms(self) -> tuple[tuple[Decimal, ...], tuple[Decimal, ...]]:
        locations: list[Decimal] = []
        weights: list[Decimal] = []
        for level, mass in zip(self.levels, self.level_masses):
            if level == 0:
                locations.append(Decimal(0))
                weights.append(mass)
            else:
                half = mass / Decimal(2)
                locations.extend([-level, level])
                weights.extend([half, half])
        return tuple(locations), tuple(weights)

    def as_mapping(self) -> dict[str, list[str]]:
        return {
            "levels": [str(value) for value in self.levels],
            "level_masses": [str(value) for value in self.level_masses],
        }


class CertifiedRiskOracle:
    def __init__(
        self,
        prior: DecimalSymmetricPrior,
        m: Decimal,
        *,
        tail_cutoff: Decimal,
        z_cells: int,
    ):
        if m <= 0 or tail_cutoff <= 0 or z_cells < 4:
            raise ValueError("m, tail_cutoff, and z_cells must be positive")
        if z_cells & (z_cells - 1):
            raise ValueError("z_cells must be a power of two for an exact Decimal partition")
        self.prior = prior
        self.m = m
        self.tail_cutoff = tail_cutoff
        self.z_cells = z_cells
        self.locations, self.weights = prior.atoms()
        if max(abs(theta) for theta in self.locations) > m:
            raise ValueError("prior support lies outside [-m,m]")
        width = (Decimal(2) * tail_cutoff) / Decimal(z_cells)
        self.z_edges = tuple(
            -tail_cutoff + Decimal(index) * width
            for index in range(z_cells + 1)
        )
        self._posterior_cache: dict[Decimal, Interval] = {}
        self._variance_cache: dict[Decimal, Interval] = {}
        self._terms_cache: dict[
            Decimal, tuple[tuple[Interval, ...], Interval]
        ] = {}
        self._cumulant_cache: dict[
            Decimal, tuple[Interval, Interval, Interval, Interval, Interval]
        ] = {}
        self._simpson_risk_cache: dict[Decimal, Interval] = {}
        self._curvature_cache: dict[
            tuple[Decimal, Decimal], Interval
        ] = {}
        self._density_boxes = tuple(
            self._normal_density_box(left, right)
            for left, right in zip(self.z_edges, self.z_edges[1:])
        )
        self.z_midpoints = tuple(
            (left + right) / Decimal(2)
            for left, right in zip(self.z_edges, self.z_edges[1:])
        )
        self.z_width = self.z_edges[1] - self.z_edges[0]
        self._midpoint_densities = tuple(
            self._normal_density_point(abs(midpoint))
            for midpoint in self.z_midpoints
        )

    def _normal_density_point(self, absolute_x: Decimal) -> Interval:
        x = Interval.point(absolute_x)
        exponent = Interval.point(Decimal("-0.5")) * x.square()
        two_pi = Interval(
            _mul(Decimal(2), PI_LOWER, ROUND_FLOOR),
            _mul(Decimal(2), PI_UPPER, ROUND_CEILING),
        )
        denominator = two_pi.sqrt()
        return exponent.exp() / denominator

    def _normal_density_box(self, left: Decimal, right: Decimal) -> Interval:
        farthest = max(abs(left), abs(right))
        nearest = Decimal(0) if left <= 0 <= right else min(abs(left), abs(right))
        far_density = self._normal_density_point(farthest)
        near_density = self._normal_density_point(nearest)
        return Interval(far_density.lower, near_density.upper)

    def _posterior_terms(
        self, y: Decimal
    ) -> tuple[tuple[Interval, ...], Interval]:
        if y in self._terms_cache:
            return self._terms_cache[y]
        exponents: list[Interval] = []
        for theta in self.locations:
            theta_interval = Interval.point(theta)
            exponent = (
                theta_interval * Interval.point(y)
                - Interval.point(Decimal("0.5")) * theta_interval.square()
            )
            exponents.append(exponent)
        shift = max(value.upper for value in exponents)
        terms = [
            Interval.point(weight)
            * (exponent - Interval.point(shift)).exp()
            for weight, exponent in zip(self.weights, exponents)
        ]
        denominator = Interval.point(0)
        for term in terms:
            denominator = denominator + term
        result = (tuple(terms), denominator)
        self._terms_cache[y] = result
        return result

    def posterior_state(self, y: Decimal) -> tuple[Interval, Interval]:
        if y in self._posterior_cache:
            return self._posterior_cache[y], self._variance_cache[y]
        terms, denominator = self._posterior_terms(y)
        numerator = Interval.point(0)
        for theta, term in zip(self.locations, terms):
            numerator = numerator + Interval.point(theta) * term
        mean = (numerator / denominator).intersect(-self.m, self.m)

        # Exact pairwise identity:
        # Var(theta|y) = sum_{j<k} a_j a_k (theta_j-theta_k)^2 / S0^2.
        variance_numerator = Interval.point(0)
        for left_index in range(len(self.locations)):
            for right_index in range(left_index + 1, len(self.locations)):
                difference = self.locations[left_index] - self.locations[right_index]
                variance_numerator = (
                    variance_numerator
                    + terms[left_index]
                    * terms[right_index]
                    * Interval.point(difference * difference)
                )
        variance = (variance_numerator / denominator.square()).intersect(
            Decimal(0), self.m * self.m
        )
        self._posterior_cache[y] = mean
        self._variance_cache[y] = variance
        return mean, variance

    def posterior_cumulants(
        self, y: Decimal
    ) -> tuple[Interval, Interval, Interval, Interval, Interval]:
        """Posterior mean and cumulants of orders two through five."""

        if y in self._cumulant_cache:
            return self._cumulant_cache[y]
        mean, variance = self.posterior_state(y)
        terms, denominator = self._posterior_terms(y)
        third = Interval.point(0)
        fourth_moment = Interval.point(0)
        fifth_moment = Interval.point(0)
        for theta, term in zip(self.locations, terms):
            probability = term / denominator
            centered = Interval.point(theta) - mean
            centered_two = centered.square()
            centered_three = centered_two * centered
            centered_four = centered_two.square()
            centered_five = centered_four * centered
            third = third + probability * centered_three
            fourth_moment = fourth_moment + probability * centered_four
            fifth_moment = fifth_moment + probability * centered_five
        fourth = (
            fourth_moment
            - Interval.point(3) * variance.square()
        )
        fifth = (
            fifth_moment
            - Interval.point(10) * third * variance
        )
        result = (mean, variance, third, fourth, fifth)
        self._cumulant_cache[y] = result
        return result

    def posterior_mean(self, y: Decimal) -> Interval:
        return self.posterior_state(y)[0]

    @staticmethod
    def _concave_product_max_upper(
        mean_lower: Decimal,
        mean_upper: Decimal,
        left: Decimal,
        right: Decimal,
    ) -> Decimal:
        lower = max(mean_lower, left)
        upper = min(mean_upper, right)
        if lower > upper:
            return Decimal(0)
        vertex = (left + right) / Decimal(2)
        point = min(max(vertex, lower), upper)
        return _mul(
            _sub(point, left, ROUND_CEILING),
            _sub(right, point, ROUND_CEILING),
            ROUND_CEILING,
        )

    def _sharp_third_support_interval(
        self,
        mean: Interval,
        variance: Interval,
    ) -> Interval:
        """Sharp support/moment enclosure for the third central moment.

        For U=theta-mu in [-A,B], E[U]=0, and E[U^2]=V,

            -V(A-V/A) <= E[U^3] <= V(B-V/B).

        The bounds follow from the nonnegative polynomials
        (U+A)(U-V/A)^2 and (B-U)(U+V/B)^2 and are sharp.  The
        interval formulas below retain the shared lower variance and also
        use the completed-square bounds -A^3/4 and B^3/4.
        """

        alpha = min(self.locations)
        beta = max(self.locations)
        left_radius = _sub(mean.upper, alpha, ROUND_CEILING)
        right_radius = _sub(beta, mean.lower, ROUND_CEILING)

        if left_radius == 0:
            lower = Decimal(0)
        else:
            coarse_lower = -_mul(
                left_radius, variance.upper, ROUND_CEILING
            )
            variance_square_lower = _mul(
                variance.lower, variance.lower, ROUND_FLOOR
            )
            quotient_lower = _div(
                variance_square_lower, left_radius, ROUND_FLOOR
            )
            variance_refined_lower = _add(
                coarse_lower, quotient_lower, ROUND_FLOOR
            )
            radius_square = _mul(
                left_radius, left_radius, ROUND_CEILING
            )
            radius_cube = _mul(
                radius_square, left_radius, ROUND_CEILING
            )
            cubic_lower = -_div(
                radius_cube, Decimal(4), ROUND_CEILING
            )
            lower = max(
                coarse_lower,
                variance_refined_lower,
                cubic_lower,
            )

        if right_radius == 0:
            upper = Decimal(0)
        else:
            coarse_upper = _mul(
                right_radius, variance.upper, ROUND_CEILING
            )
            variance_square_lower = _mul(
                variance.lower, variance.lower, ROUND_FLOOR
            )
            quotient_lower = _div(
                variance_square_lower, right_radius, ROUND_FLOOR
            )
            variance_refined_upper = _sub(
                coarse_upper, quotient_lower, ROUND_CEILING
            )
            radius_square = _mul(
                right_radius, right_radius, ROUND_CEILING
            )
            radius_cube = _mul(
                radius_square, right_radius, ROUND_CEILING
            )
            cubic_upper = _div(
                radius_cube, Decimal(4), ROUND_CEILING
            )
            upper = min(
                coarse_upper,
                variance_refined_upper,
                cubic_upper,
            )
        return Interval(lower, upper)

    def _posterior_flow_enclosure(
        self,
        y_lower: Decimal,
        y_upper: Decimal,
    ) -> tuple[Interval, Interval, Interval, Interval, Interval]:
        """Reachable interval state for (mu,V,kappa3,kappa4,kappa5)."""

        if y_lower > y_upper:
            raise ValueError("invalid observation interval")
        alpha = min(self.locations)
        beta = max(self.locations)
        center = (y_lower + y_upper) / Decimal(2)
        half_width = (y_upper - y_lower) / Decimal(2)
        mean_at_lower = self.posterior_mean(y_lower)
        mean_at_upper = self.posterior_mean(y_upper)
        mean_range = Interval(
            max(alpha, mean_at_lower.lower),
            min(beta, mean_at_upper.upper),
        )
        (
            _,
            variance_center,
            third_center,
            fourth_center,
            _,
        ) = self.posterior_cumulants(center)

        local_slope = max(
            _sub(mean_range.upper, alpha, ROUND_CEILING),
            _sub(beta, mean_range.lower, ROUND_CEILING),
        )
        positive_exponent = (
            Interval.point(local_slope)
            * Interval.point(half_width)
        ).exp()
        negative_exponent = (
            Interval.point(-local_slope)
            * Interval.point(half_width)
        ).exp()
        variance_tube = variance_center * Interval(
            negative_exponent.lower,
            positive_exponent.upper,
        )

        variance_vertex = (alpha + beta) / Decimal(2)
        if mean_range.lower <= variance_vertex <= mean_range.upper:
            diameter_upper = _sub(beta, alpha, ROUND_CEILING)
            variance_bhatia_upper = _div(
                _mul(
                    diameter_upper,
                    diameter_upper,
                    ROUND_CEILING,
                ),
                Decimal(4),
                ROUND_CEILING,
            )
        else:
            variance_mean = (
                mean_range.lower
                if variance_vertex < mean_range.lower
                else mean_range.upper
            )
            variance_bhatia_upper = _mul(
                _sub(beta, variance_mean, ROUND_CEILING),
                _sub(variance_mean, alpha, ROUND_CEILING),
                ROUND_CEILING,
            )
        variance = Interval(
            max(Decimal(0), variance_tube.lower),
            min(variance_tube.upper, variance_bhatia_upper),
        )

        local_slope_square = _mul(
            local_slope, local_slope, ROUND_CEILING
        )
        variance_upper_square = _mul(
            variance.upper, variance.upper, ROUND_CEILING
        )
        fourth_bound = _add(
            _mul(
                local_slope_square,
                variance.upper,
                ROUND_CEILING,
            ),
            _mul(
                Decimal(3),
                variance_upper_square,
                ROUND_CEILING,
            ),
            ROUND_CEILING,
        )
        third_error = _mul(
            half_width, fourth_bound, ROUND_CEILING
        )
        third_flow = third_center + Interval(
            -third_error, third_error
        )
        third_support = self._sharp_third_support_interval(
            mean_range, variance
        )
        third = third_flow.intersection(third_support)

        # Two sound fixed-point refinement passes.  Each pass narrows V using
        # V'=kappa3 and then narrows kappa3 using the sharp common-state
        # support constraint above.
        for _ in range(2):
            third_magnitude = max(abs(third.lower), abs(third.upper))
            variance_error = _mul(
                half_width, third_magnitude, ROUND_CEILING
            )
            variance_flow = variance_center + Interval(
                -variance_error, variance_error
            )
            variance = variance.intersection(
                Interval(
                    max(Decimal(0), variance_flow.lower),
                    variance_flow.upper,
                )
            )
            third = third.intersection(
                self._sharp_third_support_interval(
                    mean_range, variance
                )
            )

        variance_upper_square = _mul(
            variance.upper, variance.upper, ROUND_CEILING
        )
        local_slope_square = _mul(
            local_slope, local_slope, ROUND_CEILING
        )
        local_slope_cube = _mul(
            local_slope_square, local_slope, ROUND_CEILING
        )
        fifth_bound = _add(
            _mul(
                local_slope_cube,
                variance.upper,
                ROUND_CEILING,
            ),
            _mul(
                Decimal(10),
                _mul(
                    local_slope,
                    variance_upper_square,
                    ROUND_CEILING,
                ),
                ROUND_CEILING,
            ),
            ROUND_CEILING,
        )
        fourth_error = _mul(
            half_width, fifth_bound, ROUND_CEILING
        )
        fourth_flow = fourth_center + Interval(
            -fourth_error, fourth_error
        )
        fourth_generic = Interval(
            -_mul(
                Decimal(3),
                variance_upper_square,
                ROUND_CEILING,
            ),
            _mul(
                local_slope_square,
                variance.upper,
                ROUND_CEILING,
            ),
        )
        fourth = fourth_flow.intersection(fourth_generic)
        fifth = Interval(-fifth_bound, fifth_bound)
        return mean_range, variance, third, fourth, fifth

    def _q_third_joint_interval(
        self,
        *,
        t: Decimal,
        mean: Interval,
        variance: Interval,
        third: Interval,
    ) -> Interval:
        alpha = min(self.locations)
        beta = max(self.locations)
        q = mean - Interval.point(t)
        flow = q * third

        positive = self._concave_product_max_upper(
            mean.lower, mean.upper, t, beta
        )
        negative = self._concave_product_max_upper(
            mean.lower, mean.upper, alpha, t
        )
        shared_upper = _mul(
            variance.upper,
            max(positive, negative),
            ROUND_CEILING,
        )

        positive_negative = Decimal(0)
        if mean.upper >= t:
            positive_negative = _mul(
                _sub(mean.upper, t, ROUND_CEILING),
                _sub(mean.upper, alpha, ROUND_CEILING),
                ROUND_CEILING,
            )
        negative_negative = Decimal(0)
        if mean.lower <= t:
            negative_negative = _mul(
                _sub(t, mean.lower, ROUND_CEILING),
                _sub(beta, mean.lower, ROUND_CEILING),
                ROUND_CEILING,
            )
        shared_lower = -_mul(
            variance.upper,
            max(positive_negative, negative_negative),
            ROUND_CEILING,
        )
        return flow.intersection(
            Interval(shared_lower, shared_upper)
        )

    def _risk_integrand_point(
        self, t: Decimal, z: Decimal
    ) -> Interval:
        mean = self.posterior_mean(t + z)
        q = mean - Interval.point(t)
        return q.square() * self._normal_density_point(abs(z))

    def _risk_point_simpson(
        self,
        t: Decimal,
        *,
        panels: int | None = None,
    ) -> Interval:
        """Fourth-order outward-rounded enclosure of r(t)."""

        panel_count = self.z_cells if panels is None else panels
        if panel_count < 1:
            raise ValueError("panel count must be positive")
        if panels is None and t in self._simpson_risk_cache:
            return self._simpson_risk_cache[t]
        alpha = min(self.locations)
        beta = max(self.locations)
        base_width = self.tail_cutoff / Decimal(panel_count)
        panel_width = Decimal(2) * base_width
        total = Interval.point(0)
        for panel in range(panel_count):
            z_lower = (
                -self.tail_cutoff
                + Decimal(2 * panel) * base_width
            )
            z_center = z_lower + base_width
            z_upper = z_lower + panel_width
            h_lower = self._risk_integrand_point(t, z_lower)
            h_center = self._risk_integrand_point(t, z_center)
            h_upper = self._risk_integrand_point(t, z_upper)
            simpson = (
                (
                    Interval.point(panel_width)
                    / Interval.point(6)
                )
                * (
                    h_lower
                    + Interval.point(4) * h_center
                    + h_upper
                )
            )

            mean, variance, third, fourth, fifth = (
                self._posterior_flow_enclosure(
                    t + z_lower, t + z_upper
                )
            )
            q = mean - Interval.point(t)
            q_square = q.square()
            variance_square = variance.square()
            third_square = third.square()
            q_third = self._q_third_joint_interval(
                t=t,
                mean=mean,
                variance=variance,
                third=third,
            )

            u_zero = q_square
            u_one = (
                Interval.point(2) * q * variance
            )
            u_two = (
                Interval.point(2)
                * (variance_square + q_third)
            )
            u_three = (
                Interval.point(2)
                * (
                    Interval.point(3) * variance * third
                    + q * fourth
                )
            )
            u_four = (
                Interval.point(2)
                * (
                    Interval.point(3) * third_square
                    + Interval.point(4) * variance * fourth
                    + q * fifth
                )
            )

            z_interval = Interval(z_lower, z_upper)
            z_square = z_interval.square()
            z_cube = z_square * z_interval
            z_fourth = z_square.square()
            coefficient_two = z_square - Interval.point(1)
            coefficient_one = (
                Interval.point(3) * z_interval - z_cube
            )
            coefficient_zero = (
                z_fourth
                - Interval.point(6) * z_square
                + Interval.point(3)
            )
            bracket = (
                u_four
                - Interval.point(4) * z_interval * u_three
                + Interval.point(6) * coefficient_two * u_two
                + Interval.point(4) * coefficient_one * u_one
                + coefficient_zero * u_zero
            )
            fourth_h = bracket * self._normal_density_box(
                z_lower, z_upper
            )
            panel_width_interval = Interval.point(panel_width)
            error_coefficient = (
                panel_width_interval
                * panel_width_interval
                * panel_width_interval
                * panel_width_interval
                * panel_width_interval
                / Interval.point(2880)
            )
            total = total + simpson - error_coefficient * fourth_h

        q_tail_interval = (
            Interval(alpha, beta) - Interval.point(t)
        )
        q_tail = max(
            abs(q_tail_interval.lower),
            abs(q_tail_interval.upper),
        )
        tail_density = self._normal_density_point(self.tail_cutoff)
        tail_interval = (
            Interval.point(2)
            * Interval.point(q_tail).square()
            * tail_density
            / Interval.point(self.tail_cutoff)
        )
        result = Interval(
            max(Decimal(0), total.lower),
            _add(total.upper, tail_interval.upper, ROUND_CEILING),
        )
        if panels is None:
            self._simpson_risk_cache[t] = result
        return result

    def _q_third_joint_interval_tbox(
        self,
        *,
        t_lower: Decimal,
        t_upper: Decimal,
        mean: Interval,
        variance: Interval,
        third: Interval,
    ) -> Interval:
        alpha = min(self.locations)
        beta = max(self.locations)
        q = mean - Interval(t_lower, t_upper)
        flow = q * third
        positive = self._concave_product_max_upper(
            mean.lower, mean.upper, t_lower, beta
        )
        negative = self._concave_product_max_upper(
            mean.lower, mean.upper, alpha, t_upper
        )
        shared_upper = _mul(
            variance.upper,
            max(positive, negative),
            ROUND_CEILING,
        )

        positive_negative = Decimal(0)
        if mean.upper >= t_lower:
            positive_negative = _mul(
                _sub(mean.upper, t_lower, ROUND_CEILING),
                _sub(mean.upper, alpha, ROUND_CEILING),
                ROUND_CEILING,
            )
        negative_negative = Decimal(0)
        if mean.lower <= t_upper:
            negative_negative = _mul(
                _sub(t_upper, mean.lower, ROUND_CEILING),
                _sub(beta, mean.lower, ROUND_CEILING),
                ROUND_CEILING,
            )
        shared_lower = -_mul(
            variance.upper,
            max(positive_negative, negative_negative),
            ROUND_CEILING,
        )
        return flow.intersection(
            Interval(shared_lower, shared_upper)
        )

    def _curvature_box(
        self,
        t_lower: Decimal,
        t_upper: Decimal,
    ) -> Interval:
        """Outward-rounded enclosure of r'' on a parameter interval."""

        key = (t_lower, t_upper)
        if key in self._curvature_cache:
            return self._curvature_cache[key]
        if not (Decimal(0) <= t_lower <= t_upper <= self.m):
            raise ValueError("curvature box must lie in [0,m]")
        total = Interval.point(0)
        for z_lower, z_upper, density in zip(
            self.z_edges[:-1],
            self.z_edges[1:],
            self._density_boxes,
        ):
            mean, variance, third, _, _ = (
                self._posterior_flow_enclosure(
                    t_lower + z_lower,
                    t_upper + z_upper,
                )
            )
            variance_part = (
                variance - Interval.point(1)
            ).square()
            q_third = self._q_third_joint_interval_tbox(
                t_lower=t_lower,
                t_upper=t_upper,
                mean=mean,
                variance=variance,
                third=third,
            )
            state = variance_part + q_third
            total = (
                total
                + Interval.point(z_upper - z_lower)
                * state
                * density
            )

        alpha = min(self.locations)
        beta = max(self.locations)
        diameter = _sub(beta, alpha, ROUND_CEILING)
        variance_max = _div(
            _mul(diameter, diameter, ROUND_CEILING),
            Decimal(4),
            ROUND_CEILING,
        )
        variance_tail = Interval(
            Decimal(0), variance_max
        )
        variance_part_tail = (
            variance_tail - Interval.point(1)
        ).square()
        q_tail_interval = (
            Interval(alpha, beta)
            - Interval(t_lower, t_upper)
        )
        q_magnitude = max(
            abs(q_tail_interval.lower),
            abs(q_tail_interval.upper),
        )
        third_magnitude = _mul(
            diameter, variance_max, ROUND_CEILING
        )
        product_magnitude = _mul(
            q_magnitude, third_magnitude, ROUND_CEILING
        )
        state_tail = (
            variance_part_tail
            + Interval(-product_magnitude, product_magnitude)
        )
        tail_probability_bound = (
            Interval.point(2)
            * self._normal_density_point(self.tail_cutoff)
            / Interval.point(self.tail_cutoff)
        )
        tail_probability = Interval(
            Decimal(0), tail_probability_bound.upper
        )
        result = (
            Interval.point(2)
            * (total + tail_probability * state_tail)
        )
        self._curvature_cache[key] = result
        return result

    @staticmethod
    def _secant_majorant_upper(
        left_value: Decimal,
        right_value: Decimal,
        curvature_magnitude: Decimal,
        width: Decimal,
    ) -> Decimal:
        if curvature_magnitude <= 0 or width <= 0:
            return max(left_value, right_value)
        coefficient = (
            Interval.point(curvature_magnitude)
            * Interval.point(width).square()
            / Interval.point(2)
        )
        difference = (
            Interval.point(right_value)
            - Interval.point(left_value)
        )
        if difference.lower >= coefficient.upper:
            return right_value
        if difference.upper <= -coefficient.upper:
            return left_value
        interior = (
            Interval.point(left_value)
            + (difference + coefficient).square()
            / (Interval.point(4) * coefficient)
        )
        return max(left_value, right_value, interior.upper)

    def risk_box_cartesian(self, t_lower: Decimal, t_upper: Decimal) -> Interval:
        """First-order rectangle enclosure retained as a no-go baseline."""

        if not (Decimal(0) <= t_lower <= t_upper <= self.m):
            raise ValueError("risk box must lie in [0,m]")
        total = Interval.point(0)
        for (z_lower, z_upper), density in zip(
            zip(self.z_edges, self.z_edges[1:]), self._density_boxes
        ):
            posterior_lower = self.posterior_mean(t_lower + z_lower)
            posterior_upper = self.posterior_mean(t_upper + z_upper)
            q = Interval(
                _sub(posterior_lower.lower, t_upper, ROUND_FLOOR),
                _sub(posterior_upper.upper, t_lower, ROUND_CEILING),
            )
            width = Interval.point(z_upper - z_lower)
            total = total + width * q.square() * density

        # |q| <= 2m and 2 Phi(-A) <= 2 phi(A)/A.
        tail_density = self._normal_density_point(self.tail_cutoff)
        tail_upper = _div(
            _mul(
                _mul(Decimal(8), self.m * self.m, ROUND_CEILING),
                tail_density.upper,
                ROUND_CEILING,
            ),
            self.tail_cutoff,
            ROUND_CEILING,
        )
        return Interval(total.lower, _add(total.upper, tail_upper, ROUND_CEILING))

    def _risk_point_midpoint(self, t: Decimal) -> Interval:
        """Second-order composite-midpoint enclosure of r(t)."""

        total = Interval.point(0)
        midpoint_error = Decimal(0)
        m2 = self.m * self.m
        m3 = m2 * self.m
        m4 = m2 * m2
        for z_left, z_right, z, density, density_box in zip(
            self.z_edges[:-1],
            self.z_edges[1:],
            self.z_midpoints,
            self._midpoint_densities,
            self._density_boxes,
        ):
            mean, _ = self.posterior_state(t + z)
            q = mean - Interval.point(t)
            total = total + Interval.point(self.z_width) * q.square() * density

            # On a fixed-t z-cell, q is nondecreasing because q_z=V>=0.
            q_left = self.posterior_mean(t + z_left) - Interval.point(t)
            q_right = self.posterior_mean(t + z_right) - Interval.point(t)
            q_abs = max(abs(q_left.lower), abs(q_right.upper))
            z_abs = max(abs(z_left), abs(z_right))

            # H''/phi = 2V^2+2q*kappa3+(z^2-1)q^2-4zqV.
            # Use V<=m^2 and |kappa3|<=2m V<=2m^3.  The latter retains
            # posterior dependence and is much sharper than |kappa3|<=8m^3.
            with localcontext() as upward:
                upward.prec = _precision()
                upward.rounding = ROUND_CEILING
                second_bound = density_box.upper * (
                    Decimal(2) * m4
                    + Decimal(4) * m3 * q_abs
                    + (z_abs * z_abs + Decimal(1)) * q_abs * q_abs
                    + Decimal(4) * z_abs * q_abs * m2
                )
                midpoint_error += (
                    self.z_width**3 * second_bound / Decimal(24)
                )

        a = self.tail_cutoff
        tail_density = self._normal_density_point(a)
        tail_upper = _div(
            _mul(
                _mul(Decimal(8), m2, ROUND_CEILING),
                tail_density.upper,
                ROUND_CEILING,
            ),
            a,
            ROUND_CEILING,
        )
        lower = max(
            Decimal(0),
            _sub(total.lower, midpoint_error, ROUND_FLOOR),
        )
        upper = _add(
            _add(total.upper, midpoint_error, ROUND_CEILING),
            tail_upper,
            ROUND_CEILING,
        )
        return Interval(lower, upper)

    def _risk_derivative_point(self, t: Decimal) -> Interval:
        """First-order-in-z enclosure of r'(t)=2E[q(V-1)]."""

        total = Interval.point(0)
        midpoint_error = Decimal(0)
        one = Interval.point(1)
        two = Interval.point(2)
        m2 = self.m * self.m
        m3 = m2 * self.m
        variance_distance = max(Decimal(1), abs(m2 - Decimal(1)))
        for z_left, z_right, z, density, density_box in zip(
            self.z_edges[:-1],
            self.z_edges[1:],
            self.z_midpoints,
            self._midpoint_densities,
            self._density_boxes,
        ):
            mean, variance = self.posterior_state(t + z)
            q = mean - Interval.point(t)
            integrand = two * q * (variance - one) * density
            total = total + Interval.point(self.z_width) * integrand

            q_left = self.posterior_mean(t + z_left) - Interval.point(t)
            q_right = self.posterior_mean(t + z_right) - Interval.point(t)
            q_abs = max(abs(q_left.lower), abs(q_right.upper))
            z_abs = max(abs(z_left), abs(z_right))

            # J'=2[V(V-1)+q*kappa3-z*q(V-1)]phi.
            with localcontext() as upward:
                upward.prec = _precision()
                upward.rounding = ROUND_CEILING
                first_bound = Decimal(2) * density_box.upper * (
                    m2 * variance_distance
                    + Decimal(2) * m3 * q_abs
                    + z_abs * q_abs * variance_distance
                )
                midpoint_error += (
                    self.z_width * self.z_width * first_bound / Decimal(4)
                )

        tail_density = self._normal_density_point(self.tail_cutoff)
        tail_error = _div(
            _mul(
                Decimal(8) * self.m * variance_distance,
                tail_density.upper,
                ROUND_CEILING,
            ),
            self.tail_cutoff,
            ROUND_CEILING,
        )
        total_error = _add(midpoint_error, tail_error, ROUND_CEILING)
        return Interval(
            _sub(total.lower, total_error, ROUND_FLOOR),
            _add(total.upper, total_error, ROUND_CEILING),
        )

    def risk_box(self, t_lower: Decimal, t_upper: Decimal) -> Interval:
        """Joint flow/secant enclosure of r on a parameter interval."""

        if not (Decimal(0) <= t_lower <= t_upper <= self.m):
            raise ValueError("risk box must lie in [0,m]")
        if t_lower == t_upper:
            return self._risk_point_simpson(t_lower)
        left_risk = self._risk_point_simpson(t_lower)
        right_risk = self._risk_point_simpson(t_upper)
        curvature = self._curvature_box(t_lower, t_upper)
        curvature_magnitude = max(
            Decimal(0), -curvature.lower
        )
        upper = self._secant_majorant_upper(
            left_risk.upper,
            right_risk.upper,
            curvature_magnitude,
            t_upper - t_lower,
        )
        return Interval(
            Decimal(0),
            upper,
        )

    def bayes_risk(self) -> Interval:
        result = Interval.point(0)
        for level, mass in zip(
            self.prior.levels, self.prior.level_masses
        ):
            result = result + Interval.point(mass) * self.risk_box(level, level)
        return result


@dataclass(frozen=True)
class CertifiedMaximum:
    lower: Decimal
    upper: Decimal
    maximizer_lower_bound: Decimal
    nodes: int
    exhausted: bool


def maximize_risk_certified(
    oracle: CertifiedRiskOracle,
    *,
    tolerance: Decimal,
    max_nodes: int,
) -> CertifiedMaximum:
    if tolerance <= 0 or max_nodes < 1:
        raise ValueError("invalid tolerance/max_nodes")
    point_cache: dict[Decimal, Interval] = {}

    def point(t: Decimal) -> Interval:
        if t not in point_cache:
            point_cache[t] = oracle.risk_box(t, t)
        return point_cache[t]

    endpoint_zero = point(Decimal(0))
    endpoint_m = point(oracle.m)
    if endpoint_zero.lower >= endpoint_m.lower:
        best_lower = endpoint_zero.lower
        best_t = Decimal(0)
    else:
        best_lower = endpoint_m.lower
        best_t = oracle.m

    def record(
        lower: Decimal, upper: Decimal
    ) -> tuple[Decimal, Decimal, Decimal]:
        nonlocal best_lower, best_t
        center = (lower + upper) / Decimal(2)
        center_value = point(center)
        if center_value.lower > best_lower:
            best_lower = center_value.lower
            best_t = center
        box = oracle.risk_box(lower, upper)
        return -box.upper, lower, upper

    queue = [record(Decimal(0), oracle.m)]
    nodes = 1
    while queue:
        global_upper = -queue[0][0]
        if global_upper - best_lower <= tolerance:
            return CertifiedMaximum(best_lower, global_upper, best_t, nodes, False)
        if nodes >= max_nodes:
            return CertifiedMaximum(best_lower, global_upper, best_t, nodes, True)
        _, lower, upper = heapq.heappop(queue)
        center = (lower + upper) / Decimal(2)
        heapq.heappush(queue, record(lower, center))
        heapq.heappush(queue, record(center, upper))
        nodes += 2
    return CertifiedMaximum(best_lower, best_lower, best_t, nodes, False)


def build_certificate(
    prior: DecimalSymmetricPrior,
    m: Decimal,
    *,
    precision: int,
    tail_cutoff: Decimal,
    z_cells: int,
    separation_tolerance: Decimal,
    max_nodes: int,
) -> dict[str, object]:
    if precision < 30:
        raise ValueError("precision must be at least 30 decimal digits")
    getcontext().prec = precision
    oracle = CertifiedRiskOracle(
        prior,
        m,
        tail_cutoff=tail_cutoff,
        z_cells=z_cells,
    )
    bayes = oracle.bayes_risk()
    worst = maximize_risk_certified(
        oracle,
        tolerance=separation_tolerance,
        max_nodes=max_nodes,
    )
    gap_upper = _sub(worst.upper, bayes.lower, ROUND_CEILING)
    return {
        "format": "bounded-normal-mean-certificate-v0",
        "bound_version": "posterior-flow-sharp-third-v1",
        "status": (
            "rigorous interval enclosure under stated Decimal trusted base"
            if not worst.exhausted
            else "valid enclosure; requested separation tolerance not reached"
        ),
        "m": str(m),
        "prior": prior.as_mapping(),
        "settings": {
            "precision": precision,
            "tail_cutoff": str(tail_cutoff),
            "z_cells": z_cells,
            "point_quadrature": "outward-rounded composite Simpson",
            "parameter_bound": (
                "sharp third-moment posterior flow tube and secant majorant"
            ),
            "separation_tolerance": str(separation_tolerance),
            "max_nodes": max_nodes,
        },
        "bayes_risk": bayes.as_strings(),
        "worst_case_risk": [str(worst.lower), str(worst.upper)],
        "worst_case_lower_bound_location": str(worst.maximizer_lower_bound),
        "minimax_risk_interval": [str(bayes.lower), str(worst.upper)],
        "minimax_gap_upper": str(gap_upper),
        "separation_nodes": worst.nodes,
        "separation_exhausted": worst.exhausted,
        "proof_dependencies": [
            "delta_pi is nondecreasing because delta_pi'(y)=Var(theta|y)>=0",
            "Bhatia-Davis: V<=(beta-mu)(mu-alpha)",
            "kappa3=V*(nu-mu) for nu in [alpha,beta]",
            (
                "-V*(A-V/A)<=kappa3<=V*(B-V/B), obtained from "
                "(U+A)*(U-V/A)^2>=0 and (B-U)*(U+V/B)^2>=0"
            ),
            "kappa3'=kappa4 and kappa4'=kappa5",
            "|kappa4|<=C^2*V+3*V^2",
            "|kappa5|<=C^3*V+10*C*V^2",
            "composite Simpson remainder uses an enclosure of H''''",
            "r'' lower enclosure implies the semiconvex secant majorant",
            "2*Phi(-A)<=2*phi(A)/A",
            "b(pi)<=v_star<=sup_t r(t,delta_pi)",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prior_json", type=Path)
    parser.add_argument("--m", required=True)
    parser.add_argument("--precision", type=int, default=60)
    parser.add_argument("--tail-cutoff", default="10")
    parser.add_argument("--z-cells", type=int, default=256)
    parser.add_argument("--tolerance", default="0.001")
    parser.add_argument("--max-nodes", type=int, default=2001)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data = json.loads(args.prior_json.read_text(encoding="utf-8"))
    prior_data = data["prior"] if "prior" in data else data
    prior = DecimalSymmetricPrior.from_mapping(prior_data)
    certificate = build_certificate(
        prior,
        Decimal(args.m),
        precision=args.precision,
        tail_cutoff=Decimal(args.tail_cutoff),
        z_cells=args.z_cells,
        separation_tolerance=Decimal(args.tolerance),
        max_nodes=args.max_nodes,
    )
    serialized = json.dumps(certificate, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
