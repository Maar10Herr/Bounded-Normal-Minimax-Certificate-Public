#!/usr/bin/env python3
"""Adversarial tests reconstructed from the original moment definitions."""

from __future__ import annotations

import unittest

import numpy as np

from src.solver.solver_fast import (
    GaussianMixtureRisk,
    SymmetricPrior,
    _normal_hermite_rule,
    _posterior_flow_enclosure,
    local_joint_second_derivative_interval,
    optimize_level_masses,
)


class SharpThirdMomentTests(unittest.TestCase):
    def test_random_discrete_distributions(self) -> None:
        rng = np.random.default_rng(20260724)
        worst_lower_slack = np.inf
        worst_upper_slack = np.inf
        for _ in range(50_000):
            atom_count = int(rng.integers(2, 10))
            atoms = np.sort(rng.uniform(-5.0, 5.0, atom_count))
            weights = rng.dirichlet(np.ones(atom_count))
            mean = float(weights @ atoms)
            centered = atoms - mean
            variance = float(weights @ (centered**2))
            third = float(weights @ (centered**3))
            left_radius = mean - float(atoms[0])
            right_radius = float(atoms[-1]) - mean

            if variance == 0.0:
                lower = upper = 0.0
            else:
                lower = variance * (
                    variance / left_radius - left_radius
                )
                upper = variance * (
                    right_radius - variance / right_radius
                )
            scale = max(1.0, abs(lower), abs(third), abs(upper))
            self.assertGreaterEqual(
                third - lower,
                -2e-12 * scale,
            )
            self.assertGreaterEqual(
                upper - third,
                -2e-12 * scale,
            )
            worst_lower_slack = min(worst_lower_slack, third - lower)
            worst_upper_slack = min(worst_upper_slack, upper - third)

        self.assertGreaterEqual(worst_lower_slack, -1e-10)
        self.assertGreaterEqual(worst_upper_slack, -1e-10)

    def test_two_point_equality(self) -> None:
        rng = np.random.default_rng(1729)
        for _ in range(10_000):
            left, right = np.sort(rng.uniform(-10.0, 10.0, 2))
            probability = float(rng.uniform(1e-8, 1.0 - 1e-8))
            atoms = np.array([left, right])
            weights = np.array([probability, 1.0 - probability])
            mean = float(weights @ atoms)
            centered = atoms - mean
            variance = float(weights @ (centered**2))
            third = float(weights @ (centered**3))
            left_radius = mean - left
            right_radius = right - mean
            lower = variance * (
                variance / left_radius - left_radius
            )
            upper = variance * (
                right_radius - variance / right_radius
            )
            scale = max(1.0, abs(third))
            self.assertAlmostEqual(lower / scale, third / scale, places=11)
            self.assertAlmostEqual(upper / scale, third / scale, places=11)


class PosteriorFlowTests(unittest.TestCase):
    @staticmethod
    def _evaluators() -> list[tuple[float, GaussianMixtureRisk]]:
        specifications = [
            (1.0, [1.0], [1.0]),
            (2.0, [0.0, 2.0], [0.42041493258, 0.57958506742]),
            (
                4.0,
                [0.0, 2.0943603515625, 4.0],
                [0.3029107793, 0.4647008393, 0.2323883814],
            ),
        ]
        return [
            (
                m,
                GaussianMixtureRisk(
                    SymmetricPrior(
                        np.asarray(levels),
                        np.asarray(masses),
                    ),
                    order=128,
                ),
            )
            for m, levels, masses in specifications
        ]

    def test_random_flow_tubes_from_original_posterior(self) -> None:
        rng = np.random.default_rng(1009)
        for m, evaluator in self._evaluators():
            for _ in range(500):
                center = float(rng.uniform(-m - 5.0, m + 5.0))
                half_width = float(10 ** rng.uniform(-5.0, -0.2))
                lower = center - half_width
                upper = center + half_width
                enclosure = _posterior_flow_enclosure(
                    evaluator, lower, upper
                )
                grid = np.linspace(lower, upper, 129)
                states = np.asarray(
                    [
                        evaluator.posterior_moments_y(float(y))
                        for y in grid
                    ]
                )
                scale = max(1.0, m**3)
                self.assertGreaterEqual(
                    float(states[:, 0].min()) - enclosure[0],
                    -2e-12 * scale,
                )
                self.assertGreaterEqual(
                    enclosure[1] - float(states[:, 0].max()),
                    -2e-12 * scale,
                )
                self.assertGreaterEqual(
                    float(states[:, 1].min()) - enclosure[2],
                    -2e-12 * scale,
                )
                self.assertGreaterEqual(
                    enclosure[3] - float(states[:, 1].max()),
                    -2e-12 * scale,
                )
                self.assertGreaterEqual(
                    float(states[:, 2].min()) - enclosure[4],
                    -2e-12 * scale,
                )
                self.assertGreaterEqual(
                    enclosure[5] - float(states[:, 2].max()),
                    -2e-12 * scale,
                )

    def test_random_curvature_cells(self) -> None:
        rng = np.random.default_rng(65537)
        for m, evaluator in self._evaluators():
            for _ in range(250):
                endpoints = np.sort(rng.uniform(0.0, m, 2))
                lower = float(endpoints[0])
                upper = float(endpoints[1])
                enclosure = local_joint_second_derivative_interval(
                    evaluator, lower, upper
                )
                grid = np.linspace(lower, upper, 129)
                values = np.asarray(
                    [
                        evaluator.evaluate(float(t)).second_derivative
                        for t in grid
                    ]
                )
                scale = max(
                    1.0,
                    abs(enclosure.lower),
                    abs(enclosure.upper),
                )
                self.assertGreaterEqual(
                    float(values.min()) - enclosure.lower,
                    -2e-11 * scale,
                )
                self.assertGreaterEqual(
                    enclosure.upper - float(values.max()),
                    -2e-11 * scale,
                )


class FastOptimizerTests(unittest.TestCase):
    def test_quadrature_rule_cache_is_immutable(self) -> None:
        first = _normal_hermite_rule(192)
        second = _normal_hermite_rule(192)
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])
        self.assertFalse(first[0].flags.writeable)
        self.assertFalse(first[1].flags.writeable)

    def test_two_level_mass_kkt_equality(self) -> None:
        prior = optimize_level_masses(
            [0.0, 1.5],
            quadrature_order=192,
        )
        risks = GaussianMixtureRisk(prior, order=192).level_risks()
        self.assertGreater(float(prior.level_masses.min()), 1e-6)
        self.assertAlmostEqual(float(risks[0]), float(risks[1]), places=13)

    def test_support_location_envelope_gradient(self) -> None:
        prior = SymmetricPrior(
            np.asarray([0.0, 1.9, 4.0]),
            np.asarray([0.31, 0.46, 0.23]),
        )
        evaluator = GaussianMixtureRisk(prior, order=256)
        analytic = (
            float(prior.level_masses[1])
            * evaluator.evaluate(float(prior.levels[1])).first_derivative
        )
        step = 2e-5
        displaced_risks = []
        for direction in (-1.0, 1.0):
            displaced = SymmetricPrior(
                np.asarray([0.0, 1.9 + direction * step, 4.0]),
                prior.level_masses,
            )
            displaced_risks.append(
                GaussianMixtureRisk(displaced, order=256).bayes_risk()
            )
        finite_difference = (
            displaced_risks[1] - displaced_risks[0]
        ) / (2.0 * step)
        self.assertAlmostEqual(analytic, finite_difference, places=9)


if __name__ == "__main__":
    unittest.main()
