#!/usr/bin/env python3
"""Independent numerical reconstructions of the central identities.

These tests intentionally evaluate the definitions directly rather than
calling the solver's risk or derivative methods.
"""

from __future__ import annotations

import math
import unittest

import numpy as np


def normal_rule(order: int = 256) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return math.sqrt(2.0) * nodes, weights / math.sqrt(math.pi)


def posterior(
    y: np.ndarray,
    atoms: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = (
        y[..., None] * atoms
        - 0.5 * atoms**2
        + np.log(weights)
    )
    logits -= logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    mean = probabilities @ atoms
    centered = atoms - mean[..., None]
    variance = np.sum(probabilities * centered**2, axis=-1)
    third = np.sum(probabilities * centered**3, axis=-1)
    return mean, variance, third


def risk(
    t: float,
    atoms: np.ndarray,
    weights: np.ndarray,
    order: int = 256,
) -> float:
    z, normal_weights = normal_rule(order)
    mean = posterior(t + z, atoms, weights)[0]
    return float(normal_weights @ (mean - t) ** 2)


def bayes_risk(
    atoms: np.ndarray,
    weights: np.ndarray,
    order: int = 256,
) -> float:
    return float(
        sum(
            weight * risk(float(theta), atoms, weights, order)
            for theta, weight in zip(atoms, weights)
        )
    )


class OptimalityIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.atoms = np.asarray([-4.0, -2.1264, -0.5985, 0.5985, 2.1264, 4.0])
        level_masses = np.asarray([0.23358, 0.39943, 0.36699])
        level_masses /= level_masses.sum()
        self.weights = np.asarray(
            [
                level_masses[0] / 2,
                level_masses[1] / 2,
                level_masses[2] / 2,
                level_masses[2] / 2,
                level_masses[1] / 2,
                level_masses[0] / 2,
            ]
        )

    def test_bayes_risk_is_prior_average_of_frequentist_risk(self) -> None:
        z, normal_weights = normal_rule()
        second_moment = float(self.weights @ (self.atoms**2))
        posterior_second = 0.0
        for theta, weight in zip(self.atoms, self.weights):
            mean = posterior(theta + z, self.atoms, self.weights)[0]
            posterior_second += float(weight * (normal_weights @ mean**2))
        direct_integral_identity = second_moment - posterior_second
        average_risk = bayes_risk(self.atoms, self.weights)
        self.assertAlmostEqual(direct_integral_identity, average_risk, places=13)

    def test_directional_derivative_toward_atom(self) -> None:
        base = bayes_risk(self.atoms, self.weights)
        for t in (-4.0, -1.37, 0.0, 2.71, 4.0):
            predicted = risk(t, self.atoms, self.weights) - base
            derivatives: list[float] = []
            for epsilon in (1e-4, 5e-5):
                augmented_atoms = np.append(self.atoms, t)
                augmented_weights = np.append(
                    (1.0 - epsilon) * self.weights,
                    epsilon,
                )
                derivatives.append(
                    (
                    bayes_risk(augmented_atoms, augmented_weights) - base
                    )
                    / epsilon
                )
            # Cancel the first-order finite-epsilon term.  The derivative
            # itself remains the one-sided feasible directional derivative.
            richardson = 2.0 * derivatives[1] - derivatives[0]
            self.assertLess(abs(richardson - predicted), 8e-8)

    def test_bayes_maximum_risk_bracket_for_fixed_estimator(self) -> None:
        values = np.asarray(
            [risk(float(theta), self.atoms, self.weights) for theta in self.atoms]
        )
        average = float(self.weights @ values)
        grid = np.linspace(-4.0, 4.0, 801)
        grid_maximum = max(
            risk(float(t), self.atoms, self.weights, order=192) for t in grid
        )
        self.assertLessEqual(average, grid_maximum + 2e-13)


class PosteriorFlowIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.atoms = np.asarray([-3.0, -0.7, 1.2, 4.0])
        self.weights = np.asarray([0.12, 0.28, 0.31, 0.29])

    def test_mean_derivative_equals_variance(self) -> None:
        for y in np.linspace(-6.0, 7.0, 31):
            step = 2e-5
            left = posterior(np.asarray(y - step), self.atoms, self.weights)[0]
            right = posterior(np.asarray(y + step), self.atoms, self.weights)[0]
            derivative = float((right - left) / (2 * step))
            variance = float(
                posterior(np.asarray(y), self.atoms, self.weights)[1]
            )
            self.assertAlmostEqual(derivative, variance, places=8)

    def test_variance_derivative_equals_third_cumulant(self) -> None:
        for y in np.linspace(-5.0, 6.0, 25):
            step = 3e-5
            left = posterior(
                np.asarray(y - step), self.atoms, self.weights
            )[1]
            right = posterior(
                np.asarray(y + step), self.atoms, self.weights
            )[1]
            derivative = float((right - left) / (2 * step))
            third = float(
                posterior(np.asarray(y), self.atoms, self.weights)[2]
            )
            self.assertAlmostEqual(derivative, third, places=7)

    def test_common_mean_representation(self) -> None:
        for y in np.linspace(-20.0, 20.0, 81):
            logits = y * self.atoms - 0.5 * self.atoms**2 + np.log(self.weights)
            logits -= logits.max()
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum()
            mean = float(probabilities @ self.atoms)
            centered = self.atoms - mean
            variance = float(probabilities @ centered**2)
            third = float(probabilities @ centered**3)
            if variance < 1e-24:
                self.assertLess(abs(third), 1e-20)
                continue
            common_mean = float(
                probabilities @ (self.atoms * centered**2) / variance
            )
            self.assertGreaterEqual(common_mean, self.atoms.min() - 1e-13)
            self.assertLessEqual(common_mean, self.atoms.max() + 1e-13)
            self.assertAlmostEqual(
                third,
                variance * (common_mean - mean),
                places=12,
            )

    def test_curvature_identity(self) -> None:
        z, normal_weights = normal_rule(256)
        for t in (0.0, 0.7, 2.8):
            mean, variance, third = posterior(
                t + z, self.atoms, self.weights
            )
            formula = float(
                2
                * normal_weights
                @ ((variance - 1.0) ** 2 + (mean - t) * third)
            )
            step = 2e-3
            finite_difference = (
                risk(t + step, self.atoms, self.weights)
                - 2 * risk(t, self.atoms, self.weights)
                + risk(t - step, self.atoms, self.weights)
            ) / step**2
            self.assertAlmostEqual(formula, finite_difference, places=5)


if __name__ == "__main__":
    unittest.main()
