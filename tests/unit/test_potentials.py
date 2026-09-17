"""Inflaton potentials: analytic derivatives against numerical ones."""

from __future__ import annotations

import math

import numpy as np
import pytest

from particlesim.cosmo.potentials import (
    STAROBINSKY_EXPONENT,
    Exponential,
    Natural,
    Potential,
    PowerLaw,
    Quadratic,
    SeparableSum,
    Starobinsky,
    as_multifield,
)

POTENTIALS = [
    (Quadratic(mass=1e-5), 12.0),
    (PowerLaw(amplitude=1e-12, exponent=1.0), 8.0),
    (PowerLaw(amplitude=1e-12, exponent=4.0), 20.0),
    (PowerLaw(amplitude=1e-12, exponent=2.5), 15.0),
    (Exponential(amplitude=1e-10, rate=0.4), 1.5),
    (Starobinsky(amplitude=1e-10), 5.3),
    (Natural(amplitude=1e-10, decay_constant=7.0), 8.0),
]


def _derivative(f, x, step=1e-5):
    return (f(x + step) - f(x - step)) / (2.0 * step)


@pytest.mark.parametrize(("potential", "phi"), POTENTIALS)
def test_first_derivative_matches_a_difference(potential, phi):
    numerical = _derivative(potential.value, phi)
    assert float(potential.gradient(phi)) == pytest.approx(float(numerical), rel=1e-7)


@pytest.mark.parametrize(("potential", "phi"), POTENTIALS)
def test_second_derivative_matches_a_difference(potential, phi):
    numerical = _derivative(potential.gradient, phi)
    assert float(potential.curvature(phi)) == pytest.approx(float(numerical), rel=1e-7)


@pytest.mark.parametrize(("potential", "phi"), POTENTIALS)
def test_third_derivative_matches_a_difference(potential, phi):
    """Every built-in overrides the finite-difference fallback analytically.

    An analytic derivative that is wrong is worse than a numerical one that
    is approximate, so each override is checked against the thing it
    replaces rather than trusted.
    """
    numerical = _derivative(potential.curvature, phi, step=1e-4)
    assert float(potential.third_derivative(phi)) == pytest.approx(float(numerical), rel=1e-6)


def test_finite_difference_fallback_is_used_when_not_overridden():
    class Bare(Potential):
        typical_field = 1.0

        def value(self, phi):
            return np.exp(-0.3 * np.asarray(phi, dtype=float))

        def gradient(self, phi):
            return -0.3 * self.value(phi)

        def curvature(self, phi):
            return 0.09 * self.value(phi)

    bare = Bare()
    assert float(bare.third_derivative(2.0)) == pytest.approx(-0.027 * math.exp(-0.6), rel=1e-7)


def test_starobinsky_slow_roll_parameters_match_closed_forms():
    """Closed forms in ``x = exp(-sqrt(2/3) phi)``, derived from the potential.

        epsilon_V = (4/3) x^2/(1-x)^2
        eta_V     = -(4/3) x (1-2x)/(1-x)^2
        xi_V^2    = (16/9) x^2 (1-4x)/(1-x)^3

    These are what make the Starobinsky benchmark an algebra check on the
    potential as well as a check on the mode solver: an error in any of the
    three derivatives would move ``n_s`` without moving anything else.
    """
    potential = Starobinsky()
    for phi in (2.0, 5.35, 8.0):
        x = math.exp(-STAROBINSKY_EXPONENT * phi)
        assert float(potential.epsilon(phi)) == pytest.approx(
            (4.0 / 3.0) * x**2 / (1.0 - x) ** 2, rel=1e-12
        )
        assert float(potential.eta(phi)) == pytest.approx(
            -(4.0 / 3.0) * x * (1.0 - 2.0 * x) / (1.0 - x) ** 2, rel=1e-12
        )
        assert float(potential.xi_squared(phi)) == pytest.approx(
            (16.0 / 9.0) * x**2 * (1.0 - 4.0 * x) / (1.0 - x) ** 3, rel=1e-12
        )


def test_exponential_epsilon_is_constant():
    potential = Exponential(rate=0.4)
    values = potential.epsilon(np.array([-5.0, 0.0, 12.0]))
    assert np.allclose(values, 0.5 * 0.4**2, rtol=1e-14)


def test_natural_epsilon_matches_the_half_angle_form():
    f = 7.0
    potential = Natural(decay_constant=f)
    for phi in (1.0, 9.0, 18.0):
        tangent = math.tan(0.5 * phi / f)
        assert float(potential.epsilon(phi)) == pytest.approx(tangent**2 / (2.0 * f**2), rel=1e-12)


def test_roll_direction_follows_the_gradient():
    assert Quadratic().roll_direction(10.0) == -1.0
    assert Starobinsky().roll_direction(5.0) == -1.0
    assert Natural(decay_constant=5.0).roll_direction(5.0) == +1.0


def test_roll_direction_refuses_a_flat_point():
    with pytest.raises(ValueError, match="flat"):
        Natural(decay_constant=5.0).roll_direction(0.0)


def test_domains():
    assert Natural(decay_constant=5.0).domain == (0.0, pytest.approx(5.0 * math.pi))
    assert not bool(Natural(decay_constant=5.0).contains(16.0))
    assert bool(Quadratic().contains(3.0))
    assert not bool(Quadratic().contains(-3.0))
    assert bool(Exponential().contains(-100.0))


def test_power_law_rejects_a_non_positive_exponent():
    with pytest.raises(ValueError, match="exponent must be positive"):
        PowerLaw(exponent=0.0)


def test_natural_rejects_a_non_positive_decay_constant():
    with pytest.raises(ValueError, match="decay_constant must be positive"):
        Natural(decay_constant=0.0)


def test_separable_sum_adds_its_parts():
    parts = (Quadratic(mass=1e-5), Quadratic(mass=3e-5), Exponential(rate=0.2))
    total = SeparableSum(parts)
    phi = np.array([10.0, 4.0, 1.0])
    assert total.fields == 3
    assert float(total.value(phi)) == pytest.approx(
        sum(float(part.value(phi[i])) for i, part in enumerate(parts))
    )
    assert np.allclose(
        total.gradient(phi), [float(part.gradient(phi[i])) for i, part in enumerate(parts)]
    )
    hessian = total.hessian(phi)
    assert hessian.shape == (3, 3)
    assert np.allclose(
        np.diag(hessian), [float(part.curvature(phi[i])) for i, part in enumerate(parts)]
    )
    assert np.allclose(hessian - np.diag(np.diag(hessian)), 0.0)


def test_separable_sum_is_vectorized_over_points():
    total = SeparableSum((Quadratic(mass=1e-5), Quadratic(mass=2e-5)))
    points = np.array([[10.0, 5.0], [8.0, 4.0], [1.0, 0.5]])
    assert total.value(points).shape == (3,)
    assert total.gradient(points).shape == (3, 2)
    assert total.hessian(points).shape == (3, 2, 2)


def test_separable_sum_needs_a_part():
    with pytest.raises(ValueError, match="at least one part"):
        SeparableSum(())


def test_as_multifield_agrees_with_the_single_field_case():
    potential = Starobinsky()
    wrapped = as_multifield(potential)
    for phi in (2.0, 5.0):
        assert float(wrapped.value([phi])) == pytest.approx(float(potential.value(phi)))
        assert float(wrapped.epsilon([phi])) == pytest.approx(float(potential.epsilon(phi)))
