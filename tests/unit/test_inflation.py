"""Slow-roll algebra and exact inflationary backgrounds.

The power-law family is the reference throughout: ``epsilon_V = 1`` at
``phi = p/sqrt(2)`` and ``N = (phi^2 - phi_end^2)/(2p)`` are both exact, so
the quadrature and the inversion can be checked against closed forms rather
than against themselves.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from particlesim.cosmo.inflation import (
    InflationNeverEnds,
    efolds,
    end_of_inflation,
    evolve_fields,
    evolve_inflation,
    field_at_efolds,
    run_to_end,
    slow_roll,
    slow_roll_prediction,
)
from particlesim.cosmo.potentials import (
    Exponential,
    Natural,
    PowerLaw,
    Quadratic,
    SeparableSum,
    Starobinsky,
    as_multifield,
)

EXPONENTS = [1.0, 2.0, 2.5, 4.0]


@pytest.mark.parametrize("exponent", EXPONENTS)
def test_end_of_inflation_matches_the_power_law_closed_form(exponent):
    potential = PowerLaw(amplitude=1e-12, exponent=exponent)
    assert end_of_inflation(potential) == pytest.approx(exponent / math.sqrt(2.0), rel=1e-12)


@pytest.mark.parametrize("exponent", EXPONENTS)
def test_efolds_matches_the_power_law_closed_form(exponent):
    potential = PowerLaw(amplitude=1e-12, exponent=exponent)
    end = end_of_inflation(potential)
    for phi in (10.0, 20.0):
        expected = (phi**2 - end**2) / (2.0 * exponent)
        assert efolds(potential, phi, end) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("exponent", EXPONENTS)
def test_field_at_efolds_inverts_the_e_fold_integral(exponent):
    potential = PowerLaw(amplitude=1e-12, exponent=exponent)
    phi = field_at_efolds(potential, 60.0)
    assert efolds(potential, phi) == pytest.approx(60.0, rel=1e-12)
    end = end_of_inflation(potential)
    assert phi == pytest.approx(math.sqrt(120.0 * exponent + end**2), rel=1e-10)


@pytest.mark.parametrize("exponent", EXPONENTS)
def test_power_law_observables_match_the_closed_form(exponent):
    """``n_s = 1 - (2p+4)/(4N+p)`` and ``r = 16p/(4N+p)``, exactly in slow roll."""
    potential = PowerLaw(amplitude=1e-12, exponent=exponent)
    prediction = slow_roll_prediction(potential, 55.0)
    denominator = 4.0 * 55.0 + exponent
    assert prediction.spectral_index == pytest.approx(
        1.0 - (2.0 * exponent + 4.0) / denominator, rel=1e-10
    )
    assert prediction.tensor_to_scalar == pytest.approx(16.0 * exponent / denominator, rel=1e-10)
    assert prediction.tensor_index == pytest.approx(-2.0 * exponent / denominator, rel=1e-10)


def test_natural_inflation_end_matches_the_closed_form():
    decay = 7.0
    expected = 2.0 * decay * math.atan(math.sqrt(2.0) * decay)
    assert end_of_inflation(Natural(decay_constant=decay)) == pytest.approx(expected, rel=1e-12)


def test_natural_inflation_rolls_away_from_the_origin():
    """The one built-in whose field increases, which is why nothing assumes."""
    potential = Natural(decay_constant=7.0)
    end = end_of_inflation(potential)
    pivot = field_at_efolds(potential, 55.0)
    assert 0.0 < pivot < end < potential.domain[1]


def test_efold_quadrature_is_converged_at_the_default_order():
    potential = Starobinsky()
    end = end_of_inflation(potential)
    values = [efolds(potential, 5.5, end, order=order) for order in (24, 48, 96, 192)]
    assert max(values) - min(values) < 1e-10


def test_exponential_potential_has_no_end_of_inflation():
    with pytest.raises(InflationNeverEnds, match="no end of inflation"):
        end_of_inflation(Exponential(rate=0.4))


def test_natural_inflation_cannot_reach_sixty_e_folds_at_small_decay_constant():
    """``f`` below about a Planck mass does not fit sixty e-folds in.

    The failure is reported as :class:`InflationNeverEnds` rather than as the
    edge of the search bracket, which is the difference between a model that
    is ruled out and a number that looks like a field value.
    """
    with pytest.raises(InflationNeverEnds, match="fewer than"):
        field_at_efolds(Natural(decay_constant=0.3), 60.0)


def test_efolds_refuses_an_interval_spanning_a_turning_point():
    potential = Natural(decay_constant=5.0)
    with pytest.raises(ValueError, match="changes sign"):
        efolds(potential, 5.0, 1.6 * math.pi * 5.0)


def test_slow_roll_refuses_a_non_positive_potential():
    with pytest.raises(ValueError, match="not positive"):
        slow_roll(Natural(decay_constant=5.0), math.pi * 5.0)


def test_end_of_inflation_refuses_a_starting_point_past_the_end():
    with pytest.raises(ValueError, match="already past the end"):
        end_of_inflation(Quadratic(), start=1.0)


def test_end_of_inflation_refuses_a_point_outside_the_domain():
    with pytest.raises(ValueError, match="outside the domain"):
        end_of_inflation(Quadratic(), start=-5.0)


def test_slow_roll_amplitudes_satisfy_the_consistency_relation():
    prediction = slow_roll_prediction(Quadratic(mass=1e-5), 55.0)
    ratio = prediction.tensor_amplitude / prediction.scalar_amplitude
    assert ratio == pytest.approx(prediction.tensor_to_scalar, rel=1e-12)
    assert set(prediction.summary()) == {
        "field",
        "epsilon",
        "eta",
        "spectral_index",
        "tensor_to_scalar",
        "running",
        "scalar_amplitude",
    }


def test_exact_background_satisfies_the_raychaudhuri_equation():
    """``d ln H / dN = -epsilon``, which the integration never imposes.

    ``H`` is algebraic in the state, so the Friedmann constraint holds by
    construction and checking it would prove nothing. Its derivative is a
    different statement: it is only true if the field equation was
    integrated correctly. The residual here is the central difference's own
    ``O(h^2)``, not the solver's.
    """
    run = run_to_end(Starobinsky(), 55.0, margin=8.0)
    grid = np.linspace(2.0, run.total_efolds - 2.0, 40)
    step = 1e-4
    slope = (np.log(run.hubble_at(grid + step)) - np.log(run.hubble_at(grid - step))) / (2 * step)
    assert np.allclose(slope, -run.epsilon_at(grid), rtol=1e-6, atol=1e-12)


def test_inflation_ends_at_epsilon_one():
    run = run_to_end(Quadratic(), 55.0, margin=8.0)
    assert run.ended
    assert float(run.epsilon[-1]) == pytest.approx(1.0, abs=1e-9)
    assert float(run.efolds[-1]) == pytest.approx(run.total_efolds)


def test_the_exact_e_fold_count_exceeds_the_slow_roll_estimate():
    """Which surface ends inflation is worth more than a part in a thousand of n_s.

    Slow roll stops at ``epsilon_V = 1``; the exact evolution runs on to
    ``epsilon_H = 1``, which for a plateau potential is one and a half
    e-folds further. Measured against ``dn_s/dN ~ 2/N^2``, that is 1e-3 --
    the size of the criterion the Starobinsky benchmark is stated to, which
    is why the pivot is placed by the numerical count and not this one.
    """
    potential = Starobinsky()
    run = run_to_end(potential, 55.0, margin=8.0)
    gap = run.total_efolds - 63.0
    assert 1.0 < gap < 2.5
    assert float(run.field[-1]) < end_of_inflation(potential)


def test_starting_from_rest_reaches_the_slow_roll_attractor():
    potential = Quadratic()
    phi = field_at_efolds(potential, 60.0)
    run = evolve_inflation(potential, phi, velocity=0.0)
    for number, tolerance in ((1.0, 6e-2), (2.0, 6e-3), (3.0, 3e-3)):
        field, velocity = run.state(number)
        attractor = -float(potential.gradient(field) / potential.value(field))
        assert float(velocity) == pytest.approx(attractor, rel=tolerance)
    on_attractor = evolve_inflation(potential, phi)
    assert run.total_efolds - on_attractor.total_efolds == pytest.approx(0.333, abs=0.01)


def test_run_to_end_places_the_pivot_inside_the_run():
    run = run_to_end(Starobinsky(), 55.0, margin=8.0)
    pivot = run.at_efolds_remaining(55.0)
    assert pivot > float(run.efolds[0])
    assert run.efolds_remaining(pivot) == pytest.approx(55.0)
    with pytest.raises(ValueError, match="does not reach back"):
        run.at_efolds_remaining(200.0)


def test_evolve_inflation_refuses_a_non_inflating_start():
    with pytest.raises(ValueError, match="not inflating"):
        evolve_inflation(Quadratic(), 15.0, velocity=2.0)


def test_evolve_inflation_refuses_a_field_outside_the_domain():
    with pytest.raises(ValueError, match="outside the domain"):
        evolve_inflation(Quadratic(), -15.0)


def test_run_summary_reports_the_ends():
    run = run_to_end(Quadratic(), 55.0, margin=8.0)
    summary = run.summary()
    assert summary["ended"] is True
    assert summary["initial_field"] > summary["final_field"]
    assert summary["final_epsilon"] == pytest.approx(1.0, abs=1e-9)


def test_two_equal_masses_reduce_to_one_field():
    """A straight trajectory in two fields is single-field inflation.

    With equal masses ``V = (1/2) m^2 (phi_1^2 + phi_2^2)`` depends only on
    the radius, so the radial trajectory is an exact solution and the
    vector machinery has to reproduce the scalar machinery exactly. It does,
    to eleven digits, which is a stronger statement than either agreeing
    with a formula.
    """
    mass = 1e-5
    single = Quadratic(mass=mass)
    phi = field_at_efolds(single, 60.0)
    one = evolve_inflation(single, phi)
    two = evolve_fields(
        SeparableSum((Quadratic(mass=mass), Quadratic(mass=mass))),
        [phi / math.sqrt(2.0)] * 2,
    )
    assert two.total_efolds == pytest.approx(one.total_efolds, abs=1e-8)
    assert np.allclose(two.hubble, one.hubble, rtol=1e-9)
    radius = np.sqrt(np.sum(two.field**2, axis=-1))
    assert np.allclose(radius, one.field, rtol=1e-9)
    assert two.summary()["fields"] == 2


def test_one_field_through_the_multifield_solver_matches_the_single_field_solver():
    potential = Starobinsky()
    phi = field_at_efolds(potential, 60.0)
    one = evolve_inflation(potential, phi)
    wrapped = evolve_fields(as_multifield(potential), [phi])
    assert wrapped.total_efolds == pytest.approx(one.total_efolds, abs=1e-8)
    assert np.allclose(wrapped.field[:, 0], one.field, rtol=1e-9)


def test_unequal_masses_bend_the_trajectory():
    """Otherwise the multi-field tests would be single-field tests."""
    run = evolve_fields(SeparableSum((Quadratic(mass=1e-5), Quadratic(mass=3e-5))), [12.0, 9.0])
    ratio = run.field[:, 0] / run.field[:, 1]
    midpoint = ratio[run.efolds.size // 2]
    assert ratio[0] == pytest.approx(12.0 / 9.0)
    assert abs(midpoint / ratio[0]) > 2.0


def test_evolve_fields_checks_the_shape_of_its_initial_data():
    potential = SeparableSum((Quadratic(), Quadratic()))
    with pytest.raises(ValueError, match="expected 2 initial field values"):
        evolve_fields(potential, [10.0])


def test_evolve_fields_refuses_a_non_inflating_start():
    potential = SeparableSum((Quadratic(), Quadratic()))
    with pytest.raises(ValueError, match="not inflating"):
        evolve_fields(potential, [10.0, 10.0], velocity=[1.5, 1.5])


def test_a_custom_stopping_surface_terminates_the_run():
    potential = SeparableSum((Quadratic(mass=1e-5), Quadratic(mass=3e-5)))
    target = 100.0

    def surface(phi, _velocity):
        return float(np.sum(np.asarray(phi) ** 2)) - target

    run = evolve_fields(potential, [12.0, 9.0], stop=surface)
    assert run.ended
    assert float(np.sum(run.field[-1] ** 2)) == pytest.approx(target, rel=1e-9)
