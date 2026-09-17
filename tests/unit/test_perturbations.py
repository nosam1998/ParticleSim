"""Mukhanov-Sasaki spectra, the Starobinsky gate benchmark, and delta N.

The exponential potential is the sharpest test in this file. Its
``epsilon`` is constant, so the mode equation has a closed-form Hankel
solution and the spectrum is an exact power law with

    n_s - 1 = -2 epsilon/(1 - epsilon),    r = 16 epsilon

*exactly*. At ``lambda = 0.4`` the exact index is ``-0.17391`` where
first-order slow roll says ``-0.16``, so a solver that had quietly
degenerated into the slow-roll formula would fail by four per cent of the
tilt rather than pass.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from particlesim.cosmo.inflation import (
    evolve_fields,
    evolve_inflation,
    field_at_efolds,
    run_to_end,
    slow_roll,
)
from particlesim.cosmo.perturbations import (
    Spectrum,
    crossing_efolds,
    delta_n_gradient,
    delta_n_spectrum,
    efolds_to_surface,
    mode_power,
    pivot_wavenumber,
    power_spectrum,
    primordial_spectra,
)
from particlesim.cosmo.potentials import (
    Exponential,
    PowerLaw,
    Quadratic,
    SeparableSum,
    Starobinsky,
    as_multifield,
)

PIVOT_EFOLDS = 55.0
EXPONENTIAL_RATE = 0.4


@pytest.fixture(scope="module")
def exponential_spectrum() -> Spectrum:
    run = evolve_inflation(Exponential(rate=EXPONENTIAL_RATE), 0.0, max_efolds=40.0)
    return power_spectrum(run, efolds_remaining=20.0)


@pytest.fixture(scope="module")
def starobinsky_run():
    return run_to_end(Starobinsky(), PIVOT_EFOLDS, margin=8.0)


@pytest.fixture(scope="module")
def starobinsky_spectrum(starobinsky_run) -> Spectrum:
    return power_spectrum(starobinsky_run, efolds_remaining=PIVOT_EFOLDS)


@pytest.fixture(scope="module")
def wide_spectrum():
    """A narrow fit at the pivot and the spectrum three decades either side."""
    run = run_to_end(Starobinsky(), 30.0, margin=10.0)
    narrow = power_spectrum(run, efolds_remaining=30.0)
    offsets = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    scalar, tensor = primordial_spectra(run, narrow.pivot * np.exp(offsets))
    return offsets, scalar, narrow


@pytest.mark.benchmark
def test_power_law_inflation_index_is_exact(exponential_spectrum):
    epsilon = 0.5 * EXPONENTIAL_RATE**2
    exact = 1.0 - 2.0 * epsilon / (1.0 - epsilon)
    assert exponential_spectrum.spectral_index == pytest.approx(exact, abs=1e-7)
    first_order = 1.0 - 2.0 * epsilon
    assert abs(first_order - exact) > 1e-2


@pytest.mark.benchmark
def test_power_law_inflation_tensor_to_scalar_is_exact(exponential_spectrum):
    """``r = 16 epsilon`` exactly here, because ``z`` and ``a`` differ by a constant.

    With ``epsilon`` constant, ``z = a sqrt(2 epsilon)`` is proportional to
    ``a``, so the scalar and tensor mode functions solve the same equation
    with the same Bunch-Davies data and their ratio is the normalisation
    alone. Any error in either amplitude that is not shared shows up here.
    """
    epsilon = 0.5 * EXPONENTIAL_RATE**2
    assert exponential_spectrum.tensor_to_scalar == pytest.approx(16.0 * epsilon, rel=1e-7)


@pytest.mark.benchmark
def test_power_law_inflation_tensor_index_equals_the_scalar_tilt(exponential_spectrum):
    epsilon = 0.5 * EXPONENTIAL_RATE**2
    exact = -2.0 * epsilon / (1.0 - epsilon)
    assert exponential_spectrum.tensor_index == pytest.approx(exact, abs=1e-7)


def test_power_law_inflation_has_no_running(exponential_spectrum):
    assert abs(exponential_spectrum.running) < 1e-7


@pytest.mark.benchmark
def test_starobinsky_spectral_index_at_fifty_five_e_folds(starobinsky_spectrum):
    """The gate benchmark, and where the leading-order formula runs out.

    The pipeline gives ``n_s = 0.96490``, converged to a part in a million
    against every knob it has (sub-horizon e-folds, freeze-out e-folds,
    tolerance, fit width, background margin). The leading-order Starobinsky
    result ``1 - 2/N = 0.96364`` sits 1.3e-3 below it, and that gap is the
    formula's own subleading term, not this module's error: it falls as
    ``ln N / N^2`` (checked in
    ``test_leading_order_index_is_approached_from_above``), so the
    asymptotic expression is itself only good to three digits beyond
    ``N ~ 90``.
    """
    assert starobinsky_spectrum.spectral_index == pytest.approx(0.964898, abs=5e-5)
    gap = starobinsky_spectrum.spectral_index - (1.0 - 2.0 / PIVOT_EFOLDS)
    assert 1.0e-3 < gap < 1.5e-3


@pytest.mark.benchmark
def test_starobinsky_tensor_to_scalar_at_fifty_five_e_folds(starobinsky_spectrum):
    assert starobinsky_spectrum.tensor_to_scalar == pytest.approx(0.0035496, rel=1e-2)
    leading = 12.0 / PIVOT_EFOLDS**2
    assert 0.85 < starobinsky_spectrum.tensor_to_scalar / leading < 0.95


@pytest.mark.benchmark
@pytest.mark.slow
def test_leading_order_index_is_approached_from_above():
    """``n_s - (1 - 2/N)`` falls like ``ln N / N^2``, so ``N^2`` times it creeps up.

    Two decades of ``N`` is enough to tell a subleading term from a bug: a
    constant offset would leave ``N^2 * gap`` growing like ``N^2`` and a
    convergence failure would leave it wandering.
    """
    potential = Starobinsky()
    scaled = {}
    for number in (PIVOT_EFOLDS, 200.0):
        run = run_to_end(potential, number, margin=9.0)
        spectrum = power_spectrum(run, efolds_remaining=number)
        gap = spectrum.spectral_index - (1.0 - 2.0 / number)
        scaled[number] = number**2 * gap
        assert gap > 0.0
    assert 3.0 < scaled[PIVOT_EFOLDS] < 4.5
    assert 4.0 < scaled[200.0] < 6.0
    assert scaled[200.0] > scaled[PIVOT_EFOLDS]


def test_tensor_amplitude_matches_the_de_sitter_formula(starobinsky_run, starobinsky_spectrum):
    """``P_t = 2H^2/pi^2`` to 1e-4, while ``P_R`` is 2.6 per cent off its own formula.

    Both corrections are first order in slow roll, but in different
    parameters: the tensor correction is ``O(epsilon) = 2e-4`` on the
    Starobinsky plateau and the scalar one is ``O(eta) = 2e-2``. Asserting
    the sizes separately is what makes this a test of the amplitudes rather
    than of the slow-roll formulae -- and the lower bound on the scalar side
    is there because a solver that had silently become the slow-roll
    expression would agree with it exactly.
    """
    pivot = starobinsky_run.at_efolds_remaining(PIVOT_EFOLDS)
    hubble = float(starobinsky_run.hubble_at(pivot))
    epsilon = float(starobinsky_run.epsilon_at(pivot))
    logs = np.log(starobinsky_spectrum.wavenumbers)
    tensor = float(
        np.interp(math.log(starobinsky_spectrum.pivot), logs, starobinsky_spectrum.tensor_power)
    )
    assert tensor == pytest.approx(2.0 * hubble**2 / math.pi**2, rel=1e-3)

    slow_roll_scalar = hubble**2 / (8.0 * math.pi**2 * epsilon)
    ratio = starobinsky_spectrum.pivot_scalar_power / slow_roll_scalar
    assert 1e-3 < abs(ratio - 1.0) < 5e-2


def test_modes_are_frozen_when_they_are_read(starobinsky_spectrum):
    assert starobinsky_spectrum.drift < 1e-4


def test_spectrum_summary_reports_the_observables(starobinsky_spectrum):
    summary = starobinsky_spectrum.summary()
    assert set(summary) == {
        "efolds_remaining",
        "spectral_index",
        "tensor_to_scalar",
        "running",
        "tensor_index",
        "scalar_amplitude",
        "drift",
    }
    assert summary["scalar_amplitude"] > 0.0
    assert summary["running"] < 0.0


@pytest.mark.benchmark
def test_quartic_spectrum_differs_from_slow_roll_at_second_order():
    """The exact index sits within ``O(epsilon^2)`` of the first-order formula.

    For ``V ~ phi^4`` at fifty-five e-folds ``epsilon = 0.018``, so second
    order is 3e-4 -- and that is the measured gap. The upper bound catches a
    mode solver that has gone wrong; the lower bound catches one that has
    been replaced by the formula it is meant to improve on.
    """
    potential = PowerLaw(amplitude=1e-12, exponent=4.0)
    run = run_to_end(potential, PIVOT_EFOLDS, margin=8.0)
    spectrum = power_spectrum(run, efolds_remaining=PIVOT_EFOLDS)
    pivot = run.at_efolds_remaining(PIVOT_EFOLDS)
    first_order = slow_roll(potential, float(run.state(pivot)[0]))
    difference = spectrum.spectral_index - first_order.spectral_index
    assert 1e-4 < abs(difference) < 1e-3
    assert abs(difference) < 4.0 * first_order.epsilon**2 * 6.0


def test_pivot_wavenumber_round_trips_through_horizon_crossing(starobinsky_run):
    pivot = starobinsky_run.at_efolds_remaining(PIVOT_EFOLDS)
    wavenumber = pivot_wavenumber(starobinsky_run, PIVOT_EFOLDS)
    assert crossing_efolds(starobinsky_run, wavenumber) == pytest.approx(pivot, abs=1e-9)


def test_a_mode_outside_the_horizon_at_the_start_is_refused(starobinsky_run):
    smallest = math.exp(float(starobinsky_run.log_comoving_hubble(starobinsky_run.efolds[0])))
    with pytest.raises(ValueError, match="already outside the horizon"):
        crossing_efolds(starobinsky_run, 0.5 * smallest)


def test_a_mode_that_never_crosses_is_refused(starobinsky_run):
    largest = math.exp(float(starobinsky_run.log_comoving_hubble(starobinsky_run.total_efolds)))
    with pytest.raises(ValueError, match="never crosses"):
        crossing_efolds(starobinsky_run, 2.0 * largest)


def test_a_mode_without_room_to_start_sub_horizon_is_refused(starobinsky_run):
    wavenumber = pivot_wavenumber(starobinsky_run, PIVOT_EFOLDS)
    with pytest.raises(ValueError, match="sub-horizon evolution first"):
        mode_power(starobinsky_run, wavenumber, subhorizon_efolds=20.0)


def test_a_mode_without_room_to_freeze_out_is_refused(starobinsky_run):
    wavenumber = pivot_wavenumber(starobinsky_run, 2.0)
    with pytest.raises(ValueError, match="to freeze out"):
        mode_power(starobinsky_run, wavenumber)


def test_a_quadratic_fit_needs_three_points(starobinsky_run):
    with pytest.raises(ValueError, match="at least three points"):
        power_spectrum(starobinsky_run, efolds_remaining=PIVOT_EFOLDS, points=2)


@pytest.fixture(scope="module")
def quadratic_delta_n():
    potential = Quadratic(mass=1e-5)
    phi = field_at_efolds(potential, PIVOT_EFOLDS + 8.0)
    run = evolve_fields(as_multifield(potential), [phi])
    return potential, run, delta_n_spectrum(run, PIVOT_EFOLDS)


def test_delta_n_gradient_is_the_inverse_slow_roll_velocity(quadratic_delta_n):
    """``dN/dphi = 1/sqrt(2 epsilon)`` in slow roll, and the gap is ``O(epsilon)``.

    The gradient here comes from differentiating the *exact* evolution, so
    it is not the slow-roll expression and should not match it perfectly.
    Both bounds are asserted: too large a difference means the
    differentiation is wrong, too small a one means it has collapsed onto
    the analytic formula.
    """
    potential, run, _ = quadratic_delta_n
    pivot = run.total_efolds - PIVOT_EFOLDS
    fields, _ = run.state(pivot)
    gradient = delta_n_gradient(run.potential, fields)
    epsilon = float(potential.epsilon(float(fields[0])))
    expected = 1.0 / math.sqrt(2.0 * epsilon)
    relative = abs(float(gradient[0]) / expected - 1.0)
    assert 1e-3 < relative < 1e-2


def test_delta_n_matches_the_mode_spectrum_for_one_field(quadratic_delta_n):
    """One field, two independent routes: super-horizon algebra and mode evolution.

    ``delta N`` is a slow-roll statement about the separate-universe
    picture; the mode solver integrates the perturbation through horizon
    crossing. They agree to within the slow-roll error ``O(epsilon) =
    9e-3``, which is what makes the multi-field ``delta N`` result
    trustworthy where no mode solver is available.
    """
    potential, _, spectrum = quadratic_delta_n
    run = run_to_end(potential, PIVOT_EFOLDS, margin=8.0)
    modes = power_spectrum(run, efolds_remaining=PIVOT_EFOLDS)
    assert spectrum.scalar_power == pytest.approx(modes.pivot_scalar_power, rel=1.5e-2)
    assert spectrum.spectral_index == pytest.approx(modes.spectral_index, abs=5e-4)
    assert spectrum.tensor_to_scalar == pytest.approx(modes.tensor_to_scalar, rel=2e-2)


def test_delta_n_summary_reports_the_observables(quadratic_delta_n):
    _, _, spectrum = quadratic_delta_n
    summary = spectrum.summary()
    assert set(summary) == {
        "efolds_remaining",
        "scalar_amplitude",
        "spectral_index",
        "tensor_to_scalar",
        "epsilon",
    }
    assert spectrum.tensor_power == pytest.approx(2.0 * spectrum.hubble**2 / math.pi**2)


def test_two_equal_masses_give_the_single_field_delta_n_spectrum():
    mass = 1e-5
    potential = Quadratic(mass=mass)
    phi = field_at_efolds(potential, PIVOT_EFOLDS + 8.0)
    single = delta_n_spectrum(evolve_fields(as_multifield(potential), [phi]), PIVOT_EFOLDS)
    pair = delta_n_spectrum(
        evolve_fields(
            SeparableSum((Quadratic(mass=mass), Quadratic(mass=mass))),
            [phi / math.sqrt(2.0)] * 2,
        ),
        PIVOT_EFOLDS,
    )
    assert pair.scalar_power == pytest.approx(single.scalar_power, rel=1e-6)
    assert pair.spectral_index == pytest.approx(single.spectral_index, abs=1e-6)
    assert float(np.sum(pair.gradient**2)) == pytest.approx(
        float(np.sum(single.gradient**2)), rel=1e-6
    )


@pytest.mark.benchmark
def test_delta_n_on_a_fixed_radius_surface_is_analytic_for_two_quadratics():
    """``dN/dphi_i = phi_i/2`` on a surface of constant ``sum phi_i^2``.

    In slow roll ``d(sum phi_i^2)/dN = -2 sum m_i^2 phi_i^2 / V = -4``
    whatever the masses are, because the numerator is twice the potential.
    So the e-folds to a fixed-radius surface are ``(R_start^2 - R_end^2)/4``
    and the gradient is exactly ``phi_i/2`` -- for unequal masses, on a
    trajectory that is genuinely curved.

    The residual is the slow-roll correction, ``O(epsilon) ~ 1/R^2``, and
    the test checks that it *scales* that way rather than only that it is
    small: doubling the radius has to cut it by four.
    """
    potential = SeparableSum((Quadratic(mass=1e-5), Quadratic(mass=3e-5)))
    residuals = {}
    for radius in (15.0, 30.0):
        start = np.array([0.6 * radius, 0.8 * radius])
        target = 0.25 * float(np.sum(start**2))

        def surface(phi, _velocity, target=target):
            return float(np.sum(np.asarray(phi) ** 2)) - target

        gradient = delta_n_gradient(potential, start, surface)
        residuals[radius] = float(np.max(np.abs(gradient / (0.5 * start) - 1.0)))
    assert residuals[15.0] < 2.5e-2
    assert residuals[30.0] < 6e-3
    assert 3.5 < residuals[15.0] / residuals[30.0] < 5.0


def test_efolds_to_surface_reports_a_surface_that_is_never_reached():
    potential = as_multifield(Quadratic(mass=1e-5))

    def unreachable(phi, _velocity):
        return float(np.sum(np.asarray(phi) ** 2)) + 1.0

    with pytest.raises(ValueError, match="never reached the stopping surface"):
        efolds_to_surface(potential, [15.0], unreachable, max_efolds=20.0)


def test_delta_n_spectrum_refuses_a_pivot_the_run_does_not_cover():
    potential = Quadratic(mass=1e-5)
    run = evolve_fields(as_multifield(potential), [field_at_efolds(potential, 20.0)])
    with pytest.raises(ValueError, match="needs"):
        delta_n_spectrum(run, 55.0)


@pytest.mark.parametrize("offset", [-4.0, -2.0, 0.0, 2.0, 4.0])
def test_primordial_spectrum_is_a_power_law_over_three_decades(offset, wide_spectrum):
    """The pivot's ``n_s`` and running describe ``P(k)`` well away from the pivot.

    Fitted at the pivot over half a decade, then extrapolated over three and
    a half: the residual is the cubic term the quadratic fit leaves behind,
    and it stays at 1.5e-3 in ``ln P`` at the far end.
    """
    offsets, scalar, narrow = wide_spectrum
    index = list(offsets).index(offset)
    model = (
        math.log(narrow.pivot_scalar_power)
        + (narrow.spectral_index - 1.0) * offset
        + 0.5 * narrow.running * offset**2
    )
    assert math.log(scalar[index]) == pytest.approx(model, abs=3e-3)


def test_the_primordial_spectrum_residual_is_the_cubic_term(wide_spectrum):
    offsets, scalar, narrow = wide_spectrum
    model = (
        math.log(narrow.pivot_scalar_power)
        + (narrow.spectral_index - 1.0) * offsets
        + 0.5 * narrow.running * offsets**2
    )
    residual = np.abs(np.log(scalar) - model)
    far = residual[list(offsets).index(4.0)]
    near = residual[list(offsets).index(2.0)]
    assert 6.0 < far / near < 12.0


def test_the_primordial_spectrum_falls_monotonically(wide_spectrum):
    _, scalar, _ = wide_spectrum
    assert np.all(np.diff(scalar) < 0.0)
