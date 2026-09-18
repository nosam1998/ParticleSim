"""The coupled field-space mode system, against the two things that can check it.

Issue #124's acceptance is deliberately two statements, because either alone
can pass while the physics is wrong.

With one field the matrix solver and
:func:`particlesim.cosmo.perturbations.mode_power` solve *the same equation
in different variables* -- flat-gauge ``dphi`` against the curvature
perturbation ``R`` -- so agreement there is an identity and is held to the
exact power-law result rather than to each other.

With two fields there is no exact answer, so the check is against ``delta N``
at a pivot late enough for the trajectory to have become adiabatic, where
``delta N`` is right to ``O(epsilon)``. That alone would still pass for a
solver that had dropped the field-space coupling entirely, since an
adiabatic trajectory has nothing left to couple -- hence the second half,
which is that the entropic power is large at an earlier pivot.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.cosmo.inflation import evolve_fields, evolve_inflation
from particlesim.cosmo.multifield import (
    background_at,
    crossing_efolds,
    mass_matrix,
    mode_matrix,
    multifield_spectrum,
    pivot_wavenumber,
    trajectory_basis,
)
from particlesim.cosmo.perturbations import (
    delta_n_spectrum,
    end_of_inflation_surface,
    mode_power,
    power_spectrum,
)
from particlesim.cosmo.potentials import (
    Exponential,
    Quadratic,
    SeparableSum,
    as_multifield,
)

EXPONENTIAL_RATE = 0.4


@pytest.fixture(scope="module")
def exponential_run():
    """One field, wrapped as multi-field: the case with an exact answer."""
    return evolve_fields(as_multifield(Exponential(rate=EXPONENTIAL_RATE)), [0.0], max_efolds=40.0)


@pytest.fixture(scope="module")
def quadratic_run():
    """One quadratic field: the identity test needs a non-zero mass matrix."""
    return evolve_fields(as_multifield(Quadratic(mass=1.0)), [16.0], max_efolds=200.0)


@pytest.fixture(scope="module")
def two_field_run():
    """Quadratic fields of unequal mass, run to the end of inflation.

    The mass ratio of seven is what makes the trajectory turn: the heavy
    field rolls away first and the light one is left, so the adiabatic
    direction rotates from ``(-0.02, -1.00)`` at the start to ``(-1.00, 0)``
    by the end. A curved trajectory in field space is exactly the condition
    for the entropic mode to source the curvature perturbation.
    """
    potential = SeparableSum((Quadratic(mass=1.0), Quadratic(mass=7.0)))
    return evolve_fields(potential, [12.0, 12.0], max_efolds=200.0, stop=end_of_inflation_surface)


# --- the mass matrix -----------------------------------------------------


def test_the_mass_matrix_is_symmetric(two_field_run):
    """``V_ij`` is a Hessian and the back-reaction is an outer product plus its
    transpose, so neither piece can break the symmetry."""
    for number in (5.0, 20.0, 50.0):
        matrix = mass_matrix(two_field_run, number)
        assert matrix.shape == (2, 2)
        assert np.allclose(matrix, matrix.T, rtol=0, atol=1e-12)


def test_the_back_reaction_is_not_a_small_correction(two_field_run):
    """Dropping it is not a first version, it is a different equation.

    Checked by comparing the full mass matrix with the Hessian alone: the
    two differ by an amount comparable to the Hessian itself, not by a
    slow-roll correction to it.
    """
    number = 20.0
    phi, _, _, _, hubble_squared = background_at(two_field_run, number)
    hessian = np.asarray(two_field_run.potential.hessian(phi)) / hubble_squared
    full = mass_matrix(two_field_run, number)
    difference = np.max(np.abs(full - hessian))
    assert difference > 0.05 * np.max(np.abs(hessian)), (difference, hessian)


def test_the_single_field_mass_matrix_matches_the_curvature_equation(quadratic_run):
    """The identity that makes the one-field limit exact, checked numerically.

    Substituting ``dphi = phi' R`` into this module's equation returns the
    curvature equation iff

        M/H^2 = -phi'''/phi' - (3 - epsilon) phi''/phi'

    which is derived in the module docstring from ``epsilon' = phi' phi''``
    and the background equation. ``phi'''`` is not carried anywhere, so it is
    obtained here by differencing ``phi''`` -- which makes this a check on
    the *back-reaction term*, since the Hessian alone does not satisfy it.

    A quadratic potential, not the exponential one: see
    :func:`test_the_exponential_potential_has_no_effective_mass` for why
    that case would compare zero against zero.
    """
    step = 1e-4
    for number in (10.0, 20.0, 30.0):
        _, velocity, acceleration, epsilon, _ = background_at(quadratic_run, number)
        ahead = background_at(quadratic_run, number + step)[2]
        behind = background_at(quadratic_run, number - step)[2]
        jerk = (ahead - behind) / (2.0 * step)

        expected = -jerk / velocity - (3.0 - epsilon) * acceleration / velocity
        measured = mass_matrix(quadratic_run, number)[0, 0]
        assert abs(float(expected[0])) > 1e-3, "this must not be a zero-against-zero check"
        assert measured == pytest.approx(float(expected[0]), rel=1e-5)


def test_the_exponential_potential_has_no_effective_mass(exponential_run):
    """Why power-law inflation comes out exact, in one number.

    For ``V ~ exp(-lambda phi)`` the Hessian is ``lambda^2 V`` and the
    gravitational back-reaction cancels it *identically*: the effective mass
    matrix is zero to round-off, 5.6e-17 and 1.1e-16 at ``N = 10`` and 20.
    So the field perturbation obeys the same equation as a massless field --
    which is the same statement as ``z = a sqrt(2 epsilon)`` being
    proportional to ``a`` when ``epsilon`` is constant, and is why the
    scalar and tensor modes there share a solution and ``r = 16 epsilon``
    holds exactly.

    It also means the exponential potential cannot test the back-reaction
    term at all, since both sides of the identity above vanish.
    """
    for number in (5.0, 10.0, 20.0, 30.0):
        assert abs(float(mass_matrix(exponential_run, number)[0, 0])) < 1e-14


def test_the_trajectory_basis_is_orthonormal(two_field_run):
    for number in (2.0, 30.0, 60.0):
        _, velocity, _, _, _ = background_at(two_field_run, number)
        along, entropic = trajectory_basis(velocity)
        assert along.shape == (2,)
        assert entropic.shape == (2, 1)
        assert float(np.linalg.norm(along)) == pytest.approx(1.0, abs=1e-12)
        assert np.allclose(entropic.T @ entropic, np.eye(1), atol=1e-12)
        assert np.allclose(entropic.T @ along, 0.0, atol=1e-12)
        # and it points along the trajectory, not merely somewhere
        assert np.allclose(along, velocity / np.linalg.norm(velocity), atol=1e-12)


def test_a_trajectory_at_rest_has_no_adiabatic_direction():
    with pytest.raises(ValueError, match="come to rest"):
        trajectory_basis(np.zeros(2))


# --- acceptance 1: the single-field limit is an identity ------------------


@pytest.mark.benchmark
def test_one_field_reproduces_the_exact_power_law_index(exponential_run):
    """Issue #124's first acceptance criterion, met at 8e-11 against 1e-7.

    ``n_s - 1 = -2 epsilon/(1 - epsilon)`` is exact for a constant-``epsilon``
    background. This solver evolves flat-gauge ``dphi`` and projects, where
    :func:`mode_power` evolves ``R`` directly; they are the same equation
    after ``dphi = phi' R``, so this is an identity and not a tolerance.

    Measured: ``n_s = 0.8260869566015`` against an exact
    ``0.8260869565217``, a difference of 8.0e-11.
    """
    epsilon = 0.5 * EXPONENTIAL_RATE**2
    exact = 1.0 - 2.0 * epsilon / (1.0 - epsilon)
    spectrum = multifield_spectrum(exponential_run, efolds_remaining=20.0)

    assert spectrum.spectral_index == pytest.approx(exact, abs=1e-7)
    assert abs(spectrum.spectral_index - exact) < 1e-9, spectrum.spectral_index
    # Not the first-order formula, which is four per cent of the tilt away.
    assert abs((1.0 - 2.0 * epsilon) - exact) > 1e-2


@pytest.mark.benchmark
def test_one_field_reproduces_the_single_field_amplitude(exponential_run):
    """Not only the tilt: the amplitude too, against the ``R`` solver.

    The two normalisations are written independently -- ``1/(2 k a^2)`` for a
    field perturbation here, ``1/(4 a^2 epsilon k)`` for the curvature
    perturbation there -- so agreeing on the amplitude checks the projection
    ``R = phi' dphi/(2 epsilon)`` and both normalisations at once.
    """
    scalar_run = evolve_inflation(Exponential(rate=EXPONENTIAL_RATE), 0.0, max_efolds=40.0)
    reference = power_spectrum(scalar_run, efolds_remaining=20.0)
    matrix = multifield_spectrum(exponential_run, efolds_remaining=20.0)

    assert matrix.pivot == pytest.approx(reference.pivot, rel=1e-10)
    assert matrix.pivot_curvature_power == pytest.approx(reference.pivot_scalar_power, rel=1e-6)


def test_one_field_has_no_entropic_direction(exponential_run):
    """Zero exactly, not numerically: there is nothing orthogonal to evolve."""
    result = mode_matrix(exponential_run, pivot_wavenumber(exponential_run, 20.0))
    assert result.fields == 1
    assert result.entropy_power == 0.0
    assert result.cross_power == 0.0
    assert result.correlation == 0.0


def test_one_field_agrees_mode_by_mode_with_the_curvature_solver(exponential_run):
    """The identity at the level of a single ``k``, before any fit."""
    scalar_run = evolve_inflation(Exponential(rate=EXPONENTIAL_RATE), 0.0, max_efolds=40.0)
    wavenumber = pivot_wavenumber(exponential_run, 20.0)
    assert crossing_efolds(exponential_run, wavenumber) == pytest.approx(
        scalar_run.total_efolds - 20.0, abs=1e-6
    )
    matrix = mode_matrix(exponential_run, wavenumber)
    reference = mode_power(scalar_run, wavenumber)
    assert matrix.curvature_power == pytest.approx(reference.scalar_power, rel=1e-6)


# --- acceptance 2: two fields, against delta N ----------------------------


@pytest.mark.slow
@pytest.mark.benchmark
def test_two_fields_agree_with_delta_n_once_the_trajectory_is_adiabatic(two_field_run):
    """Issue #124's second acceptance criterion, both halves.

    ``delta N`` is a super-horizon slow-roll statement, right to
    ``O(epsilon)``. At a pivot where the entropic mode has decayed, the mode
    matrix has to land on it; measured, with the mode read as late in the run
    as it reaches:

        e-folds left   P_R (modes)   P_R (delta N)   ratio    P_S/P_R
             60          5.10e+02       1.18e+03     0.431    2.40e-01
             50          5.66e+02       5.95e+02     0.952    1.96e-02
             40          1.74e+02       1.79e+02     0.972    2.02e-16
             30          1.53e+01       1.50e+01     1.015    2.22e-27

    The last two rows are the agreement, at 2.8% and 1.5% against
    ``epsilon`` of 0.078 and 0.017 -- so the discrepancy is a fraction of
    ``epsilon``, which is what ``O(epsilon)`` means here.

    The first row is the other half of the criterion. At sixty e-folds
    remaining the entropic power is a quarter of the curvature power and the
    two methods differ by 57%, because ``R`` is still being sourced between
    that reading and the end of inflation. A solver that had dropped the
    field-space coupling would agree everywhere and fail this.
    """
    run = two_field_run

    def compare(remaining: float):
        reference = delta_n_spectrum(
            run, efolds_remaining=remaining, surface=end_of_inflation_surface
        )
        wavenumber = pivot_wavenumber(run, remaining)
        crossing = crossing_efolds(run, wavenumber)
        freeze = min(20.0, run.total_efolds - crossing - 0.5)
        result = mode_matrix(run, wavenumber, freeze_efolds=freeze)
        return result, reference

    # Adiabatic: agreement to a fraction of epsilon, and no entropy left.
    for remaining in (40.0, 30.0):
        result, reference = compare(remaining)
        discrepancy = abs(result.curvature_power / reference.scalar_power - 1.0)
        assert discrepancy < reference.epsilon, (remaining, discrepancy, reference.epsilon)
        assert result.entropy_power / result.curvature_power < 1e-10

    # Not yet adiabatic: a live entropic mode, and the two methods part.
    result, reference = compare(60.0)
    assert result.entropy_power / result.curvature_power > 0.1
    assert abs(result.curvature_power / reference.scalar_power - 1.0) > 0.3


@pytest.mark.slow
def test_the_entropic_mode_decays_along_the_trajectory(two_field_run):
    """The turn, seen in the spectra rather than in the background.

    The heavy field rolls away first, so the adiabatic direction rotates and
    the entropic power falls by more than twenty orders of magnitude between
    sixty and thirty e-folds before the end. That monotone decay is the
    physical content of "the trajectory becomes adiabatic".
    """
    run = two_field_run
    fractions = []
    for remaining in (60.0, 50.0, 40.0):
        wavenumber = pivot_wavenumber(run, remaining)
        crossing = crossing_efolds(run, wavenumber)
        freeze = min(20.0, run.total_efolds - crossing - 0.5)
        result = mode_matrix(run, wavenumber, freeze_efolds=freeze)
        fractions.append(result.entropy_power / result.curvature_power)
    assert fractions == sorted(fractions, reverse=True), fractions
    assert fractions[0] > 0.1
    assert fractions[-1] < 1e-10


def test_the_mode_matrix_starts_at_the_identity(two_field_run):
    """Each field in its own vacuum, so the matrix is diagonal before coupling.

    Off-diagonal entries later are the coupling doing its work; if the
    initial data were not independent, the spectra would be sums over
    correlated solutions and would come out wrong by a factor that looks
    like a normalisation error.
    """
    from particlesim.cosmo.multifield import log_comoving_hubble

    wavenumber = pivot_wavenumber(two_field_run, 40.0)
    crossing = crossing_efolds(two_field_run, wavenumber)
    # k/aH at crossing is one, by definition of the crossing.
    assert float(np.exp(np.log(wavenumber) - log_comoving_hubble(two_field_run, crossing))) == (
        pytest.approx(1.0, rel=1e-9)
    )


# --- refusals ------------------------------------------------------------


def test_a_mode_that_never_crosses_is_refused(two_field_run):
    # ln(aH) reaches 72.5 on this run, so k must exceed exp(72.5) = 2.9e31.
    with pytest.raises(ValueError, match="never crosses the horizon"):
        crossing_efolds(two_field_run, 1e35)


def test_a_mode_already_outside_at_the_start_is_refused(two_field_run):
    with pytest.raises(ValueError, match="already outside the horizon"):
        crossing_efolds(two_field_run, 1e-30)


def test_a_pivot_the_run_does_not_reach_is_refused(two_field_run):
    with pytest.raises(ValueError, match="does not reach back"):
        pivot_wavenumber(two_field_run, 500.0)


def test_a_quadratic_fit_needs_three_points(exponential_run):
    with pytest.raises(ValueError, match="at least three points"):
        multifield_spectrum(exponential_run, efolds_remaining=20.0, points=2)
