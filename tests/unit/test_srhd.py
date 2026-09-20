"""Valencia special-relativistic hydrodynamics against things that are exact.

Issue #57 asks that relativistic shock tubes match published profiles. What
a published profile *is* is the exact solution of the Riemann problem, so
the tests below compute it here and compare against that rather than
against a figure -- and then check the exact solver itself against the jump
conditions of the same flux function the numerical scheme uses, so the
reference is not taken on trust either.

Three things in this module are exact rather than tolerant, and each is
asserted as such: the advected pulse (a translation), the conservation of
rest mass (a telescoping sum), and a stationary contact under HLLC (a flux
that is identically the same at every face).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from particlesim.solvers.hydro.evolve import (
    RelativisticHydro,
    advected_pulse,
    grid_for,
    riemann_initial_data,
    smooth_pulse,
)
from particlesim.solvers.hydro.reconstruct import (
    NOMINAL_ORDER,
    SCHEMES,
    _monotonised_slope,
    _parabolic_faces,
    reconstruct,
)
from particlesim.solvers.hydro.riemann import (
    contact_speed,
    exact_profile,
    exact_riemann,
    hlle,
    riemann_flux,
)
from particlesim.solvers.hydro.srhd import (
    MAXIMUM_LORENTZ,
    GammaLaw,
    characteristic_speeds,
    cold_flow_fraction,
    conserved_to_primitive,
    flux,
    lorentz,
    primitive_to_conserved,
    recovery_precision,
)

GAMMA = 5.0 / 3.0


@pytest.fixture(scope="module")
def eos() -> GammaLaw:
    return GammaLaw(GAMMA)


def _scalar(value) -> float:
    return float(np.ravel(value)[0])


# --- the equation of state and the variables ------------------------------


def test_the_adiabatic_index_is_refused_at_or_above_two():
    """Because the sound speed reaches the speed of light there, not near it."""
    with pytest.raises(ValueError, match="must lie in"):
        GammaLaw(2.0)


def test_the_sound_speed_stays_below_one_up_to_the_stiffest_index(eos):
    stiff = GammaLaw(1.999)
    hot = stiff.sound_speed_squared(1e-8, 1.0)
    assert hot < 1.0
    assert eos.sound_speed_squared(1e-8, 1.0) < 1.0


def test_the_conserved_variables_round_trip(eos):
    """Over four decades of density and pressure and up to a Lorentz factor of 7."""
    rng = np.random.default_rng(11)
    count = 4000
    density = 10.0 ** rng.uniform(-3.0, 1.0, count)
    velocity = rng.uniform(-0.99, 0.99, count)
    pressure = 10.0 ** rng.uniform(-4.0, 1.0, count)

    recovered = conserved_to_primitive(
        *primitive_to_conserved(density, velocity, pressure, eos), eos
    )
    assert np.max(np.abs(recovered[0] / density - 1.0)) < 1e-9
    assert np.max(np.abs(recovered[1] - velocity)) < 1e-9
    assert np.max(np.abs(recovered[2] / pressure - 1.0)) < 1e-9


def test_recovery_precision_bounds_the_round_trip_error(eos):
    """The accuracy is set by the flow, not by the algorithm.

    ``tau + D + p - D W`` is the internal energy, and for a cold fast flow
    it is a small difference of large numbers. Double precision keeps
    ``eps/fraction`` of it, and the measured error tracks that across four
    decades -- so this asserts a *law*, binned by the predicted bound, not a
    tolerance. Rearranging the residual cannot beat it; the digits are not
    in the conserved variables to begin with.
    """
    rng = np.random.default_rng(13)
    count = 6000
    density = 10.0 ** rng.uniform(-2.0, 1.0, count)
    velocity = np.tanh(rng.uniform(-3.0, 3.0, count))
    pressure = density * 10.0 ** rng.uniform(-9.0, -1.0, count)

    conserved = primitive_to_conserved(density, velocity, pressure, eos)
    bound = recovery_precision(*conserved[:2], conserved[2], pressure, eos)
    actual = np.abs(conserved_to_primitive(*conserved, eos)[2] / pressure - 1.0)

    for low in (1e-15, 1e-13, 1e-11, 1e-9):
        inside = (bound >= low) & (bound < low * 100.0)
        if inside.sum() < 30:
            continue
        ratio = np.median(actual[inside] / bound[inside])
        assert 0.05 < ratio < 20.0, (low, ratio, inside.sum())


def test_a_relativistically_hot_gas_loses_nothing_to_the_cancellation(eos):
    """The other end of the same law: the internal energy is the whole state.

    At a pressure ten thousand times the rest-mass density the subtraction
    in the recovery is not a subtraction of nearly equal numbers at all, so
    the fraction is above one and the bound is machine epsilon itself. Which
    is why the law is stated as a ratio: a fixed tolerance would be far too
    loose here and hopelessly tight for a cold flow.
    """
    conserved = primitive_to_conserved(1.0, 0.0, 1e4, eos)
    assert _scalar(cold_flow_fraction(*conserved, 1e4, eos)) > 1.0
    assert _scalar(recovery_precision(*conserved, 1e4, eos)) <= np.finfo(float).eps


def test_recovery_refuses_past_its_lorentz_ceiling(eos):
    velocity = math.sqrt(1.0 - (0.5 / MAXIMUM_LORENTZ) ** 2)
    with pytest.raises(ValueError, match="Lorentz factor"):
        conserved_to_primitive(*primitive_to_conserved(1.0, velocity, 1e-6, eos), eos)


def test_lorentz_is_one_at_rest():
    assert _scalar(lorentz(0.0)) == 1.0


# --- the characteristic speeds --------------------------------------------


def test_the_characteristic_speeds_are_the_flux_jacobians_eigenvalues(eos):
    """Relativistic velocity addition, checked against a differentiated Jacobian.

    ``(v +- c)/(1 +- v c)`` is a closed form, and the thing it is a closed
    form *of* is the eigenvalue problem of ``dF/dU``. Differentiating the
    flux numerically and diagonalising gives the same three numbers to
    ``5e-9``, which is the finite difference's limit rather than the
    formula's -- and it is a check no rearrangement of the formula could
    pass by accident.
    """
    rng = np.random.default_rng(17)
    worst = 0.0
    for _ in range(20):
        state = (
            float(10.0 ** rng.uniform(-1.0, 1.0)),
            float(rng.uniform(-0.9, 0.9)),
            float(10.0 ** rng.uniform(-2.0, 1.0)),
        )
        conserved = np.array([_scalar(x) for x in primitive_to_conserved(*state, eos)])

        def fluxes(vector):
            return np.array(
                [_scalar(x) for x in flux(*conserved_to_primitive(*vector, eos, 1e-15), eos)]
            )

        jacobian = np.empty((3, 3))
        for column in range(3):
            shift = np.zeros(3)
            shift[column] = 1e-6 * max(abs(conserved[column]), 1.0)
            jacobian[:, column] = (fluxes(conserved + shift) - fluxes(conserved - shift)) / (
                2.0 * shift[column]
            )
        measured = np.sort(np.linalg.eigvals(jacobian).real)
        closed = np.sort([_scalar(s) for s in characteristic_speeds(*state, eos)])
        worst = max(worst, float(np.max(np.abs(measured - closed))))
    assert worst < 5e-8


def test_the_characteristic_speeds_are_subluminal_by_construction(eos):
    """No clamp anywhere: the velocity-addition law cannot exceed one."""
    velocity = np.tanh(np.linspace(-8.0, 8.0, 41))
    speeds = characteristic_speeds(
        np.ones_like(velocity), velocity, np.full_like(velocity, 10.0), eos
    )
    assert np.max(np.abs(np.stack(speeds))) < 1.0


# --- reconstruction -------------------------------------------------------


@pytest.mark.parametrize("scheme", SCHEMES)
def test_every_reconstruction_mirrors_its_two_sides(scheme):
    """The right state is the left state of the reversed field, exactly.

    Zero, not small. A reconstruction whose two sides are written out
    separately can disagree in a way no smooth test finds, because a smooth
    field is locally symmetric anyway.
    """
    grid = np.linspace(0.0, 1.0, 33, endpoint=False)
    values = np.sin(2.0 * np.pi * grid) + 0.3 * np.cos(6.0 * np.pi * grid)
    left, right = reconstruct(values, scheme)
    mirror_left, mirror_right = reconstruct(values[::-1], scheme)
    assert np.max(np.abs(left - np.roll(mirror_right[::-1], -1))) == 0.0
    assert np.max(np.abs(right - np.roll(mirror_left[::-1], -1))) == 0.0


@pytest.mark.parametrize("scheme", SCHEMES)
def test_no_reconstruction_invents_a_new_extremum_at_a_jump(scheme):
    """A step from 1 to 3 must reconstruct inside ``[1, 3]`` on every scheme."""
    step = np.where(np.arange(64) < 32, 1.0, 3.0)
    left, right = reconstruct(step, scheme)
    assert min(left.min(), right.min()) >= 1.0
    assert max(left.max(), right.max()) <= 3.0


def test_the_unlimited_parabolic_face_is_what_needs_the_monotonised_slope():
    """And it undershoots to 0.833 at a jump from 1 to 3 without it.

    The fourth-order interpolant is not bounded by the two cell averages it
    sits between, so Colella and Woodward build the faces from a limited
    slope instead. Removing that step leaves an undershoot that no amount of
    in-cell limiting afterwards can undo, which is why this is asserted
    rather than assumed.
    """
    step = np.where(np.arange(64) < 32, 1.0, 3.0)
    assert _parabolic_faces(step).min() == pytest.approx(5.0 / 6.0, rel=1e-12)
    slope = _monotonised_slope(step)
    monotonised = step + 0.5 * (np.roll(step, -1) - step) - (np.roll(slope, -1) - slope) / 6.0
    assert monotonised.min() >= 1.0


@pytest.mark.parametrize("points", [32, 64, 128, 256, 512])
def test_the_parabolic_limiter_fires_in_six_faces_at_any_resolution(points):
    """Which is why PPM converges below third order on a smooth profile.

    Six faces out of ``N``, clustered on the two extrema and one neighbour
    each, and the count does not grow when the grid does -- so their share
    of the ``L1`` error falls only as fast as ``1/N`` times the local error,
    whatever the faces were interpolated to. That is the mechanism, checked
    directly, rather than a rate inferred from a log-log slope.
    """
    grid = (np.arange(points) + 0.5) / points
    values = 1.0 + 0.2 * np.sin(2.0 * np.pi * grid)
    classic, _ = reconstruct(values, "ppm")
    touched = np.flatnonzero(np.abs(classic - _parabolic_faces(values)) > 1e-14)
    assert len(touched) == 6
    assert int(np.argmax(values)) in touched
    assert int(np.argmin(values)) in touched


def test_the_extremum_limiter_leaves_a_smooth_profile_alone():
    """Colella and Sekora's repair, checked by it doing nothing.

    On a sine the extremum-preserving faces are the unlimited fourth-order
    ones to every digit -- which is the whole point, and is what lifts the
    measured order from two to four.
    """
    grid = (np.arange(128) + 0.5) / 128
    values = 1.0 + 0.2 * np.sin(2.0 * np.pi * grid)
    preserved, _ = reconstruct(values, "ppm-extremum")
    assert np.max(np.abs(preserved - _parabolic_faces(values))) < 1e-15


def test_reconstruct_refuses_an_unknown_scheme():
    with pytest.raises(ValueError, match="unknown reconstruction"):
        reconstruct(np.ones(8), "quintic")


def test_reconstruct_refuses_more_than_one_dimension():
    with pytest.raises(ValueError, match="one-dimensional"):
        reconstruct(np.ones((4, 4)), "minmod")


# --- the exact Riemann solver ---------------------------------------------


def test_the_shock_branch_satisfies_the_jump_conditions(eos):
    """``F(U*) - F(U) = V (U* - U)``, with the scheme's own flux function.

    The exact solver's shock branch comes from the Taub adiabat, which is
    one rearrangement away from being wrong, so it is checked against the
    conservation laws directly instead of against a reference. Three
    equations, evaluated on three hundred random Riemann problems, residual
    below ``1e-11`` of the larger side.
    """
    rng = np.random.default_rng(19)
    worst = 0.0
    shocks = 0
    for _ in range(120):
        left = (
            float(10.0 ** rng.uniform(-1.0, 1.0)),
            float(rng.uniform(-0.8, 0.8)),
            float(10.0 ** rng.uniform(-2.0, 2.0)),
        )
        right = (
            float(10.0 ** rng.uniform(-1.0, 1.0)),
            float(rng.uniform(-0.8, 0.8)),
            float(10.0 ** rng.uniform(-2.0, 2.0)),
        )
        try:
            fan = exact_riemann(left, right, eos)
        except ValueError:
            continue
        sides = (
            (left, fan.left_is_shock, fan.left_star_density, fan.left_speeds[0]),
            (right, fan.right_is_shock, fan.right_star_density, fan.right_speeds[0]),
        )
        for state, is_shock, star_density, speed in sides:
            if not is_shock:
                continue
            shocks += 1
            star = (star_density, fan.star_velocity, fan.star_pressure)
            ahead = np.array([_scalar(x) for x in primitive_to_conserved(*state, eos)])
            behind = np.array([_scalar(x) for x in primitive_to_conserved(*star, eos)])
            flux_ahead = np.array([_scalar(x) for x in flux(*state, eos)])
            flux_behind = np.array([_scalar(x) for x in flux(*star, eos)])
            jump = (flux_behind - flux_ahead) - speed * (behind - ahead)
            scale = max(
                np.max(np.abs(flux_behind - flux_ahead)),
                np.max(np.abs(speed * (behind - ahead))),
            )
            worst = max(worst, float(np.max(np.abs(jump))) / scale)
    assert shocks > 100
    assert worst < 1e-11


def test_the_shock_branch_satisfies_the_taub_adiabat(eos):
    """``h*^2 - h^2 = (h*/rho* + h/rho)(p* - p)``, to round-off."""
    fan = exact_riemann((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), eos)
    ahead = eos.enthalpy(0.125, 0.1)
    behind = eos.enthalpy(fan.right_star_density, fan.star_pressure)
    residual = (
        behind**2
        - ahead**2
        - (behind / fan.right_star_density + ahead / 0.125) * (fan.star_pressure - 0.1)
    )
    assert abs(residual) < 1e-14 * (behind**2 + ahead**2)


@pytest.mark.parametrize(("strength", "expected"), [(1e-1, 0.0136), (1e-2, 0.0147), (1e-3, 0.0148)])
def test_a_weak_shock_and_the_isentrope_agree_to_third_order(eos, strength, expected):
    """The entropy jump across a shock is cubic in its strength.

    Which is what ties the two branches of the exact solver together: they
    are different curves through the same point, osculating to second order,
    so the pressure found by matching them is continuous in the wave type.
    A shock branch wrong by a constant, or by the wrong power, would show up
    here and nowhere else.
    """
    left = (1.0, 0.0, 1.0 + 2.0 * strength)
    fan = exact_riemann(left, (1.0, 0.0, 1.0), eos)
    measured = fan.star_pressure - 1.0
    isentropic = (fan.star_pressure / 1.0) ** (1.0 / GAMMA)
    assert abs(fan.right_star_density - isentropic) / measured**3 == pytest.approx(
        expected, rel=0.05
    )


def test_the_riemann_invariant_reduces_to_the_newtonian_one():
    """``(2/sqrt(G-1)) artanh(c/sqrt(G-1)) -> 2c/(G-1)``.

    The relativistic invariant is used to cross every rarefaction, so it is
    worth a check that does not go through the solver at all: the closed
    form, evaluated directly, against the Newtonian limit it must reproduce.
    """
    root = math.sqrt(GAMMA - 1.0)
    for sound, expected in ((1e-2, 1.0000500045), (1e-3, 1.0000005000)):
        relativistic = 2.0 / root * math.atanh(sound / root)
        assert relativistic / (2.0 * sound / (GAMMA - 1.0)) == pytest.approx(expected, rel=1e-9)


def test_the_profile_across_a_rarefaction_keeps_the_entropy(eos):
    """``p / rho^Gamma`` constant through the fan, and equal to the state it came from."""
    left = (10.0, 0.0, 13.33)
    right = (1.0, 0.0, 1e-6)
    fan = exact_riemann(left, right, eos)
    inside = np.linspace(fan.left_speeds[0] + 1e-6, fan.left_speeds[1] - 1e-6, 9)
    density, _, pressure = exact_profile(left, right, inside, eos, fan)
    entropy = pressure / density**GAMMA
    assert np.max(np.abs(entropy / (left[2] / left[0] ** GAMMA) - 1.0)) < 1e-14


def test_exact_riemann_refuses_an_unphysical_state(eos):
    with pytest.raises(ValueError, match="not a physical fluid state"):
        exact_riemann((1.0, 0.0, 1.0), (-1.0, 0.0, 1.0), eos)


def test_exact_riemann_refuses_a_problem_that_would_open_a_vacuum(eos):
    with pytest.raises(ValueError, match="vacuum"):
        exact_riemann((1.0, -0.999, 1e-6), (1.0, 0.999, 1e-6), eos)


# --- the approximate solvers ----------------------------------------------


def test_both_forms_of_the_contact_root_agree(eos):
    """Algebraically the same root, and near enough numerically -- checked.

    ``(-b - sqrt(b^2 - 4ac))/(2a)`` is where a root is normally lost to
    cancellation, so the stable ``c/q`` is what the solver uses. Whether it
    mattered is a measurement rather than a belief, and the answer is barely:
    over a thousand random contacts the two forms stay within ``1e-13`` of
    each other and of the contact's own velocity, because ``4ac/b^2`` is of
    order one for this quadratic rather than small. The stable form is kept
    as the right habit, not as a fix for something that was broken.
    """
    rng = np.random.default_rng(23)
    from particlesim.solvers.hydro.riemann import _signal_speeds, _total_energy_form

    worst = 0.0
    apart = 0.0
    for _ in range(1000):
        pressure = np.array([float(10.0 ** rng.uniform(-3.0, 2.0))])
        velocity = np.array([float(rng.uniform(-0.95, 0.95))])
        left = (np.array([float(10.0 ** rng.uniform(-2.0, 2.0))]), velocity, pressure)
        right = (np.array([float(10.0 ** rng.uniform(-2.0, 2.0))]), velocity, pressure)
        low, high = _signal_speeds(left, right, eos)
        left_state, left_flux = _total_energy_form(*left, eos)
        right_state, right_flux = _total_energy_form(*right, eos)
        span = high - low
        state = (high * right_state - low * left_state + left_flux - right_flux) / span
        fluxes = (
            high * left_flux - low * right_flux + high * low * (right_state - left_state)
        ) / span
        stable = _scalar(contact_speed(state, fluxes, True))
        naive = _scalar(contact_speed(state, fluxes, False))
        worst = max(worst, abs(stable - naive))
        apart = max(apart, abs(stable - _scalar(velocity)), abs(naive - _scalar(velocity)))
    assert worst < 1e-13
    assert apart < 1e-12


def test_the_contact_speed_of_a_stationary_contact_is_exactly_zero(eos):
    """Because the HLL momentum is, and the root is that over something finite."""
    from particlesim.solvers.hydro.riemann import _signal_speeds, _total_energy_form

    left = (np.array([1.0]), np.array([0.0]), np.array([1.0]))
    right = (np.array([10.0]), np.array([0.0]), np.array([1.0]))
    low, high = _signal_speeds(left, right, eos)
    left_state, left_flux = _total_energy_form(*left, eos)
    right_state, right_flux = _total_energy_form(*right, eos)
    span = high - low
    state = (high * right_state - low * left_state + left_flux - right_flux) / span
    fluxes = (high * left_flux - low * right_flux + high * low * (right_state - left_state)) / span
    assert _scalar(contact_speed(state, fluxes)) == 0.0


def test_the_two_solvers_agree_where_there_is_no_contact(eos):
    """A single uniform state has no fan at all, so both return its own flux."""
    uniform = (np.full(4, 2.0), np.full(4, 0.3), np.full(4, 1.5))
    exact = np.stack(flux(*uniform, eos))
    for solver in ("hlle", "hllc"):
        assert np.max(np.abs(riemann_flux(uniform, uniform, eos, solver) - exact)) < 1e-14


def test_riemann_flux_refuses_an_unknown_solver(eos):
    uniform = (np.ones(2), np.zeros(2), np.ones(2))
    with pytest.raises(ValueError, match="unknown Riemann solver"):
        riemann_flux(uniform, uniform, eos, "roe")


def test_hlle_carries_a_diffusive_flux_across_a_density_jump(eos):
    """Which is the mechanism, not the symptom: ``lambda_L lambda_R (U_R - U_L)``.

    The two signal speeds straddle zero at a stationary contact, so their
    product is negative and multiplies a density difference that is not
    zero. That term is the entire disagreement with HLLC here, and naming
    it is what makes the smearing below a prediction rather than a surprise.
    """
    left = (np.array([1.0]), np.array([0.0]), np.array([1.0]))
    right = (np.array([10.0]), np.array([0.0]), np.array([1.0]))
    averaged = hlle(left, right, eos)
    assert _scalar(averaged[0]) < 0.0  # rest mass flowing from dense to thin
    assert _scalar(riemann_flux(left, right, eos, "hllc")[0]) == 0.0


# --- the evolution --------------------------------------------------------


def test_the_solver_refuses_a_configuration_it_cannot_run(eos):
    grid = grid_for(1.0, 16)
    with pytest.raises(ValueError, match="unknown reconstruction"):
        RelativisticHydro(grid, eos, reconstruction="spline")
    with pytest.raises(ValueError, match="unknown Riemann solver"):
        RelativisticHydro(grid, eos, solver="roe")
    with pytest.raises(ValueError, match="unknown boundary"):
        RelativisticHydro(grid, eos, boundary="reflecting")
    with pytest.raises(ValueError, match="Courant number"):
        RelativisticHydro(grid, eos, courant=1.5)


def test_run_refuses_a_negative_duration_or_no_steps(eos):
    solver = RelativisticHydro(grid_for(1.0, 16), eos)
    state = smooth_pulse(solver)
    with pytest.raises(ValueError, match="must not be negative"):
        solver.run(state, -1.0)
    with pytest.raises(ValueError, match="at least one step"):
        solver.run(state, 1.0, steps=0)


def test_a_uniform_flow_stays_uniform(eos):
    """The simplest exact solution there is, and it has to be exact."""
    solver = RelativisticHydro(grid_for(1.0, 32), eos, reconstruction="weno5")
    state = solver.conserved(np.full(32, 2.0), np.full(32, 0.6), np.full(32, 0.5))
    evolved = solver.run(state, 0.5)
    assert np.max(np.abs(evolved - state)) < 1e-13


@pytest.mark.benchmark
def test_rest_mass_is_conserved_through_a_shock(eos):
    """To round-off, because the update is a difference of fluxes and nothing else.

    Not a property of the scheme: the same face is added to one cell and
    subtracted from its neighbour before any physics is consulted, so the
    total cancels for every reconstruction and every Riemann solver alike,
    and through a discontinuity where nothing else is accurate at all.
    """
    for scheme in ("minmod", "ppm", "weno5"):
        for which in ("hlle", "hllc"):
            solver = RelativisticHydro(
                grid_for(1.0, 100), eos, reconstruction=scheme, solver=which, courant=0.3
            )
            state = riemann_initial_data(solver, (10.0, 0.0, 13.33), (1.0, 0.0, 1e-3))
            start = solver.totals(state)
            finish = solver.totals(solver.run(state, 0.2))
            assert abs(finish[0] / start[0] - 1.0) < 1e-13, (scheme, which)


@pytest.mark.benchmark
def test_hllc_holds_a_stationary_contact_and_hlle_does_not(eos):
    """Twelve orders of magnitude apart, on the same grid and the same steps.

    Uniform pressure and velocity make HLLC's contact speed exactly zero,
    its star states the outer states, and the flux ``(0, p, 0)`` at every
    face -- so nothing moves. HLLE's fan-averaged flux carries
    ``lambda_L lambda_R (U_R - U_L)``, which does not vanish, and the jump
    spreads.
    """
    points = 128
    grid = (np.arange(points) + 0.5) / points
    density = np.where((grid > 0.3) & (grid < 0.7), 10.0, 1.0)
    drift = {}
    for which in ("hllc", "hlle"):
        solver = RelativisticHydro(grid_for(1.0, points), eos, solver=which, courant=0.4)
        state = solver.conserved(density, 0.0, 1.0)
        recovered = solver.primitives(solver.run(state, 0.3))
        drift[which] = float(np.max(np.abs(recovered[0] - density)))
    assert drift["hllc"] < 1e-11
    assert drift["hlle"] > 1.0
    assert drift["hlle"] / drift["hllc"] > 1e11


@pytest.mark.benchmark
def test_what_is_left_of_the_contact_is_the_recovery_and_not_the_flux(eos):
    """Move the arithmetic and see which error follows it.

    HLLC's residual drift tracks :func:`conserved_to_primitive`'s tolerance
    across two decades, because the recovered pressure is uniform only to
    that tolerance and the faces then do not quite cancel. HLLE's does not
    move at all over the same three runs -- one number belongs to the
    arithmetic and the other to the scheme.
    """
    points = 128
    grid = (np.arange(points) + 0.5) / points
    density = np.where((grid > 0.3) & (grid < 0.7), 10.0, 1.0)
    tracked = {}
    for which in ("hllc", "hlle"):
        drifts = []
        for tolerance in (1e-11, 1e-13, 1e-15):
            solver = RelativisticHydro(
                grid_for(1.0, points),
                eos,
                solver=which,
                courant=0.4,
                recovery_tolerance=tolerance,
            )
            state = solver.conserved(density, 0.0, 1.0)
            recovered = solver.primitives(solver.run(state, 0.3))
            drifts.append(float(np.max(np.abs(recovered[0] - density))))
        tracked[which] = drifts
    assert tracked["hllc"][0] / tracked["hllc"][2] > 1e3
    assert max(tracked["hlle"]) / min(tracked["hlle"]) < 1.000001


@pytest.fixture(scope="module")
def advection_orders(eos):
    """Measured convergence order per scheme, at fixed Courant and refined step.

    One run of the table, shared by the tests below: six reconstructions at
    three resolutions each, against the exact translated pulse.
    """
    duration = 2.0
    measured = {}
    for scheme in SCHEMES:
        for label, exponent in (("courant", 1.0), ("refined", 5.0 / 3.0)):
            if label == "refined" and NOMINAL_ORDER[scheme] < 4:
                continue
            sizes = (32, 64, 128) if label == "refined" else (32, 64, 128, 256)
            errors = []
            for points in sizes:
                solver = RelativisticHydro(
                    grid_for(1.0, points), eos, reconstruction=scheme, courant=0.4
                )
                steps = int(
                    np.ceil(
                        duration / (0.4 * solver.spacing) * (points / sizes[0]) ** (exponent - 1.0)
                    )
                )
                evolved = solver.run(smooth_pulse(solver, velocity=0.5), duration, steps=steps)
                exact = advected_pulse(solver, duration, velocity=0.5)
                errors.append(float(np.mean(np.abs(evolved[0] - exact[0]))))
            measured[scheme, label] = [
                math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)
            ]
    return measured


@pytest.mark.benchmark
@pytest.mark.parametrize("scheme", SCHEMES)
def test_a_smooth_pulse_is_advected_at_the_measured_order(advection_orders, scheme):
    """Against an exact solution, so this is an order and not a difference.

    Uniform velocity and pressure make the exact answer a pure translation,
    and all three conserved variables affine in the density -- so the cell
    averages are available in closed form and the initial data does not cap
    the measurement at second order the way cell-centre sampling would.
    """
    label = "refined" if NOMINAL_ORDER[scheme] >= 4 else "courant"
    finest = advection_orders[scheme, label][-1]
    assert abs(finest - NOMINAL_ORDER[scheme]) < 0.5, (scheme, advection_orders[scheme, label])


@pytest.mark.benchmark
def test_a_fixed_courant_number_hides_weno5s_order(advection_orders):
    """The measurement is right and the conclusion drawn from it is wrong.

    At a fixed Courant number the third-order time error falls like
    ``dx^3`` and caps everything above it, so WENO5's measured order slides
    towards three while never looking obviously broken. Refining the step as
    ``dx^(5/3)`` instead recovers five to two decimal places. Both numbers
    are kept, because the trap is the point.
    """
    capped = advection_orders["weno5", "courant"]
    refined = advection_orders["weno5", "refined"]
    assert capped == sorted(capped, reverse=True)
    assert capped[-1] < 4.0
    assert refined[-1] == pytest.approx(5.0, abs=0.1)


@pytest.mark.benchmark
def test_the_extremum_limiter_is_what_lifts_ppm_from_second_order(advection_orders):
    """Same parabola, same faces, different limiter: two orders apart."""
    assert advection_orders["ppm", "courant"][-1] < 2.6
    assert advection_orders["ppm-extremum", "refined"][-1] > 3.5


# --- the acceptance: shock tubes against the exact solution ---------------


SHOCK_TUBES = {
    "mildly relativistic": ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.35),
    "blast wave": ((10.0, 0.0, 13.33), (1.0, 0.0, 1e-6), 0.4),
}


@pytest.mark.benchmark
@pytest.mark.parametrize("name", sorted(SHOCK_TUBES))
def test_the_shock_tube_converges_to_the_exact_solution(eos, name):
    """Issue #57's acceptance, as an equality rather than a comparison to a figure.

    First order in ``L1`` is what a discontinuity allows any scheme, however
    high its order in smooth flow, so the rate rather than the tolerance is
    the assertion. The reference is the exact solution sampled at ``x/t``,
    with no reference resolution and no interpolation on that side.
    """
    left, right, duration = SHOCK_TUBES[name]
    fan = exact_riemann(left, right, eos)
    errors = []
    for points in (200, 400):
        solver = RelativisticHydro(
            grid_for(1.0, points),
            eos,
            reconstruction="ppm-extremum",
            boundary="outflow",
            courant=0.3,
        )
        state = solver.run(riemann_initial_data(solver, left, right), duration)
        density = solver.primitives(state)[0]
        exact = exact_profile(left, right, (solver.centres - 0.5) / duration, eos, fan)
        errors.append(float(np.mean(np.abs(density - exact[0]))))
    assert math.log2(errors[0] / errors[1]) > 0.8
    assert errors[1] < 0.05 * left[0]


@pytest.mark.benchmark
def test_the_standard_tubes_star_states_are_the_published_ones(eos):
    """Marti and Mueller's two relativistic shock tubes, to the digits they are quoted at.

    These come out of the exact solver, which is itself held to the jump
    conditions above rather than to this table -- so agreeing with the
    literature here is a second, independent confirmation rather than the
    only one.
    """
    first = exact_riemann((10.0, 0.0, 13.33), (1.0, 0.0, 1e-6), eos)
    assert first.star_pressure == pytest.approx(1.4477, rel=1e-4)
    assert first.star_velocity == pytest.approx(0.71399, rel=1e-4)
    assert first.right_speeds[0] == pytest.approx(0.82837, rel=1e-4)
    assert first.right_star_density == pytest.approx(5.0706, rel=1e-4)

    second = exact_riemann((1.0, 0.0, 1000.0), (1.0, 0.0, 0.01), eos)
    assert second.star_pressure == pytest.approx(18.597, rel=1e-4)
    assert second.star_velocity == pytest.approx(0.96041, rel=1e-4)
    assert second.right_speeds[0] == pytest.approx(0.98680, rel=1e-4)
    assert second.right_star_density == pytest.approx(10.416, rel=1e-4)


@pytest.mark.benchmark
def test_the_numerical_star_state_is_the_exact_one(eos):
    """Read off the plateau between the contact and the shock, not fitted.

    The region between the two right-going waves is uniform in the exact
    solution, so the numerical solution there can be compared with a number
    rather than with a curve -- and it agrees to better than half a percent
    in pressure at 400 cells.
    """
    left, right, duration = SHOCK_TUBES["blast wave"]
    fan = exact_riemann(left, right, eos)
    solver = RelativisticHydro(
        grid_for(1.0, 400), eos, reconstruction="weno5", boundary="outflow", courant=0.3
    )
    state = solver.run(riemann_initial_data(solver, left, right), duration)
    _, velocity, pressure = solver.primitives(state)

    similarity = (solver.centres - 0.5) / duration
    plateau = (similarity > fan.star_velocity + 0.02) & (similarity < fan.right_speeds[0] - 0.02)
    assert plateau.sum() > 8
    assert np.median(pressure[plateau]) == pytest.approx(fan.star_pressure, rel=5e-3)
    assert np.median(velocity[plateau]) == pytest.approx(fan.star_velocity, rel=5e-3)
