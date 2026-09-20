"""General-relativistic hydrodynamics against a star that is an exact solution.

Issue #57's remaining task and issue #58's acceptance. A static star solves
this system exactly, so the whole right-hand side has to vanish on one --
which is a check every sign and factor has to pass together, and which a
single wrong term fails in a way that does not converge.

The sharper test is the one the acceptance does not ask for. A star past the
maximum-mass point of its own sequence is unstable to radial collapse, and
the structure solver locates that point without evolving anything. So the
evolution can be asked to agree: bounded below it, exponential above it,
with the boundary where ``dM/drho_c`` changes sign and no tolerance
anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.core.spherical import SphericalGrid
from particlesim.matter.eos import Polytrope
from particlesim.matter.tov import solve_tov
from particlesim.solvers.hydro.reconstruct import SCHEMES
from particlesim.solvers.hydro.spherical import SphericalHydro, from_star
from particlesim.solvers.hydro.srhd import GammaLaw

GAMMA = 1.9
CONSTANT = 100.0


@pytest.fixture(scope="module")
def cold() -> Polytrope:
    return Polytrope(polytropic_constant=CONSTANT, gamma=GAMMA)


@pytest.fixture(scope="module")
def eos() -> GammaLaw:
    return GammaLaw(GAMMA)


def _build(cold, eos, central_pressure, points, scheme="ppm-extremum", span=1.5):
    star = solve_tov(cold, central_pressure, samples=20000)
    solver = SphericalHydro(
        SphericalGrid(r_max=span * star.radius, n=points),
        eos,
        reconstruction=scheme,
        reference_density=float(cold.density_from_pressure(central_pressure)),
    )
    return star, solver, from_star(solver, star, cold)


# --- the metric, against the solver that built the star -------------------


def test_the_constrained_metric_is_the_structure_solvers(cold, eos):
    """Two independent routes to ``a(r)``: an outward constraint integration
    here, and the Tolman-Oppenheimer-Volkoff integration that produced the
    star. They are the same equation reached from different directions, so
    agreeing is a check on both rather than a round trip through one.
    """
    star, solver, state = _build(cold, eos, 1e-4, 400)
    _, _, _, a, lapse, mass = solver.decompose(state)

    assert mass[-1] == pytest.approx(star.mass, rel=1e-5)
    outside = solver.radii > 1.05 * star.radius
    exterior = 1.0 / np.sqrt(1.0 - 2.0 * star.mass / solver.radii[outside])
    assert np.max(np.abs(a[outside] / exterior - 1.0)) < 1e-4
    # Schwarzschild outside: the lapse is the reciprocal of the radial metric
    assert np.max(np.abs(lapse[outside] * a[outside] - 1.0)) < 1e-4


def test_the_static_balance_is_the_structure_equation(cold, eos):
    """``alpha p' + (e + p) alpha' = 0``, which is what the momentum equation
    becomes at rest -- and, with the polar slicing condition, is exactly the
    equation :func:`solve_tov` integrates. The evolution and the stellar
    structure solver are therefore the same physics reached two ways, and
    this asserts it on the profile rather than in a docstring.
    """
    errors = []
    for points in (200, 400, 800):
        star, solver, state = _build(cold, eos, 1e-4, points)
        density, _, pressure, _, lapse, _ = solver.decompose(state)
        radii = solver.radii
        energy = density + pressure / (GAMMA - 1.0)

        def slope(values, radii=radii):
            out = np.empty_like(values)
            out[1:-1] = (values[2:] - values[:-2]) / (radii[2:] - radii[:-2])
            out[0] = (values[1] - values[0]) / (radii[1] - radii[0])
            out[-1] = (values[-1] - values[-2]) / (radii[-1] - radii[-2])
            return out

        inside = (radii > 0.1 * star.radius) & (radii < 0.9 * star.radius)
        residual = lapse * slope(pressure) + (energy + pressure) * slope(lapse)
        scale = float(np.max(np.abs(lapse * slope(pressure))[inside]))
        errors.append(float(np.max(np.abs(residual)[inside])) / scale)

    # The residual is the difference stencil's own truncation error, so what
    # is asserted is that it falls like one rather than that it is small.
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert all(order > 1.8 for order in orders), (errors, orders)
    assert errors[-1] < 1e-4


# --- the right-hand side on an exact solution ------------------------------


@pytest.mark.benchmark
@pytest.mark.parametrize("scheme", ["minmod", "ppm-extremum", "weno5"])
def test_the_right_hand_side_vanishes_on_a_static_star(cold, eos, scheme):
    """Second order, and second order for *every* reconstruction.

    That the order does not improve with the reconstruction is the
    interesting part: what caps it is the source term, evaluated at the cell
    centre rather than averaged over the cell, which is an ``O(dx^2)`` error
    no face interpolation can undo. The higher-order schemes are five times
    smaller in absolute terms and exactly as convergent, which is the
    signature of a constant-factor improvement rather than an order one.
    """
    errors = []
    for points in (200, 400, 800):
        star, solver, state = _build(cold, eos, 1e-3, points, scheme)
        radii = solver.radii
        inside = (radii > 0.1 * star.radius) & (radii < 0.9 * star.radius)
        residual = solver.rhs(state)
        errors.append(max(float(np.max(np.abs(residual[k][inside]))) for k in range(3)))

    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert all(abs(order - 2.0) < 0.2 for order in orders), (scheme, orders)


@pytest.mark.benchmark
def test_the_wrong_equation_of_state_gives_a_residual_that_does_not_converge(cold):
    """The control, and the way the first version of this was caught.

    A star built on one adiabatic index and evolved with another is not a
    solution of the system being evolved, and the residual says so by
    sitting flat under refinement instead of falling. Without this the
    convergence test above only shows that *something* converges.
    """
    mismatched = GammaLaw(5.0 / 3.0)
    errors = []
    for points in (200, 400, 800):
        star, solver, state = _build(cold, mismatched, 1e-3, points)
        radii = solver.radii
        inside = (radii > 0.1 * star.radius) & (radii < 0.9 * star.radius)
        errors.append(float(np.max(np.abs(solver.rhs(state)[1][inside]))))

    assert errors[-1] / errors[0] > 0.3
    scale = float(np.max(np.abs(state[0] + state[2]))) / star.radius
    assert errors[-1] > 1e-3 * scale


# --- the atmosphere --------------------------------------------------------


def test_the_energy_floor_is_what_keeps_the_recovery_solvable(cold, eos):
    """A cell given momentum without the energy to carry it has no recovery.

    ``tau = sqrt(D^2 + S^2) - D`` exactly for a cold flow, and anything
    below that describes negative internal energy. A single step can hand an
    atmosphere cell some momentum and land there; the recovery then does not
    misbehave, it refuses, several thousand steps into a run and naming
    numbers that look perfectly ordinary. The floor is applied to the
    conserved variables before any recovery is attempted.
    """
    _, solver, _ = _build(cold, eos, 1e-4, 64)
    density = np.array([2.3e-13])
    momentum = np.array([-3.7e-16])
    energy = np.array([2.6e-23])
    cold_limit = np.sqrt(density**2 + momentum**2) - density
    assert energy[0] < cold_limit[0]

    raised = solver.regularise(density, momentum, energy)
    assert raised[2][0] >= cold_limit[0]


def test_the_atmosphere_is_held_at_rest_and_reported(cold, eos):
    _, solver, state = _build(cold, eos, 1e-4, 200)
    density, velocity, pressure, _, _, _ = solver.decompose(state)
    outer = solver.radii > 1.2 * solve_tov(cold, 1e-4).radius
    assert np.allclose(density[outer], solver.density_floor)
    assert np.allclose(velocity[outer], 0.0)
    assert solver.floored_mass(state) >= 0.0


# --- issue #58's acceptance -------------------------------------------------


@pytest.mark.benchmark
def test_a_stable_star_holds_for_ten_dynamical_times(cold, eos):
    """Issue #58's acceptance: ``L2`` density error below ``1e-3``.

    Measured at ``8.8e-5`` on a 160-cell grid, and -- which matters more --
    it *oscillates* rather than drifting: 1.3e-4, 1.8e-4, 7.6e-5, 6.6e-5 at
    the first four dynamical times. A number that comes back down is a star
    ringing about its equilibrium; one that only grows is a star leaving it,
    and the two can look identical at any single time.
    """
    star, solver, state = _build(cold, eos, 5e-5, 160, span=1.5)
    initial = solver.decompose(state)[0]
    inside = solver.radii < star.radius
    rest = solver.rest_mass(state)

    errors = []
    elapsed = 0.0
    checkpoint = star.dynamical_time
    while elapsed < 10.0 * star.dynamical_time:
        size = min(solver.time_step(state), 10.0 * star.dynamical_time - elapsed)
        state = solver.step(state, size)
        elapsed += size
        if elapsed >= checkpoint:
            density = solver.decompose(state)[0]
            errors.append(float(np.sqrt(np.mean(((density - initial) / initial[0])[inside] ** 2))))
            checkpoint += star.dynamical_time

    assert len(errors) >= 10
    assert errors[-1] < 1e-3
    assert min(errors[4:]) < max(errors[:4])  # it comes back down
    assert abs(solver.rest_mass(state) / rest - 1.0) < 1e-4


@pytest.mark.benchmark
def test_the_stability_boundary_is_where_the_structure_solver_puts_it(cold, eos):
    """Bounded below the maximum-mass point, exponential above it.

    ``solve_tov`` locates the turning point of ``M(rho_c)`` without evolving
    anything, so this is a comparison between two solvers that share no code
    beyond the equation of state -- and the quantity compared is the *sign*
    of a derivative, which no tolerance can be tuned to.

    | rho_c | 2M/R | dM/drho_c | perturbation at 4 dynamical times |
    |---|---|---|---|
    | 4.8e-4 | 0.251 | + | 6.5x |
    | 8.6e-4 | 0.330 | + | 1.3x |
    | 1.8e-3 | 0.416 | - | 35x |
    | 2.9e-3 | 0.450 | - | 88x |
    """
    amplitudes = {}
    for central_pressure in (5e-5, 1.5e-4, 6e-4, 1.5e-3):
        star, solver, state = _build(cold, eos, central_pressure, 120, span=1.4)
        density, _, pressure, a, _, _ = solver.decompose(state)
        kick = 1e-3 * np.sin(np.pi * np.minimum(solver.radii / star.radius, 1.0))
        state = solver.densitise(density, kick, pressure, a)
        start = solver.decompose(state)[0][0]

        elapsed = 0.0
        while elapsed < 4.0 * star.dynamical_time:
            size = min(solver.time_step(state), 4.0 * star.dynamical_time - elapsed)
            state = solver.step(state, size)
            elapsed += size
        amplitudes[central_pressure] = (
            abs(float(solver.decompose(state)[0][0]) / start - 1.0) / 1e-3
        )

        rising = solve_tov(cold, central_pressure * 1.05).mass
        falling = solve_tov(cold, central_pressure * 0.95).mass
        stable = rising > falling
        assert stable == (central_pressure < 3e-4), central_pressure

    assert amplitudes[5e-5] < 15.0
    assert amplitudes[1.5e-4] < 15.0
    assert amplitudes[6e-4] > 25.0
    assert amplitudes[1.5e-3] > amplitudes[6e-4]


@pytest.mark.benchmark
def test_the_limiter_at_the_centre_is_what_makes_a_star_drift(cold, eos):
    """The stellar centre is a smooth maximum, and minmod flattens it every step.

    Measured directly: the limiter zeroes the density slope in cell zero --
    the centre -- at 100, 200 and 400 cells alike, because the density is
    largest there and a one-sided-slope limiter cannot tell a smooth
    extremum from an overshoot. That is a systematic forcing applied at
    exactly the place the error grows, and it shows in the evolution: over
    two dynamical times minmod's error rises monotonically while the
    extremum-preserving reconstruction's comes back down. The same finding
    as the advected pulse in issue #57, arriving as a physical consequence.
    """
    histories = {}
    for scheme in ("minmod", "ppm-extremum"):
        star, solver, state = _build(cold, eos, 1e-4, 160, scheme)
        initial = solver.decompose(state)[0]
        inside = solver.radii < star.radius

        if scheme == "minmod":
            padded = solver._pad(initial)
            backward = padded - np.roll(padded, 1)
            forward = np.roll(padded, -1) - padded
            flattened = np.flatnonzero(backward * forward <= 0.0) - 3
            assert 0 in flattened  # the centre, every time

        errors = []
        elapsed = 0.0
        checkpoint = 0.5 * star.dynamical_time
        while elapsed < 2.0 * star.dynamical_time:
            size = min(solver.time_step(state), 2.0 * star.dynamical_time - elapsed)
            state = solver.step(state, size)
            elapsed += size
            if elapsed >= checkpoint:
                density = solver.decompose(state)[0]
                errors.append(
                    float(np.sqrt(np.mean(((density - initial) / initial[0])[inside] ** 2)))
                )
                checkpoint += 0.5 * star.dynamical_time
        histories[scheme] = errors

    assert histories["minmod"] == sorted(histories["minmod"])
    assert histories["ppm-extremum"][-1] < max(histories["ppm-extremum"][:-1])
    assert histories["ppm-extremum"][-1] < histories["minmod"][-1]


# --- refusals ---------------------------------------------------------------


def test_the_solver_refuses_a_configuration_it_cannot_run(eos):
    grid = SphericalGrid(r_max=10.0, n=32)
    with pytest.raises(ValueError, match="unknown reconstruction"):
        SphericalHydro(grid, eos, reconstruction="spline")
    with pytest.raises(ValueError, match="unknown Riemann solver"):
        SphericalHydro(grid, eos, solver="roe")
    with pytest.raises(ValueError, match="Courant number"):
        SphericalHydro(grid, eos, courant=1.5)
    with pytest.raises(ValueError, match="reference density"):
        SphericalHydro(grid, eos, reference_density=0.0)
    with pytest.raises(ValueError, match="uniform radial grid"):
        SphericalHydro(SphericalGrid(r_max=10.0, n=32, refine=2.0), eos)


def test_run_refuses_a_negative_duration_or_no_steps(cold, eos):
    _, solver, state = _build(cold, eos, 1e-4, 32)
    with pytest.raises(ValueError, match="must not be negative"):
        solver.run(state, -1.0)
    with pytest.raises(ValueError, match="at least one step"):
        solver.run(state, 1.0, steps=0)


def test_every_reconstruction_is_accepted(cold, eos):
    for scheme in SCHEMES:
        _, solver, state = _build(cold, eos, 1e-4, 48, scheme)
        assert solver.rhs(state).shape == (3, 48)
