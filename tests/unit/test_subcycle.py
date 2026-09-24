"""Berger-Oliger subcycling for the spherical solver (issue #111)."""

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

import particlesim.solvers.nr.subcycle as subcycle
from particlesim.core.interpolate import hermite
from particlesim.core.spherical import SphericalGrid
from particlesim.solvers.nr.hierarchy import RATIO, Hierarchy, Level, _interpolate
from particlesim.solvers.nr.polar import PolarSlicingBreakdown
from particlesim.solvers.nr.spherical import ScalarCollapse, gaussian_pulse
from particlesim.solvers.nr.subcycle import (
    ANCHOR_CELLS,
    INTERFACE_CELLS,
    Interface,
    Subcycler,
    mass_flux,
    normalised_metric,
)

R_MAX, AMPLITUDE, CENTRE, WIDTH = 10.0, 3e-3, 3.0, 0.7
COURANT, DURATION = 0.25, 1.0

#: Inside the shell, so matter lies on both sides of the refinement boundary.
#: The boundary placement that decides; see the module docstring.
INSIDE_THE_SHELL = 2.5


def level_at(n: int, r_max: float) -> Level:
    grid = SphericalGrid(r_max=r_max, n=n)
    state = gaussian_pulse(grid, amplitude=AMPLITUDE, r0=CENTRE, width=WIDTH, ingoing=True)
    return Level(grid, state.Phi, state.Pi)


def nested(n: int, boundaries: list[float]) -> Hierarchy:
    """A coarse level of ``n`` cells and one child inside each boundary."""
    levels = [level_at(n, R_MAX)]
    for depth, boundary in enumerate(boundaries, start=1):
        spacing = R_MAX / n / RATIO**depth
        cells = int(round(boundary / spacing)) + INTERFACE_CELLS
        levels.append(level_at(cells, cells * spacing))
    return Hierarchy(levels)


def finest_after(n: int, boundaries: list[float], probe: np.ndarray) -> np.ndarray:
    sub = Subcycler(nested(n, boundaries), courant=COURANT)
    for _ in range(int(round(DURATION / sub.dt))):
        sub.step()
    # A cubic spline, because the probe's own order caps what can be seen:
    # np.interp is second order and reads a fourth-order scheme as broken.
    return CubicSpline(sub.sims[-1].r, sub.states[-1].Phi)(probe)


def uniform_after(n: int, probe: np.ndarray) -> np.ndarray:
    sim = ScalarCollapse(SphericalGrid(r_max=R_MAX, n=n), courant=COURANT)
    state = gaussian_pulse(sim.grid, amplitude=AMPLITUDE, r0=CENTRE, width=WIDTH, ingoing=True)
    for _ in range(int(round(DURATION / sim.dt))):
        state = sim.step(state, sim.dt)
    return CubicSpline(sim.r, state.Phi)(probe)


def distances(boundaries: list[float], resolutions=(100, 200)) -> list[float]:
    """``|hierarchy - uniform|`` at the finest level's spacing, per resolution."""
    probe = np.linspace(0.3, 0.8 * boundaries[-1], 25)
    out = []
    for n in resolutions:
        reference = uniform_after(n * RATIO ** len(boundaries), probe)
        out.append(float(np.max(np.abs(finest_after(n, boundaries, probe) - reference))))
    return out


# --- construction -------------------------------------------------------


def test_a_single_level_subcycler_is_the_ordinary_solver():
    """With nothing to subcycle, the scheme must reduce to the uniform
    solver exactly -- bit for bit, not to a tolerance."""
    grid = SphericalGrid(r_max=R_MAX, n=100)
    state = gaussian_pulse(grid, amplitude=AMPLITUDE, r0=CENTRE, width=WIDTH, ingoing=True)
    sim = ScalarCollapse(grid, courant=COURANT)
    sub = Subcycler(Hierarchy([Level(grid, state.Phi, state.Pi)]), courant=COURANT)
    for _ in range(20):
        state = sim.step(state, sim.dt)
        sub.step()
    np.testing.assert_array_equal(sub.states[0].Phi, state.Phi)
    np.testing.assert_array_equal(sub.states[0].Pi, state.Pi)
    assert sub.t == pytest.approx(state.t)


def test_a_refined_level_needs_room_for_its_interface():
    coarse = level_at(40, R_MAX)
    cells = 2 * INTERFACE_CELLS - 1
    grid = SphericalGrid(r_max=cells * coarse.spacing / RATIO, n=cells)
    tiny = Level(grid, np.zeros(cells), np.zeros(cells))
    with pytest.raises(ValueError, match="driven by its parent"):
        Subcycler(Hierarchy([coarse, tiny]))


def test_hermite_in_time_is_fourth_order_on_the_solvers_own_data():
    """A child's boundary needs its parent's fields at instants the parent
    never lands on. Interpolating between the two endpoints is second order;
    a cubic through both values and both right-hand sides -- which the
    parent has already computed -- is fourth, for nothing."""
    sim = ScalarCollapse(SphericalGrid(r_max=R_MAX, n=400), courant=COURANT)
    start = gaussian_pulse(sim.grid, amplitude=AMPLITUDE, r0=CENTRE, width=WIDTH, ingoing=True)

    def errors(rule):
        out = []
        for factor in (1, 2, 4):
            dt = sim.dt / factor
            end = sim.step(start, dt)
            truth = sim.step(start, 0.5 * dt)
            got = rule(start.Phi, end.Phi, sim.rhs(start)[0], sim.rhs(end)[0], dt, 0.5)
            out.append(float(np.max(np.abs(got - truth.Phi))))
        return out

    def linear(s, e, ds, de, step, f):
        return s + f * (e - s)

    for rule, low, high in ((hermite, 3.8, 4.2), (linear, 1.8, 2.2)):
        e = errors(rule)
        orders = [np.log2(e[i] / e[i + 1]) for i in range(len(e) - 1)]
        assert all(low < o < high for o in orders), f"{rule.__name__}: orders {orders}"


# --- the lapse ----------------------------------------------------------


@pytest.mark.parametrize(("boundary", "own_error"), [(5.0, 1e-4), (INSIDE_THE_SHELL, 5e-2)])
def test_a_child_cannot_normalise_its_own_lapse(boundary, own_error):
    """The mass is local and a child solves for it correctly. The lapse is
    not: its constant is fixed at the asymptotic boundary, and a child's own
    solve fixes it at the child's edge instead. That is only right when the
    edge is in vacuum. With matter outside it the child's lapse is off by a
    constant factor -- about 11% here -- at every resolution, and a child
    running at the wrong rate of coordinate time is worse than no child.

    Measured at ``t = 0`` because nothing more is needed to see it, against
    the parent's lapse at a radius both levels cover.
    """
    radius = np.array([1.0])
    for n in (100, 200):
        coarse = level_at(n, R_MAX)
        spacing = coarse.spacing / RATIO
        cells = int(round(boundary / spacing)) + INTERFACE_CELLS
        sub = Subcycler(Hierarchy([coarse, level_at(cells, cells * spacing)]))
        parent, child = sub.sims
        start = sub.states[0]
        at_rest = Interface(parent, start, start, parent.rhs(start), parent.rhs(start), step=1.0)

        _, parent_lapse = parent.solve_metric(start.Phi, start.Pi)
        _, own = child.solve_metric(sub.states[1].Phi, sub.states[1].Pi)
        _, inherited = normalised_metric(child, sub.states[1], at_rest, 0.0)

        def at(sim, alpha):
            return float(_interpolate(Level(sim.grid, alpha, alpha), radius, alpha, odd=False)[0])

        truth = at(parent, parent_lapse)
        assert abs(at(child, inherited) - truth) < 5e-5
        if boundary == INSIDE_THE_SHELL:
            assert abs(at(child, own) - truth) > own_error, "the defect should be visible here"
        else:
            assert abs(at(child, own) - truth) < own_error, "in vacuum the child's own is fine"


# --- convergence --------------------------------------------------------


@pytest.mark.slow
@pytest.mark.benchmark
def test_fourth_order_survives_subcycling_with_matter_across_the_boundary():
    """The acceptance criterion of issue #111, in time as well as space.

    With the refinement boundary inside the shell the pulse crosses it, which
    is what a collapse run does on its way in and again on its way out. The
    hierarchy is compared with a uniform grid at the child's spacing and the
    difference falls by about sixteen per doubling: nothing is lost through
    the interface. It does not fall *to* the uniform run's accuracy, and is
    not meant to -- the pulse spends part of its trip on the coarse level
    before it enters the child, and refining afterwards cannot recover what
    was lost before.
    """
    near, far = distances([INSIDE_THE_SHELL])
    assert far < near / 12.0, f"ratio {near / far:.1f}, distances {near:.3e}, {far:.3e}"


@pytest.mark.slow
def test_three_levels_subcycle_at_fourth_order():
    """Each level takes its boundary data and its lapse from the one above,
    so a question asked by the finest level about its own instant has to be
    carried all the way to the coarsest, through every intermediate step.
    Two levels would not exercise that chain."""
    near, far = distances([5.0, INSIDE_THE_SHELL])
    assert far < near / 12.0, f"ratio {near / far:.1f}, distances {near:.3e}, {far:.3e}"


@pytest.mark.slow
def test_a_child_normalising_its_own_lapse_does_not_converge_at_all(monkeypatch):
    """The contrast for the fourth-order test, measured rather than asserted.
    With the child's lapse fixed at its own edge the error is a constant --
    6.5e-3 and 6.7e-3 at successive resolutions -- because a wrong rate of
    coordinate time is not a truncation error and refining does nothing to
    it."""
    real = subcycle.normalised_metric
    monkeypatch.setattr(
        subcycle,
        "normalised_metric",
        lambda sim, state, interface, fraction, anchor=None: real(
            sim, state, None, fraction, anchor
        ),
    )
    near, far = distances([INSIDE_THE_SHELL])
    assert far > 1e-3 and near / far < 1.5, f"distances {near:.3e}, {far:.3e}"


@pytest.mark.slow
def test_linear_interpolation_in_time_costs_order(monkeypatch):
    """The other contrast. Linear boundary data in time is the textbook
    choice; here it gives a ratio near seven where Hermite gives sixteen,
    and an error three times larger at the finer resolution. The ratio is
    not a clean four because the interior's fourth-order error is still
    mixed in at these resolutions."""
    hermite_near, hermite_far = distances([INSIDE_THE_SHELL])
    monkeypatch.setattr(subcycle, "hermite", lambda s, e, ds, de, step, f: s + f * (e - s))
    linear_near, linear_far = distances([INSIDE_THE_SHELL])
    assert linear_near / linear_far < 10.0, f"linear ratio {linear_near / linear_far:.1f}"
    assert linear_far > 2.0 * hermite_far, f"linear {linear_far:.3e}, hermite {hermite_far:.3e}"


@pytest.mark.slow
@pytest.mark.benchmark
def test_the_adm_mass_is_conserved_through_a_refinement_boundary():
    """The other half of #111's acceptance criterion for a boundary. The
    pulse is ingoing and nothing leaves the grid, so the ADM mass is fixed
    and any change is error. Read from the composite constraint solve, which
    takes each radius from the finest level that owns it; the drift falls by
    about seventeen per doubling with the matter straddling the boundary."""

    def adm(hierarchy: Hierarchy) -> float:
        a_levels, _ = hierarchy.solve_metric()
        radius = hierarchy.levels[0].radii[hierarchy.owned(0)][-1]
        return 0.5 * radius * (1.0 - 1.0 / a_levels[0][-1] ** 2)

    drifts = []
    for n in (100, 200):
        sub = Subcycler(nested(n, [INSIDE_THE_SHELL]), courant=COURANT)
        start = adm(sub.hierarchy())
        for _ in range(int(round(2.0 * DURATION / sub.dt))):
            sub.step()
        drifts.append(abs(adm(sub.hierarchy()) - start) / start)
    assert drifts[1] < drifts[0] / 12.0, f"drifts {drifts}"
    assert drifts[1] < 1e-4


# --- a parent's covered interior ----------------------------------------


def test_an_anchored_solve_given_the_true_mass_is_the_plain_solve():
    grid = SphericalGrid(r_max=R_MAX, n=400)
    sim = ScalarCollapse(grid)
    state = gaussian_pulse(grid, amplitude=AMPLITUDE, r0=CENTRE, width=WIDTH, ingoing=True)
    a, alpha = sim.solve_metric(state.Phi, state.Pi)
    mass = 0.5 * sim.r * (1.0 - 1.0 / a**2)
    anchored = sim.solve_anchored_metric(state.Phi, state.Pi, 100, float(mass[100]))
    np.testing.assert_allclose(anchored[0], a, rtol=1e-14)
    np.testing.assert_allclose(anchored[1], alpha, rtol=1e-14)


def test_the_mass_flux_is_the_rate_the_mass_changes():
    """``dm/dt = 4 pi r^2 alpha Phi Pi / a^3`` at a fixed radius, against the
    mass the constraint solve returns there, differenced in time."""
    sim = ScalarCollapse(SphericalGrid(r_max=R_MAX, n=400))
    state = gaussian_pulse(sim.grid, amplitude=AMPLITUDE, r0=CENTRE, width=WIDTH, ingoing=True)
    cell = 100
    times, masses, fluxes = [], [], []
    for _ in range(int(2.0 / sim.dt)):
        metric = sim.solve_metric(state.Phi, state.Pi)
        times.append(state.t)
        masses.append(0.5 * sim.r[cell] * (1.0 - 1.0 / metric[0][cell] ** 2))
        fluxes.append(mass_flux(sim, state, metric, cell))
        state = sim.step(state, sim.dt)
    rate = np.gradient(np.array(masses), np.array(times))
    fluxes = np.array(fluxes)
    assert np.abs(rate - fluxes)[2:-2].max() < 2e-3 * np.abs(fluxes).max()


def test_nothing_in_a_parent_s_covered_interior_reaches_outside_it():
    """A parent steps its own copy of the region its child covers, and near
    threshold that copy holds structure far below its spacing. Integrating
    the mass through it once stopped a dispersing run with a ``2m/r >= 1`` at
    the base grid's first cell, and fed a wrong mass into every level's lapse.

    So fill the covered interior with values no solve could accept and step:
    nothing may refuse, and the child -- whose data and lapse both come from
    the parent -- must come out exactly as it does from clean data.
    """
    clean = nested(200, [INSIDE_THE_SHELL])
    parent = clean.levels[0]
    Phi, Pi = np.array(parent.Phi), np.array(parent.Pi)
    anchor = int(round(clean.levels[1].outer / parent.spacing)) - ANCHOR_CELLS
    reach = 12  # one step: four stages through the dissipation's three cells
    Phi[: anchor - reach] = 40.0
    Pi[: anchor - reach] = 40.0
    with pytest.raises(PolarSlicingBreakdown):
        ScalarCollapse(parent.grid).solve_metric(Phi, Pi)
    dirty = Hierarchy([Level(parent.grid, Phi, Pi), clean.levels[1]])

    ends = []
    for hierarchy in (clean, dirty):
        sub = Subcycler(hierarchy, courant=COURANT)
        sub.step()
        ends.append(sub.states[1])
    np.testing.assert_allclose(ends[1].Phi, ends[0].Phi, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(ends[1].Pi, ends[0].Pi, rtol=0.0, atol=1e-13)
