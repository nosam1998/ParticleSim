"""Regridding the spherical hierarchy, and the adaptive solver built on it (#111)."""

import functools

import numpy as np
import pytest

from particlesim.analysis.critical_collapse import evolve_to_verdict
from particlesim.analysis.spherical_diagnostics import ricci_scalar
from particlesim.core.spherical import SphericalGrid
from particlesim.solvers.nr.adaptive import AdaptiveCollapse
from particlesim.solvers.nr.hierarchy import RATIO, Hierarchy, Level
from particlesim.solvers.nr.spherical import ScalarCollapse, gaussian_pulse
from particlesim.solvers.nr.subcycle import INTERFACE_CELLS, Subcycler

# The thin shell of test_critical_collapse.py, 5.6% below its threshold: it
# bounces through the origin with a curvature near 148 and disperses.
R_MAX, R0, WIDTH, SUBCRITICAL = 10.0, 4.0, 0.5, 8.0e-4

#: The same bounce on uniform grids at the default dissipation, peak ``|R|``
#: inside ``r < 1.5``: 153.2 at n = 400, 147.46 at 800, 147.81 at 1600.
UNIFORM_3200_PEAK = 147.91


def shell(sim, amplitude: float = SUBCRITICAL):
    return gaussian_pulse(sim.grid, amplitude=amplitude, r0=R0, width=WIDTH, ingoing=True)


def nested(n: int, boundary: float) -> Hierarchy:
    """A level of ``n`` cells over ``[0, 10]`` and a child inside ``boundary``,
    with the ingoing pulse of ``test_subcycle.py`` straddling the boundary."""

    def level(cells: int, r_max: float) -> Level:
        grid = SphericalGrid(r_max=r_max, n=cells)
        state = gaussian_pulse(grid, amplitude=3e-3, r0=3.0, width=0.7, ingoing=True)
        return Level(grid, state.Phi, state.Pi)

    spacing = R_MAX / n / RATIO
    cells = int(round(boundary / spacing)) + INTERFACE_CELLS
    return Hierarchy([level(n, R_MAX), level(cells, cells * spacing)])


def near_the_origin(sim):
    """Weak, but with something inside every level the tests force."""
    return gaussian_pulse(sim.grid, amplitude=1e-3, r0=1.0, width=0.5)


def bounce(sim, t_end: float):
    """Peak ``|R|`` inside ``r < 1.5`` and the depth after every step."""
    state, peak, depths = shell(sim), 0.0, []
    for _ in range(int(round(t_end / sim.dt))):
        state = sim.step(state, sim.dt)
        a, _ = sim.solve_metric(state.Phi, state.Pi)
        inner = sim.r < 1.5
        peak = max(peak, float(np.abs(ricci_scalar(a, state.Phi, state.Pi))[inner].max()))
        depths.append(getattr(sim, "depth", 1))
    return peak, depths


# --- the criterion ------------------------------------------------------


def test_cells_across_counts_cells_per_curvature_radius():
    """``1 / sqrt(8 pi rho)`` with ``rho = (Phi^2 + Pi^2) / (2 a^2)``, in cells
    of proper length ``a dr``: the metric cancels."""
    grid = SphericalGrid(r_max=1.0, n=100)
    Phi, Pi = np.full(100, 0.3), np.full(100, 0.4)
    sub = Subcycler(Hierarchy([Level(grid, Phi, Pi)]), cells_per_radius=16.0)
    assert sub.cells_across(0) == pytest.approx(1.0 / (0.01 * np.sqrt(4.0 * np.pi * 0.25)))
    Phi[60:] = 3.0  # only counted when inside the extent asked about
    sub = Subcycler(Hierarchy([Level(grid, Phi, Pi)]), cells_per_radius=16.0)
    assert sub.cells_across(0, extent=0.5) == pytest.approx(56.4189, rel=1e-5)
    assert sub.cells_across(0) < 11.0


def test_an_under_resolved_level_gets_a_child_over_its_inner_part():
    grid = SphericalGrid(r_max=R_MAX, n=400)
    state = near_the_origin(ScalarCollapse(grid))
    level = Level(grid, state.Phi, state.Pi)
    sub = Subcycler(Hierarchy([level]), cells_per_radius=1e6, level_cells=160, max_depth=3)
    sub.step()
    assert sub.depth == 2
    child = sub.sims[1]
    assert child.dr == pytest.approx(grid.dr / RATIO)
    assert child.grid.r_max == pytest.approx(160 * grid.dr / RATIO)
    sub.step()
    assert sub.depth == 3, "and so on, one level per check"
    sub.step()
    assert sub.depth == 3, "never past max_depth"


def test_level_cells_must_nest():
    grid = SphericalGrid(r_max=R_MAX, n=100)
    level = Level(grid, np.zeros(100), np.zeros(100))
    for bad in (161, 200, 4):
        with pytest.raises(ValueError, match="level_cells"):
            Subcycler(Hierarchy([level]), cells_per_radius=16.0, level_cells=bad)


@functools.cache
def bounce_run(trigger: str, seed_extent: float | None):
    """The strong bounce to ``t = 9``, once per configuration."""
    kw = (
        {"cells_per_radius": 16.0}
        if trigger == "curvature"
        else {
            "cells_per_radius": None,
            "tolerance": 1e-6,
        }
    )
    sim = AdaptiveCollapse(SphericalGrid(r_max=R_MAX, n=400), seed_extent=seed_extent, **kw)
    peak, depths = bounce(sim, t_end=9.0)
    return peak, depths, sim.subcycler.seed_depth, list(sim.subcycler.regrids)


CONFIGURATIONS = [("curvature", None), ("curvature", 2.0), ("richardson", 2.0)]


@pytest.mark.slow
@pytest.mark.parametrize(("trigger", "seed_extent"), CONFIGURATIONS)
def test_levels_follow_a_bounce_in_and_leave_after_it(trigger, seed_extent):
    """Created while the shell focuses, retired once it has dispersed, and
    each only once: nothing thrashes at the origin afterwards."""
    _, depths, seed_depth, events = bounce_run(trigger, seed_extent)
    deepest = max(depths)
    assert deepest >= seed_depth + 1
    assert depths[-1] == seed_depth
    assert len(events) == 2 * (deepest - seed_depth), events


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.parametrize(("trigger", "seed_extent"), CONFIGURATIONS)
def test_a_refined_bounce_approaches_a_uniform_grid_eight_times_finer(trigger, seed_extent):
    """The base grid is n = 400, where the uniform solver overshoots the peak
    by 3.6%. Every configuration lands within 1e-3 of n = 3200: 7.5e-4 with
    the curvature trigger alone, 7e-6 with a level seeded over the shell's
    trip in, and 4.9e-4 with the Richardson estimate."""
    peak, _, _, _ = bounce_run(trigger, seed_extent)
    assert peak == pytest.approx(UNIFORM_3200_PEAK, rel=1e-3)


@pytest.mark.slow
def test_the_richardson_estimate_refines_the_infall_and_the_curvature_does_not():
    """The difference between the two triggers. The shell is weak until it
    is nearly at the origin, so its curvature radius is long all the way in;
    its error is not, and the estimate adds a level more than a unit of time
    before the curvature does."""
    first = {t: bounce_run(t, 2.0)[3][0][0] for t in ("richardson", "curvature")}
    assert first["richardson"] < 4.0 < first["curvature"], first


def test_the_richardson_estimate_is_one_steps_local_error():
    """Fifteen times the child's error per parent step, which at fourth order
    is fifth order in the spacing: a factor of 32 per doubling, measured
    away from the interface, whose own error the cells next to it carry."""
    maxima = []
    for n in (100, 200, 400):
        sub = Subcycler(nested(n, 2.5), tolerance=1e9)
        while sub.t < 0.5 - 1e-12:
            sub.step()
        radii, error = sub.errors[1]
        maxima.append(float(error[radii < 2.0].max()))
    ratios = [maxima[i] / maxima[i + 1] for i in range(2)]
    assert all(28.0 < q < 36.0 for q in ratios), ratios


def test_a_fresh_level_s_first_estimate_is_discarded():
    """Prolonging then restricting is not the identity, so a new level's first
    estimate would measure its own construction."""
    grid = SphericalGrid(r_max=R_MAX, n=200)
    state = near_the_origin(ScalarCollapse(grid))
    sub = Subcycler(
        Hierarchy([Level(grid, state.Phi, state.Pi)]),
        tolerance=1e9,
        cells_per_radius=1e6,
        level_cells=100,
        max_depth=2,
    )
    sub.step()
    assert sub.depth == 2 and 1 not in sub.errors
    sub.step()
    assert 1 not in sub.errors, "the first estimate is dropped"
    sub.step()
    assert 1 in sub.errors


def test_a_richardson_estimate_needs_two_levels_to_start():
    grid = SphericalGrid(r_max=R_MAX, n=100)
    level = Level(grid, np.zeros(100), np.zeros(100))
    with pytest.raises(ValueError, match="compares a level with its parent"):
        Subcycler(Hierarchy([level]), tolerance=1e-6)
    Subcycler(Hierarchy([level]), tolerance=1e-6, cells_per_radius=16.0)


# --- the adapter --------------------------------------------------------


def test_with_nothing_to_refine_the_adaptive_solver_is_the_uniform_one():
    grid = SphericalGrid(r_max=R_MAX, n=200)
    uniform, adaptive = ScalarCollapse(grid), AdaptiveCollapse(grid)
    a, b = shell(uniform, 1e-4), shell(adaptive, 1e-4)
    for _ in range(20):
        a, b = uniform.step(a, uniform.dt), adaptive.step(b, adaptive.dt)
    assert adaptive.depth == 1
    np.testing.assert_array_equal(a.Phi, b.Phi)
    np.testing.assert_array_equal(a.Pi, b.Pi)


def test_the_first_state_must_be_on_the_base_grid():
    sim = AdaptiveCollapse(SphericalGrid(r_max=R_MAX, n=200))
    elsewhere = shell(ScalarCollapse(SphericalGrid(r_max=R_MAX, n=100)))
    with pytest.raises(ValueError, match="base grid"):
        sim.step(elsewhere, sim.dt)


def test_an_adaptive_run_is_stepped_from_its_own_state():
    """It holds the hierarchy; a state handed back from anywhere else would be
    silently ignored, so it is refused instead."""
    sim = AdaptiveCollapse(SphericalGrid(r_max=R_MAX, n=200))
    start = shell(sim)
    after = sim.step(start, sim.dt)
    with pytest.raises(ValueError, match="last returned"):
        sim.step(start, sim.dt)
    sim.step(after, sim.dt)


def test_its_metric_is_the_hierarchys():
    """Normalised once at the outer boundary, not at each level's edge."""
    grid = SphericalGrid(r_max=R_MAX, n=400)
    sim = AdaptiveCollapse(grid, cells_per_radius=1e6, max_depth=3)
    state = near_the_origin(sim)
    for _ in range(3):
        state = sim.step(state, sim.dt)
    assert sim.depth == 3 and len(sim.r) == len(state.Phi)
    a, alpha = sim.solve_metric(state.Phi, state.Pi)
    want_a, want_alpha = sim.subcycler.hierarchy().composite_metric()
    np.testing.assert_allclose(a, want_a, rtol=1e-13)
    np.testing.assert_allclose(alpha, want_alpha, rtol=1e-13)
    with pytest.raises(ValueError, match="last step returned"):
        sim.solve_metric(state.Phi.copy(), state.Pi)


def test_the_finest_level_reports_the_curvature_it_passes_through():
    """Between two base steps the finest level takes several of its own, and
    near threshold the peak can fall between base steps; the adapter reports
    the largest ``|R|`` any of them saw."""
    grid = SphericalGrid(r_max=R_MAX, n=400)
    sim = AdaptiveCollapse(grid, cells_per_radius=1e6, max_depth=3)
    state = near_the_origin(sim)
    state = sim.step(state, sim.dt)
    assert sim.ricci_between_steps() == 0.0, "nothing is tracked until asked"
    sim.watch_ricci(1.5)
    for _ in range(3):
        state = sim.step(state, sim.dt)
        finest = sim.subcycler.sims[-1]
        last = sim.subcycler.states[-1]
        a, _ = finest.solve_metric(last.Phi, last.Pi)
        inside = finest.r < 1.5
        at_the_end = float(np.abs(ricci_scalar(a, last.Phi, last.Pi))[inside].max())
        assert sim.ricci_between_steps() >= at_the_end > 0.0


@pytest.mark.slow
def test_the_collapse_search_runs_on_the_adaptive_solver():
    """``evolve_to_verdict`` takes it as it takes the uniform solver, and 2%
    either side of the threshold the verdicts split."""
    make = lambda: AdaptiveCollapse(SphericalGrid(r_max=R_MAX, n=400))  # noqa: E731
    below = make()
    above = make()
    low = evolve_to_verdict(below, shell(below, 8.30e-4), t_end=10.0)
    high = evolve_to_verdict(above, shell(above, 8.65e-4), t_end=10.0)
    assert low.verdict == "subcritical"
    assert high.verdict == "supercritical"
