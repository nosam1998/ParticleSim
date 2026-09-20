"""Fluid collapse: two fates from one star, and a theorem about the outside.

Issue #60's acceptance is that collapse forms a horizon and matches the
Schwarzschild exterior. The second half is Birkhoff's theorem, which says
the exterior should not move at all while the interior falls in -- so it is
a theorem to verify rather than a tolerance to meet, and it is the sharper
of the two.

The first half has to be read in the slicing. Polar-areal coordinates are
horizon-avoiding, so a trapped surface never forms in finite coordinate
time: ``2m/r`` asymptotes to one while the lapse collapses, and the
constraint integration then refuses to continue. Both signals are required
before a run is called a collapse, because either alone is also what a grid
too coarse to resolve the approach produces.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.core.spherical import SphericalGrid
from particlesim.matter.eos import Polytrope
from particlesim.scenarios.singularity.fluid_collapse import (
    LAPSE_THRESHOLD,
    exterior_residual,
    horizon_radius,
    is_unstable,
    lapse_collapsed,
    polytropic_collapse,
)
from particlesim.solvers.hydro.spherical import SphericalHydro
from particlesim.solvers.hydro.srhd import GammaLaw, conserved_to_primitive

UNSTABLE = 1.5e-3
STABLE = 5e-5


@pytest.fixture(scope="module")
def cold() -> Polytrope:
    return Polytrope(polytropic_constant=100.0, gamma=1.9)


@pytest.fixture(scope="module")
def collapsed():
    return polytropic_collapse(UNSTABLE, kick=-0.02, points=120).run(6.0)


@pytest.fixture(scope="module")
def dispersed():
    return polytropic_collapse(UNSTABLE, kick=+0.02, points=120).run(6.0)


# --- which stars can collapse at all --------------------------------------


def test_the_structure_solver_says_which_stars_can_collapse(cold):
    """Settled without evolving anything, which is why it is worth asking first.

    A star on the stable branch will not collapse however hard it is pushed,
    and time spent wondering why is time spent looking for a bug in a
    correct answer -- which is exactly how the turning point came to be
    checked here at all.
    """
    assert not is_unstable(cold, STABLE)
    assert is_unstable(cold, UNSTABLE)


# --- the two fates ---------------------------------------------------------


@pytest.mark.benchmark
def test_an_unstable_star_pushed_inward_collapses(collapsed):
    """Issue #60's first half, read in the slicing that produces it.

    ``2m/r`` reaches 0.988 and the lapse falls to ``5e-5``, and the run ends
    when the constraint integration refuses to step past ``2m/r = 1``.
    That refusal is the physical end of the run in these coordinates, not an
    exception to be caught and worked around.
    """
    assert collapsed.outcome == "collapsed"
    assert collapsed.peak_compactness > 0.95
    assert collapsed.minimum_lapse < 1e-3
    assert collapsed.horizon_radius is not None
    assert collapsed.formed_horizon
    assert collapsed.slicing_broke_down
    assert collapsed.central_density_ratio > 2.0


@pytest.mark.benchmark
def test_the_same_star_pushed_outward_disperses(dispersed, collapsed):
    """Same star, same kick, opposite sign. The equilibrium is the unstable one.

    Nothing else about the two runs differs, so this is as clean a
    demonstration as the problem allows that the ending is decided by the
    perturbation and not by the scheme.
    """
    assert dispersed.outcome == "dispersed"
    assert dispersed.central_density_ratio < 0.2
    assert dispersed.peak_compactness < collapsed.peak_compactness
    assert dispersed.minimum_lapse > 100.0 * collapsed.minimum_lapse
    assert dispersed.horizon_radius is None


@pytest.mark.benchmark
def test_a_star_on_the_stable_branch_does_not_collapse(cold):
    """However it is pushed. The same inward kick leaves it bounded."""
    result = polytropic_collapse(STABLE, kick=-0.02, points=120).run(4.0)
    assert result.outcome == "bounded"
    assert not result.formed_horizon
    assert result.peak_compactness < 0.5
    assert 0.5 < result.central_density_ratio < 2.0


# --- the sharp half: Birkhoff ----------------------------------------------


@pytest.mark.benchmark
def test_the_exterior_stays_schwarzschild_while_the_interior_collapses(collapsed, dispersed):
    """A spherically symmetric vacuum is Schwarzschild, statically, regardless.

    So the metric outside the star should not move at all while the star
    falls in, and it does not: parts in ``1e-4`` through a collapse that
    takes ``2m/r`` from 0.53 to 0.99, and parts in ``1e-6`` on the runs that
    stay mild. That is a theorem being verified, and it is what makes the
    interior's violence credible rather than merely dramatic.
    """
    assert collapsed.exterior_residual < 1e-3
    assert dispersed.exterior_residual < 1e-5
    assert any("no longer vacuum" in note for note in dispersed.notes)


def test_the_birkhoff_comparison_stops_when_the_region_stops_being_vacuum(dispersed):
    """Otherwise the number is a metric compared against a region full of matter.

    A dispersing star expands past the sampling radius, and without the
    guard the residual reads 0.1 and looks like a failed theorem instead of
    a misapplied one.
    """
    assert any("no longer vacuum" in note for note in dispersed.notes)
    assert dispersed.exterior_residual < 1e-5


def test_exterior_residual_is_zero_on_an_exact_schwarzschild_metric():
    radii = np.linspace(5.0, 20.0, 64)
    mass = 1.5
    a = 1.0 / np.sqrt(1.0 - 2.0 * mass / radii)
    assert exterior_residual(radii, a, 1.0 / a, mass, 4.0) < 1e-15


def test_exterior_residual_refuses_a_sampling_radius_off_the_grid():
    radii = np.linspace(5.0, 20.0, 16)
    a = np.ones(16)
    with pytest.raises(ValueError, match="no grid point lies beyond"):
        exterior_residual(radii, a, a, 1.0, 100.0)


# --- the diagnostics -------------------------------------------------------


def test_the_horizon_needs_both_signals():
    """Either alone is also what an unresolved approach produces."""
    radii = np.linspace(1.0, 10.0, 32)
    compact = np.full(32, 0.99)
    a = 1.0 / np.sqrt(1.0 - compact)
    assert horizon_radius(radii, a) == pytest.approx(radii[0])
    assert horizon_radius(radii, np.ones(32)) is None
    assert lapse_collapsed(np.full(32, 0.5 * LAPSE_THRESHOLD))
    assert not lapse_collapsed(np.ones(32))


def test_the_scenario_produces_a_battery_row(cold):
    row = polytropic_collapse(STABLE, kick=-0.02, points=80).as_battery_result(1.0)
    assert row.scenario == "fluid_collapse"
    assert row.outcome in ("collapsed", "dispersed", "bounded")
    assert set(row.as_row()) == {
        "scenario",
        "outcome",
        "bounced",
        "max_density",
        "min_scale_factor",
    }


def test_the_collapse_refuses_data_it_cannot_use(cold):
    from particlesim.matter.tov import solve_tov
    from particlesim.scenarios.singularity.fluid_collapse import FluidCollapse

    bare = solve_tov(cold, STABLE)
    with pytest.raises(ValueError, match="carries no profile"):
        FluidCollapse(bare, cold, GammaLaw(1.9))

    star = solve_tov(cold, STABLE, samples=1000)
    with pytest.raises(ValueError, match="must be subluminal"):
        FluidCollapse(star, cold, GammaLaw(1.9), kick=1.5)
    with pytest.raises(ValueError, match="positive duration"):
        FluidCollapse(star, cold, GammaLaw(1.9)).run(0.0)


# --- the floor that was not doing anything ---------------------------------


def test_the_energy_floor_clears_the_round_off_in_the_limit_it_floors():
    """The margin has to exceed the cancellation in the quantity it floors.

    Found deep in a collapse, at a Lorentz factor of nine. The floor held
    ``tau`` above ``sqrt(D^2 + S^2) - D``, but its margin was an absolute
    ``1e-23`` while the cancellation in that difference is ``5e-19`` -- so
    the floor fired, left the state exactly at the cold limit where the
    internal energy is zero, and the recovery then had no root. A floor
    that is present, fires, and does nothing is worse than none, because it
    is the first thing ruled out.
    """
    solver = SphericalHydro(
        SphericalGrid(r_max=10.0, n=32), GammaLaw(1.9), reference_density=2.9e-3
    )
    density = np.array([2.74240e-4])
    momentum = np.array([-2.33233e-3])
    cold_limit = np.sqrt(density**2 + momentum**2) - density
    raised = solver.regularise(density, momentum, cold_limit.copy())

    assert raised[2][0] > cold_limit[0]
    recovered = conserved_to_primitive(raised[0], raised[1], raised[2], solver.eos)
    assert recovered[2][0] > 0.0
    assert abs(recovered[1][0]) < 1.0
