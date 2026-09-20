"""Nested refinement levels for the spherical solver (issue #111)."""

import numpy as np
import pytest

from particlesim.core.spherical import SphericalGrid
from particlesim.solvers.nr.hierarchy import (
    RATIO,
    Hierarchy,
    Level,
    prolong,
    refine,
    restrict,
)
from particlesim.solvers.nr.spherical import ScalarCollapse, gaussian_pulse

R_MAX, AMPLITUDE, CENTRE, WIDTH = 10.0, 3e-3, 3.0, 0.7


def level_at(n: int, r_max: float = R_MAX) -> Level:
    grid = SphericalGrid(r_max=r_max, n=n)
    state = gaussian_pulse(grid, amplitude=AMPLITUDE, r0=CENTRE, width=WIDTH)
    return Level(grid, state.Phi, state.Pi)


def uniform_metric(n: int):
    sim = ScalarCollapse(SphericalGrid(r_max=R_MAX, n=n))
    state = gaussian_pulse(sim.grid, amplitude=AMPLITUDE, r0=CENTRE, width=WIDTH)
    a, alpha = sim.solve_metric(state.Phi, state.Pi)
    return sim.adm_mass(a), float(alpha[0])


def two_level(n: int, boundary: float = R_MAX / 2) -> Hierarchy:
    """A coarse level of ``n`` cells and a child inside ``boundary``, refined 2:1."""
    fine_cells = int(round(boundary / R_MAX * n)) * RATIO
    return Hierarchy([level_at(n), level_at(fine_cells, boundary)])


# --- the levels themselves ---------------------------------------------


def test_a_level_must_be_uniform_and_shaped_like_its_grid():
    graded = SphericalGrid(r_max=1.0, n=8, refine=2.0)
    with pytest.raises(ValueError, match="uniform"):
        Level(graded, np.zeros(8), np.zeros(8))
    uniform = SphericalGrid(r_max=1.0, n=8)
    with pytest.raises(ValueError, match="one value per cell"):
        Level(uniform, np.zeros(7), np.zeros(8))


def test_refinement_snaps_outward_so_the_grids_nest_exactly():
    """Restriction is an average of whole child cells, which only works if
    each parent cell is exactly the union of ``RATIO`` of them. An extent
    that fell mid-cell would make it a resampling instead, so the extent is
    snapped outward to a whole number of parent cells rather than taken at
    its word.
    """
    parent = level_at(40)
    child = refine(parent, extent=2.55)  # 10.2 parent cells
    assert child.outer == pytest.approx(2.75)  # 11 of them
    assert child.grid.n == RATIO * 11
    assert child.spacing == pytest.approx(parent.spacing / RATIO)


def test_a_child_must_be_a_strict_sub_interval():
    parent = level_at(40)
    for extent in (0.0, -1.0, R_MAX, R_MAX + 1):
        with pytest.raises(ValueError, match="strict sub-interval"):
            refine(parent, extent)


def test_restriction_is_exact_on_a_linear_field_and_second_order_otherwise():
    """Averaging two children is exactly the parent *cell average*, and this
    solver stores point values at cell centres -- its derivatives, its parity
    reflection and its midpoint interpolation all read them that way. Against
    a point value the average is off by ``dr^2 f'' / 32``, so it is exact
    where ``f'' = 0`` and second order elsewhere.

    Pinned rather than wished away, because the evolution will restrict and a
    second-order restriction inside a fourth-order evolution caps the
    evolution. The metric solve does not restrict at all, so nothing is
    capped by it today.
    """

    def round_trip(n, phi_of, pi_of):
        grid = SphericalGrid(r_max=R_MAX, n=n)
        parent = Level(grid, phi_of(grid.radii()), pi_of(grid.radii()))
        child_grid = refine(parent, extent=R_MAX / 2).grid
        child = Level(child_grid, prolong(parent, child_grid), prolong(parent, child_grid, "Pi"))
        back_phi, back_pi = restrict(child, parent)
        covered = parent.radii < child.outer
        # Outside the child nothing may be touched at all.
        np.testing.assert_array_equal(back_phi[~covered], parent.Phi[~covered])
        return max(
            float(np.max(np.abs(back_phi[covered] - parent.Phi[covered]))),
            float(np.max(np.abs(back_pi[covered] - parent.Pi[covered]))),
        )

    # Phi is odd across the origin and Pi is even; a field without a definite
    # parity is not valid input and would show a fixed error at the centre.
    assert round_trip(40, lambda r: 3.0 * r, lambda r: np.ones_like(r)) == 0.0

    errors = [round_trip(n, lambda r: r**3, lambda r: r**2) for n in (40, 80, 160)]
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert all(1.8 < o < 2.2 for o in orders), f"orders were {orders}, errors {errors}"


def test_prolongation_carries_the_parity_of_the_field_it_moves():
    """``Phi`` is odd across the origin and ``Pi`` is even. The origin is not
    a boundary but the centre of a sphere, so the stencil reaches across it
    with the sign the field's rank demands. Interpolating an odd field as
    though it were even puts a cusp at ``r = 0`` on the level that can least
    afford one, and the ``2 f Phi / r`` term turns that into growth.
    """
    grid = SphericalGrid(r_max=1.0, n=32)
    radii = grid.radii()
    odd, even = radii**3, np.cos(radii)
    parent = Level(grid, odd, even)
    child = SphericalGrid(r_max=0.5, n=32)
    np.testing.assert_allclose(prolong(parent, child), child.radii() ** 3, rtol=1e-6)
    np.testing.assert_allclose(prolong(parent, child, "Pi"), np.cos(child.radii()), rtol=1e-6)


# --- the hierarchy ------------------------------------------------------


def test_a_hierarchy_checks_that_its_levels_nest_and_refine_by_two():
    coarse = level_at(40)
    with pytest.raises(ValueError, match="at least one level"):
        Hierarchy([])
    with pytest.raises(ValueError, match="decrease inward"):
        Hierarchy([coarse, level_at(80, R_MAX)])
    # Right extent, wrong spacing: 40 cells over half the radius is the
    # parent's own spacing, not half of it.
    with pytest.raises(ValueError, match="refine by"):
        Hierarchy([coarse, level_at(40 // RATIO, R_MAX / 2)])


def test_each_cell_is_owned_by_exactly_one_level():
    """The composite grid is a partition, not an overlay. A cell covered by
    a finer level is not the coarse level's to report, or the same matter
    would be counted twice in the mass integral.
    """
    h = two_level(100)
    composite = h.composite_radii()
    assert len(composite) == sum(int(h.owned(k).sum()) for k in range(h.depth))
    assert np.all(np.diff(composite) > 0), "the composite grid must increase outward"
    inner = h.levels[1].outer
    assert np.all(h.levels[0].radii[h.owned(0)] > inner)
    assert np.all(h.levels[1].radii[h.owned(1)] < inner)


def test_the_hierarchy_metric_agrees_with_a_uniform_grid_of_the_finest_spacing():
    """A level boundary is not supposed to be visible in the answer. The
    two-level solve at ``n`` cells buys what a uniform grid buys at ``2n``,
    which is the whole point of refining: the accuracy follows the finest
    spacing where the finest spacing is.
    """
    hierarchy_mass, _ = _hierarchy_values(two_level(100))
    uniform_fine, _ = uniform_metric(200)
    reference, _ = uniform_metric(12800)
    assert abs(hierarchy_mass - reference) == pytest.approx(abs(uniform_fine - reference), rel=0.05)


def _hierarchy_values(h: Hierarchy):
    a_levels, alpha_levels = h.solve_metric()
    coarse = h.levels[0]
    adm = 0.5 * coarse.radii[-1] * (1.0 - 1.0 / a_levels[0][-1] ** 2)
    return float(adm), float(alpha_levels[-1][0])


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.parametrize("boundary", [2.0, CENTRE])
def test_fourth_order_survives_the_refinement_boundary(boundary):
    """Issue #111's acceptance criterion, and the reason the gap between two
    levels is crossed by a Runge-Kutta step rather than an Euler one.

    Both metric functions are measured, because they are separate
    integrations and were not always the same order: the mass solve was
    fourth order while the lapse beside it was second, which no single error
    norm would have shown.

    Two boundary placements, because where the boundary sits relative to the
    matter decides how hard the gap is. Away from the shell the order is a
    clean 4.00; placed at the shell's own peak it settles near 3.6, and stays
    there rather than drifting -- 3.61, 3.78, 3.71, 3.57 as the ladder is
    extended -- because that is where the crossing has to reconstruct the
    field from the coarse level over the widest gap. Both are tested rather
    than the flattering one, since a refinement scheme that only holds its
    order away from the solution is no use to a solver that refines toward
    one.
    """
    reference_mass, reference_lapse = uniform_metric(12800)
    masses, lapses = [], []
    for n in (100, 200, 400, 800):
        mass, lapse = _hierarchy_values(two_level(n, boundary))
        masses.append(abs(mass - reference_mass))
        lapses.append(abs(lapse - reference_lapse))

    for name, errors in (("mass", masses), ("lapse", lapses)):
        orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
        assert all(3.4 < o < 4.4 for o in orders), f"{name} at r={boundary}: orders {orders}"


@pytest.mark.slow
def test_crossing_the_level_gap_with_an_euler_step_would_be_second_order():
    """The failure this design exists to prevent, measured rather than
    asserted. Between the outermost point of one level and the innermost
    point of the next lies a stretch belonging to neither. Crossing it with
    a single Euler step is one local ``O(dr^2)`` error at one point, and
    that alone takes the whole metric from fourth order to second -- a
    refinement scheme that quietly stops being fourth order at its own
    boundaries, which is the one place a refinement scheme is judged.

    Measured with the boundary at the shell's peak. Put it out in the tail
    instead and the Euler error is small enough to hide: the mass reads
    3.77, 3.31, 2.64 there, still falling toward two but not obviously
    broken at any resolution anyone would run. Refinement follows the
    solution, so the case that matters is the one with matter on both sides
    of the boundary.
    """
    import particlesim.solvers.nr.hierarchy as module

    def euler(level, inner_radius, inner_mass, inner_log, outer_radius):
        step = outer_radius - inner_radius
        phi = module._interpolate(level, np.array([outer_radius]), level.Phi, odd=True)[0]
        pi = module._interpolate(level, np.array([outer_radius]), level.Pi, odd=False)[0]
        density = float(pi**2 + phi**2)
        free = 1.0 - 2.0 * inner_mass / outer_radius
        d_mass = 2.0 * np.pi * outer_radius**2 * density * free
        d_log = inner_mass / (outer_radius**2 * free) + 2.0 * np.pi * outer_radius * density
        return inner_mass + step * d_mass, inner_log + step * d_log

    reference_mass, reference_lapse = uniform_metric(12800)
    original = module._cross_gap
    module._cross_gap = euler
    try:
        measured = [_hierarchy_values(two_level(n, CENTRE)) for n in (100, 200, 400)]
    finally:
        module._cross_gap = original

    for name, exact, index in (("mass", reference_mass, 0), ("lapse", reference_lapse, 1)):
        errors = [abs(row[index] - exact) for row in measured]
        orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
        assert all(1.8 < o < 2.3 for o in orders), f"{name}: expected second order, got {orders}"


@pytest.mark.slow
def test_the_gap_must_be_crossed_for_the_mass_and_the_lapse_together():
    """The subtler version of the same failure, and the one that was
    actually written first.

    The slicing condition's slope is ``(m + 4 pi r^3 S)/(r(r - 2m))``, so
    carrying ``ln alpha`` across the gap needs to know how ``m`` varies along
    it. An earlier design crossed the two separately and held the mass at its
    end value while the lapse crossed, which is wrong by
    ``O(dm/dr * gap^2)`` -- second order, and exactly zero wherever there is
    no matter left to accumulate. It measured a clean 4.00 with the boundary
    in a shell's tail and was wrong the whole time.
    """
    import particlesim.solvers.nr.hierarchy as module

    original = module._cross_gap

    def frozen_mass(level, inner_radius, inner_mass, inner_log, outer_radius):
        """Cross for the mass properly, then for the lapse with m held fixed."""
        mass, _ = original(level, inner_radius, inner_mass, inner_log, outer_radius)

        def d_log(radius):
            phi = module._interpolate(level, np.array([radius]), level.Phi, odd=True)[0]
            pi = module._interpolate(level, np.array([radius]), level.Pi, odd=False)[0]
            free = 1.0 - 2.0 * mass / radius
            return mass / (radius**2 * free) + 2.0 * np.pi * radius * float(pi**2 + phi**2)

        step = outer_radius - inner_radius
        half = inner_radius + 0.5 * step
        crossed = step / 6.0 * (d_log(inner_radius) + 4.0 * d_log(half) + d_log(outer_radius))
        return mass, inner_log + crossed

    _, reference_lapse = uniform_metric(12800)
    module._cross_gap = frozen_mass
    try:
        errors = [
            abs(_hierarchy_values(two_level(n, CENTRE))[1] - reference_lapse)
            for n in (200, 400, 800)
        ]
    finally:
        module._cross_gap = original

    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert all(o < 3.0 for o in orders), (
        f"freezing the mass across the gap should cost order, got {orders}"
    )
