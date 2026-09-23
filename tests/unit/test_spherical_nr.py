"""Spherically symmetric scalar collapse (design doc Section 5.4, Milestone 1)."""

import numpy as np
import pytest
from scipy.integrate import quad

from particlesim.core.interpolate import midpoints
from particlesim.core.spherical import SphericalGrid
from particlesim.solvers.nr.polar import (
    midpoint_mass,
    solve_lapse,
    solve_polar_metric,
)
from particlesim.solvers.nr.spherical import (
    ScalarCollapse,
    SphericalState,
    gaussian_pulse,
)


def sim_at(n: int, r_max: float = 20.0, **kw) -> ScalarCollapse:
    return ScalarCollapse(SphericalGrid(r_max=r_max, n=n), **kw)


def test_vacuum_is_exactly_flat():
    """Any deviation here is pure discretization error in the constraint
    solve, so it must be zero rather than small."""
    sim = sim_at(200)
    zero = np.zeros_like(sim.r)
    a, alpha = sim.solve_metric(zero, zero)
    np.testing.assert_array_equal(a, 1.0)
    np.testing.assert_allclose(alpha, 1.0, rtol=0, atol=1e-15)
    assert sim.adm_mass(a) == 0.0


def test_graded_grids_are_refused():
    grid = SphericalGrid(r_max=10.0, n=50, refine=4.0)
    with pytest.raises(ValueError, match="uniform spacing"):
        ScalarCollapse(grid)


def test_adm_mass_matches_the_energy_integral_in_the_linear_limit():
    """Pins the 2*pi in the Hamiltonian constraint.

    The deviation must vanish with amplitude, not with resolution: it is the
    gravitational binding energy, which is physics, and it scales as the
    mass itself.
    """
    deviations = {}
    for amp in (1e-4, 1e-5):
        sim = sim_at(400)
        st = gaussian_pulse(sim.grid, amplitude=amp, r0=8.0, width=1.5)
        a, _ = sim.solve_metric(st.Phi, st.Pi)
        ratio = sim.adm_mass(a) / sim.energy_integral(st.Phi, st.Pi)
        deviations[amp] = abs(ratio - 1.0)
        assert ratio == pytest.approx(1.0, abs=5e-3)
    # Energy scales as amplitude squared, so a hundredfold drop is expected.
    assert deviations[1e-4] / deviations[1e-5] == pytest.approx(100.0, rel=0.2)


def test_binding_deviation_is_physical_not_numerical():
    """Same amplitude, four times the resolution: if the deviation were
    discretization error it would fall, and it does not."""
    values = []
    for n in (200, 800):
        sim = sim_at(n)
        st = gaussian_pulse(sim.grid, amplitude=1e-5, r0=8.0, width=1.5)
        a, _ = sim.solve_metric(st.Phi, st.Pi)
        values.append(sim.adm_mass(a) / sim.energy_integral(st.Phi, st.Pi) - 1.0)
    assert values[0] == pytest.approx(values[1], rel=0.02)


def test_lapse_is_normalised_to_schwarzschild_at_the_outer_boundary():
    sim = sim_at(400)
    st = gaussian_pulse(sim.grid, amplitude=1e-3, r0=8.0, width=1.5)
    a, alpha = sim.solve_metric(st.Phi, st.Pi)
    assert alpha[-1] * a[-1] == pytest.approx(1.0, rel=1e-12)
    # The lapse decreases inward, as it must where mass is enclosed.
    assert alpha[0] < alpha[-1]


def test_evolution_conserves_the_adm_mass():
    sim = sim_at(200, courant=0.25)
    st = gaussian_pulse(sim.grid, amplitude=1e-4, r0=8.0, width=1.5, ingoing=True)
    a0, _ = sim.solve_metric(st.Phi, st.Pi)
    m0 = sim.adm_mass(a0)
    for _ in range(int(3.0 / sim.dt)):
        st = sim.step(st, sim.dt)
    a, _ = sim.solve_metric(st.Phi, st.Pi)
    assert np.isfinite(st.Phi).all() and np.isfinite(st.Pi).all()
    assert abs(sim.adm_mass(a) - m0) / m0 < 1e-3


def test_ingoing_pulse_reaches_the_origin_and_disperses():
    sim = sim_at(200)
    st = gaussian_pulse(sim.grid, amplitude=1e-4, r0=8.0, width=1.0, ingoing=True)
    central = []
    for i in range(int(12.0 / sim.dt)):
        st = sim.step(st, sim.dt)
        if i % 40 == 0:
            central.append(abs(st.Pi[0]))
    # The pulse arrives at the centre and then leaves again.
    assert max(central) > 10 * central[0]
    assert central[-1] < max(central) / 5


def test_state_is_immutable_and_replaces_cleanly():
    sim = sim_at(50)
    st = gaussian_pulse(sim.grid, 1e-4, 5.0, 1.0)
    new = st.with_fields(st.Phi * 2, st.Pi, t=1.0)
    assert isinstance(new, SphericalState)
    assert new.t == 1.0 and st.t == 0.0
    with pytest.raises(AttributeError):
        st.t = 5.0  # type: ignore[misc]


@pytest.mark.benchmark
@pytest.mark.slow
def test_fourth_order_convergence():
    """The scheme claims fourth order in space and time; measured through the
    ADM mass drift, which is a physical error rather than a norm of an
    arbitrary residual. Dissipation is off so only the scheme is measured."""
    drifts = []
    for n in (200, 400, 800):
        sim = sim_at(n, courant=0.25, dissipation=0.0)
        st = gaussian_pulse(sim.grid, amplitude=1e-4, r0=8.0, width=1.5, ingoing=True)
        a0, _ = sim.solve_metric(st.Phi, st.Pi)
        m0 = sim.adm_mass(a0)
        for _ in range(int(4.0 / sim.dt)):
            st = sim.step(st, sim.dt)
        a, _ = sim.solve_metric(st.Phi, st.Pi)
        drifts.append(abs(sim.adm_mass(a) - m0) / m0)
    orders = [np.log2(drifts[i] / drifts[i + 1]) for i in range(len(drifts) - 1)]
    assert all(3.7 < o < 4.3 for o in orders), f"orders were {orders}"


@pytest.mark.benchmark
@pytest.mark.slow
def test_strong_data_collapses_and_weak_data_does_not():
    """Polar slicing is horizon-avoiding, so collapse shows up either as the
    lapse going to zero with 2m/r approaching one from below, or as the
    constraint integration refusing to step past 2m/r = 1. Both are the same
    physics; which one appears depends on how fast the field concentrates
    relative to the grid."""
    from particlesim.solvers.nr.spherical import PolarSlicingBreakdown

    strong = sim_at(400)
    st = gaussian_pulse(strong.grid, amplitude=0.005, r0=8.0, width=1.0, ingoing=True)
    min_alpha, max_compactness, broke_down = 1.0, 0.0, False
    try:
        for i in range(int(10.0 / strong.dt)):
            st = strong.step(st, strong.dt)
            if i % 25 == 0:
                a, alpha = strong.solve_metric(st.Phi, st.Pi)
                assert (a >= 1.0 - 1e-12).all(), "a < 1 means negative enclosed mass"
                min_alpha = min(min_alpha, float(alpha.min()))
                max_compactness = max(max_compactness, float((1 - 1 / a**2).max()))
    except PolarSlicingBreakdown:
        broke_down = True

    assert broke_down or ScalarCollapse.lapse_collapsed(np.array([min_alpha]))
    assert max_compactness > 0.9
    assert max_compactness < 1.0  # never crossed, by construction

    weak = sim_at(300)
    st = gaussian_pulse(weak.grid, amplitude=1e-4, r0=8.0, width=1.0, ingoing=True)
    worst_alpha = 1.0
    for i in range(int(10.0 / weak.dt)):
        st = weak.step(st, weak.dt)
        if i % 25 == 0:
            a, alpha = weak.solve_metric(st.Phi, st.Pi)
            assert (a >= 1.0 - 1e-12).all()
            worst_alpha = min(worst_alpha, float(alpha.min()))
    assert not ScalarCollapse.lapse_collapsed(np.array([worst_alpha]))


def test_constraint_solve_refuses_data_it_cannot_resolve():
    """Regression: a Runge-Kutta stage stepping past 2m/r = 1 used to land on
    a negative mass, giving a < 1. That is unphysical, finite and plottable,
    which is the worst combination, and a check on the accepted value alone
    missed it because the bad value sits below one from underneath."""
    from particlesim.solvers.nr.spherical import PolarSlicingBreakdown

    coarse = sim_at(300)
    st = gaussian_pulse(coarse.grid, amplitude=0.01, r0=8.0, width=1.0, ingoing=True)
    with pytest.raises(PolarSlicingBreakdown, match="2m/r"):
        coarse.solve_metric(st.Phi, st.Pi)


@pytest.mark.slow
def test_refining_the_grid_resolves_data_a_coarse_grid_refuses():
    """The same data the coarse grid refuses is fine once resolved, which is
    what distinguishes a resolution limit from a horizon."""
    fine = sim_at(1600)
    st = gaussian_pulse(fine.grid, amplitude=0.01, r0=8.0, width=1.0, ingoing=True)
    a, _ = fine.solve_metric(st.Phi, st.Pi)
    assert (a >= 1.0 - 1e-12).all()
    assert fine.adm_mass(a) > 0
    assert (1 - 1 / a**2).max() > 0.99


def test_the_midpoint_stencil_falls_back_rather_than_reading_past_the_end():
    """Two points do not span the fourth-order stencil. The interpolation is
    used by the lapse solve on whatever radial extent it is handed, so the
    short case has to degrade to the average rather than index out of range.
    """
    values = np.array([1.0, 3.0])
    np.testing.assert_allclose(midpoints(values), [2.0])
    assert len(midpoints(np.arange(3.0))) == 2


def test_the_midpoint_stencil_is_exact_for_a_cubic():
    """Fourth-order accuracy means the four-point stencil reproduces any
    cubic exactly, endpoints included -- the endpoints use a one-sided
    quadratic, so they are held to the weaker claim they actually make.
    """
    x = np.linspace(0.0, 3.0, 8)
    cubic = 2.0 - 0.5 * x + 0.25 * x**2 - 0.125 * x**3
    mid = x[:-1] + 0.5 * (x[1] - x[0])
    exact = 2.0 - 0.5 * mid + 0.25 * mid**2 - 0.125 * mid**3
    np.testing.assert_allclose(midpoints(cubic)[1:-1], exact[1:-1], rtol=1e-12)


@pytest.mark.parametrize(
    ("label", "peak", "r_max"),
    [("shell away from the origin", 2.0, 6.0), ("matter filling the centre", 0.0, 4.0)],
)
def test_the_lapse_quadrature_is_fourth_order_in_the_mass_it_samples(label, peak, r_max):
    """The slicing condition's slope depends on the mass, so Simpson's rule
    needs the mass at each cell midpoint and the accuracy of *that* sets the
    accuracy of the lapse. This measures the quadrature alone: the mass and
    its derivative are handed in exactly, so the mass solve cannot
    contribute, and the answer is an integral scipy evaluates independently.

    Both placements are tested because they fail differently. Away from the
    origin, interpolating the midpoint mass from values alone is enough --
    the four-point stencil gets fourth order there and only the two-point
    average is second. With matter at the centre, where the mass is cubic in
    the radius and the slope carries ``m / r^2``, the stencil is second order
    too: 2.00 against Hermite's 4.00, and at 800 cells twenty thousand times
    less accurate. A shell-only test would have passed the stencil.
    """
    amplitude, width = (0.05, 0.35) if peak else (0.02, 1.5)

    def mass_of(r):
        return amplitude * r**3 * np.exp(-(((r - peak) / width) ** 2))

    def dmass_of(r):
        shape = np.exp(-(((r - peak) / width) ** 2))
        return amplitude * shape * (3 * r**2 - 2 * r**3 * (r - peak) / width**2)

    def slope_of(r, m):
        return m / (r**2 * (1.0 - 2.0 * m / r))

    errors = []
    for n in (50, 100, 200, 400):
        dr = r_max / n
        radii = (np.arange(n) + 0.5) * dr
        mass = mass_of(radii)
        alpha = solve_lapse(
            radii,
            dr,
            mass,
            lambda i, mid, r, m: slope_of(r, m),
            lambda i, mid, r, m: dmass_of(r),
        )
        # The grid's last point is r_max - dr/2, not r_max. Integrating the
        # truth to r_max instead leaves a half cell out, which is O(dr) and
        # swamps everything being measured -- it reads as first order.
        exact, _ = quad(lambda r: slope_of(r, mass_of(r)), 0.0, radii[-1], limit=400, epsabs=1e-14)
        first_cell = 0.5 * radii[0] * slope_of(radii[0], mass[0])
        errors.append(abs(float(np.log(alpha[-1] / alpha[0])) + first_cell - exact))

    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert all(3.5 < o < 4.3 for o in orders), f"{label}: orders were {orders}, errors {errors}"


def test_the_midpoint_mass_is_exact_where_the_mass_is_cubic():
    """The Misner-Sharp mass goes as ``C r^3`` at the origin and the lapse
    slope carries ``m / r^2``, so a relative error in the midpoint mass at
    the innermost cell is divided by ``h^2`` and survives as ``O(h^2)`` in
    the integral. Both rules that use values alone are badly wrong there and
    in opposite directions, which is why swapping one for the other changes
    the sign of the lapse perturbation at the centre rather than just its
    size. Hermite fits a cubic through the values and the derivatives, so on
    a cubic it is not approximately right but exact.
    """
    spacing = 0.25
    radii = (np.arange(6) + 0.5) * spacing
    coefficient = 1.7
    mass = coefficient * radii**3

    def mass_slope(index, mid, radius, value):
        return 3.0 * coefficient * radius**2

    hermite = midpoint_mass(radii, spacing, mass, mass_slope)
    truth = coefficient * (radii[:-1] + 0.5 * spacing) ** 3
    np.testing.assert_allclose(hermite, truth, rtol=1e-13)

    # What the value-only rules do at that first midpoint, for the record.
    average = 0.5 * (mass[0] + mass[1])
    stencil = midpoints(mass)[0]
    assert average / truth[0] == pytest.approx(1.75, rel=1e-12)
    assert stencil / truth[0] == pytest.approx(0.625, rel=1e-12)


def test_the_metric_solve_hands_the_mass_derivative_to_the_lapse():
    """The quadrature tests call ``solve_lapse`` directly, so they would not
    notice the mass source quietly ceasing to be passed through. That
    threading is the whole fix: without it the lapse falls back to
    interpolating from values alone, which is second order wherever there is
    matter at the centre. This pins the wiring rather than the arithmetic.
    """
    radii = (np.arange(40) + 0.5) * 0.1
    density = np.exp(-((radii / 0.8) ** 2))
    source = 2.0 * np.pi * radii**2 * density
    source_mid = midpoints(source)
    density_mid = midpoints(density)

    def mass_slope(index, mid, radius, mass):
        value = source_mid[index] if mid else source[index]
        return value * (1.0 - 2.0 * mass / radius)

    def lapse_slope(index, mid, radius, mass):
        value = density_mid[index] if mid else density[index]
        return mass / (radius**2 * (1.0 - 2.0 * mass / radius)) + 2.0 * np.pi * radius * value

    _, alpha, mass = solve_polar_metric(radii, 0.1, mass_slope, lapse_slope)
    hermite = solve_lapse(radii, 0.1, mass, lapse_slope, mass_slope)
    values_only = solve_lapse(radii, 0.1, mass, lapse_slope)

    np.testing.assert_allclose(alpha, hermite, rtol=0, atol=0)
    assert not np.allclose(alpha, values_only, rtol=1e-9), (
        "the two rules agree, so this data cannot tell them apart and the test proves nothing"
    )


def _callable_metric(sim, Phi, Pi):
    """The metric exactly as ``solve_metric`` computed it before vectorising."""
    from particlesim.solvers.nr.polar import solve_polar_metric

    density = Pi**2 + Phi**2
    source = 2.0 * np.pi * sim.r**2 * density
    source_mid, density_mid = midpoints(source), midpoints(density)

    def mass_slope(i, mid, radius, mass):
        return (source_mid[i] if mid else source[i]) * (1.0 - 2.0 * mass / radius)

    def lapse_slope(i, mid, radius, mass):
        value = density_mid[i] if mid else density[i]
        return mass / (radius**2 * (1.0 - 2.0 * mass / radius)) + 2.0 * np.pi * radius * value

    a, alpha, _ = solve_polar_metric(sim.r, sim.dr, mass_slope, lapse_slope)
    return a, alpha


@pytest.mark.parametrize(
    ("r_max", "n", "amplitude", "r0", "width"),
    [
        (10.0, 400, 3e-3, 3.0, 0.7),  # weak shell
        (20.0, 1600, 0.01, 8.0, 1.0),  # strong, resolved: 2m/r reaches 0.998
        (20.0, 600, 0.01, 8.0, 1.0),  # near a horizon on a coarse grid
    ],
)
def test_the_vectorised_metric_is_the_loop_to_rounding(r_max, n, amplitude, r0, width):
    """Vectorising changed the arithmetic's order, not the arithmetic.

    That was the requirement, because a different discretisation of the same
    equation was tried first and failed: an integrating-factor solution is
    exact in the continuum and fourth order, and near a horizon it read
    ``max 2m/r = 0.999296`` where this reads 0.998379, against a converged
    0.998385. The comparison runs where the lapse spans forty-eight decades,
    which is where rounding differences would compound if anything did.
    """
    sim = sim_at(n, r_max=r_max)
    st = gaussian_pulse(sim.grid, amplitude=amplitude, r0=r0, width=width, ingoing=True)
    a_new, alpha_new = sim.solve_metric(st.Phi, st.Pi)
    a_old, alpha_old = _callable_metric(sim, st.Phi, st.Pi)
    np.testing.assert_allclose(a_new, a_old, rtol=1e-12, atol=0)
    np.testing.assert_allclose(alpha_new, alpha_old, rtol=1e-11, atol=0)


def test_the_vectorised_metric_refuses_where_the_loop_refused():
    """The loop raised when any Runge-Kutta stage stepped to ``2m/r >= 1``.
    Every stage is affine in the mass, so the vectorised solve checks all of
    them and must raise at the same radius, with the same words."""
    from particlesim.solvers.nr.spherical import PolarSlicingBreakdown

    sim = sim_at(300)
    st = gaussian_pulse(sim.grid, amplitude=0.01, r0=8.0, width=1.0, ingoing=True)
    with pytest.raises(PolarSlicingBreakdown) as old:
        _callable_metric(sim, st.Phi, st.Pi)
    with pytest.raises(PolarSlicingBreakdown) as new:
        sim.solve_metric(st.Phi, st.Pi)
    assert str(new.value) == str(old.value)
