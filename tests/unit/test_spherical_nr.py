"""Spherically symmetric scalar collapse (design doc Section 5.4, Milestone 1)."""

import numpy as np
import pytest

from particlesim.core.spherical import SphericalGrid
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
    sim = sim_at(200, courant=0.25, dissipation=0.02)
    st = gaussian_pulse(sim.grid, amplitude=1e-4, r0=8.0, width=1.5, ingoing=True)
    a0, _ = sim.solve_metric(st.Phi, st.Pi)
    m0 = sim.adm_mass(a0)
    for _ in range(int(3.0 / sim.dt)):
        st = sim.step(st, sim.dt)
    a, _ = sim.solve_metric(st.Phi, st.Pi)
    assert np.isfinite(st.Phi).all() and np.isfinite(st.Pi).all()
    assert abs(sim.adm_mass(a) - m0) / m0 < 1e-3


def test_ingoing_pulse_reaches_the_origin_and_disperses():
    sim = sim_at(200, dissipation=0.02)
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

    strong = sim_at(400, dissipation=0.02)
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

    weak = sim_at(300, dissipation=0.02)
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

    coarse = sim_at(300, dissipation=0.02)
    st = gaussian_pulse(coarse.grid, amplitude=0.01, r0=8.0, width=1.0, ingoing=True)
    with pytest.raises(PolarSlicingBreakdown, match="2m/r"):
        coarse.solve_metric(st.Phi, st.Pi)


@pytest.mark.slow
def test_refining_the_grid_resolves_data_a_coarse_grid_refuses():
    """The same data the coarse grid refuses is fine once resolved, which is
    what distinguishes a resolution limit from a horizon."""
    fine = sim_at(1600, dissipation=0.02)
    st = gaussian_pulse(fine.grid, amplitude=0.01, r0=8.0, width=1.0, ingoing=True)
    a, _ = fine.solve_metric(st.Phi, st.Pi)
    assert (a >= 1.0 - 1e-12).all()
    assert fine.adm_mass(a) > 0
    assert (1 - 1 / a**2).max() > 0.99
