"""The Ashtekar-Olmedo-Singh polymerised black-hole interior (issue #24).

Every number the plugin reports is reached two ways, or held to a closed
form:

- the solution is checked against Hamilton's equations integrated
  numerically from the effective Hamiltonian, which never sees the closed
  form;
- the polymerisation parameters are checked against the two area
  conditions they come from, solved exactly;
- at zero area gap everything is Schwarzschild, symbolically through the
  GR-limit harness and numerically to round-off.

The benchmark is the one the prescription exists for: curvature at the
transition surface is bounded by a value independent of the mass.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
from scipy.integrate import solve_ivp

from particlesim.theories.limits import check_gr_limit
from particlesim.theories.lqg.polymer_bh import AREA_GAP, BARBERO_IMMIRZI, PolymerBlackHole
from particlesim.theories.registry import list_theories

AOS = PolymerBlackHole()
GAMMA = BARBERO_IMMIRZI


def test_the_plugin_is_registered_as_tier_b():
    assert list_theories()["lqg.polymer_bh"] is PolymerBlackHole
    assert PolymerBlackHole.tier == "B"
    assert AOS.values == {"area_gap": AREA_GAP, "immirzi": BARBERO_IMMIRZI}


def test_the_gr_limit_is_schwarzschild_symbolically():
    """The harness sets the area gap to zero and compares with Schwarzschild exactly."""
    report = check_gr_limit(PolymerBlackHole)
    assert report.passed, report.reasons
    assert report.metric_family_matches is True


def test_a_wrong_declared_limit_is_caught():
    """The same check has to be able to fail: a nonzero area gap is not Schwarzschild."""

    class Misdeclared(PolymerBlackHole):
        id = "test.misdeclared_polymer_bh"

        def gr_limit(self):
            return {"area_gap": 1.0}

    report = check_gr_limit(Misdeclared)
    assert not report.passed
    assert report.metric_family_matches is False


def test_at_zero_area_gap_the_interior_is_schwarzschild_to_round_off():
    """``r = 2m e^T``, ``g_xx = 2m/r - 1``, ``N^2 = r^2/(2m/r - 1)``, ``K = 48 m^2/r^6``."""
    classical = PolymerBlackHole(area_gap=0.0)
    mass, time = 3.0, np.linspace(-6.0, -0.01, 9)
    state = classical.interior(mass, time)
    radius = 2 * mass * np.exp(time)
    assert np.allclose(np.sqrt(state["p_c"]), radius, rtol=1e-14)
    assert np.allclose(state["g_xx"], 2 * mass / radius - 1, rtol=1e-12)
    assert np.allclose(state["lapse"] ** 2, radius**2 / (2 * mass / radius - 1), rtol=1e-12)
    curvature = classical.kretschmann(mass, time)
    assert np.allclose(curvature, 48 * mass**2 / radius**6, rtol=1e-13)
    assert classical.transition_time(mass) == -np.inf
    assert classical.white_hole_time(mass) == -np.inf
    assert np.isnan(classical.white_hole_mass(mass))
    with pytest.raises(ValueError, match="unbounded"):
        classical.maximum_kretschmann(mass)


@pytest.mark.parametrize("mass", [10.0, 1e3])
def test_the_closed_form_solves_hamiltons_equations(mass):
    """Integrated from just inside the horizon to past the white-hole side of the bounce.

    The equations come from the effective Hamiltonian alone; agreement to
    1e-10 over the transition surface is what says the closed form, and the
    stable rewriting of it, are right. Both of AOS's mass functions stay at
    ``m`` along the numerical solution.
    """
    polymerisation = AOS.polymerisation(mass)
    start, end = -0.01, 0.9 * AOS.white_hole_time(mass)
    initial = AOS.interior(mass, start)
    state0 = [float(initial[name]) for name in ("b", "p_b", "c", "p_c")]
    solution = solve_ivp(
        AOS.hamilton_equations(mass),
        (start, end),
        state0,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
        dense_output=True,
    )
    times = np.linspace(start, end, 41)
    numerical = solution.sol(times)
    exact = AOS.interior(mass, times)
    for index, name in enumerate(("b", "p_b", "c", "p_c")):
        assert np.allclose(numerical[index], exact[name], rtol=1e-10, atol=0.0), name
    assert times.min() < AOS.transition_time(mass) < times.max()
    o_b, o_c = AOS.masses(numerical, polymerisation)
    assert np.allclose(o_b, mass, rtol=1e-10)
    assert np.allclose(o_c, mass, rtol=1e-10)


def test_the_metric_family_in_areal_radius_is_the_same_interior():
    """``metric_family`` rewrites the closed form in ``r``; it has to agree with it in ``T``."""
    mass = 50.0
    metric = AOS.metric_family({"mass": mass})
    r = sp.Symbol("r", positive=True)
    g_xx = sp.lambdify(r, metric[0, 0])
    g_rr = sp.lambdify(r, metric[1, 1])
    times = np.linspace(0.8 * AOS.transition_time(mass), -0.05, 7)
    state = AOS.interior(mass, times)
    radius = np.sqrt(state["p_c"])
    assert np.allclose([float(g_xx(value)) for value in radius], state["g_xx"], rtol=1e-9)
    # g_rr dr^2 = -N^2 dT^2, with dr/dT = p_c' / (2 r) = r cos(delta_c c).
    expected = -(state["lapse"] ** 2) / (radius * state["cos_c"]) ** 2
    assert np.allclose([float(g_rr(value)) for value in radius], expected, rtol=1e-9)


def test_the_prescription_is_the_double_root_of_the_area_conditions():
    """The exact conditions have two roots, one each side of AOS's form, closing as ``m^-1/3``."""
    masses = [1e4, 1e6, 1e8, 1e10]
    offsets = []
    for mass in masses:
        closed = AOS.polymerisation(mass)
        lower, upper = AOS.plaquette_polymerisation(mass)
        assert lower[0] < closed[0] < upper[0]
        assert upper[1] < closed[1] < lower[1]
        offsets.append(upper[0] / closed[0] - 1.0)
        assert abs(lower[0] / closed[0] - 1.0) == pytest.approx(offsets[-1], rel=0.01)
    ratios = [a / b for a, b in zip(offsets, offsets[1:], strict=False)]
    for ratio in ratios:
        assert ratio == pytest.approx(100 ** (1 / 3), rel=0.01)


def test_the_transition_surface_grows_as_the_cube_root_of_the_mass():
    """``r_T = sqrt(m gamma L_o delta_c)``, which the prescription makes ``m^(1/3)`` exactly."""
    expected = np.sqrt(GAMMA ** (4 / 3) * AREA_GAP ** (2 / 3) / (2 * (4 * np.pi**2) ** (1 / 3)))
    for mass in (1e2, 1e5, 1e8, 1e12):
        assert AOS.transition_radius(mass) / mass ** (1 / 3) == pytest.approx(expected, rel=1e-12)
        state = AOS.interior(mass, AOS.transition_time(mass))
        assert np.sqrt(state["p_c"]) == pytest.approx(AOS.transition_radius(mass), rel=1e-12)


@pytest.mark.benchmark
def test_the_curvature_bound_is_independent_of_the_mass():
    """AOS's scaling: curvature peaks at the transition surface, at one value for every large mass.

    The radius there goes as ``m^(1/3)``, so ``m^2/r^6`` does not depend on
    ``m``. Measured: 82188.3642 Planck units at ``m = 10^6``, ``10^9`` and
    ``10^12``, the same to 2e-08; at ``m = 100`` the peak is 0.06% lower.
    Classically the same interior reaches any curvature at all.
    """
    bound = 82188.364163
    for mass in (1e6, 1e9, 1e12):
        time, peak = AOS.maximum_kretschmann(mass)
        assert time == pytest.approx(AOS.transition_time(mass), abs=1e-3)
        assert peak == pytest.approx(bound, rel=3e-8)
    _, small = AOS.maximum_kretschmann(100.0)
    assert small / bound - 1 == pytest.approx(-6.03e-4, abs=1e-6)

    mass = 1e6
    times = np.linspace(0.99 * AOS.white_hole_time(mass), -0.01, 2001)
    assert np.nanmax(AOS.kretschmann(mass, times)) <= bound * (1 + 1e-8)


def test_near_the_horizon_the_curvature_is_classical():
    """A Planck-scale correction: 9e-06 of Schwarzschild's at ``T = -0.5`` for ``m = 10^6``."""
    mass, times = 1e6, np.array([-0.5, -2.0, -5.0])
    state = AOS.interior(mass, times)
    relative = AOS.kretschmann(mass, times) / (48 * mass**2 / state["p_c"] ** 3) - 1
    assert np.all(np.abs(relative) < [2e-5, 2e-4, 5e-3])
    assert np.all(np.diff(np.abs(relative)) > 0)


def test_the_white_hole_has_the_black_hole_mass_less_a_log_suppressed_deficit():
    """``m_WH/m - 1 = eps (ln(eps/4) + 1) + O(eps^2)`` with ``eps = gamma^2 delta_b^2``.

    Expanding ``p_c`` at ``cos(delta_b b) = -1`` with ``K = gamma^4
    delta_b^4 / 16``, which is what the prescription makes it. The deficit
    falls as ``m^(-2/3) ln m``: 7e-02 at ``m = 100``, 5.5e-06 at ``10^9``.
    """
    for mass in (1e6, 1e9, 1e12):
        epsilon = (GAMMA * AOS.polymerisation(mass)[0]) ** 2
        predicted = epsilon * (np.log(epsilon / 4) + 1)
        assert AOS.white_hole_mass(mass) / mass - 1 == pytest.approx(predicted, rel=1e-3)
    assert AOS.white_hole_mass(100.0) / 100.0 - 1 == pytest.approx(-0.0712, abs=1e-4)


@pytest.mark.parametrize("mass", [0.0, -1.0])
def test_a_non_positive_mass_is_refused(mass):
    with pytest.raises(ValueError, match="positive"):
        AOS.polymerisation(mass)
