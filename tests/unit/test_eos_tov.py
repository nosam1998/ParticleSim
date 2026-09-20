"""Equations of state, and stellar structure checked against what is exact.

Issue #58's stated acceptance -- a TOV star stable for ten dynamical times --
needs a hydrodynamic evolution, which belongs to #57 and #59. What is
checkable from the equation-of-state side is sharper than a tolerance, and
this file is built around the four things that are exactly true: the first
law, the causal density limit, the Schwarzschild interior solution, and the
Lane-Emden radius of an ``n = 1`` polytrope.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from particlesim.matter.eos import (
    EquationOfState,
    IdealGas,
    PiecewisePolytrope,
    Polytrope,
    Tabulated,
    UniformDensity,
    as_plugin,
)
from particlesim.matter.tov import (
    BUCHDAHL_LIMIT,
    Star,
    mass_radius_sequence,
    newtonian_polytrope_radius,
    solve_tov,
    solve_uniform_density,
    uniform_density_central_pressure,
    uniform_density_pressure,
)
from particlesim.theories.base import TheoryStack
from particlesim.theories.gr import GeneralRelativity

STIFF = Polytrope(polytropic_constant=100.0, gamma=2.0)
SOFT = Polytrope(polytropic_constant=100.0, gamma=1.8)
PIECEWISE = PiecewisePolytrope(transitions=(1e-3,), exponents=(1.5, 2.5), polytropic_constant=100.0)


# --- the first law ---------------------------------------------------------


def test_every_equation_of_state_satisfies_the_first_law():
    """``dh = dp/rho`` along the cold branch, by central difference.

    This relates three of the quantities a plugin returns, so a
    ``specific_energy`` that does not match its own ``pressure`` fails here
    and nowhere else -- each function on its own looks perfectly reasonable.
    """
    for eos in (STIFF, SOFT, IdealGas(100.0, 2.0), PIECEWISE):
        for density in (1e-5, 1e-4, 1e-2):
            step = density * 1e-6
            enthalpy = (
                float(eos.enthalpy(density + step)) - float(eos.enthalpy(density - step))
            ) / (2.0 * step)
            pressure = (
                float(eos.pressure(density + step)) - float(eos.pressure(density - step))
            ) / (2.0 * step)
            assert enthalpy == pytest.approx(pressure / density, rel=1e-6)


def test_the_tabulated_reader_integrates_its_own_energy():
    """``eps`` comes from ``d eps = (p/rho^2) d rho``, not from a third column.

    So a table sampled from a polytrope reconstructs that polytrope's
    specific energy -- up to the reference constant, which is physics, and to
    interpolation accuracy, which is arithmetic.
    """
    densities = np.logspace(-6, -2, 400)
    table = Tabulated.from_polytrope(STIFF, densities)
    probes = np.array([1e-5, 1e-4, 1e-3])
    reference = float(STIFF.specific_energy(densities[0]))
    expected = np.asarray(STIFF.specific_energy(probes)) - reference
    assert table.specific_energy(probes) == pytest.approx(expected, rel=2e-3)


def test_log_interpolation_of_a_power_law_is_exact():
    """Which is why the pressure round-trips to round-off, not to a tolerance.

    A linear interpolant between points a decade apart would be wrong by
    tens of percent in the middle; in log-log a power law is a straight line.
    """
    table = Tabulated.from_polytrope(STIFF, np.logspace(-6, -2, 16))
    probes = np.array([3.7e-6, 2.2e-5, 8.1e-4])
    assert table.pressure(probes) == pytest.approx(STIFF.pressure(probes), rel=1e-12)
    assert table.density_from_pressure(STIFF.pressure(probes)) == pytest.approx(probes, rel=1e-12)


# --- causality -------------------------------------------------------------


def test_a_polytrope_stiffer_than_two_is_superluminal_above_a_known_density():
    """``rho_max = [(G-1)/(G K (G-2))]^(1/(G-1))``, where ``c_s^2`` is exactly one.

    Not a bound to be careful about: an exact density, at which the closed
    form returns 1.00000000. Below ``Gamma = 2`` the sound speed approaches
    one from beneath and never crosses, so the limit is infinite.
    """
    for gamma in (2.5, 3.0):
        eos = Polytrope(polytropic_constant=100.0, gamma=gamma)
        limit = eos.causal_density_limit()
        expected = ((gamma - 1.0) / (gamma * 100.0 * (gamma - 2.0))) ** (1.0 / (gamma - 1.0))
        assert limit == pytest.approx(expected, rel=1e-12)
        assert float(eos.sound_speed_squared(limit)) == pytest.approx(1.0, abs=1e-9)
        assert float(eos.sound_speed_squared(limit * 1.5)) > 1.0

    for gamma in (1.5, 2.0):
        assert Polytrope(100.0, gamma).causal_density_limit() == math.inf
        assert float(Polytrope(100.0, gamma).sound_speed_squared(1e6)) < 1.0


def test_a_superluminal_density_is_refused_rather_than_answered():
    eos = Polytrope(polytropic_constant=100.0, gamma=3.0)
    eos.check_causal(eos.causal_density_limit() * 0.5)
    with pytest.raises(ValueError, match="superluminal above"):
        eos.check_causal(eos.causal_density_limit() * 2.0)


def test_the_closed_form_sound_speed_agrees_with_the_numerical_one():
    """A plugin that overrides the default must not disagree with it.

    The base class differentiates ``pressure`` and ``energy_density``; the
    polytrope supplies a formula. They are independent routes to the same
    number and the suite holds them together.
    """
    densities = np.array([1e-5, 1e-3, 1e-1])
    for eos in (STIFF, SOFT, Polytrope(50.0, 2.5)):
        analytic = np.asarray(eos.sound_speed_squared(densities))
        numeric = np.asarray(EquationOfState.sound_speed_squared(eos, densities))
        assert analytic == pytest.approx(numeric, rel=1e-6)


# --- the piecewise construction --------------------------------------------


def test_continuity_fixes_the_segment_constants_rather_than_the_user():
    """``K_(i+1) = K_i rho_i^(G_i - G_(i+1))``, and both ``p`` and ``eps`` join.

    A table of independently chosen constants is a discontinuous equation of
    state that still evaluates, which is exactly the failure worth designing
    out: nothing downstream would report the jump.
    """
    boundary = PIECEWISE.transitions[0]
    constants = PIECEWISE.constants
    assert constants[1] == pytest.approx(
        constants[0] * boundary ** (PIECEWISE.exponents[0] - PIECEWISE.exponents[1])
    )
    below, above = boundary * (1.0 - 1e-9), boundary * (1.0 + 1e-9)
    assert float(PIECEWISE.pressure(below)) == pytest.approx(
        float(PIECEWISE.pressure(above)), rel=1e-7
    )
    assert float(PIECEWISE.specific_energy(below)) == pytest.approx(
        float(PIECEWISE.specific_energy(above)), rel=1e-7
    )


def test_the_piecewise_inverse_is_the_inverse():
    densities = np.array([1e-4, 5e-4, 5e-3, 2e-2])
    assert PIECEWISE.density_from_pressure(PIECEWISE.pressure(densities)) == pytest.approx(
        densities, rel=1e-10
    )


def test_a_malformed_piecewise_specification_is_refused():
    with pytest.raises(ValueError, match="exponents"):
        PiecewisePolytrope(transitions=(1e-3, 1e-2), exponents=(2.0, 2.0))
    with pytest.raises(ValueError, match="increasing"):
        PiecewisePolytrope(transitions=(1e-2, 1e-3), exponents=(2.0, 2.0, 2.0))
    with pytest.raises(ValueError, match="must exceed one"):
        PiecewisePolytrope(transitions=(1e-3,), exponents=(1.0, 2.0))


def test_a_table_that_is_not_monotonic_is_refused():
    """A falling pressure has a negative sound speed and is not matter."""
    densities = np.logspace(-6, -2, 32)
    pressures = np.asarray(STIFF.pressure(densities))
    pressures[10] = pressures[9] * 0.5
    with pytest.raises(ValueError, match="strictly increasing"):
        Tabulated(densities=densities, pressures=pressures)
    with pytest.raises(ValueError, match="at least two points"):
        Tabulated(densities=np.array([1e-3]), pressures=np.array([1e-6]))


# --- the ideal gas and the incompressible case ------------------------------


def test_the_ideal_gas_carries_a_thermal_branch_and_its_own_isentrope():
    """``p = (G-1) rho eps`` with ``eps`` free, and the cold branch inherited.

    A star built on the cold branch and evolved on the thermal one has to be
    the same gas, which is why both live in one plugin.
    """
    gas = IdealGas(polytropic_constant=100.0, gamma=2.0)
    assert float(gas.thermal_pressure(1e-3, 0.1)) == pytest.approx(1e-4)
    cold = float(gas.specific_energy(1e-3))
    assert float(gas.thermal_pressure(1e-3, cold)) == pytest.approx(float(gas.pressure(1e-3)))
    assert float(gas.thermal_sound_speed_squared(1e-3, cold)) == pytest.approx(
        float(gas.sound_speed_squared(1e-3)), rel=1e-12
    )


def test_incompressible_matter_refuses_a_pressure_density_relation():
    """It has none, and saying so beats returning something."""
    eos = UniformDensity(density=1e-3)
    with pytest.raises(NotImplementedError, match="no p\\(rho\\)"):
        eos.pressure(1e-3)
    assert float(eos.energy_density_from_pressure(1.0)) == pytest.approx(1e-3)
    assert float(eos.sound_speed_squared(1e-3)) == math.inf


# --- stellar structure, against the closed form -----------------------------


@pytest.mark.parametrize("compactness", [0.2, 0.5, 0.8])
def test_the_incompressible_star_matches_the_schwarzschild_interior(compactness):
    """Radius, mass *and* the whole profile, to one part in ``1e9``.

    The integrator is compared with arithmetic rather than with another
    integrator, at a compactness of 0.8 among others -- far beyond any real
    star, and well inside where a small error would show.
    """
    density = 1e-3
    star = solve_uniform_density(density, compactness, samples=80)
    radius = math.sqrt(compactness / (8.0 / 3.0 * math.pi * density))
    mass = 4.0 / 3.0 * math.pi * density * radius**3

    assert star.radius == pytest.approx(radius, rel=1e-9)
    assert star.mass == pytest.approx(mass, rel=1e-9)
    assert star.compactness == pytest.approx(compactness, rel=1e-9)

    inside = star.radii < 0.98 * radius
    exact = uniform_density_pressure(star.radii[inside], radius, mass, density)
    assert star.pressures[inside] == pytest.approx(exact, rel=1e-8)


def test_the_newtonian_limit_is_approached_at_first_order_in_the_compactness():
    """``R -> sqrt(pi K / 2)``, with the error tracking ``2M/R``.

    A tolerance would pass for any integrator landing nearby. What is
    asserted is the *rate*: the departure from the Lane-Emden radius divided
    by the compactness holds near 0.7 across three decades of central
    density, which is the relativistic correction general relativity
    predicts rather than a number that happens to be small.
    """
    newtonian = newtonian_polytrope_radius(100.0)
    ratios = []
    for central in (1e-4, 1e-5, 1e-6):
        star = solve_tov(STIFF, float(STIFF.pressure(central)), maximum_radius=50.0 * newtonian)
        ratios.append(abs(star.radius / newtonian - 1.0) / star.compactness)
    for ratio in ratios:
        assert ratio == pytest.approx(0.71, abs=0.03)


def test_the_lane_emden_radius_does_not_depend_on_the_central_density():
    """The peculiarity of ``Gamma = 2``, and why it makes a clean check.

    An integrator with the equations slightly wrong would not reproduce a
    radius that does not move.
    """
    newtonian = newtonian_polytrope_radius(100.0)
    radii = [
        solve_tov(STIFF, float(STIFF.pressure(rho)), maximum_radius=50.0 * newtonian).radius
        for rho in (1e-6, 3e-7)
    ]
    assert radii[0] == pytest.approx(radii[1], rel=1e-3)
    assert radii[1] == pytest.approx(newtonian, rel=1e-3)
    assert newtonian_polytrope_radius(400.0) == pytest.approx(2.0 * newtonian)


# --- the Buchdahl bound -----------------------------------------------------


def test_the_central_pressure_diverges_at_the_buchdahl_bound():
    """``2M/R = 8/9`` is where the closed form's denominator vanishes.

    No static star of any equation of state is more compact, which is why
    the bound is a property of the solution rather than of the matter.
    """
    density = 1e-3
    pressures = []
    for compactness in (0.80, 0.86, 0.88):
        radius = math.sqrt(compactness / (8.0 / 3.0 * math.pi * density))
        mass = 4.0 / 3.0 * math.pi * density * radius**3
        pressures.append(uniform_density_central_pressure(radius, mass, density))
    for lower, higher in zip(pressures, pressures[1:], strict=False):
        assert higher > lower
    assert pressures[-1] > 10.0 * pressures[0]
    assert BUCHDAHL_LIMIT == pytest.approx(8.0 / 9.0)


def test_a_star_past_the_bound_is_refused_rather_than_integrated():
    with pytest.raises(ValueError, match="Buchdahl"):
        solve_uniform_density(1e-3, BUCHDAHL_LIMIT)
    with pytest.raises(ValueError, match="no static star is more compact"):
        uniform_density_pressure(0.0, 1.0, 0.5, 1e-3)


# --- the sequence and the reporting ----------------------------------------


def test_a_mass_radius_sequence_rises_on_the_stable_branch():
    stars = mass_radius_sequence(STIFF, [1e-6, 3e-6, 1e-5, 3e-5])
    masses = [star.mass for star in stars]
    for lighter, heavier in zip(masses, masses[1:], strict=False):
        assert heavier > lighter
    for star in stars:
        assert 0.0 < star.compactness < BUCHDAHL_LIMIT


def test_a_star_reports_its_compactness_and_dynamical_time():
    star = Star(radius=10.0, mass=1.0, central_pressure=1e-4)
    assert star.compactness == pytest.approx(0.2)
    assert star.dynamical_time == pytest.approx(math.sqrt(1000.0))
    assert set(star.as_row()) == {
        "radius",
        "mass",
        "compactness",
        "central_pressure",
        "dynamical_time",
    }


def test_the_integrator_refuses_nonsense_and_reports_a_missing_surface():
    with pytest.raises(ValueError, match="central pressure must be positive"):
        solve_tov(STIFF, 0.0)
    with pytest.raises(ValueError, match="had not fallen to the surface"):
        solve_tov(STIFF, float(STIFF.pressure(1e-6)), maximum_radius=1.0)


# --- the stack -------------------------------------------------------------


def test_an_equation_of_state_composes_into_a_theory_stack():
    """Issue #58's third task, and the consistency it actually buys.

    The stack checks the frame, which is the thing an equation of state can
    get wrong invisibly: a constitutive relation written against one metric
    and evaluated on another is wrong by the conformal factor, and both
    halves look fine on their own.
    """
    plugin = as_plugin(STIFF)
    stack = TheoryStack(gravity=GeneralRelativity(), eos=plugin())
    assert stack.describe()["eos"]["id"] == "eos.polytrope"
    assert plugin.tier == "C"
    assert plugin().gr_limit() == {}
    assert plugin().observable_predictions()["causal_density_limit"] == math.inf


def test_a_frame_mismatch_between_gravity_and_matter_is_caught():
    with pytest.raises(ValueError, match="jordan frame"):
        TheoryStack(gravity=GeneralRelativity(), eos=as_plugin(STIFF, frame="jordan")())


def test_a_polytrope_says_where_it_stops_being_causal():
    assert "causal at every density" in Polytrope(100.0, 2.0).validity_statement
    assert "causal only below" in Polytrope(100.0, 3.0).validity_statement
