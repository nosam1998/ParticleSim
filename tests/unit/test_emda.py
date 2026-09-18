"""EMDA against the interior it is supposed to change.

Issue #71 asks for mass inflation to be reported and compared with general
relativity. The comparison is not between two rates. In general relativity
the mass function measured behind the Cauchy horizon diverges at a rate that
is the inner surface gravity, and that is measured here from an integrated
null ray. Under EMDA there is no Cauchy horizon at all for any dilaton
coupling above zero, and the transition is discontinuous -- so the second
half of this file is about establishing the absence rather than a number.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from particlesim.scenarios.singularity.interior import (
    PriceTail,
    advance_to_inner_surface,
    blueshift_rate,
    crossing_mass,
    kretschmann_exponent_measured,
    outgoing_ray,
    report_mass_inflation,
)
from particlesim.theories.emda import EMDA, DilatonBlackHole
from particlesim.theories.limits import check_gr_limit
from particlesim.theories.registry import list_theories

MASS, CHARGE = 1.0, 0.9


def rn() -> DilatonBlackHole:
    return DilatonBlackHole(mass=MASS, charge=CHARGE, dilaton_coupling=0.0)


# --- the static solution --------------------------------------------------


def test_zero_dilaton_coupling_is_reissner_nordstrom():
    """``r_pm = M pm sqrt(M^2 - Q^2)``, ``b = 1``, and the areal radius is ``r``."""
    hole = rn()
    root = math.sqrt(MASS**2 - CHARGE**2)
    assert hole.exponent == pytest.approx(1.0)
    assert hole.outer_radius == pytest.approx(MASS + root)
    assert hole.inner_radius == pytest.approx(MASS - root)
    assert hole.areal_radius(1.0) == pytest.approx(1.0)


def test_an_uncharged_hole_is_schwarzschild_at_every_dilaton_coupling():
    """The dilaton is invisible without a Maxwell field to couple to.

    ``r_- = 0`` for every ``a`` once ``Q = 0``, so the whole one-parameter
    family collapses to one metric. That is worth asserting because it is
    what makes the general-relativistic limit below a statement about the
    charge rather than about the coupling.
    """
    for coupling in (0.0, 0.5, 1.0, math.sqrt(3.0)):
        hole = DilatonBlackHole(mass=MASS, charge=0.0, dilaton_coupling=coupling)
        assert hole.outer_radius == pytest.approx(2.0 * MASS)
        assert hole.inner_radius == 0.0
        assert hole.areal_radius(3.0) == pytest.approx(3.0)


def test_the_extremality_bound_moves_with_the_dilaton_coupling():
    """``M^2 >= (1 - a^2) Q^2``, so it dissolves entirely at ``a = 1``.

    A dilaton black hole can carry charge that would over-extremalise a
    Reissner-Nordstrom hole of the same mass. Refusing the configuration that
    has no horizon, rather than returning a complex radius, is the behaviour
    worth pinning.
    """
    with pytest.raises(ValueError, match="extremality"):
        DilatonBlackHole(mass=1.0, charge=1.2, dilaton_coupling=0.0)
    assert DilatonBlackHole(mass=1.0, charge=1.2, dilaton_coupling=1.0).outer_radius == (
        pytest.approx(2.0)
    )


def test_the_heterotic_hole_has_a_surface_gravity_independent_of_its_charge():
    """``kappa_+ = 1/(4M)`` at ``a = 1``, whatever the charge.

    The exponent ``b`` vanishes there, so the factor ``(1 - r_-/r_+)^b`` that
    carries the charge dependence in Reissner-Nordstrom is identically one.
    The Hawking temperature of the heterotic black hole therefore does not
    know about its charge, which Reissner-Nordstrom's plainly does.
    """
    for charge in (0.1, 0.5, 0.9, 1.4):
        heterotic = DilatonBlackHole(mass=1.0, charge=charge, dilaton_coupling=1.0)
        assert heterotic.outer_surface_gravity() == pytest.approx(0.25)
    assert rn().outer_surface_gravity() == pytest.approx(0.21141437933446)
    assert DilatonBlackHole(1.0, 0.1, 0.0).outer_surface_gravity() == pytest.approx(0.2499984217)


def test_only_an_uncoupled_charged_hole_has_a_cauchy_horizon():
    assert rn().has_cauchy_horizon
    assert not DilatonBlackHole(MASS, CHARGE, 0.5).has_cauchy_horizon
    assert not DilatonBlackHole(MASS, 0.0, 0.0).has_cauchy_horizon


def test_an_inner_surface_gravity_is_refused_where_there_is_no_inner_horizon():
    """For ``a > 0`` the derivative of ``f`` at ``r_-`` is infinite, not a number.

    Returning something anyway is the error available here: the formula
    ``(r_+ - r_-)/(2 r_-^2)`` evaluates perfectly well and would describe
    nothing.
    """
    with pytest.raises(ValueError, match="no inner horizon"):
        DilatonBlackHole(MASS, CHARGE, 1.0).inner_surface_gravity()
    assert rn().inner_surface_gravity() == pytest.approx(1.369774385431092)


# --- mass inflation, in general relativity --------------------------------


def test_the_outgoing_ray_approaches_the_cauchy_horizon_at_the_surface_gravity():
    """``-d ln(r - r_-)/dv -> kappa_-``, measured from the integration.

    The rate is a property of the solution of a nonlinear ordinary
    differential equation; the surface gravity is a closed form in ``r_pm``.
    They agree to better than one part in ``1e10``.
    """
    hole = rn()
    assert blueshift_rate(hole) == pytest.approx(hole.inner_surface_gravity(), rel=1e-10)


def test_the_mass_function_inflates_at_the_inner_surface_gravity():
    """``d ln m/dv = kappa_- - p/v``, so one run measures both constants."""
    hole = rn()
    report = report_mass_inflation(hole)
    assert report.inflates
    assert report.arrival_time == math.inf
    assert report.measured_rate == pytest.approx(hole.inner_surface_gravity(), rel=1e-9)


def test_the_rate_does_not_depend_on_the_tail_that_drove_it():
    """Change the Price exponent and the mass changes by orders; the rate does not.

    A rate that moved with ``p`` would mean the fit was absorbing the tail
    rather than measuring the geometry. Across ``p = 8, 12, 16`` the final
    mass spans seventeen decades and the rate agrees to one part in ``1e9``.
    """
    hole = rn()
    reports = [report_mass_inflation(hole, PriceTail(exponent=p)) for p in (8.0, 12.0, 16.0)]
    for report in reports:
        assert report.measured_rate == pytest.approx(hole.inner_surface_gravity(), rel=1e-9)
    masses = [report.final_mass for report in reports]
    assert masses[0] / masses[-1] > 1e15


def test_the_rate_is_geometry_and_the_amplitudes_only_set_the_scale():
    """Flip either sign, move either amplitude six decades: the rate does not move.

    The model has three inputs the geometry does not fix -- the tail's
    amplitude, the outgoing shell's mass, and the sign of each. The mass
    function tracks all of them linearly and the exponential rate tracks none,
    which is what it means for ``kappa_-`` to be a property of the Cauchy
    horizon rather than of the perturbation crossing it.
    """
    hole = rn()
    surface_gravity = hole.inner_surface_gravity()
    cases = [
        report_mass_inflation(hole, PriceTail(amplitude=0.05)),
        report_mass_inflation(hole, PriceTail(amplitude=-0.05)),
        report_mass_inflation(hole, PriceTail(amplitude=1e-6)),
        report_mass_inflation(hole, shell_mass=-1e-3),
        report_mass_inflation(hole, shell_mass=1e-8),
    ]
    for report in cases:
        assert report.measured_rate == pytest.approx(surface_gravity, rel=1e-9)
    assert cases[1].final_mass == pytest.approx(-cases[0].final_mass, rel=1e-12)
    assert cases[2].final_mass / cases[0].final_mass == pytest.approx(2e-5, rel=1e-6)


def test_a_perturbation_that_vanishes_produces_a_mass_that_diverges():
    """The whole phenomenon, as two numbers from one run.

    At ``v = 120`` the ingoing tail that drives the instability has decayed
    to about ``1e-35`` of the hole's mass, and the mass function behind the
    crossing has reached about ``1e32``. Sixty-seven orders of magnitude
    separate cause from effect, which is why the Cauchy horizon does not
    survive an arbitrarily small perturbation.
    """
    tail = PriceTail(exponent=16.0)
    report = report_mass_inflation(rn(), tail)
    assert tail.mass_excess(120.0) < 1e-30
    assert report.final_mass > 1e30


def test_the_crossing_relation_is_ordinary_mass_addition_far_from_the_horizon():
    """``m_D -> m_B + m_C - m_A`` as the crossing radius grows.

    Mass inflation is entirely the ``1/f_A`` factor switching on near the
    inner horizon. Away from it the relation has to degenerate to masses
    simply adding, and the residual falls by a decade per decade of radius --
    first order in ``1/r``, as the expansion of ``f`` says it should.
    """
    hole = rn()
    residuals = [abs(crossing_mass(hole, gap, 0.2, 0.3) - 1.5) for gap in (1e2, 1e4, 1e6)]
    assert residuals[0] < 2e-3
    for coarse, fine in zip(residuals, residuals[1:], strict=False):
        assert coarse / fine == pytest.approx(100.0, rel=0.1)


# --- and what EMDA does instead -------------------------------------------


def test_the_crossing_relation_refuses_a_dilaton_black_hole():
    """There is no mass inflation to report, so no number is returned for it."""
    with pytest.raises(ValueError, match="needs a Cauchy horizon"):
        crossing_mass(DilatonBlackHole(MASS, CHARGE, 1.0), 1e-3, 0.1, 0.1)
    with pytest.raises(ValueError, match="only asymptotes"):
        outgoing_ray(DilatonBlackHole(MASS, CHARGE, 1.0))


def test_the_ray_reaches_the_inner_surface_at_finite_advanced_time():
    """Finite for every ``a > 0``, infinite at ``a = 0``, and monotone between.

    ``dv = 2 dr/|f|`` behaves as ``(r - r_-)^(-b)`` at the inner surface, and
    that integral converges for every ``b < 1``. Since ``b = (1-a^2)/(1+a^2)``
    the borderline is exactly ``a = 0``: the Cauchy horizon is the one case
    that is never reached.
    """
    assert advance_to_inner_surface(rn()) == math.inf
    arrivals = [
        advance_to_inner_surface(DilatonBlackHole(MASS, CHARGE, a)) for a in (0.1, 0.3, 0.6, 1.0)
    ]
    assert all(math.isfinite(value) for value in arrivals)
    for early, late in zip(arrivals, arrivals[1:], strict=False):
        assert early > late


def test_the_arrival_time_diverges_like_one_over_the_missing_exponent():
    """``Delta v (1 - b)`` stays bounded while ``Delta v`` itself runs away.

    The divergence is not incidental: it is the same ``1/(1-b)`` that the
    endpoint integral produces, so the way the finite answer fails at
    ``a = 0`` is exactly the way a Cauchy horizon forms.
    """
    products = []
    for coupling in (0.02, 0.05, 0.1):
        hole = DilatonBlackHole(MASS, CHARGE, coupling)
        arrival = advance_to_inner_surface(hole)
        products.append(arrival * (1.0 - hole.exponent))
    assert advance_to_inner_surface(DilatonBlackHole(MASS, CHARGE, 0.02)) > 500.0
    assert all(0.7 < value < 0.8 for value in products)


def test_the_kretschmann_exponent_is_measured_not_quoted():
    """``2 + 4a^2/(1+a^2)``, read off the curvature tensor at four couplings.

    The scalar is built from the metric by the symbolic machinery and
    evaluated at fifty digits, because the gaps involved would round to zero
    in double precision. Slope of ``ln K`` against ``ln(r - r_-)`` against the
    closed form, to better than ``1e-5``.
    """
    for coupling in (0.0, 0.5, 1.0, math.sqrt(3.0)):
        hole = DilatonBlackHole(MASS, CHARGE, coupling)
        measured = kretschmann_exponent_measured(hole)
        assert measured == pytest.approx(hole.kretschmann_exponent(), abs=1e-5)


def test_the_cauchy_horizon_is_regular_and_the_limit_is_discontinuous():
    """The exponent tends to ``2`` as ``a -> 0`` but is ``0`` at ``a = 0``.

    So the Cauchy horizon's regularity is not a continuous property of the
    dilaton coupling. An arbitrarily small coupling replaces a finite-
    curvature null surface with a singularity at which ``K`` diverges as
    ``(r - r_-)^-2``. That is why the comparison the issue asks for cannot be
    two rates: the arena is removed, not modified.
    """
    assert rn().kretschmann_exponent() == 0.0
    assert kretschmann_exponent_measured(rn()) == pytest.approx(0.0, abs=1e-4)
    approaching = [DilatonBlackHole(MASS, CHARGE, a).kretschmann_exponent() for a in (0.1, 0.01)]
    assert approaching[-1] == pytest.approx(2.0, abs=1e-3)
    assert all(value > 2.0 for value in approaching)


def test_the_report_says_no_rather_than_finding_inflation_everywhere():
    """A diagnostic that could only report mass inflation would prove nothing."""
    report = report_mass_inflation(DilatonBlackHole(MASS, CHARGE, 1.0))
    assert not report.inflates
    assert report.measured_rate is None
    assert math.isfinite(report.arrival_time)
    assert "no Cauchy horizon" in report.notes[0]
    assert report.as_row()["kretschmann_exponent"] == pytest.approx(4.0)


def test_a_tail_that_does_not_decay_is_refused():
    with pytest.raises(ValueError, match="does not decay"):
        PriceTail(exponent=0.0)


# --- the plugin -----------------------------------------------------------


def test_the_plugin_is_registered_and_recovers_schwarzschild():
    """Including the metric family, which the limit harness compares symbolically."""
    assert list_theories()["string.eft4d.emda"] is EMDA
    report = check_gr_limit(EMDA, check_action=False)
    assert report.checked
    assert report.passed, report.reasons
    assert report.metric_family_matches


def test_the_plugin_metric_family_is_the_charged_solution():
    radius, polar = sp.symbols("r theta", positive=True)
    theory = EMDA(dilaton_coupling=1.0, charge=0.9)
    metric = sp.Matrix(theory.metric_family({"mass": 1.0}))
    expected = DilatonBlackHole(1.0, 0.9, 1.0).symbolic_metric(radius, polar)
    assert sp.simplify(metric - expected).is_zero_matrix


def test_the_plugin_predicts_no_mass_inflation_once_the_dilaton_couples():
    """The prediction is what the harness would hold it to, so it has to be stated."""
    assert EMDA(charge=0.9).observable_predictions()["mass_inflation"] is True
    assert EMDA(dilaton_coupling=1.0, charge=0.9).observable_predictions()["mass_inflation"] is (
        False
    )
