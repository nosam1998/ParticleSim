"""The NS-NS reduction, checked rather than quoted.

Issue #70 asks that the symbolic reduction match the literature action. Three
things have to hold for that to mean anything: the frame change differs from
an identity by a total derivative, the dualisation's factor of ``3!`` is
computed rather than assumed, and the two-form's Bianchi identity becomes the
axion's equation of motion. The third is what a duality *is*.
"""

from __future__ import annotations

import pytest
import sympy as sp

from particlesim.theories.dilaton import (
    Dilaton,
    axion_equation,
    axion_kinetic_coefficient,
    divergence,
    dual_three_form,
    einstein_frame_density,
    exterior_derivative,
    frame_boundary_current,
    levi_civita_contraction,
    string_frame_density,
)
from particlesim.theories.limits import check_gr_limit
from particlesim.theories.registry import list_theories

COORDS = list(sp.symbols("t x y z", real=True))


@pytest.fixture(scope="module")
def friedmann():
    """``(g_E, g_S, phi, chi)`` on a Friedmann metric, where all of this is tractable."""
    time = COORDS[0]
    scale = sp.Function("a", positive=True)(time)
    dilaton = sp.Function("phi", real=True)(time)
    axion = sp.Function("chi", real=True)(time)
    einstein = sp.diag(-1, scale**2, scale**2, scale**2)
    return einstein, sp.exp(2 * dilaton) * einstein, dilaton, axion


# --- the frame change -----------------------------------------------------


def test_the_frames_differ_by_a_total_derivative(friedmann):
    """Not term by term -- by the divergence of ``-6 sqrt(-g) grad phi``, exactly.

    This is the statement that makes the two actions the same theory. A
    reduction that merely reproduced the terms of the literature action would
    be a *different* action containing the same terms, and no comparison of
    coefficients would notice.
    """
    einstein, string, dilaton, _ = friedmann

    difference = string_frame_density(string, COORDS, dilaton) - einstein_frame_density(
        einstein, COORDS, dilaton
    )
    boundary = divergence(frame_boundary_current(einstein, COORDS, dilaton), COORDS)
    assert sp.simplify(difference - boundary) == 0


def test_the_boundary_term_is_not_zero(friedmann):
    """Otherwise the test above would pass for the wrong reason.

    On Friedmann it is ``d/dt(6 a^3 phidot)``: a real total derivative, not a
    vanishing one, so the two densities genuinely are not equal pointwise.
    """
    einstein, _, dilaton, _ = friedmann
    current = frame_boundary_current(einstein, COORDS, dilaton)
    assert sp.simplify(current[0]) != 0

    time, scale = COORDS[0], sp.Function("a", positive=True)(COORDS[0])
    expected = sp.diff(6 * scale**3 * sp.diff(dilaton, time), time)
    assert sp.simplify(divergence(current, COORDS) - expected) == 0


def test_a_constant_dilaton_leaves_the_frames_identical(friedmann):
    """No dilaton gradient, no boundary term, and the string frame is Einstein's."""
    einstein, _, _, _ = friedmann
    constant = sp.Symbol("phi0", real=True)
    current = frame_boundary_current(einstein, COORDS, constant)
    assert all(sp.simplify(component) == 0 for component in current)


# --- the dualisation ------------------------------------------------------


def test_the_levi_civita_contraction_is_three_factorial():
    """Computed from the symbol, because it is the whole ``1/12 -> 1/2``."""
    assert levi_civita_contraction(4) == 6
    assert axion_kinetic_coefficient(4) == sp.Rational(1, 2)


def test_the_dilaton_exponent_flips_under_dualisation():
    """``e^(-4 phi)`` on the two-form, ``e^(+4 phi)`` on the axion.

    The sign reversal is the point of the dual description -- the axion is
    strongly coupled where the two-form was weak -- and it is the easiest
    thing in this reduction to get backwards, since every other coefficient
    survives unchanged.
    """
    predictions = Dilaton(coupling=0.3).observable_predictions()
    assert predictions["two_form_kinetic_exponent"] == -4
    assert predictions["axion_kinetic_exponent"] == +4


def test_the_bianchi_identity_is_the_axion_equation_of_motion(friedmann):
    """What a duality *is*, checked exactly rather than by inspection.

    ``dH = 0`` holds identically for ``H = dB``. After substituting the dual
    it becomes ``grad_mu(e^(4 phi) grad^mu chi) = 0`` -- a field equation,
    not an identity. The two expressions come out proportional with a
    constant ratio of ``-1``, which is the orientation convention of the
    Levi-Civita symbol and not a physical sign: what matters is that the
    ratio is a *constant*, so one vanishes exactly when the other does.
    """
    einstein, _, dilaton, axion = friedmann

    bianchi = exterior_derivative(dual_three_form(einstein, COORDS, dilaton, axion), COORDS)
    equation = axion_equation(einstein, COORDS, dilaton, axion)

    assert sp.simplify(bianchi) != 0  # there is something to compare
    assert sp.simplify(bianchi + equation) == 0


def test_a_constant_axion_gives_no_three_form(friedmann):
    """``H`` is built from ``d chi``, so a frozen axion is no field at all."""
    einstein, _, dilaton, _ = friedmann
    components = dual_three_form(einstein, COORDS, dilaton, sp.Symbol("chi0", real=True))
    assert all(sp.simplify(value) == 0 for value in components.values())


def test_the_einstein_density_carries_the_axion_with_a_half(friedmann):
    """``- e^(4 phi) (grad chi)^2 / 2``, the coefficient the contraction fixes."""
    einstein, _, dilaton, axion = friedmann
    inverse = einstein.inv()
    gradient = sum(
        inverse[i, j] * sp.diff(axion, COORDS[i]) * sp.diff(axion, COORDS[j])
        for i in range(4)
        for j in range(4)
    )

    without = einstein_frame_density(einstein, COORDS, dilaton)
    with_axion = einstein_frame_density(einstein, COORDS, dilaton, axion_squared=gradient)
    root = sp.sqrt(-einstein.det())
    expected = -root * sp.exp(4 * dilaton) * gradient / 2
    assert sp.simplify(with_axion - without - expected) == 0


# --- the plugin -----------------------------------------------------------


def test_the_plugin_is_registered_and_reduces_to_gr():
    assert list_theories()["string.eft4d.dilaton"] is Dilaton
    report = check_gr_limit(Dilaton, check_action=False)
    assert report.checked
    assert report.passed, report.reasons


def test_the_limit_is_at_zero_coupling():
    """GR where a test can evaluate it, as elsewhere in this repository."""
    assert Dilaton().gr_limit() == {"coupling": 0.0}
    assert Dilaton(coupling=0.0).observable_predictions()["dilaton_decoupled"]
    assert not Dilaton(coupling=0.7).observable_predictions()["dilaton_decoupled"]


def test_the_plugin_declares_the_einstein_frame():
    """Which is why its effective stress-energy is GR's.

    The Einstein-frame action has a canonical Einstein-Hilbert term by
    construction, so the dilaton and axion are matter rather than a
    modification of gravity. In the string frame that would be false, and
    the declared frame is what makes the statement checkable.
    """
    assert Dilaton.frame == "einstein"
    einstein = sp.Matrix(4, 4, lambda i, j: sp.Symbol(f"G{i}{j}"))
    stress = Dilaton(coupling=0.5).effective_stress_energy(einstein, None)
    assert sp.simplify(stress - einstein / (8 * sp.pi)) == sp.zeros(4, 4)
