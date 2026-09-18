"""Toroidal compactification against what can be derived by hand.

Issue #76 asks that the emitted plugin's couplings match hand-derived values.
Taken literally that is satisfiable by writing the same formula twice, so the
weight here is on the two checks that are not:

* the **no-scale identity**, which is asserted symbolically and comes out at
  exactly ``3`` -- and the tests below show it is a count of logarithms, so
  two moduli give an anti-de Sitter potential, four a positive one, and only
  three the exact cancellation; and
* the **Kaluza-Klein scale**, which the module reads off the same
  :class:`~particlesim.theories.kk.Torus` whose lattice spectrum is
  diagonalised in :mod:`tests.unit.test_kaluza_klein`, and which is compared
  here against the closed form ``min_i 1/R_i`` computed independently.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
import sympy as sp

from particlesim.theories.compactify import (
    DEFAULT_VACUUM,
    Modulus,
    ToroidalCompactification,
    ToroidalVacuum,
    emit_plugin,
    kahler_metric,
    kahler_potential,
    modulus_symbols,
    no_scale_identity,
    scalar_potential,
)
from particlesim.theories.limits import check_gr_limit
from particlesim.theories.registry import list_theories

Z3 = (Fraction(1, 3), Fraction(1, 3), Fraction(-2, 3))
Z2 = (Fraction(1, 2), Fraction(-1, 2), Fraction(0))


def kahler_sector(count: int):
    """``(K, fields, conjugates)`` for ``count`` logarithmic moduli."""
    moduli = [Modulus(f"T{i + 1}", "kahler") for i in range(count)]
    fields, conjugates = modulus_symbols(moduli)
    return kahler_potential(moduli, fields, conjugates), fields, conjugates


# --- the no-scale identity ------------------------------------------------


def test_the_no_scale_identity_is_exactly_three():
    """``K^(i jbar) K_i K_jbar = 3``, symbolically and with no tolerance.

    This is the whole reason the Kahler moduli are flat. It is an exact
    algebraic identity in the fields, not a value at a point, so the
    assertion is against the sympy integer.
    """
    potential, fields, conjugates = kahler_sector(3)
    assert no_scale_identity(potential, fields, conjugates) == 3


def test_the_identity_counts_logarithms_so_three_is_structural():
    """One, two, three, four moduli give one, two, three, four.

    Without this the ``3`` above could be a coincidence of the particular
    metric. It is not: the identity returns the number of logarithms in
    ``K``, which is why a ``T^2 x T^2 x T^2`` -- three factors, three
    logarithms -- is the geometry that gives the no-scale structure.
    """
    for count in (1, 2, 3, 4):
        potential, fields, conjugates = kahler_sector(count)
        assert no_scale_identity(potential, fields, conjugates) == count


def test_a_constant_superpotential_gives_exactly_zero_potential():
    """``V = 0`` identically, as an expression, not at a sampled point.

    The ``3`` from the identity cancels the ``-3|W|^2`` of ``N = 1``
    supergravity term by term, so the result is the sympy zero and the
    Kahler moduli are exactly flat at tree level.
    """
    potential, fields, conjugates = kahler_sector(3)
    flux = sp.Symbol("W0", positive=True)
    assert scalar_potential(potential, flux, fields, conjugates) == 0


def test_the_cancellation_is_a_knife_edge_at_three():
    """Two moduli give ``V < 0``, four give ``V > 0``, three give zero.

    The ``-3`` in the supergravity potential is fixed by four-dimensional
    ``N = 1``, so the sign of ``V`` is set by how the modulus count compares
    with it. Asserting the two signs either side is what makes the zero in
    the middle a cancellation rather than a coincidence.
    """
    flux = sp.Symbol("W0", positive=True)
    signs = {}
    for count in (2, 3, 4):
        potential, fields, conjugates = kahler_sector(count)
        expression = scalar_potential(potential, flux, fields, conjugates)
        point = {symbol: sp.Integer(1) for symbol in fields + conjugates}
        point[flux] = sp.Integer(1)
        signs[count] = sp.sign(expression.subs(point))
    assert signs == {2: -1, 3: 0, 4: 1}


def test_including_the_dilaton_gives_four_and_breaks_the_cancellation():
    """The no-scale structure belongs to the Kahler sector specifically.

    Adding the axio-dilaton -- which enters ``K`` logarithmically just like
    the ``T_i`` -- takes the identity to ``4``, and the same constant
    superpotential then leaves ``V = e^K |W|^2 > 0``. A module that summed
    over whatever moduli were in scope would report a cancellation that does
    not happen.
    """
    moduli = [Modulus(f"T{i + 1}", "kahler") for i in range(3)]
    moduli.append(Modulus("S", "axio_dilaton"))
    fields, conjugates = modulus_symbols(moduli)
    potential = kahler_potential(moduli, fields, conjugates)

    assert no_scale_identity(potential, fields, conjugates) == 4

    flux = sp.Symbol("W0", positive=True)
    expression = scalar_potential(potential, flux, fields, conjugates)
    point = {symbol: sp.Integer(1) for symbol in fields + conjugates}
    assert expression.subs(point).subs(flux, 1) == sp.Rational(1, 16)


def test_the_kahler_metric_is_diagonal_with_the_expected_entries():
    """``K_(i jbar) = delta_(ij) / (T_i + Tbar_i)^2``.

    A factorised ``K`` cannot mix the factors, and the diagonal entry is the
    second derivative of a single logarithm. This is the input the identity
    is built on, so it is worth pinning separately.
    """
    potential, fields, conjugates = kahler_sector(3)
    metric = kahler_metric(potential, fields, conjugates)
    for i in range(3):
        for j in range(3):
            expected = 1 / (fields[i] + conjugates[i]) ** 2 if i == j else 0
            assert sp.simplify(metric[i, j] - expected) == 0


# --- field content --------------------------------------------------------


def test_a_plain_torus_keeps_three_kahler_three_complex_structure_and_the_dilaton():
    """Seven untwisted moduli: ``T_1..3``, ``U_1..3``, ``S``."""
    moduli = ToroidalVacuum().moduli()
    assert [m.name for m in moduli] == ["T1", "T2", "T3", "U1", "U2", "U3", "S"]
    assert sum(1 for m in moduli if m.kind == "kahler") == 3
    assert sum(1 for m in moduli if m.kind == "complex_structure") == 3


def test_a_z3_twist_removes_every_untwisted_complex_structure_modulus():
    """``h^(2,1)`` of the untwisted sector is zero, and the count says so.

    The rule is computed, not tabulated: ``U_i`` survives only when ``2 v_i``
    is an integer, and a ``Z_3`` twist has ``v_i = 1/3``. The three Kahler
    moduli are invariant under any diagonal phase and stay, which is why the
    count falls from seven to four rather than to one.
    """
    moduli = ToroidalVacuum(twist=Z3).moduli()
    assert [m.name for m in moduli] == ["T1", "T2", "T3", "S"]


def test_a_z2_twist_keeps_the_complex_structure_moduli():
    """``z -> -z`` is compatible with any ``tau``, so all three ``U`` survive.

    This is the companion to the ``Z_3`` case and is what stops the orbifold
    rule from being "a twist removes the ``U``". The order of the twist is
    what matters, and the module works it out from ``2 v_i``.
    """
    moduli = ToroidalVacuum(twist=Z2).moduli()
    assert [m.name for m in moduli] == ["T1", "T2", "T3", "U1", "U2", "U3", "S"]
    assert len(ToroidalVacuum(twist=Z2).kahler_moduli()) == 3


def test_the_tree_level_superpotential_is_exactly_zero():
    """No flux and no non-perturbative effects, so ``W = 0`` rather than small."""
    assert ToroidalVacuum().superpotential() == 0


# --- the couplings the issue asks about -----------------------------------


def test_the_gauge_coupling_comes_back_through_the_dilaton():
    """``g_s -> Re S = 1/g_s -> g^2 = 1/Re S``, and the round trip closes.

    The hand derivation is ``f = S`` for the heterotic gauge kinetic
    function, so ``Re f = Re S = 1/g_s`` sits in front of the field strength
    and the coupling is its inverse.
    """
    vacuum = ToroidalVacuum(string_coupling=0.25)
    assert vacuum.dilaton_vev() == pytest.approx(4.0)
    assert vacuum.gauge_coupling() == pytest.approx(0.25)
    assert vacuum.gauge_coupling() * vacuum.dilaton_vev() == pytest.approx(1.0)


def test_the_kaluza_klein_scale_matches_the_closed_form_from_the_other_module():
    """``min_i 1/R_i``, computed here and read off ``Torus.spectrum`` there.

    This is the cross-module check the acceptance is really about. The
    compactification does not carry its own tower formula: it asks the
    Kaluza-Klein module, whose lattice spectrum is diagonalised rather than
    quoted, and the answer has to agree with the hand value.
    """
    radii = (2.0, 3.0, 5.0)
    vacuum = ToroidalVacuum(radii=radii)
    assert vacuum.kaluza_klein_scale() == pytest.approx(min(1.0 / r for r in radii))


def test_the_kaluza_klein_scale_falls_like_one_over_the_radius():
    """Double every radius and the lightest tower state halves.

    A scale that matched one geometry but did not move with it would be a
    number rather than a derivation.
    """
    base = ToroidalVacuum(radii=(1.0, 1.0, 1.0)).kaluza_klein_scale()
    doubled = ToroidalVacuum(radii=(2.0, 2.0, 2.0)).kaluza_klein_scale()
    assert doubled == pytest.approx(base / 2.0)


# --- the emitted plugin ---------------------------------------------------


def test_the_emitted_plugin_carries_the_hand_derived_values():
    """``derived`` matches what the hand derivation gives for the same vacuum."""
    vacuum = ToroidalVacuum(radii=(2.0, 3.0, 5.0), string_coupling=0.25)
    plugin = emit_plugin(vacuum)
    assert plugin.derived["gauge_coupling"] == pytest.approx(0.25)
    assert plugin.derived["kaluza_klein_scale"] == pytest.approx(1.0 / 5.0)
    assert plugin.derived["modulus_count"] == 7


def test_a_twisted_vacuum_emits_a_plugin_with_the_reduced_modulus_count():
    """The orbifold projection reaches the plugin, not just the moduli list."""
    plugin = emit_plugin(ToroidalVacuum(twist=Z3))
    assert plugin.derived["modulus_count"] == 4
    assert plugin().observable_predictions()["modulus_count"] == 4


def test_the_emitted_plugin_passes_the_general_relativity_limit():
    """Emitting a ``Theory`` rather than a dictionary is the point of the exercise.

    Both derived couplings vanish in the limit -- no gauge sector, an
    infinitely heavy tower -- so the plugin has a genuine General Relativity
    point and goes through the same harness every hand-written plugin does.
    """
    report = check_gr_limit(emit_plugin(ToroidalVacuum()), check_action=False)
    assert report.checked
    assert report.passed, report.reasons


def test_the_emitted_plugin_reports_its_moduli_as_unstabilised():
    """``W = 0`` means flat directions, and the prediction says so.

    Reporting a vacuum with unstabilised moduli as a finished model would be
    the substantive error available here, so it is asserted rather than left
    to the docstring.
    """
    predictions = emit_plugin(ToroidalVacuum())().observable_predictions()
    assert predictions["moduli_stabilised"] is False
    assert "untwisted" in emit_plugin(ToroidalVacuum()).validity_statement


def test_the_plugin_name_can_be_chosen():
    """One vacuum per plugin, so the identifier has to be settable."""
    assert emit_plugin(ToroidalVacuum(), name="string.compactify.z3").id == "string.compactify.z3"


# --- what the vacuum refuses ----------------------------------------------


def test_a_vacuum_needs_three_torus_factors():
    with pytest.raises(ValueError, match="three torus factors"):
        ToroidalVacuum(radii=(1.0, 1.0))


def test_radii_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        ToroidalVacuum(radii=(1.0, -1.0, 1.0))


def test_a_strong_string_coupling_is_refused_rather_than_extrapolated_into():
    """At ``g_s >= 1`` the tree-level derivation is not the right one.

    Returning a coupling anyway would be the failure mode worth guarding:
    the number would look fine and mean nothing.
    """
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        ToroidalVacuum(string_coupling=1.5)


def test_a_twist_outside_su_three_is_refused():
    """``sum v_i`` has to be an integer or no supersymmetry survives.

    ``(1/4, 1/4, 1/4)`` sums to ``3/4``, so the holonomy is not in ``SU(3)``
    and the Kahler-potential derivation above has nothing to stand on.
    """
    with pytest.raises(ValueError, match="SU\\(3\\)"):
        ToroidalVacuum(twist=(Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)))


def test_a_twist_needs_three_entries():
    with pytest.raises(ValueError, match="three entries"):
        ToroidalVacuum(twist=(Fraction(1, 2), Fraction(1, 2)))


def test_an_unknown_modulus_kind_is_refused():
    with pytest.raises(ValueError, match="unknown modulus kind"):
        Modulus("X", "winding")


def test_the_default_vacuum_is_registered_and_gated_like_every_other_plugin():
    """``string.compactify.torus`` is discoverable, so ``check-limits`` covers it.

    One plugin per vacuum means the registry has to name one, and an
    unregistered emitter would be checked only by this file. Registering the
    default vacuum puts it through the same gate as the hand-written plugins.
    """
    assert list_theories()["string.compactify.torus"] is ToroidalCompactification
    report = check_gr_limit(ToroidalCompactification, check_action=False)
    assert report.checked
    assert report.passed, report.reasons
    assert ToroidalCompactification.derived["gauge_coupling"] == pytest.approx(
        DEFAULT_VACUUM.gauge_coupling()
    )
