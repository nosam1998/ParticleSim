"""The exact-CFT reference module, and the harness screening that uses it.

Issue #75 asks that the harness score a hypothesis against the module before
any run. What makes that worth having is that the module can say *no*: string
theory resolves the conical singularity of ``C/Z_N`` and does not resolve the
null orbifold's, so a hypothesis claiming universal singularity resolution is
refuted by a dictionary lookup. Everything the module asserts is recomputed
here rather than compared against a copy of itself -- the Lorentz elements are
checked to be Lorentz elements, and their invariants against closed forms the
module never evaluates.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from particlesim.scenarios.singularity.harness import evaluate, prescreen
from particlesim.theories.base import Coupling, Theory
from particlesim.theories.exact_cft import (
    CRYSTALLOGRAPHIC,
    MINKOWSKI,
    REFERENCES,
    BoostOrbifold,
    ConicalOrbifold,
    CosetBlackHole,
    Reference,
    catalogue,
    score_hypothesis,
)

UNTWISTED = Fraction(-1, 12)


class _Universalist(Theory):
    """Claims every singularity is resolved -- which string theory refutes."""

    id = "test.universalist"
    tier = "B"
    couplings = [Coupling("unused", 1.0)]
    provenance = "test fixture making a claim the reference module already answers"
    validity_statement = "none; this plugin overreaches on purpose"

    def gr_limit(self) -> dict[str, float]:
        return {"unused": 1.0}

    def observable_predictions(self) -> dict[str, object]:
        return {"all_singularities_resolved": True, "bounce": True}


class _Unrunnable(_Universalist):
    """Same claim, and dynamics that cannot be integrated at all."""

    id = "test.unrunnable"

    def reduced_equations(self, symmetry: str):
        raise RuntimeError("this plugin cannot be evolved")


# --- the conical orbifold, which strings do not mind ----------------------


def test_the_twisted_ground_state_is_lifted_above_the_untwisted_one():
    """``-1/12 + v(1-v)/2``, in exact rationals, and always above ``-1/12``.

    A twisted sector costs energy rather than producing a divergence, which
    is the mechanism by which the conical singularity carries states instead
    of breaking the theory.
    """
    for order in range(2, 9):
        orbifold = ConicalOrbifold(order)
        for twist in orbifold.twists():
            assert orbifold.ground_state_energy(twist) > UNTWISTED


def test_the_two_ends_of_the_ground_state_formula_are_the_known_bosons():
    """``v = 0`` is two periodic bosons, ``v = 1/2`` two antiperiodic ones.

    ``-1/12 = 2 x (-1/24)`` and ``1/24 = 2 x 1/48``. Without these the
    formula could carry any normalisation and still look plausible.
    """
    orbifold = ConicalOrbifold(2)
    assert orbifold.ground_state_energy(Fraction(0)) == 2 * Fraction(-1, 24)
    assert orbifold.ground_state_energy(Fraction(1, 2)) == 2 * Fraction(1, 48)
    assert ConicalOrbifold(3).ground_state_energy(Fraction(1, 3)) == Fraction(1, 36)


def test_a_twist_outside_one_turn_is_refused():
    with pytest.raises(ValueError, match="fraction of a turn"):
        ConicalOrbifold(3).ground_state_energy(Fraction(4, 3))


def test_the_sectors_close_under_the_modular_group():
    """``N^2`` sectors, and ``S`` and ``T`` map the set to itself."""
    for order in range(2, 9):
        orbifold = ConicalOrbifold(order)
        assert len(orbifold.sectors()) == order**2
        assert orbifold.is_modular_closed()


def test_dropping_the_twisted_sectors_breaks_modular_invariance():
    """The negative control: closure is why the twisted sectors have to exist.

    Keeping only the untwisted boundary conditions ``g = 0`` leaves a set
    that ``S`` immediately escapes, since ``S`` exchanges the two cycles and
    a nonzero ``h`` becomes a nonzero ``g``. So the twisted sectors are not
    an addition to the theory, they are a consistency requirement.
    """
    orbifold = ConicalOrbifold(3)
    untwisted_only = {(0, h) for h in range(3)}
    escapes = [s for s in untwisted_only if not orbifold.modular_images(s) <= untwisted_only]
    assert escapes


def test_the_fixed_point_count_comes_from_the_rotation_matrix():
    """``|det(1 - theta)|`` per plane: 4, 3, 2, 1 on ``T^2``, and 27 on ``T^6/Z_3``.

    Computed from ``theta`` rather than tabulated. The 27 is the same number
    as the twisted sectors of ``T^6/Z_3``, which is where most of that
    orbifold's chiral matter lives -- and what ``string.compactify``'s
    untwisted-only counts leave out.
    """
    assert [ConicalOrbifold(n).fixed_points(2) for n in CRYSTALLOGRAPHIC] == [4, 3, 2, 1]
    assert ConicalOrbifold(3).fixed_points(6) == 27
    assert ConicalOrbifold(2).fixed_points(6) == 64


def test_the_image_sum_over_a_finite_group_is_finite():
    """``N`` terms, which is the whole reason this background is harmless."""
    for order in (2, 3, 7):
        assert ConicalOrbifold(order).images() == order
    assert ConicalOrbifold(3).status == "resolved"


def test_an_orbifold_needs_an_order_of_at_least_two():
    with pytest.raises(ValueError, match="order at least 2"):
        ConicalOrbifold(1)


# --- the infinite groups, which strings do mind ---------------------------


def test_the_identifications_really_are_lorentz_transformations():
    """Preserve the metric and compose additively in the parameter.

    Both are checked at a power the closed form was not built at, so the
    matrices are being tested rather than restated. Without this the
    invariants below would be arithmetic on an arbitrary matrix.
    """
    for kind in ("boost", "null_rotation"):
        orbifold = BoostOrbifold(rapidity=0.3, kind=kind)
        element = orbifold.element(3)
        assert np.max(np.abs(element.T @ MINKOWSKI @ element - MINKOWSKI)) < 1e-13
        composed = orbifold.element(1) @ orbifold.element(2)
        assert np.max(np.abs(composed - element)) < 1e-13


def test_the_null_orbifold_invariant_grows_polynomially():
    """``s_n = 4 + n^2 b^2`` exactly, from the matrix.

    The parabolic element is the one that makes the null orbifold *nearly*
    all right: the growth is only polynomial. It is still unbounded, and the
    group is still infinite, which is enough.
    """
    orbifold = BoostOrbifold(rapidity=0.3, kind="null_rotation")
    for n in range(8):
        assert orbifold.image_invariant(n) == pytest.approx(4.0 + (n * 0.3) ** 2, rel=1e-12)
    assert orbifold.growth == "polynomial"


def test_the_milne_invariant_grows_exponentially():
    """``s_n = 2 + 2 cosh(n b)``, which is worse and by a widening margin."""
    orbifold = BoostOrbifold(rapidity=0.3, kind="boost")
    for n in range(8):
        assert orbifold.image_invariant(n) == pytest.approx(
            2.0 + 2.0 * math.cosh(n * 0.3), rel=1e-12
        )
    assert orbifold.growth == "exponential"


def test_the_boost_overtakes_the_null_rotation():
    """Same parameter, and the ratio runs away rather than settling.

    A hyperbolic element and a parabolic one are different conjugacy classes,
    not different constants, and the invariants have to show it.
    """
    milne = BoostOrbifold(rapidity=0.4, kind="boost")
    null = BoostOrbifold(rapidity=0.4, kind="null_rotation")
    ratios = [milne.image_invariant(n) / null.image_invariant(n) for n in (5, 15, 30)]
    for early, late in zip(ratios, ratios[1:], strict=False):
        assert late > early
    assert ratios[-1] > 100.0


def test_both_infinite_quotients_diverge_and_are_marked_unstable():
    for kind in ("boost", "null_rotation"):
        orbifold = BoostOrbifold(kind=kind)
        assert orbifold.image_sum_diverges(power=2.0)
        assert orbifold.status == "unstable"


def test_an_identification_needs_a_nonzero_parameter():
    with pytest.raises(ValueError, match="must be positive"):
        BoostOrbifold(rapidity=0.0)
    with pytest.raises(ValueError, match="unknown identification"):
        BoostOrbifold(kind="rotation")


# --- the two-dimensional black hole ---------------------------------------


def test_the_critical_level_is_exactly_nine_quarters():
    """``c = 3k/(k-2) - 1 = 26`` solved symbolically, and ``c(9/4) = 26`` in rationals."""
    assert CosetBlackHole.critical_level() == Fraction(9, 4)
    assert CosetBlackHole(Fraction(9, 4)).central_charge() == 26


def test_the_large_level_limit_is_the_semiclassical_two():
    """Two bosons: the metric and the dilaton, and nothing else."""
    assert CosetBlackHole.semiclassical_charge() == 2


def test_the_alpha_prime_corrections_are_six_twelve_twenty_four():
    """``c = 2 + 6/k + 12/k^2 + 24/k^3``, each coefficient twice the last.

    The leading ``2`` is the semiclassical answer; everything after it is an
    alpha-prime correction, so a background claiming to be exact while
    stopping at the metric is missing the ``6/k``.
    """
    coefficients = CosetBlackHole.alpha_prime_coefficients(5)
    assert coefficients == [2, 6, 12, 24, 48]
    for coarse, fine in zip(coefficients[1:], coefficients[2:], strict=False):
        assert fine == 2 * coarse


def test_a_level_at_or_below_two_is_refused():
    with pytest.raises(ValueError, match="must exceed 2"):
        CosetBlackHole(Fraction(2))


# --- the catalogue and the scoring ----------------------------------------


def test_the_catalogue_covers_the_four_backgrounds_with_valid_statuses():
    known = catalogue()
    assert set(known) == {
        "conical_orbifold",
        "two_dimensional_black_hole",
        "null_orbifold",
        "milne",
    }
    assert {r.status for r in REFERENCES} == {"resolved", "exactly_solved", "unstable"}
    assert all(r.claim and r.evidence for r in REFERENCES)


def test_an_unknown_status_is_refused():
    with pytest.raises(ValueError, match="unknown status"):
        Reference("x", "conical", "probably fine", "claim", "evidence")


def test_a_universal_resolution_claim_is_contradicted():
    """The module's reason for existing, in one call.

    String theory resolves the conical singularity and does *not* resolve the
    null orbifold's -- that background is unstable instead. So "every
    singularity is resolved" is refuted without evolving anything.
    """
    score = score_hypothesis({"all_singularities_resolved": True})
    assert score.refuted
    assert "null_orbifold is unstable" in score.contradicted[0]


def test_the_scoring_can_agree_as_well_as_disagree():
    """A scorer that only ever objects would be as useless as one that never does."""
    score = score_hypothesis(
        {"conical_singularity_resolved": True, "null_singularity_resolved": False}
    )
    assert not score.refuted
    assert len(score.supported) == 2


def test_getting_the_conical_case_backwards_is_also_caught():
    """Claiming strings break down on ``C/Z_N`` is wrong in the other direction."""
    score = score_hypothesis({"conical_singularity_resolved": False})
    assert score.refuted


def test_claims_the_module_has_nothing_to_say_about_are_left_alone():
    """Silence is recorded as silence rather than scored as agreement."""
    score = score_hypothesis({"bounce": True, "echoes_in_the_ringdown": False})
    assert not score.refuted
    assert set(score.unaddressed) == {"bounce", "echoes_in_the_ringdown"}
    assert score.as_row()["refuted"] is False


# --- the harness step -----------------------------------------------------


def test_the_prescreen_runs_nothing_at_all():
    """Scored even when the dynamics cannot be integrated.

    This is what makes "before any evolution runs" checkable rather than a
    claim about the order of lines: ``_Unrunnable`` raises the moment anything
    asks for its equations, and the screening still returns a verdict.
    """
    score = prescreen(_Unrunnable())
    assert score.refuted
    with pytest.raises(RuntimeError):
        _Unrunnable().reduced_equations("flrw")


def test_the_report_card_carries_the_refutation_and_fails_on_it():
    """A hypothesis refuted before the run cannot be rescued by the run."""
    card = evaluate(_Universalist())
    assert card.reference_score is not None
    assert card.reference_score.refuted
    assert not card.passed
    assert any("null_orbifold" in entry for entry in card.contradicted)
    assert "REFUTED BY EXACT RESULTS" in card.render()
    assert card.summary()["reference_score"]["refuted"] is True


def test_a_plugin_the_module_has_no_opinion_on_is_unaffected():
    """Screening must not turn into a second opinion on everything.

    Loop quantum cosmology says nothing about any background in the
    catalogue, so its report card is exactly what it was before the step
    existed.
    """
    from particlesim.theories import get_theory

    card = evaluate(get_theory("lqg.lqc"))
    assert card.reference_score is not None
    assert not card.reference_score.refuted
    assert card.passed
