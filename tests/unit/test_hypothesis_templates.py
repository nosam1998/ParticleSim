"""The four hypothesis templates, exercised so they cannot rot silently."""

import pytest

from examples.hypotheses.action_term import CurvatureSquared
from examples.hypotheses.limiting_curvature import LimitingCurvature
from examples.hypotheses.matter_boundary import StiffMatter
from examples.hypotheses.metric_family import RegularBlackHole
from particlesim.scenarios.singularity.harness import evaluate
from particlesim.theories.limits import check_gr_limit

TEMPLATES = [CurvatureSquared, LimitingCurvature, RegularBlackHole, StiffMatter]


@pytest.mark.parametrize("cls", TEMPLATES, ids=lambda c: c.id)
def test_every_template_declares_itself_completely(cls):
    theory = cls()
    assert theory.provenance
    assert theory.validity_statement and theory.validity_statement != "unrestricted"
    assert theory.tier in ("A", "B")


@pytest.mark.parametrize("cls", TEMPLATES, ids=lambda c: c.id)
def test_every_template_has_a_checkable_gr_limit(cls):
    """A limit at infinity cannot be instantiated, so it can never be checked.
    Every template must avoid that trap, since they are what people copy."""
    report = check_gr_limit(cls)
    assert report.checked, f"{cls.id} limit is unverifiable: {report.reasons}"
    assert report.passed, f"{cls.id} does not recover GR: {report.reasons}"


@pytest.mark.parametrize("cls", TEMPLATES, ids=lambda c: c.id)
def test_every_template_runs_through_the_harness(cls):
    card = evaluate(cls())
    assert card.declaration_complete
    assert card.converged
    assert isinstance(card.render(), str)


def test_limiting_curvature_corrects_the_design_document_appendix():
    """Appendix B declares the limit at infinite density, which is
    unverifiable; the template uses the inverse so the limit is a value."""
    theory = LimitingCurvature()
    assert theory.gr_limit() == {"inverse_rho_c": 0.0}
    card = evaluate(theory)
    assert card.passed
    assert set(card.confirmed) >= {"bounce", "singularity_resolved"}
    assert card.max_density == pytest.approx(0.41, rel=1e-2)


def test_stiff_matter_template_is_flagged_outside_its_own_regime():
    """The template that fails honestly: superluminal matter passes the
    mechanical checks while sitting outside the validity it declares."""
    card = evaluate(StiffMatter(w=2.0))
    assert StiffMatter(w=2.0).superluminal
    assert not card.in_regime
    assert "outside its own declared regime" in card.render()
    assert any("should not be trusted" in w for w in card.warnings)
    # And at a causal equation of state it is in regime.
    assert evaluate(StiffMatter(w=0.5)).in_regime


def test_regular_black_hole_metric_is_finite_at_the_origin():
    theory = RegularBlackHole(length=1.0)
    # The Hayward form has f -> 1 as r -> 0, not the divergence of Schwarzschild.
    assert theory.lapse_function(1e-8, 1.0) == pytest.approx(1.0, abs=1e-10)
    assert abs(theory.lapse_function(1e-3, 1.0)) < 1.0
    # With the length switched off it is Schwarzschild, which does diverge.
    assert theory.gr_limit() == {"length": 0.0}
    schwarzschild = RegularBlackHole(length=0.0)
    assert schwarzschild.lapse_function(1e-6, 1.0) < -1e5


def test_action_template_declares_a_formulation_it_has_earned():
    """Higher-derivative gravity is not automatically well posed, so claiming
    'standard' would be a hyperbolicity claim this hypothesis has not made."""
    assert CurvatureSquared.formulation == "order_reduced"
    report = check_gr_limit(CurvatureSquared)
    assert report.lagrangian_matches is True
