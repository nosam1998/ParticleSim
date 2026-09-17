"""The hypothesis harness, including its ability to say no (design doc Section 8)."""

import pytest

from particlesim.scenarios.singularity.harness import evaluate, run_battery
from particlesim.theories import get_theory
from particlesim.theories.base import Coupling, Theory
from particlesim.theories.lqg.lqc import EffectiveLQC


class _Liar(Theory):
    """Claims the singularity is resolved and supplies nothing that resolves it.

    This is the fixture the harness exists for. A report card that cannot
    return a failure for this plugin is a demonstration, not a test.
    """

    id = "test.liar"
    tier = "B"
    couplings = [Coupling("unused", 1.0)]
    provenance = "test fixture with a deliberately false claim"
    validity_statement = "none; this plugin is wrong on purpose"

    def gr_limit(self) -> dict[str, float]:
        return {"unused": 1.0}

    def observable_predictions(self) -> dict[str, object]:
        return {"bounce": True, "singularity_resolved": True}


class _Undeclared(Theory):
    """Says nothing about what it is."""

    id = "test.undeclared"
    tier = "B"
    couplings = []

    def gr_limit(self) -> dict[str, float]:
        return {}


class _Vague(EffectiveLQC):
    """Predicts something the battery cannot measure."""

    id = "test.vague"

    def observable_predictions(self) -> dict[str, object]:
        return {"echoes_in_the_ringdown": True, "bounce": True}


def test_lqc_survives_and_its_claims_are_confirmed():
    card = evaluate(get_theory("lqg.lqc"))
    assert card.passed
    assert card.singularity_resolved
    assert card.density_bounded
    assert set(card.confirmed) >= {"bounce", "singularity_resolved"}
    assert not card.contradicted
    assert card.max_density == pytest.approx(0.41, rel=1e-2)


def test_general_relativity_reaches_a_singularity_and_claims_nothing():
    card = evaluate(get_theory("gr"))
    assert not card.singularity_resolved
    assert not card.density_bounded
    assert card.confirmed == [] and card.contradicted == []
    collapse = next(b for b in card.battery if b.scenario == "flrw_collapse")
    assert collapse.outcome == "singularity"
    assert not collapse.bounced


def test_a_false_claim_is_contradicted_and_the_hypothesis_fails():
    card = evaluate(_Liar())
    assert not card.passed
    assert card.contradicted
    assert any("bounce" in c for c in card.contradicted)
    assert any("claimed True" in c and "observed False" in c for c in card.contradicted)
    assert "FAILED" in card.render()
    assert "CONTRADICTED" in card.render()


def test_an_undeclared_plugin_fails_before_any_physics_runs():
    card = evaluate(_Undeclared())
    assert not card.declaration_complete
    assert not card.passed
    assert any("provenance" in w for w in card.warnings)
    assert any("regime of validity" in w for w in card.warnings)


def test_unmeasurable_predictions_are_untested_not_confirmed():
    """Silence is not agreement. A claim the battery cannot reach must be
    recorded as untested, or every plugin could pass by predicting things
    nothing measures."""
    card = evaluate(_Vague())
    assert "echoes_in_the_ringdown" in card.untested
    assert "echoes_in_the_ringdown" not in card.confirmed
    assert "bounce" in card.confirmed


def test_battery_covers_collapse_and_expansion():
    results = run_battery(get_theory("gr"))
    assert {b.scenario for b in results} == {"flrw_collapse", "flrw_expansion"}
    collapse = next(b for b in results if b.scenario == "flrw_collapse")
    assert collapse.min_scale_factor < 1e-5


def test_report_card_summary_is_serialisable():
    import json

    card = evaluate(get_theory("lqg.lqc"))
    text = json.dumps(card.summary(), default=str)
    assert "lqg.lqc" in text
    parsed = json.loads(text)
    assert parsed["passed"] is True
    assert parsed["gr_limit_passed"] is True
    assert len(parsed["battery"]) == 2


def test_convergence_is_part_of_the_verdict():
    """A bounce that only appears at one tolerance is a solver artefact."""
    card = evaluate(get_theory("lqg.lqc"))
    assert card.converged
    assert evaluate(get_theory("gr")).converged
