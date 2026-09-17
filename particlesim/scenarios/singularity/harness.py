"""Hypothesis harness and report card (design doc Section 8).

A hypothesis about singularities is testable here only once it is written as
a theory plugin. This module takes such a plugin and answers a fixed set of
questions about it, always the same questions in the same order, so that two
hypotheses can be compared rather than each being argued on its own terms.

The report card is designed to be able to say no. Its most important field is
``contradicted``: the predictions a plugin declared through
``observable_predictions`` that the runs did not bear out. A harness that can
only report success is a demonstration, not a test, so the suite exercises
this one against a plugin that claims a bounce and does not deliver it.

What is checked, in increasing cost:

1. **Declaration.** Tier, formulation, provenance and a regime of validity
   are present. A plugin that does not say what it is cannot be held to it.
2. **General-relativistic limit.** Delegated to the limit harness, which
   compares actions, stress-energy splits, reduced dynamics and metric
   families as applicable.
3. **Cosmological battery.** Collapse and expansion under the plugin's own
   corrected dynamics, looking for a bounce, a singularity, bounded density
   and bounded curvature.
4. **Convergence.** The same run at several tolerances, to distinguish a
   physical result from an artefact of the integrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from particlesim.scenarios.singularity.flrw import W_MATTER, FLRWBackground
from particlesim.theories.base import Theory
from particlesim.theories.limits import GRLimitReport, check_gr_limit

#: Scale factor below which a collapsing solution is treated as singular.
SINGULAR_SCALE_FACTOR = 1e-6


@dataclass
class BatteryResult:
    """One scenario's outcome."""

    scenario: str
    outcome: str
    bounced: bool
    max_density: float
    min_scale_factor: float
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "outcome": self.outcome,
            "bounced": self.bounced,
            "max_density": self.max_density,
            "min_scale_factor": self.min_scale_factor,
        }


@dataclass
class ReportCard:
    """The fixed set of answers a hypothesis gets."""

    theory_id: str
    tier: str
    provenance: str
    validity_statement: str
    declaration_complete: bool
    gr_limit: GRLimitReport
    battery: list[BatteryResult]
    singularity_resolved: bool
    max_density: float
    density_bounded: bool
    converged: bool
    in_regime: bool = True
    predictions: dict[str, Any] = field(default_factory=dict)
    confirmed: list[str] = field(default_factory=list)
    contradicted: list[str] = field(default_factory=list)
    untested: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether the hypothesis survived: a recovered GR limit and nothing
        contradicted. Silence on a prediction is not a pass, but it is not a
        failure either; it lands in ``untested``."""
        return (
            self.declaration_complete
            and self.gr_limit.passed
            and not self.contradicted
            and self.converged
        )

    def summary(self) -> dict[str, Any]:
        return {
            "theory": self.theory_id,
            "tier": self.tier,
            "passed": self.passed,
            "declaration_complete": self.declaration_complete,
            "gr_limit_passed": self.gr_limit.passed,
            "singularity_resolved": self.singularity_resolved,
            "density_bounded": self.density_bounded,
            "max_density": self.max_density,
            "converged": self.converged,
            "in_regime": self.in_regime,
            "confirmed": list(self.confirmed),
            "contradicted": list(self.contradicted),
            "untested": list(self.untested),
            "warnings": list(self.warnings),
            "battery": [b.as_row() for b in self.battery],
            "provenance": self.provenance,
            "validity": self.validity_statement,
        }

    def render(self) -> str:
        """A short human-readable verdict."""
        verdict = "SURVIVED" if self.passed else "FAILED"
        if self.passed and not self.in_regime:
            verdict += " (outside its own declared regime of validity)"
        lines = [
            f"Hypothesis: {self.theory_id}  (tier {self.tier})",
            f"Verdict: {verdict}",
            f"  declaration complete : {self.declaration_complete}",
            f"  GR limit recovered   : {self.gr_limit.passed}",
            f"  singularity resolved : {self.singularity_resolved}",
            f"  density bounded      : {self.density_bounded} (max {self.max_density:.6g})",
            f"  converged            : {self.converged}",
            f"  within declared regime: {self.in_regime}",
        ]
        for b in self.battery:
            lines.append(f"  scenario {b.scenario:<22} {b.outcome}")
        if self.confirmed:
            lines.append(f"  confirmed   : {', '.join(self.confirmed)}")
        if self.contradicted:
            lines.append(f"  CONTRADICTED: {', '.join(self.contradicted)}")
        if self.untested:
            lines.append(f"  untested    : {', '.join(self.untested)}")
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        return "\n".join(lines)


def _declaration_complete(theory: Theory, warnings: list[str]) -> bool:
    ok = True
    if not theory.provenance:
        warnings.append("no provenance: the plugin does not say which paper or truncation it is")
        ok = False
    if not theory.validity_statement or theory.validity_statement == "unrestricted":
        warnings.append(
            "no regime of validity declared; every effective theory has one, and "
            "'unrestricted' is a claim rather than a default"
        )
        ok = False
    if theory.tier not in ("A", "B", "C"):
        warnings.append(f"unknown tier {theory.tier!r}")
        ok = False
    return ok


def _density_correction(theory: Theory):
    """The plugin's Friedmann correction, if it offers one."""
    if hasattr(theory, "density_correction"):
        return theory.density_correction()
    return None


def run_battery(theory: Theory, matter_density: float = 1e-3) -> list[BatteryResult]:
    """Collapse and expansion under the plugin's corrected dynamics."""
    correction = _density_correction(theory)
    results: list[BatteryResult] = []

    for name, expanding, a0 in (
        ("flrw_collapse", False, 1.0),
        ("flrw_expansion", True, 1.0),
    ):
        bg = FLRWBackground(components={W_MATTER: matter_density}, density_correction=correction)
        sol = bg.evolve(a0=a0, t_max=5000.0, expanding=expanding)
        notes = []
        if sol.outcome == "ran_to_t_max":
            notes.append("did not reach a singularity or a turning point within the time budget")
        results.append(
            BatteryResult(
                scenario=name,
                outcome=sol.outcome,
                bounced=sol.bounced,
                max_density=float(sol.density.max()),
                min_scale_factor=float(sol.min_scale_factor),
                notes=notes,
            )
        )
    return results


def _converges(theory: Theory, matter_density: float) -> tuple[bool, list[str]]:
    """Whether the collapse outcome is stable under tightening the integrator.

    A bounce that appears at one tolerance and not another is a property of
    the solver, not of the theory, and reporting it as physics is the failure
    this check exists to prevent.
    """
    correction = _density_correction(theory)
    outcomes, densities = [], []
    for rtol in (1e-8, 1e-10, 1e-12):
        bg = FLRWBackground(components={W_MATTER: matter_density}, density_correction=correction)
        sol = bg.evolve(a0=1.0, t_max=5000.0, expanding=False, rtol=rtol, atol=rtol * 1e-2)
        outcomes.append(sol.outcome)
        densities.append(float(sol.density.max()))

    warnings: list[str] = []
    if len(set(outcomes)) != 1:
        warnings.append(f"collapse outcome changes with integrator tolerance: {outcomes}")
        return False, warnings
    spread = (max(densities) - min(densities)) / max(max(densities), 1e-30)
    if spread > 1e-4:
        warnings.append(f"maximum density varies by {spread:.2g} across tolerances")
        return False, warnings
    return True, warnings


def collapse_solutions(theory: Theory, matter_density: float = 1e-3):
    """The collapsing solution under the plugin and under plain general
    relativity, on identical initial data.

    Returned as a pair because a bounce means nothing on its own: an
    integrator that stops early also produces a smallest scale factor. The
    baseline is what turns the corrected run into evidence.
    """
    baseline = FLRWBackground(components={W_MATTER: matter_density}).evolve(
        a0=1.0, t_max=5000.0, expanding=False
    )
    corrected = FLRWBackground(
        components={W_MATTER: matter_density},
        density_correction=_density_correction(theory),
    ).evolve(a0=1.0, t_max=5000.0, expanding=False)
    return baseline, corrected


def write_report(
    card: ReportCard,
    path,
    theory: Theory | None = None,
    matter_density: float = 1e-3,
):
    """Render a report card to a self-contained HTML page with its figures."""
    from particlesim.viz.report import render_html_report
    from particlesim.viz.singularity_views import hypothesis_figures

    figures: dict[str, bytes] = {}
    if theory is not None:
        baseline, corrected = collapse_solutions(theory, matter_density)
        figures = hypothesis_figures(card, baseline, corrected)
    return render_html_report(
        card.summary(), path, figures, title=f"Hypothesis report: {card.theory_id}"
    )


def evaluate(theory: Theory, matter_density: float = 1e-3) -> ReportCard:
    """Run the full battery against ``theory`` and score its declared claims."""
    warnings: list[str] = []
    declaration_ok = _declaration_complete(theory, warnings)
    limit_report = check_gr_limit(type(theory))
    if not limit_report.checked:
        warnings.append(
            "the GR limit could not be checked at all; the plugin exposes neither an "
            "action, a stress-energy split, reduced equations nor a metric family"
        )

    battery = run_battery(theory, matter_density)
    collapse = next(b for b in battery if b.scenario == "flrw_collapse")
    converged, conv_warnings = _converges(theory, matter_density)
    warnings.extend(conv_warnings)

    resolved = collapse.bounced or collapse.min_scale_factor > SINGULAR_SCALE_FACTOR
    max_density = max(b.max_density for b in battery)
    density_bounded = bool(np.isfinite(max_density)) and max_density < 1e12

    # The design document asks for a regime-of-validity flag on every run, and
    # it earns its place: a hypothesis can pass every other check while every
    # number it produced sits outside the range it claimed to be good for.
    # That is not a failure of the hypothesis, it is a statement about what
    # the result is worth, so it flags rather than fails.
    try:
        in_regime = bool(theory.regime_of_validity(max_density))
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"regime_of_validity raised: {exc!r}; assuming in-regime")
        in_regime = True
    if not in_regime:
        warnings.append(
            f"the run reached a density of {max_density:.6g}, outside the regime this "
            "plugin declares itself valid in; its own statement says these numbers "
            "should not be trusted"
        )

    predictions = theory.observable_predictions()
    confirmed: list[str] = []
    contradicted: list[str] = []
    untested: list[str] = []
    for name, claimed in predictions.items():
        observed = _observe(name, collapse, resolved, max_density)
        if observed is None:
            untested.append(name)
        elif _agrees(claimed, observed):
            confirmed.append(name)
        else:
            contradicted.append(f"{name}: claimed {claimed!r}, observed {observed!r}")

    return ReportCard(
        theory_id=theory.id,
        tier=theory.tier,
        provenance=theory.provenance,
        validity_statement=theory.validity_statement,
        declaration_complete=declaration_ok,
        gr_limit=limit_report,
        battery=battery,
        singularity_resolved=resolved,
        max_density=max_density,
        density_bounded=density_bounded,
        converged=converged,
        in_regime=in_regime,
        predictions=predictions,
        confirmed=confirmed,
        contradicted=contradicted,
        untested=untested,
        warnings=warnings,
    )


def _observe(name: str, collapse: BatteryResult, resolved: bool, max_density: float):
    """Map a declared prediction onto something the battery measured.

    Returning ``None`` means the battery has nothing to say, which is
    recorded as untested rather than quietly treated as agreement.
    """
    if name in ("bounce", "bounces"):
        return collapse.bounced
    if name in ("singularity_resolved", "singularity_avoided"):
        return resolved
    if name in ("max_density", "critical_density"):
        return max_density
    return None


def _agrees(claimed: Any, observed: Any, rel: float = 5e-2) -> bool:
    if isinstance(claimed, bool) or isinstance(observed, bool):
        return bool(claimed) == bool(observed)
    try:
        c, o = float(claimed), float(observed)
    except (TypeError, ValueError):
        return claimed == observed
    if not np.isfinite(c):
        return not np.isfinite(o)
    return abs(c - o) <= rel * max(abs(c), 1e-30)
