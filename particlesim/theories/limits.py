"""Automatic GR-limit checking for theory plugins (design doc Sections 4.2, 4.3).

Every modified theory has to answer one question before any of its results
are worth reading: does it reduce to general relativity where general
relativity is already confirmed? A plugin answers by declaring
:meth:`Theory.gr_limit`, the coupling values at which it should become GR.
This module takes that declaration and checks it, so the claim is tested
rather than trusted.

Two checks, in increasing cost:

* **Stress-energy form.** ``effective_stress_energy`` is fed a generic
  symbolic Einstein tensor and metric and compared with the GR reference
  ``G_ab / 8 pi``. This catches an extra term that survives the limit,
  which is the common failure, and costs nothing because no curvature is
  computed.
* **Action.** The Lagrangian is evaluated on a test metric with non-zero
  curvature and compared with GR's. This is the stronger check and the
  slower one, since it computes a Ricci scalar. A conformally flat metric
  is used by default: Minkowski and Schwarzschild both have vanishing Ricci
  scalar, so they would pass a theory that adds a cosmological constant and
  tell you nothing.

A harness that can only pass is worthless, so the test suite exercises this
one against a plugin whose declared limit is deliberately wrong and requires
that it be caught.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from particlesim.theories.base import Theory
from particlesim.theories.gr import GeneralRelativity

TAU, CHI, UPS, ZETA = sp.symbols("t x y z", real=True)
TEST_COORDS = [TAU, CHI, UPS, ZETA]


def conformally_flat_metric() -> sp.Matrix:
    """``a(t)^2 (-dt^2 + dx^2 + dy^2 + dz^2)``, which has non-zero Ricci scalar.

    Chosen so that a theory adding a cosmological constant, or any term that
    does not vanish in vacuum, is actually distinguishable from GR.
    """
    a = sp.Function("a")(TAU)
    return sp.diag(-(a**2), a**2, a**2, a**2)


def _generic_symmetric(name: str, n: int = 4) -> sp.Matrix:
    m = sp.zeros(n, n)
    for i in range(n):
        for j in range(i, n):
            m[i, j] = m[j, i] = sp.Symbol(f"{name}{i}{j}")
    return m


@dataclass
class GRLimitReport:
    theory_id: str
    tier: str
    checked: bool
    passed: bool
    stress_energy_matches: bool | None = None
    lagrangian_matches: bool | None = None
    reduced_equations_match: bool | None = None
    metric_family_matches: bool | None = None
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        return {
            "theory": self.theory_id,
            "tier": self.tier,
            "checked": self.checked,
            "passed": self.passed,
            "stress_energy_matches": self.stress_energy_matches,
            "lagrangian_matches": self.lagrangian_matches,
            "reduced_equations_match": self.reduced_equations_match,
            "metric_family_matches": self.metric_family_matches,
            "reasons": list(self.reasons),
        }


def check_gr_limit(
    theory_cls: type[Theory],
    check_action: bool = True,
    metric: sp.Matrix | None = None,
) -> GRLimitReport:
    """Instantiate ``theory_cls`` at its declared GR limit and compare with GR.

    Returns a report rather than raising, so a suite can check every
    registered plugin and show all failures at once. ``checked`` is False
    when the plugin does not implement the pieces needed (a Tier B or C
    plugin with no action, for instance); that is recorded, not silently
    treated as a pass.
    """
    report = GRLimitReport(
        theory_id=theory_cls.id, tier=theory_cls.tier, checked=False, passed=False
    )

    try:
        limit = theory_cls().gr_limit()
    except Exception as exc:  # noqa: BLE001 - a broken plugin must not break the suite
        report.reasons.append(f"could not read gr_limit(): {exc!r}")
        return report

    infinite = {k: v for k, v in limit.items() if not sp.S(v).is_finite}
    if infinite:
        report.reasons.append(
            f"limit sends {sorted(infinite)} to infinity, which cannot be instantiated; "
            "the plugin should expose a finite parameterisation (for example an inverse "
            "coupling) if its GR limit is at infinity"
        )
        return report

    try:
        theory = theory_cls(**limit)
    except Exception as exc:  # noqa: BLE001
        report.reasons.append(f"could not instantiate at the declared limit: {exc!r}")
        return report

    reference = GeneralRelativity()
    g_generic = _generic_symmetric("g")
    einstein = _generic_symmetric("G")

    try:
        t_theory = sp.Matrix(theory.effective_stress_energy(einstein, g_generic))
        t_gr = sp.Matrix(reference.effective_stress_energy(einstein, g_generic))
    except NotImplementedError:
        if theory_cls.tier == "A":
            # For Tier A this is a gap; for Tier B there is no action to vary,
            # so its absence is the contract rather than an omission.
            report.reasons.append("no effective_stress_energy; nothing to compare algebraically")
    except Exception as exc:  # noqa: BLE001
        report.reasons.append(f"effective_stress_energy raised: {exc!r}")
        return report
    else:
        diff = sp.simplify(t_theory - t_gr)
        report.stress_energy_matches = bool(diff.is_zero_matrix)
        report.checked = True
        if not report.stress_energy_matches:
            nonzero = [
                (i, j, sp.simplify(diff[i, j]))
                for i in range(4)
                for j in range(4)
                if sp.simplify(diff[i, j]) != 0
            ]
            report.reasons.append(
                f"effective stress-energy differs from G/8pi at the declared limit: {nonzero[:3]}"
            )

    if check_action:
        test_metric = conformally_flat_metric() if metric is None else metric
        try:
            l_theory = theory.lagrangian(test_metric, TEST_COORDS)
            l_gr = reference.lagrangian(test_metric, TEST_COORDS)
        except NotImplementedError:
            if theory_cls.tier == "A":
                report.reasons.append("no lagrangian; action check skipped")
        except Exception as exc:  # noqa: BLE001
            report.reasons.append(f"lagrangian raised: {exc!r}")
            return report
        else:
            report.lagrangian_matches = bool(sp.simplify(l_theory - l_gr) == 0)
            report.checked = True
            if not report.lagrangian_matches:
                report.reasons.append(
                    "Lagrangian differs from the Einstein-Hilbert action at the declared limit"
                )

    _check_reduced(theory, reference, report)
    _check_metric_family(theory, reference, report)

    results = [
        r
        for r in (
            report.stress_energy_matches,
            report.lagrangian_matches,
            report.reduced_equations_match,
            report.metric_family_matches,
        )
        if r is not None
    ]
    report.passed = report.checked and all(results) and bool(results)
    return report


def _check_reduced(theory: Theory, reference: Theory, report: GRLimitReport) -> None:
    """Compare symmetry-reduced dynamics against GR's, sampling the argument.

    This is what makes a Tier B declaration checkable: those plugins have no
    action to vary, so the only statement of their content is the reduced
    equation itself.
    """
    import numpy as np

    try:
        theory_eq = theory.reduced_equations("flrw")
        gr_eq = reference.reduced_equations("flrw")
    except NotImplementedError:
        return
    except Exception as exc:  # noqa: BLE001
        report.reasons.append(f"reduced_equations raised: {exc!r}")
        return

    # Sample across many decades: a correction that only bites at high
    # density would be invisible if only small values were tried.
    samples = np.logspace(-8, 2, 40)
    try:
        theory_vals = np.array([float(theory_eq(float(x))) for x in samples])
        gr_vals = np.array([float(gr_eq(float(x))) for x in samples])
    except Exception as exc:  # noqa: BLE001
        report.reasons.append(f"evaluating reduced equations raised: {exc!r}")
        return

    scale = np.maximum(np.abs(gr_vals), 1e-30)
    worst = float(np.max(np.abs(theory_vals - gr_vals) / scale))
    report.reduced_equations_match = worst < 1e-10
    report.checked = True
    if not report.reduced_equations_match:
        report.reasons.append(
            f"reduced FLRW dynamics differ from GR at the declared limit by a relative {worst:.3g}"
        )


def _check_metric_family(theory: Theory, reference: Theory, report: GRLimitReport) -> None:
    """Compare a Tier B metric family against Schwarzschild symbolically."""
    params = {"mass": 1.0}
    try:
        theory_metric = sp.Matrix(theory.metric_family(params))
        gr_metric = sp.Matrix(reference.metric_family(params))
    except NotImplementedError:
        return
    except Exception as exc:  # noqa: BLE001
        report.reasons.append(f"metric_family raised: {exc!r}")
        return

    diff = sp.simplify(theory_metric - gr_metric)
    report.metric_family_matches = bool(diff.is_zero_matrix)
    report.checked = True
    if not report.metric_family_matches:
        report.reasons.append("metric family differs from Schwarzschild at the declared limit")


def check_all(check_action: bool = True) -> dict[str, GRLimitReport]:
    """Run :func:`check_gr_limit` over every discoverable theory plugin."""
    from particlesim.theories.registry import list_theories

    return {tid: check_gr_limit(cls, check_action) for tid, cls in list_theories().items()}
