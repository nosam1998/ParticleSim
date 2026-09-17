"""General relativity with a cosmological constant.

This is the worked example in ``docs/theory_authoring.md``, shipped as a real
registered plugin rather than a code block. A documented example that nothing
executes drifts out of date silently; this one is exercised by the test suite
and by ``particlesim check-limits`` on every run.

Field equations, signature (-,+,+,+), G = c = 1:

    G_ab + Lambda g_ab = 8 pi T_ab

so the matter a metric *requires* is ``T_ab = (G_ab + Lambda g_ab) / 8 pi``.
Lambda is counted as geometry, not matter, which is the choice that makes a
de Sitter vacuum need no source at all. The opposite convention, moving
Lambda to the right-hand side as a perfect fluid with ``p = -rho``, is equally
valid and gives different energy-condition verdicts. That is exactly why
``effective_stress_energy`` is a plugin decision the framework makes each
theory state rather than assuming one convention globally.
"""

from __future__ import annotations

import sympy as sp

from particlesim.symbolic.curvature import MetricGeometry
from particlesim.theories.base import Coupling, FieldSpec, Theory


class GRWithLambda(Theory):
    """GR plus a cosmological constant, with Lambda kept on the geometry side."""

    id = "gr.lambda"
    tier = "A"
    dimension = 4
    fields = [FieldSpec("g", "metric", rank=2)]
    couplings = [Coupling("Lambda", 0.0, units="1/length^2", bounds=(-1.0, 1.0))]
    frame = "einstein"
    formulation = "standard"
    provenance = (
        "Einstein-Hilbert action with cosmological constant, "
        "S = (1/16 pi) integral sqrt(-g) (R - 2 Lambda); "
        "Lambda counted as geometry, not matter."
    )
    validity_statement = "classical; no quantum-gravity regime"

    def lagrangian(self, metric: sp.Matrix, coords: list[sp.Symbol]) -> sp.Expr:
        geom = MetricGeometry(metric, coords)
        lam = self.values["Lambda"]
        return sp.sqrt(-metric.det()) * (geom.ricci_scalar - 2 * lam) / (16 * sp.pi)

    def effective_stress_energy(self, einstein: sp.Matrix, metric: sp.Matrix) -> sp.Matrix:
        lam = self.values["Lambda"]
        return (einstein + lam * metric) / (8 * sp.pi)

    def gr_limit(self) -> dict[str, float]:
        return {"Lambda": 0.0}

    def observable_predictions(self) -> dict[str, object]:
        lam = self.values["Lambda"]
        return {
            "de_sitter_horizon_radius": (3.0 / lam) ** 0.5 if lam > 0 else None,
            "accelerating": lam > 0,
        }
