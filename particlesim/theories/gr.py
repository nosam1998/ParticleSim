"""General relativity: the baseline Tier A plugin."""

from __future__ import annotations

import sympy as sp

from particlesim.symbolic.curvature import MetricGeometry
from particlesim.theories.base import FieldSpec, Theory


class GeneralRelativity(Theory):
    id = "gr"
    tier = "A"
    dimension = 4
    fields = [FieldSpec("g", "metric", rank=2)]
    couplings = []
    frame = "einstein"
    formulation = "standard"
    provenance = "Einstein-Hilbert action, minimal coupling, geometric units G = c = 1."
    validity_statement = "classical; no quantum-gravity regime"

    def lagrangian(self, metric: sp.Matrix, coords: list[sp.Symbol]) -> sp.Expr:
        geom = MetricGeometry(metric, coords)
        return sp.sqrt(-metric.det()) * geom.ricci_scalar / (16 * sp.pi)

    def effective_stress_energy(self, einstein: sp.Matrix, metric: sp.Matrix) -> sp.Matrix:
        return einstein / (8 * sp.pi)

    def gr_limit(self) -> dict[str, float]:
        return {}
