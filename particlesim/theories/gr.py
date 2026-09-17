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

    def reduced_equations(self, symmetry: str):
        """Reference reduced dynamics, used by the GR-limit harness.

        A Tier B plugin declares that it becomes general relativity at some
        coupling value. Checking that claim needs GR's own reduced equations
        to compare against, so they live here rather than being reimplemented
        inside the harness.
        """
        if symmetry != "flrw":
            raise NotImplementedError(f"no reduced equations for {symmetry!r}")
        import numpy as np

        def hubble_squared(rho: float) -> float:
            return (8.0 * np.pi / 3.0) * rho

        return hubble_squared

    def metric_family(self, params: dict[str, float]) -> sp.Matrix:
        """Schwarzschild, the reference static spherically symmetric solution."""
        mass = float(params["mass"])
        r, th = sp.symbols("r theta", positive=True)
        f = 1 - 2 * mass / r
        return sp.diag(-f, 1 / f, r**2, r**2 * sp.sin(th) ** 2)

    def gr_limit(self) -> dict[str, float]:
        return {}
