"""Form 1 of 4: a hypothesis written as an extra term in the action.

**Hypothesis.** Gravity picks up a curvature-squared correction,
``S = (1/16 pi) integral sqrt(-g) (R + alpha R^2)``.

This is the most demanding form, and the most rewarding: because there is an
action, the limit harness can compare it against Einstein-Hilbert
symbolically, which is a stronger statement than any numerical agreement.

Note what the template does *not* claim. An action alone does not tell you
whether the resulting evolution is well posed, which is why ``formulation``
exists and why a plugin declaring ``order_reduced`` is refused for
strong-field runs. Writing ``standard`` here would be a claim about
hyperbolicity that this hypothesis has not earned, so it says so.
"""

from __future__ import annotations

import sympy as sp

from particlesim.symbolic.curvature import MetricGeometry
from particlesim.theories.base import Coupling, FieldSpec, Theory


class CurvatureSquared(Theory):
    id = "user.r_squared"
    tier = "A"
    fields = [FieldSpec("g", "metric", rank=2)]
    couplings = [Coupling("alpha", 0.0, units="length^2", bounds=(-10.0, 10.0))]
    frame = "einstein"
    # Not "standard": higher-derivative gravity is not automatically well
    # posed, and the framework refuses strong-field 3D runs for this value
    # rather than producing plausible nonsense.
    formulation = "order_reduced"
    provenance = "User hypothesis: R + alpha R^2, treated perturbatively in alpha."
    validity_statement = "weak coupling only; alpha R must stay far below one"

    def lagrangian(self, metric: sp.Matrix, coords: list[sp.Symbol]) -> sp.Expr:
        geom = MetricGeometry(metric, coords)
        r = geom.ricci_scalar
        alpha = self.values["alpha"]
        return sp.sqrt(-metric.det()) * (r + alpha * r**2) / (16 * sp.pi)

    def effective_stress_energy(self, einstein: sp.Matrix, metric: sp.Matrix) -> sp.Matrix:
        # To first order in alpha the extra term contributes nothing in vacuum,
        # which is why this is an order-reduced treatment rather than an exact
        # one. Stated here so nobody reads the simplicity as a derivation.
        return einstein / (8 * sp.pi)

    def gr_limit(self) -> dict[str, float]:
        return {"alpha": 0.0}

    def observable_predictions(self) -> dict[str, object]:
        return {"modifies_vacuum_dynamics": self.values["alpha"] != 0.0}
