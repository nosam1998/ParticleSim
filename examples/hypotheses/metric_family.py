"""Form 3 of 4: a hypothesis written as a corrected metric family.

**Hypothesis.** The Schwarzschild mass function is smeared over a minimum
length, so the metric function never produces a curvature singularity:

    f(r) = 1 - 2 M r^2 / (r^3 + 2 M l^2)

This is the Hayward form. It is a guess at a solution, not a solution of any
field equation, and a template that pretended otherwise would teach the wrong
habit. Tier B exists for exactly this: a corrected geometry with no action
behind it.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from particlesim.theories.base import Coupling, FieldSpec, Theory


class RegularBlackHole(Theory):
    id = "user.regular_bh"
    tier = "B"
    fields = [FieldSpec("g", "metric", rank=2)]
    couplings = [Coupling("length", 0.0, units="length", bounds=(0.0, 10.0))]
    provenance = "User hypothesis, Hayward-type regular metric; a guessed geometry."
    validity_statement = (
        "a metric ansatz, not a solution of field equations; its interior is "
        "an illustration of what a minimum length would do, not a prediction"
    )

    def metric_family(self, params: dict[str, float]) -> sp.Matrix:
        mass = float(params["mass"])
        ell = self.values["length"]
        r, th = sp.symbols("r theta", positive=True)
        f = 1 - 2 * mass * r**2 / (r**3 + 2 * mass * ell**2)
        return sp.diag(-f, 1 / f, r**2, r**2 * sp.sin(th) ** 2)

    def lapse_function(self, r: np.ndarray | float, mass: float) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        ell = self.values["length"]
        return 1.0 - 2.0 * mass * r**2 / (r**3 + 2.0 * mass * ell**2)

    def gr_limit(self) -> dict[str, float]:
        return {"length": 0.0}

    def observable_predictions(self) -> dict[str, object]:
        return {"curvature_bounded_at_origin": self.values["length"] > 0.0}
