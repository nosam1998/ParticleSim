"""Form 2 of 4: a hypothesis written as modified reduced equations.

**Hypothesis.** The effective energy density saturates, so the Friedmann
equation loses its source before the scale factor reaches zero and the big
bang is replaced by a bounce.

This is the worked example from Appendix B of the design document, with one
correction that matters. The appendix declares the general-relativistic limit
as ``rho_c -> infinity``. That cannot be instantiated, so the limit harness
reports the plugin as *unchecked* rather than passing, and an unchecked limit
is not a recovered one. Parameterising by ``1 / rho_c`` puts the limit at
zero, where it is an ordinary value the harness can actually test.

Run it:

    from examples.hypotheses.limiting_curvature import LimitingCurvature
    from particlesim.scenarios.singularity.harness import evaluate
    print(evaluate(LimitingCurvature()).render())
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from particlesim.theories.base import Coupling, FieldSpec, Theory


class LimitingCurvature(Theory):
    id = "user.limiting_curvature"
    tier = "B"
    fields = [FieldSpec("g", "metric", rank=2)]
    couplings = [
        Coupling("inverse_rho_c", 1.0 / 0.41, units="1/planck_density", bounds=(0.0, 1e6))
    ]
    provenance = (
        "User hypothesis: density saturates at rho_c. Structurally identical to "
        "the holonomy correction of effective loop quantum cosmology, but "
        "asserted rather than derived."
    )
    validity_statement = "homogeneous isotropic sector; no derivation, so no regime is earned"

    @property
    def rho_critical(self) -> float:
        inv = self.values["inverse_rho_c"]
        return float("inf") if inv == 0.0 else 1.0 / inv

    def density_correction(self) -> Callable[[float], float]:
        inv = self.values["inverse_rho_c"]
        return lambda rho: 1.0 - rho * inv

    def reduced_equations(self, symmetry: str) -> Callable[..., Any]:
        if symmetry != "flrw":
            raise NotImplementedError("this hypothesis only covers the FLRW sector")
        inv = self.values["inverse_rho_c"]
        return lambda rho: (8.0 * np.pi / 3.0) * rho * (1.0 - rho * inv)

    def gr_limit(self) -> dict[str, float]:
        return {"inverse_rho_c": 0.0}

    def observable_predictions(self) -> dict[str, Any]:
        return {
            "bounce": True,
            "singularity_resolved": True,
            "max_density": self.rho_critical,
        }

    def regime_of_validity(self, state: Any) -> bool:
        try:
            return float(getattr(state, "density", state)) <= self.rho_critical
        except (TypeError, ValueError):
            return True
