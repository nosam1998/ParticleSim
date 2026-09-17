"""Form 4 of 4: a hypothesis written as a matter model.

**Hypothesis.** At high density matter stiffens beyond the usual bound, with
``p = w rho`` and ``w > 1``, so pressure resists collapse more strongly than
any causal fluid normally allows.

The point of this template is that it fails honestly. ``w > 1`` is
superluminal: the sound speed exceeds light. The plugin says so in its
validity statement rather than burying it, and the harness has no business
reporting a bounce from such a fluid as a physical result. A template that
only showed a hypothesis surviving would teach the wrong lesson about what
this harness is for.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from particlesim.theories.base import Coupling, FieldSpec, Theory


class StiffMatter(Theory):
    id = "user.stiff_matter"
    tier = "B"
    fields = [FieldSpec("rho", "scalar")]
    couplings = [Coupling("w", 1.0, units="dimensionless", bounds=(0.0, 3.0))]
    provenance = "User hypothesis: equation of state p = w rho with w above the stiff limit."
    validity_statement = (
        "w > 1 means a sound speed above the speed of light; this is an "
        "exploration of what such a fluid would do, not a physical matter model"
    )

    @property
    def superluminal(self) -> bool:
        """Sound speed squared is ``w``, so anything above one is acausal."""
        return self.values["w"] > 1.0

    def reduced_equations(self, symmetry: str) -> Callable[..., Any]:
        if symmetry != "flrw":
            raise NotImplementedError("this hypothesis only covers the FLRW sector")
        return lambda rho: (8.0 * np.pi / 3.0) * rho

    def gr_limit(self) -> dict[str, float]:
        # The matter content changes, not the gravity; at w = 0 this is dust
        # evolving under ordinary general relativity.
        return {"w": 0.0}

    def observable_predictions(self) -> dict[str, Any]:
        return {"bounce": False, "acausal": self.superluminal}

    def regime_of_validity(self, state: Any) -> bool:
        return not self.superluminal
