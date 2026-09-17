"""Effective loop quantum cosmology (design doc Sections 4.3, 4.5).

The symmetry-reduced sector of loop quantum gravity replaces the Friedmann
equation with

    H^2 = (8 pi / 3) rho (1 - rho / rho_c)

The correction is invisible at low density and takes over as ``rho``
approaches ``rho_c``, where the expansion rate passes through zero: the big
bang becomes a bounce. That is the single prediction this plugin exists to
make testable, and the battery checks it by finding the maximum density of a
collapsing solution and comparing it with ``rho_c``.

**The coupling is the inverse critical density, not the critical density.**
General relativity is recovered as ``rho_c -> infinity``, which cannot be
instantiated, so a plugin parameterised that way could never have its
declared limit checked: the harness would report it as unchecked rather than
passing. Using ``1 / rho_c`` puts the limit at zero, where it is an ordinary
value, and the automatic GR-limit test then means something.

Provenance: Ashtekar and Singh 2011 (Class. Quantum Grav. 28, 213001) for
the effective equation; the critical density ``rho_c ~ 0.41 rho_Planck``
follows from the area gap of loop quantum gravity and is quoted in that
review. This implements the effective dynamics only, not the underlying
quantum theory, and says nothing about inhomogeneous or anisotropic sectors.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from particlesim.theories.base import Coupling, FieldSpec, Theory

#: Critical density in Planck units, from the loop quantum gravity area gap.
RHO_CRITICAL_PLANCK = 0.41


class EffectiveLQC(Theory):
    """Effective LQC as a Tier B plugin: corrected FLRW dynamics only."""

    id = "lqg.lqc"
    tier = "B"
    dimension = 4
    fields = [FieldSpec("g", "metric", rank=2), FieldSpec("phi", "scalar")]
    couplings = [
        Coupling(
            "inverse_rho_c",
            1.0 / RHO_CRITICAL_PLANCK,
            units="1/planck_density",
            bounds=(0.0, 1e6),
        )
    ]
    frame = "einstein"
    formulation = "standard"
    provenance = (
        "Effective loop quantum cosmology, Ashtekar and Singh 2011 "
        "(CQG 28, 213001), holonomy-corrected Friedmann equation only. "
        "Parameterised by 1/rho_c so that the GR limit sits at zero."
    )
    validity_statement = (
        "homogeneous isotropic sector only; effective dynamics, not the full "
        "quantum theory; says nothing about anisotropic or inhomogeneous cases"
    )

    @property
    def rho_critical(self) -> float:
        """Critical density, or infinity in the general-relativistic limit."""
        inv = self.values["inverse_rho_c"]
        return float("inf") if inv == 0.0 else 1.0 / inv

    def density_correction(self) -> Callable[[float], float]:
        """The factor multiplying ``rho`` in the Friedmann equation."""
        inv = self.values["inverse_rho_c"]

        def correction(rho: float) -> float:
            return 1.0 - rho * inv

        return correction

    def reduced_equations(self, symmetry: str) -> Callable[..., Any]:
        """Corrected ``H^2(rho)`` for the FLRW sector."""
        if symmetry != "flrw":
            raise NotImplementedError(
                f"{self.id} provides reduced equations for the FLRW sector only, "
                f"not {symmetry!r}: the effective dynamics are derived in the "
                "homogeneous isotropic case and do not carry over unchanged"
            )
        import numpy as np

        inv = self.values["inverse_rho_c"]

        def hubble_squared(rho: float) -> float:
            return (8.0 * np.pi / 3.0) * rho * (1.0 - rho * inv)

        return hubble_squared

    def gr_limit(self) -> dict[str, float]:
        return {"inverse_rho_c": 0.0}

    def observable_predictions(self) -> dict[str, Any]:
        return {
            "bounce": self.values["inverse_rho_c"] > 0.0,
            "max_density": self.rho_critical,
            "singularity_resolved": self.values["inverse_rho_c"] > 0.0,
        }

    def regime_of_validity(self, state: Any) -> bool:
        """Valid while the density stays at or below the critical value."""
        rho = getattr(state, "density", state)
        try:
            return bool(float(rho) <= self.rho_critical)
        except (TypeError, ValueError):
            return True
