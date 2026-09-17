"""Oppenheimer-Snyder dust collapse (design doc Section 3.5).

A ball of pressureless dust, uniform in density and initially at rest,
collapsing from radius ``R0``. Its interior is a closed
Friedmann-Lemaitre-Robertson-Walker patch and its exterior is Schwarzschild,
matched at the surface. The whole solution is closed form, which is exactly
why it belongs in the battery: any collapse code that cannot reproduce a
cycloid is not going to be trusted on a scalar field.

Parametric solution, with ``eta`` running from 0 to pi:

    R(eta) = (R0 / 2) (1 + cos eta)
    tau(eta) = sqrt(R0^3 / (8 M)) (eta + sin eta)

so the surface reaches ``R = 0`` at proper time
``tau_collapse = (pi / 2) sqrt(R0^3 / (2 M))`` and crosses its own horizon
``R = 2M`` strictly before that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class OppenheimerSnyder:
    """Uniform dust ball of gravitational mass ``mass`` released from ``r0``."""

    mass: float
    r0: float

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError("mass must be positive")
        if self.r0 <= 2.0 * self.mass:
            raise ValueError(
                f"initial radius {self.r0} is inside the horizon 2M = {2 * self.mass}; "
                "the dust must start outside its own Schwarzschild radius"
            )

    @property
    def collapse_proper_time(self) -> float:
        """Proper time at the surface from release to the central singularity."""
        return 0.5 * np.pi * np.sqrt(self.r0**3 / (2.0 * self.mass))

    @property
    def horizon_radius(self) -> float:
        return 2.0 * self.mass

    def radius(self, eta: np.ndarray) -> np.ndarray:
        return 0.5 * self.r0 * (1.0 + np.cos(np.asarray(eta, dtype=float)))

    def proper_time(self, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=float)
        return np.sqrt(self.r0**3 / (8.0 * self.mass)) * (eta + np.sin(eta))

    def eta_at_radius(self, radius: float) -> float:
        """Cycloid parameter when the surface passes through ``radius``."""
        if not 0.0 <= radius <= self.r0:
            raise ValueError(f"radius {radius} is outside (0, r0]")
        return float(np.arccos(2.0 * radius / self.r0 - 1.0))

    def proper_time_to_radius(self, radius: float) -> float:
        return float(self.proper_time(self.eta_at_radius(radius)))

    @property
    def horizon_crossing_proper_time(self) -> float:
        """When the surface crosses ``R = 2M``, which precedes the singularity."""
        return self.proper_time_to_radius(self.horizon_radius)

    def density(self, eta: np.ndarray) -> np.ndarray:
        """Uniform interior density ``3M / (4 pi R^3)``, diverging at the crunch."""
        r = self.radius(eta)
        with np.errstate(divide="ignore"):
            return 3.0 * self.mass / (4.0 * np.pi * r**3)

    def kretschmann_at_surface(self, eta: np.ndarray) -> np.ndarray:
        """``48 M^2 / R^6`` just outside the surface, where the exterior is vacuum."""
        return 48.0 * self.mass**2 / self.radius(eta) ** 6

    def eta_at_proper_time(self, tau: float) -> float:
        """Invert ``tau(eta)``, which has no closed-form inverse."""
        if not 0.0 <= tau <= self.collapse_proper_time:
            raise ValueError(f"proper time {tau} is outside [0, {self.collapse_proper_time}]")
        if tau == 0.0:
            return 0.0
        return float(brentq(lambda e: self.proper_time(e) - tau, 0.0, np.pi, xtol=1e-14))

    def trajectory(self, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """``(proper_time, radius)`` sampled uniformly in the cycloid parameter."""
        eta = np.linspace(0.0, np.pi, n)
        return self.proper_time(eta), self.radius(eta)
