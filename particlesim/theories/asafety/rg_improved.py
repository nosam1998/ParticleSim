"""Renormalization-group improved Schwarzschild (design doc Sections 4.3, 4.5).

Asymptotic safety posits that Newton's constant runs with scale and
approaches a non-trivial fixed point in the ultraviolet. Bonanno and Reuter
(2000, PRD 62, 043008) substituted the running coupling into the
Schwarzschild metric, identifying the renormalization scale with the inverse
radius, to obtain

    f(r) = 1 - 2 G(r) M / r,
    G(r) = G0 r^3 / (r^3 + omega G0 (r + gamma G0 M))

The point of the construction is that ``G(r) -> 0`` as ``r -> 0``, which
weakens gravity in the deep interior and removes the curvature singularity.
It also predicts that a black hole below a critical mass has no horizon at
all, which is a falsifiable statement and the one this plugin is tested on.

This is an improvement of a *solution*, not a derivation from a
renormalization-group-improved action. It is a Tier B plugin for exactly
that reason: there is no Lagrangian here to vary, only a corrected metric
family. Reading its interior as a prediction of asymptotic safety, rather
than as an illustration of what a running coupling does to one solution,
would be overclaiming.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp

from particlesim.theories.base import Coupling, FieldSpec, Theory

#: Value quoted by Bonanno and Reuter from a one-loop matching.
OMEGA_DEFAULT = 118.0 / (15.0 * np.pi)


class RGImprovedSchwarzschild(Theory):
    """Bonanno-Reuter running-coupling Schwarzschild as a Tier B metric family."""

    id = "asafety.rg_improved"
    tier = "B"
    dimension = 4
    fields = [FieldSpec("g", "metric", rank=2)]
    couplings = [
        Coupling("omega", OMEGA_DEFAULT, units="dimensionless", bounds=(0.0, 1e3)),
        Coupling("gamma", 9.0 / 2.0, units="dimensionless", bounds=(0.0, 1e3)),
    ]
    frame = "einstein"
    formulation = "standard"
    provenance = (
        "Bonanno and Reuter 2000 (PRD 62, 043008): Schwarzschild with the "
        "running Newton constant substituted, scale identified with 1/r. "
        "An improved solution, not a solution of an improved action."
    )
    validity_statement = (
        "a qualitative effective description; the scale identification is a "
        "choice, and the result is not a solution of any field equation"
    )

    def running_g(self, r: np.ndarray | float, mass: float) -> np.ndarray:
        """``G(r)`` in units where the infrared Newton constant is one."""
        r = np.asarray(r, dtype=float)
        omega, gamma = self.values["omega"], self.values["gamma"]
        return r**3 / (r**3 + omega * (r + gamma * mass))

    def lapse_function(self, r: np.ndarray | float, mass: float) -> np.ndarray:
        """``f(r) = 1 - 2 G(r) M / r``."""
        r = np.asarray(r, dtype=float)
        return 1.0 - 2.0 * self.running_g(r, mass) * mass / r

    def metric_family(self, params: dict[str, float]) -> sp.Matrix:
        """The static spherically symmetric metric for a given mass."""
        mass = float(params["mass"])
        r, th = sp.symbols("r theta", positive=True)
        omega, gamma = self.values["omega"], self.values["gamma"]
        g_run = r**3 / (r**3 + omega * (r + gamma * mass))
        f = 1 - 2 * g_run * mass / r
        return sp.diag(-f, 1 / f, r**2, r**2 * sp.sin(th) ** 2)

    def horizon_radii(self, mass: float, r_max: float | None = None) -> list[float]:
        """Radii where ``f(r) = 0``, of which there may be two, one or none.

        Below a critical mass the metric function never reaches zero and the
        object has no horizon. That is the plugin's sharpest prediction and
        the reason this returns a list rather than a single value.
        """
        from scipy.optimize import brentq

        upper = r_max if r_max is not None else max(10.0 * mass, 10.0)
        r = np.linspace(1e-6, upper, 20000)
        f = self.lapse_function(r, mass)
        roots: list[float] = []
        sign_changes = np.nonzero(np.sign(f[1:]) != np.sign(f[:-1]))[0]
        for i in sign_changes:
            roots.append(
                float(brentq(lambda x: float(self.lapse_function(x, mass)), r[i], r[i + 1]))
            )
        return roots

    def critical_mass(self, lo: float = 0.1, hi: float = 100.0) -> float:
        """Smallest mass that still has a horizon, found by bisection."""
        if self.horizon_radii(hi) == []:
            raise ValueError("no horizon even at the upper mass bound")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.horizon_radii(mid):
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-9:
                break
        return hi

    def gr_limit(self) -> dict[str, float]:
        return {"omega": 0.0}

    def observable_predictions(self) -> dict[str, Any]:
        return {
            "horizon_disappears_below_critical_mass": self.values["omega"] > 0.0,
            "newton_constant_vanishes_at_origin": self.values["omega"] > 0.0,
        }
