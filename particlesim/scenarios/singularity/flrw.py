"""FLRW backgrounds with big bang and big crunch diagnostics (design doc 3.1, 3.5).

Integrates

    (a'/a)^2 = (8 pi / 3) sum_i rho_i0 a^(-3(1+w_i)) - k / a^2 + correction

where the optional correction is what a Tier B plugin supplies: loop quantum
cosmology's ``(1 - rho / rho_c)`` factor is the canonical example, and it
turns the big bang into a bounce.

The integration stops at a singularity rather than running into it. A solver
that keeps stepping as the scale factor goes to zero produces ever-larger
numbers that are meaningless, and the interesting output is *when* it
stopped and *why*, not the values afterwards.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

# Equation-of-state parameters for the usual components.
W_RADIATION = 1.0 / 3.0
W_MATTER = 0.0
W_CURVATURE = -1.0 / 3.0
W_LAMBDA = -1.0


@dataclass
class FLRWSolution:
    t: np.ndarray
    a: np.ndarray
    hubble: np.ndarray
    density: np.ndarray
    outcome: str
    bounced: bool = False
    bounce_time: float | None = None
    min_scale_factor: float = 0.0

    def summary(self) -> dict[str, object]:
        return {
            "outcome": self.outcome,
            "bounced": self.bounced,
            "bounce_time": self.bounce_time,
            "min_scale_factor": self.min_scale_factor,
            "max_density": float(self.density.max()),
            "final_time": float(self.t[-1]),
        }


@dataclass
class FLRWBackground:
    """A homogeneous isotropic background.

    ``components`` maps an equation-of-state parameter ``w`` to its density
    today (at ``a = 1``). ``curvature`` is ``k``, positive for a closed
    universe, which is the case that recollapses.

    ``density_correction`` multiplies the density in the Friedmann equation
    and is the hook a Tier B theory plugin fills. The default is one, which
    is general relativity.
    """

    components: dict[float, float] = field(default_factory=lambda: {W_MATTER: 1.0})
    curvature: float = 0.0
    density_correction: Callable[[float], float] | None = None

    def density(self, a: float | np.ndarray) -> float | np.ndarray:
        a = np.asarray(a, dtype=float)
        total = np.zeros_like(a)
        for w, rho0 in self.components.items():
            total = total + rho0 * a ** (-3.0 * (1.0 + w))
        return total

    def hubble_squared(self, a: float) -> float:
        rho = float(self.density(a))
        effective = rho * (self.density_correction(rho) if self.density_correction else 1.0)
        return (8.0 * np.pi / 3.0) * effective - self.curvature / a**2

    def evolve(
        self,
        a0: float = 1.0,
        t_max: float = 100.0,
        expanding: bool = False,
        a_floor: float = 1e-8,
        rtol: float = 1e-10,
        atol: float = 1e-12,
        n_out: int = 400,
    ) -> FLRWSolution:
        """Integrate from ``a0``; ``expanding`` picks the branch of the square root.

        Stops on a singularity (``a`` reaching ``a_floor``), on a turning
        point where ``H^2`` would go negative, or at ``t_max``.
        """
        sign = 1.0 if expanding else -1.0

        def rhs(_t, y):
            a = max(float(y[0]), a_floor)
            h2 = self.hubble_squared(a)
            return [sign * a * np.sqrt(h2) if h2 > 0 else 0.0]

        def hit_floor(_t, y):
            return float(y[0]) - a_floor

        hit_floor.terminal = True
        hit_floor.direction = -1

        def turning_point(_t, y):
            return self.hubble_squared(max(float(y[0]), a_floor))

        turning_point.terminal = True
        turning_point.direction = -1

        sol = solve_ivp(
            rhs,
            (0.0, t_max),
            [a0],
            method="DOP853",
            t_eval=np.linspace(0.0, t_max, n_out),
            events=[hit_floor, turning_point],
            rtol=rtol,
            atol=atol,
        )
        t, a = sol.t, sol.y[0]
        if sol.t_events[0].size:
            t = np.append(t, sol.t_events[0][0])
            a = np.append(a, a_floor)
            outcome = "singularity"
        elif sol.t_events[1].size:
            t = np.append(t, sol.t_events[1][0])
            a = np.append(a, sol.y_events[1][0][0])
            outcome = "turning_point"
        else:
            outcome = "ran_to_t_max"

        h = np.array([np.sqrt(max(self.hubble_squared(ai), 0.0)) * sign for ai in a])
        rho = np.asarray(self.density(a))
        bounced = outcome == "turning_point" and not expanding
        return FLRWSolution(
            t=t,
            a=a,
            hubble=h,
            density=rho,
            outcome=outcome,
            bounced=bounced,
            bounce_time=float(t[-1]) if bounced else None,
            min_scale_factor=float(a.min()),
        )


def lqc_correction(rho_critical: float) -> Callable[[float], float]:
    """Loop quantum cosmology's ``(1 - rho / rho_c)`` factor.

    This is the canonical Tier B modification: it leaves the low-density
    Friedmann equation untouched and replaces the big bang with a bounce at
    ``rho = rho_c``, where the correction vanishes and the expansion rate
    goes to zero. It is supplied here as a hook illustration; the full plugin
    with its provenance belongs in the LQC theory module (issue #24).
    """
    if rho_critical <= 0:
        raise ValueError("critical density must be positive")

    def correction(rho: float) -> float:
        return 1.0 - rho / rho_critical

    return correction


def matter_dominated_scale_factor(t: np.ndarray, t_big_bang: float = 0.0) -> np.ndarray:
    """``a ∝ (t - t_bb)^(2/3)``, the flat matter-only closed form."""
    return np.maximum(np.asarray(t, dtype=float) - t_big_bang, 0.0) ** (2.0 / 3.0)


def radiation_dominated_scale_factor(t: np.ndarray, t_big_bang: float = 0.0) -> np.ndarray:
    """``a ∝ (t - t_bb)^(1/2)``, the flat radiation-only closed form."""
    return np.maximum(np.asarray(t, dtype=float) - t_big_bang, 0.0) ** 0.5
