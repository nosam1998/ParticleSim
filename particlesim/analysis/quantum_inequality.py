"""Ford-Roman quantum inequality check (design doc Sections 3.2, 5.5).

Quantum field theory permits locally negative energy density, but not
arbitrarily much of it for arbitrarily long. For a free, massless, minimally
coupled scalar field in four-dimensional Minkowski space, sampled along an
inertial worldline with a Lorentzian sampling function of characteristic
proper time ``tau0``, Ford and Roman (1995, PRD 51, 4277) proved

    <rho> = (tau0 / pi) * integral rho(tau) / (tau^2 + tau0^2) dtau
          >= -3 / (32 pi^2 tau0^4)

in units G = c = hbar = 1.

**What this module is for.** The design document calls this a plausibility
bound, and that is exactly how it should be read here. The theorem is proved
for one specific field, in flat space, for inertial observers. Applying it to
a stress-energy tensor that a warp metric *requires*, whose matter content is
unspecified, is an extrapolation, not a derivation. It is still informative:
if a configuration exceeds the bound by many orders of magnitude, no
free-scalar-like source can produce it, which is the substance of the
Pfenning-Ford result that Alcubierre bubbles need walls near the Planck
scale. Treat a violation as strong evidence against a source, and treat
compliance as necessary but nowhere near sufficient.

The inequality is an integral over *all* proper time, but any numerical
worldline is finite, so the treatment of the region outside the window is a
choice that changes the answer. It is made explicit rather than hidden:

* ``outside="zero"`` (the default) assumes the density vanishes beyond the
  window. That is exact for a localized source such as a warp bubble wall,
  where an observer sits in flat space before and after, and no
  renormalization is applied. ``weight_captured`` below one is then harmless,
  because the neglected weight multiplies zero.
* ``outside="extend"`` assumes the density outside resembles the window
  average, and renormalizes by the captured weight. Use it for a sustained
  density that the window merely samples.

Getting this wrong is not a rounding error. Renormalizing a window that
misses half the weight of a localized pulse inflates the average by a factor
of two, in the direction that manufactures a violation.

The sampling time must also be short compared to the local curvature scale,
since the flat-space theorem says nothing beyond that (``curvature_ok``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


def ford_roman_bound(tau0: float) -> float:
    """The Ford-Roman lower bound ``-3 / (32 pi^2 tau0^4)`` in geometric units."""
    if tau0 <= 0:
        raise ValueError("sampling time tau0 must be positive")
    return -3.0 / (32.0 * np.pi**2 * tau0**4)


def lorentzian_weight(tau: np.ndarray, tau0: float) -> np.ndarray:
    """Unnormalised Lorentzian sampling function ``(tau0/pi) / (tau^2 + tau0^2)``."""
    return (tau0 / np.pi) / (np.asarray(tau, dtype=float) ** 2 + tau0**2)


@dataclass
class QuantumInequalityResult:
    tau0: float
    sampled_energy: float
    bound: float
    satisfied: bool
    violation_factor: float
    weight_captured: float
    curvature_ok: bool | None
    outside: str = "zero"

    def summary(self) -> dict[str, float | bool | None]:
        return {
            "tau0": self.tau0,
            "sampled_energy": self.sampled_energy,
            "bound": self.bound,
            "satisfied": self.satisfied,
            "violation_factor": self.violation_factor,
            "weight_captured": self.weight_captured,
            "curvature_ok": self.curvature_ok,
            "outside": self.outside,
        }


def check_ford_roman(
    tau: np.ndarray,
    rho: np.ndarray,
    tau0: float,
    kretschmann_max: float | None = None,
    outside: Literal["zero", "extend"] = "zero",
) -> QuantumInequalityResult:
    """Sample ``rho(tau)`` against the Ford-Roman bound.

    ``tau`` is proper time along the worldline, centred so that the region of
    interest sits near ``tau = 0`` (the peak of the sampling function).
    ``rho`` is the energy density that observer measures, ``T_ab u^a u^b``.

    ``outside`` says what to assume beyond the window, as described in the
    module docstring: ``"zero"`` integrates without renormalizing, ``"extend"``
    divides by the captured weight.

    ``violation_factor`` is ``sampled_energy / bound``: greater than one means
    the sampled energy is more negative than the bound allows, and its
    magnitude says by how much.

    When ``kretschmann_max`` is given, the local curvature radius is estimated
    as ``K^(-1/4)`` and ``curvature_ok`` reports whether ``tau0`` is smaller
    than it, which is the regime where quoting a flat-space theorem is
    defensible at all.
    """
    tau = np.asarray(tau, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if tau.shape != rho.shape or tau.ndim != 1 or tau.size < 3:
        raise ValueError("tau and rho must be 1D arrays of equal length >= 3")
    if not np.all(np.diff(tau) > 0):
        raise ValueError("tau must be strictly increasing")

    if outside not in ("zero", "extend"):
        raise ValueError('outside must be "zero" or "extend"')
    w = lorentzian_weight(tau, tau0)
    norm = float(np.trapezoid(w, tau))
    if norm <= 0:
        raise ValueError("sampling window has no weight")
    raw = float(np.trapezoid(w * rho, tau))
    sampled = raw if outside == "zero" else raw / norm
    bound = ford_roman_bound(tau0)

    curvature_ok: bool | None = None
    if kretschmann_max is not None and kretschmann_max > 0:
        curvature_ok = bool(tau0 < kretschmann_max ** (-0.25))

    return QuantumInequalityResult(
        tau0=tau0,
        sampled_energy=sampled,
        bound=bound,
        satisfied=sampled >= bound,
        violation_factor=float(sampled / bound) if bound != 0 else float("inf"),
        weight_captured=norm,
        curvature_ok=curvature_ok,
        outside=outside,
    )


def scan_sampling_times(
    tau: np.ndarray,
    rho: np.ndarray,
    tau0_values: np.ndarray,
    kretschmann_max: float | None = None,
    outside: Literal["zero", "extend"] = "zero",
) -> list[QuantumInequalityResult]:
    """Run :func:`check_ford_roman` over several sampling times.

    The bound weakens as ``tau0^-4``, so short sampling times are easy to
    satisfy and long ones are not. Scanning shows where a configuration
    crosses over.
    """
    return [check_ford_roman(tau, rho, float(t), kretschmann_max, outside) for t in tau0_values]
