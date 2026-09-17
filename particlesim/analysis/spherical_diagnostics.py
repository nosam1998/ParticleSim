"""Diagnostics for spherically symmetric spacetimes (design doc Section 3.5, M1).

What a collapse run needs to report is not the field values but whether a
horizon formed, how large curvature grew, and whether the quantities that
should be conserved were.

A note on what "horizon" means here. In polar-areal slicing the outgoing
null expansion is ``2 / (a r)``, which is strictly positive: no marginally
trapped surface ever appears, because this slicing is horizon-avoiding by
construction. What does happen is that the compactness ``2m/r`` rises toward
one while the lapse collapses. The functions below therefore locate the
surface where ``2m/r = 1``, which for a static exterior is exactly the event
horizon and for a dynamical one is the apparent horizon of the Misner-Sharp
foliation. Calling that "the horizon" is a statement about spherical
symmetry, where the Misner-Sharp mass is well defined, and does not
generalise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq


def compactness_from_a(a: np.ndarray) -> np.ndarray:
    """``2m/r = 1 - 1/a^2`` from the radial metric function.

    Note what this can and cannot produce: in exact arithmetic the result is
    strictly below one for any finite real ``a``. A slicing whose metric
    function stays finite therefore never reports a trapped surface, which is
    not a limitation of the diagnostic but the defining property of
    horizon-avoiding coordinates.

    In floating point the story ends differently and it matters. Once ``a``
    exceeds about ``1 / sqrt(eps)``, roughly 6.7e7 in double precision,
    ``1 / a^2`` underflows against one and the result is *exactly* 1.0. A
    compactness of exactly one therefore means the metric function has
    diverged past what double precision can resolve, not that a trapped
    surface formed. :func:`saturated` reports that case so a run can
    distinguish the two rather than reading a rounding artefact as physics.
    """
    return 1.0 - 1.0 / np.asarray(a, dtype=float) ** 2


def saturated(a: np.ndarray) -> np.ndarray:
    """Where ``compactness_from_a`` has hit the floating-point ceiling of one.

    These samples carry no information beyond "``a`` is enormous here"; the
    true compactness is below one but unresolvable.
    """
    return compactness_from_a(a) >= 1.0


def compactness_from_mass(r: np.ndarray, m: np.ndarray | float) -> np.ndarray:
    """``2m/r`` from a mass profile, which *can* exceed one."""
    return 2.0 * np.asarray(m, dtype=float) / np.asarray(r, dtype=float)


def misner_sharp_mass(r: np.ndarray, a: np.ndarray) -> np.ndarray:
    return 0.5 * np.asarray(r, dtype=float) * compactness_from_a(a)


def horizon_radius(r: np.ndarray, compactness: np.ndarray) -> float | None:
    """Radius where compactness crosses one, interpolated between grid points.

    Takes the compactness profile rather than the metric function, because
    the metric function cannot express a trapped region: inside a horizon
    ``1 - 2m/r`` is negative and its square root is not real. Feeding a mass
    profile through :func:`compactness_from_mass` is what lets this locate a
    horizon at all.

    Returns ``None`` when the profile never reaches one, which is the correct
    answer for subcritical data and for any horizon-avoiding slicing.

    The crossing is found on a cubic spline rather than by taking the nearest
    grid point: reporting a horizon radius to the nearest cell would be
    reporting the resolution rather than the physics. Non-finite samples are
    dropped, since a diverging metric function should not turn a diagnostic
    into an exception.
    """
    r = np.asarray(r, dtype=float)
    f = np.asarray(compactness, dtype=float) - 1.0
    good = np.isfinite(f) & np.isfinite(r)
    r, f = r[good], f[good]
    if r.size < 4:
        return None
    order = np.argsort(r)
    r, f = r[order], f[order]
    sign_change = np.nonzero(np.sign(f[1:]) != np.sign(f[:-1]))[0]
    if sign_change.size == 0:
        return None
    i = int(sign_change[0])
    spline = CubicSpline(r, f)
    try:
        return float(brentq(spline, float(r[i]), float(r[i + 1]), xtol=1e-14, rtol=1e-15))
    except ValueError:  # pragma: no cover - guarded by the sign change above
        return None


def ricci_scalar(a: np.ndarray, Phi: np.ndarray, Pi: np.ndarray) -> np.ndarray:
    """Ricci scalar of a massless-scalar spacetime, ``R = 8 pi (Phi^2 - Pi^2) / a^2``.

    For a massless minimally coupled field the trace of the stress-energy is
    ``T = -(grad phi)^2``, so ``R = -8 pi T`` closes in the field variables
    with no derivatives of the metric at all. That matters numerically: the
    metric functions come from an ODE solve, and differentiating them twice
    would give up several orders of accuracy for a quantity available exactly.
    """
    a = np.asarray(a, dtype=float)
    return 8.0 * np.pi * (np.asarray(Phi) ** 2 - np.asarray(Pi) ** 2) / a**2


def vacuum_kretschmann(r: np.ndarray, mass: np.ndarray | float) -> np.ndarray:
    """``48 m^2 / r^6``, the Kretschmann scalar where the stress-energy vanishes.

    Exact in vacuum, and a good estimate in the near-vacuum exterior of a
    collapsing region. It is deliberately not offered as a general result:
    inside matter the scalar picks up Ricci terms this expression omits, and
    computing those from the numerically solved metric functions needs second
    derivatives of an ODE solution. That is deferred rather than approximated
    silently, since a curvature invariant quoted two orders of accuracy below
    everything around it is worse than none.
    """
    return 48.0 * np.asarray(mass) ** 2 / np.asarray(r, dtype=float) ** 6


@dataclass
class ConservationCheck:
    """Drift in a quantity that ought to be constant."""

    name: str
    initial: float
    current: float

    @property
    def absolute_drift(self) -> float:
        return self.current - self.initial

    @property
    def relative_drift(self) -> float:
        if self.initial == 0.0:
            return 0.0 if self.current == 0.0 else float("inf")
        return self.absolute_drift / self.initial

    def within(self, tolerance: float) -> bool:
        return abs(self.relative_drift) <= tolerance


@dataclass
class SphericalDiagnostics:
    """One slice's worth of diagnostics."""

    t: float
    adm_mass: float
    max_compactness: float
    min_lapse: float
    horizon_radius: float | None
    max_abs_ricci: float
    max_vacuum_kretschmann: float

    def as_row(self) -> dict[str, float]:
        """Flat scalars for a run monitor or time series."""
        return {
            "t": self.t,
            "adm_mass": self.adm_mass,
            "max_compactness": self.max_compactness,
            "min_lapse": self.min_lapse,
            "horizon_radius": -1.0 if self.horizon_radius is None else self.horizon_radius,
            "max_abs_ricci": self.max_abs_ricci,
            "max_vacuum_kretschmann": self.max_vacuum_kretschmann,
        }

    @property
    def collapsing(self) -> bool:
        """Whether this slice shows the signature of collapse in polar slicing."""
        return self.max_compactness > 0.99 and self.min_lapse < 1e-2


def diagnose(
    t: float,
    r: np.ndarray,
    a: np.ndarray,
    alpha: np.ndarray,
    Phi: np.ndarray,
    Pi: np.ndarray,
) -> SphericalDiagnostics:
    """Full diagnostic set for one slice."""
    mass = misner_sharp_mass(r, a)
    return SphericalDiagnostics(
        t=float(t),
        adm_mass=float(mass[-1]),
        max_compactness=float(compactness_from_a(a).max()),
        min_lapse=float(np.min(alpha)),
        horizon_radius=horizon_radius(r, compactness_from_mass(r, mass)),
        max_abs_ricci=float(np.abs(ricci_scalar(a, Phi, Pi)).max()),
        max_vacuum_kretschmann=float(vacuum_kretschmann(r, mass).max()),
    )


DIAGNOSTIC_COLUMNS = [
    "t",
    "adm_mass",
    "max_compactness",
    "min_lapse",
    "horizon_radius",
    "max_abs_ricci",
    "max_vacuum_kretschmann",
]
