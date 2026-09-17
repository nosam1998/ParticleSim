"""Conformal compactification and Penrose diagrams (design doc Section 3.5).

A Penrose diagram is a coordinate change that brings infinity to a finite
distance while preserving which events can reach which. It is the standard
way to show causal structure, and it is worth having because the two things
a singularity hypothesis most needs to display, whether a horizon forms and
whether the singularity is spacelike or timelike, are exactly what the
diagram makes visible.

The maps here are exact for the metrics they name, not fits. Where a claim
is made about the resulting picture it is checked: the Schwarzschild horizon
lands on lines at forty-five degrees and the singularity lands on a line of
constant conformal time, both to machine precision, because those follow
from the arctangent identities rather than from how the figure is drawn.
"""

from __future__ import annotations

import numpy as np


def minkowski(t: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compactify Minkowski space: ``(t, r) -> (T, X)``.

    Uses ``U = arctan(t - r)``, ``V = arctan(t + r)`` and ``T = V + U``,
    ``X = V - U``. Radial null rays stay at forty-five degrees, which is the
    property the whole construction exists to preserve.
    """
    t = np.asarray(t, dtype=float)
    r = np.asarray(r, dtype=float)
    u = np.arctan(t - r)
    v = np.arctan(t + r)
    return v + u, v - u


def tortoise(r: np.ndarray, mass: float) -> np.ndarray:
    """``r* = r + 2M ln|r / 2M - 1|``, which pushes the horizon to minus infinity."""
    r = np.asarray(r, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return r + 2.0 * mass * np.log(np.abs(r / (2.0 * mass) - 1.0))


def kruskal(t: np.ndarray, r: np.ndarray, mass: float) -> tuple[np.ndarray, np.ndarray]:
    """Kruskal-Szekeres null coordinates for Schwarzschild.

    Both the exterior and the interior are covered, with the branch chosen by
    whether ``r`` is outside or inside the horizon. Their product satisfies
    ``U V = (1 - r / 2M) exp(r / 2M)``, so the horizon is ``U V = 0`` and the
    singularity is ``U V = 1``, which is what makes the picture's two
    distinctive features exact rather than approximate.
    """
    t = np.asarray(t, dtype=float)
    r = np.asarray(r, dtype=float)
    rs = tortoise(r, mass)
    outside = r > 2.0 * mass
    scale = np.exp(rs / (4.0 * mass))
    u = np.where(outside, -scale * np.exp(-t / (4.0 * mass)), scale * np.exp(-t / (4.0 * mass)))
    v = scale * np.exp(t / (4.0 * mass))
    return u, v


def schwarzschild(t: np.ndarray, r: np.ndarray, mass: float) -> tuple[np.ndarray, np.ndarray]:
    """Penrose coordinates for Schwarzschild: ``(t, r) -> (T, X)``."""
    u, v = kruskal(t, r, mass)
    return np.arctan(v) + np.arctan(u), np.arctan(v) - np.arctan(u)


def schwarzschild_singularity_conformal_time() -> float:
    """The conformal time of ``r = 0``, which is exactly ``pi / 2``.

    On the singularity ``U V = 1``, so ``V = 1 / U`` with both positive, and
    ``arctan(U) + arctan(1/U) = pi / 2`` identically. The singularity is
    therefore a horizontal line in the diagram: it is spacelike, which is the
    whole reason a Schwarzschild interior cannot be avoided once entered.
    """
    return float(np.pi / 2.0)


def reissner_nordstrom_horizons(mass: float, charge: float) -> list[float]:
    """Horizon radii ``M +/- sqrt(M^2 - Q^2)``, of which there may be two, one
    or none.

    The three cases are physically different, and a function returning a
    single number could not express that: two horizons for a sub-extremal
    hole, a degenerate pair at extremality, and no horizon at all above it,
    where the singularity is naked.
    """
    if mass <= 0:
        raise ValueError("mass must be positive")
    disc = mass * mass - charge * charge
    if disc < 0:
        return []
    root = float(np.sqrt(disc))
    # Only positive radii are horizons. At zero charge the inner root sits at
    # r = 0, which is the singularity itself, not a surface: Schwarzschild has
    # one horizon, and returning two would misdescribe it.
    candidates = [mass - root, mass + root]
    radii = sorted({float(r) for r in candidates if r > 0.0})
    return radii


def reissner_nordstrom_lapse(r: np.ndarray, mass: float, charge: float) -> np.ndarray:
    """``f(r) = 1 - 2M/r + Q^2/r^2``."""
    r = np.asarray(r, dtype=float)
    return 1.0 - 2.0 * mass / r + charge**2 / r**2


def singularity_character(mass: float, charge: float) -> str:
    """Whether ``r = 0`` is spacelike or timelike.

    For Schwarzschild the metric function goes to minus infinity at the
    origin, so ``r`` is a time coordinate there and the singularity is
    spacelike: unavoidable once inside. Adding charge flips the sign at small
    radius, making ``r`` spacelike and the singularity timelike, which is
    what allows the Reissner-Nordström diagram to continue past it. The
    distinction is read off the sign rather than asserted.
    """
    if charge == 0.0:
        return "spacelike"
    near_origin = float(reissner_nordstrom_lapse(np.array([1e-8]), mass, charge)[0])
    return "timelike" if near_origin > 0 else "spacelike"
