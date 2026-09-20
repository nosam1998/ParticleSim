"""The polar-areal metric solve, shared by whatever the matter happens to be.

Issue #57. Spherically symmetric, polar-areal coordinates:

    ds^2 = -alpha(t, r)^2 dt^2 + a(t, r)^2 dr^2 + r^2 dOmega^2

The metric functions are not evolved. They are recovered at every stage by
integrating the Hamiltonian constraint and the polar slicing condition
outward from the origin, which is what makes the constraints unable to
drift: they are solved, not monitored.

**The integration is over the Misner-Sharp mass, not over ``a``.** The
Hamiltonian constraint written for ``a`` carries a ``(1 - a^2)/2r`` term
that is zero over zero at the origin; the equivalent equation for the mass,

    dm/dr = 4 pi r^2 rho,        m(0) = 0

has a source vanishing as ``r^2`` and a manifestly regular boundary
condition, and ``a = (1 - 2m/r)^(-1/2)`` follows algebraically.

**The sources are callables because they may depend on the answer.** A
scalar field's energy density carries a factor ``1/a^2``, so its mass
equation is ``dm/dr = 2 pi r^2 (Pi^2 + Phi^2)(1 - 2m/r)`` -- the source
depends on the mass being integrated for. A fluid's does not: ``tau + D``
is what an Eulerian observer measures and is metric-free. Passing arrays
would serve the second and quietly mis-serve the first, so the interface
takes functions of ``(index, midpoint, radius, mass)`` and each caller
closes over whatever it has already interpolated.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class PolarSlicingBreakdown(RuntimeError):
    """Raised when a trapped region forms, which polar-areal slicing cannot cover."""


def midpoints(values: np.ndarray) -> np.ndarray:
    """Fourth-order interpolation to cell midpoints, ``len(values) - 1`` of them.

    A second-order average here would cap the whole metric solve at second
    order regardless of the Runge-Kutta stage count, which is the usual way
    an ostensibly fourth-order code turns out to be second order.
    """
    out = np.empty(len(values) - 1)
    out[1:-1] = (-values[:-3] + 9 * values[1:-2] + 9 * values[2:-1] - values[3:]) / 16.0
    out[0] = (3 * values[0] + 6 * values[1] - values[2]) / 8.0
    out[-1] = (3 * values[-1] + 6 * values[-2] - values[-3]) / 8.0
    return out


Source = Callable[[int, bool, float, float], float]


def solve_mass(radii: np.ndarray, spacing: float, slope: Source) -> np.ndarray:
    """Fourth-order Runge-Kutta for the Misner-Sharp mass, outward from the origin.

    The bound ``2m/r < 1`` is enforced on every *stage argument*, not only
    on the accepted value, because the failure mode is subtler than an
    overshoot that stays overshot. A single stage can step past ``2m/r = 1``
    where the slope turns large and negative, and the Runge-Kutta
    combination then lands on a *negative* mass -- which has ``2m/r < 0``,
    comfortably below one, so a check on the accepted value alone passes it
    through and the caller receives a metric with ``a < 1`` and negative
    mass: unphysical, finite, and plottable.
    """
    middle = radii[:-1] + 0.5 * spacing
    mass = np.empty_like(radii)
    # The source behaves as r^2 near the origin, whose integral from zero to
    # the first point is exactly that source times the radius over three.
    mass[0] = slope(0, False, radii[0], 0.0) * radii[0] / 3.0

    def guarded(index: int, mid: bool, radius: float, value: float) -> float:
        if 2.0 * value >= radius:
            raise PolarSlicingBreakdown(
                f"the constraint integration stepped to 2m/r >= 1 at r = {radius:.4f}. "
                "Polar-areal coordinates do not cover a trapped region, so either the "
                "data is forming a horizon, or the radial grid is too coarse to resolve "
                "the approach to one. Refine the grid to tell the two apart"
            )
        return slope(index, mid, radius, value)

    for index in range(len(radii) - 1):
        first = guarded(index, False, radii[index], mass[index])
        second = guarded(index, True, middle[index], mass[index] + 0.5 * spacing * first)
        third = guarded(index, True, middle[index], mass[index] + 0.5 * spacing * second)
        fourth = guarded(index + 1, False, radii[index + 1], mass[index] + spacing * third)
        mass[index + 1] = mass[index] + spacing / 6.0 * (
            first + 2.0 * second + 2.0 * third + fourth
        )
        if 2.0 * mass[index + 1] >= radii[index + 1]:
            raise PolarSlicingBreakdown(
                f"2m/r reached one at r = {float(radii[index + 1]):.4f}: a trapped region "
                "has formed and polar-areal coordinates do not cover it. This is the "
                "physical end of the run, not a solver failure"
            )
    return mass


def solve_lapse(radii: np.ndarray, spacing: float, mass: np.ndarray, slope: Source) -> np.ndarray:
    """Simpson's rule for ``ln alpha``, normalised so ``alpha a -> 1`` at the edge.

    The boundary condition is that the outermost point matches Schwarzschild,
    which is what makes the lapse a gauge choice with a physical asymptote
    rather than an arbitrary scale.
    """
    middle = radii[:-1] + 0.5 * spacing
    mass_middle = 0.5 * (mass[:-1] + mass[1:])
    logarithm = np.empty_like(radii)
    # Near the origin the integrand is linear in r, so the first interval
    # integrates to the integrand at the first point times half its radius.
    logarithm[0] = 0.5 * radii[0] * slope(0, False, radii[0], mass[0])
    for index in range(len(radii) - 1):
        first = slope(index, False, radii[index], mass[index])
        second = slope(index, True, middle[index], mass_middle[index])
        fourth = slope(index + 1, False, radii[index + 1], mass[index + 1])
        logarithm[index + 1] = logarithm[index] + spacing / 6.0 * (first + 4.0 * second + fourth)
    outer = 1.0 / np.sqrt(1.0 - 2.0 * mass[-1] / radii[-1])
    return np.exp(logarithm - logarithm[-1]) / outer


def solve_polar_metric(radii: np.ndarray, spacing: float, mass_slope: Source, lapse_slope: Source):
    """``(a, alpha, m)``. The lapse solve runs second because it needs the mass."""
    mass = solve_mass(radii, spacing, mass_slope)
    lapse = solve_lapse(radii, spacing, mass, lapse_slope)
    return 1.0 / np.sqrt(1.0 - 2.0 * mass / radii), lapse, mass


def mass_aspect(radii: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Misner-Sharp mass ``m(r) = (r/2)(1 - 1/a^2)``, inverted from the metric."""
    return 0.5 * radii * (1.0 - 1.0 / a**2)


__all__ = [
    "PolarSlicingBreakdown",
    "Source",
    "mass_aspect",
    "midpoints",
    "solve_lapse",
    "solve_mass",
    "solve_polar_metric",
]
