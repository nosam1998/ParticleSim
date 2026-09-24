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

from particlesim.core.interpolate import midpoints


class PolarSlicingBreakdown(RuntimeError):
    """Raised when a trapped region forms, which polar-areal slicing cannot cover."""


Source = Callable[[int, bool, float, float], float]


def solve_mass(
    radii: np.ndarray,
    spacing: float,
    slope: Source,
    first_value: float | None = None,
) -> np.ndarray:
    """Fourth-order Runge-Kutta for the Misner-Sharp mass, outward from the origin.

    ``first_value`` is the mass already accumulated at ``radii[0]``, which
    is what lets a refinement hierarchy be integrated one level at a time.
    The mass at a radius depends only on the matter inside it, all of which
    lives on that level or a finer one, so a level's solve needs nothing
    from its parent beyond where the integration had reached. That locality
    is the whole reason the constraint survives refinement without a
    composite grid: nothing is interpolated across a level boundary, so
    nothing can be lost there.

    The stretch between one level's last point and the next one's first
    belongs to neither and is not this function's to cross. Whoever owns
    the gap crosses it and hands the result in here -- see
    :func:`particlesim.solvers.nr.hierarchy._cross_gap`, which has to do it
    for the mass and the lapse together because the lapse's slope depends
    on the mass.

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
    if first_value is None:
        # The source behaves as r^2 near the origin, whose integral from zero
        # to the first point is exactly that source times the radius over
        # three. Only the innermost level of a hierarchy gets this treatment;
        # every other one is handed where the level below it finished.
        mass[0] = slope(0, False, radii[0], 0.0) * radii[0] / 3.0
    else:
        mass[0] = first_value

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


def solve_linear_mass(
    radii: np.ndarray,
    spacing: float,
    source: np.ndarray,
    source_mid: np.ndarray,
    first_value: float | None = None,
) -> np.ndarray:
    """:func:`solve_mass` for ``dm/dr = sigma (1 - 2m/r)``, the same arithmetic, vectorised.

    ``first_value`` is the mass at ``radii[0]`` when the integration starts
    somewhere other than the origin, as :func:`solve_mass` takes it.

    A massless scalar's mass equation is *linear* in ``m``, with ``sigma``
    known at every node and midpoint before the integration starts. So each
    classical Runge-Kutta step is an affine map, ``m[i+1] = A[i] m[i] + B[i]``,
    whose coefficients depend on the sources alone, and the whole outward
    pass is a first-order linear recurrence that a cumulative product and a
    cumulative sum solve at once. The result is :func:`solve_mass` to
    rounding, not a different discretisation of the same equation.

    That distinction was earned. An integrating-factor solution of the same
    linear equation is also exact in the continuum, also fourth order, and
    about as fast -- and near a horizon it is much worse: ``exp(integral
    4 pi r rho)`` varies exponentially fast in a strong field, polynomial
    quadrature of it is poor. On data where this recurrence reads
    ``max 2m/r = 0.998379`` at 600 cells, within 6e-6 of its converged
    0.998385, the integrating factor read 0.999296 -- a hundred and fifty
    times further off -- and at 300 cells returned ``2m/r = 1.029``.
    Vectorising the arithmetic that already worked keeps its accuracy
    exactly, and keeps its refusals: every stage argument is affine in
    ``m[i]`` too, so the ``2m/r >= 1`` guard is applied to all of them and
    raises at the same place, with the same message, as the loop.

    The sum ``m[i] = sum_j B[j] prod_{k>j} A[k]`` has only positive terms
    wherever the step is inside Runge-Kutta's stability region, so it
    carries no cancellation; outside it the guard fires first.

    Measured at 400 cells: 1.02 ms per mass solve for the loop, 0.069 ms for
    this.
    """
    h = spacing
    middle = radii[:-1] + 0.5 * h
    # m' = s - p m, with s = sigma and p = 2 sigma / r.
    s1, s4, s2 = source[:-1], source[1:], source_mid
    p1, p4, p2 = 2.0 * s1 / radii[:-1], 2.0 * s4 / radii[1:], 2.0 * s2 / middle

    # Every stage, written as (coefficient of m[i], constant).
    k1a, k1b = -p1, s1
    y2a, y2b = 1.0 + 0.5 * h * k1a, 0.5 * h * k1b
    k2a, k2b = -p2 * y2a, s2 - p2 * y2b
    y3a, y3b = 1.0 + 0.5 * h * k2a, 0.5 * h * k2b
    k3a, k3b = -p2 * y3a, s2 - p2 * y3b
    y4a, y4b = 1.0 + h * k3a, h * k3b
    k4a, k4b = -p4 * y4a, s4 - p4 * y4b
    A = 1.0 + h / 6.0 * (k1a + 2.0 * k2a + 2.0 * k3a + k4a)
    B = h / 6.0 * (k1b + 2.0 * k2b + 2.0 * k3b + k4b)

    # The source behaves as r^2 near the origin, whose integral from zero to
    # the first point is exactly that source times the radius over three.
    first = source[0] * radii[0] / 3.0 if first_value is None else float(first_value)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        product = np.concatenate([[1.0], np.cumprod(A)])
        mass = product * (first + np.concatenate([[0.0], np.cumsum(B / product[1:])]))

        # The loop's guard, applied to every stage argument at once. The first
        # failure in the loop's own order is the one reported.
        m = mass[:-1]
        stage = np.stack([m, y2a * m + y2b, y3a * m + y3b, y4a * m + y4b])
        where = np.stack([radii[:-1], middle, middle, radii[1:]])
        stepped = np.any(~(2.0 * stage < where), axis=0)
        reached = ~(2.0 * mass[1:] < radii[1:])
    bad = stepped | reached
    if bad.any():
        index = int(np.argmax(bad))
        if stepped[index]:
            radius = float(where[:, index][~(2.0 * stage[:, index] < where[:, index])][0])
            raise PolarSlicingBreakdown(
                f"the constraint integration stepped to 2m/r >= 1 at r = {radius:.4f}. "
                "Polar-areal coordinates do not cover a trapped region, so either the "
                "data is forming a horizon, or the radial grid is too coarse to resolve "
                "the approach to one. Refine the grid to tell the two apart"
            )
        raise PolarSlicingBreakdown(
            f"2m/r reached one at r = {float(radii[index + 1]):.4f}: a trapped region "
            "has formed and polar-areal coordinates do not cover it. This is the "
            "physical end of the run, not a solver failure"
        )
    return mass


def midpoint_mass(
    radii: np.ndarray,
    spacing: float,
    mass: np.ndarray,
    mass_slope: Source | None = None,
) -> np.ndarray:
    """The mass at cell midpoints, for the lapse quadrature to sample.

    **Hermite, not interpolation from values alone, and the reason is the
    origin.** The Misner-Sharp mass behaves as ``m = C r^3`` near ``r = 0``,
    and the lapse slope carries ``m / r^2``, so a *relative* error in the
    mass at the first midpoint is divided by ``h^2`` and survives in the
    integral as ``O(h^2)``. Both obvious rules are badly wrong exactly there.
    With cell centres at ``(i + 1/2) h`` the first midpoint sits at ``r = h``,
    and against the true ``C h^3``:

    ===========================  ==========
    two-point average             ``+75%``
    one-sided quadratic stencil   ``-37.5%``
    cubic Hermite                 exact
    ===========================  ==========

    The two are not merely inaccurate but wrong in opposite directions, which
    is worse than it sounds: swapping one for the other flips the sign of the
    lapse perturbation at the centre, and the origin is where this solver is
    least forgiving. Doing that alone -- replacing the average with the
    fourth-order stencil, which is the smaller error of the two -- reawakened
    a growing mode at the origin that the average had happened to suppress.

    Hermite avoids the choice. It fits a cubic through ``m`` and ``dm/dr`` at
    the two bracketing nodes, so it is exact for a cubic and therefore exact
    where the mass is one, and it is fourth-order accurate everywhere else
    with no one-sided stencil at either end. The derivative is free: the mass
    source is already a callable and is what the mass solve integrated.

    Without ``mass_slope`` there is no derivative to use and this falls back
    to :func:`midpoints`, which is fourth order away from the origin and
    carries the ``-37.5%`` first-midpoint error described above.
    """
    if mass_slope is None:
        return midpoints(mass)
    derivative = np.array(
        [mass_slope(index, False, radii[index], mass[index]) for index in range(len(radii))]
    )
    return 0.5 * (mass[:-1] + mass[1:]) + spacing / 8.0 * (derivative[:-1] - derivative[1:])


def accumulate_lapse(
    radii: np.ndarray,
    spacing: float,
    mass: np.ndarray,
    slope: Source,
    mass_slope: Source | None = None,
    first_value: float | None = None,
) -> np.ndarray:
    """``ln alpha`` up to an additive constant, by Simpson's rule outward.

    Left unnormalised on purpose. The slicing condition fixes ``d(ln
    alpha)/dr`` locally and the boundary condition fixes one constant for
    the whole slice, so a refinement hierarchy can integrate level by level
    and normalise once at the end -- which is why a fine level's lapse needs
    nothing from its parent except the value where the integration reached
    it.

    ``mass_slope`` matters here for the same reason it does in
    :func:`solve_lapse`, and more: every level of a hierarchy but the
    outermost has matter at small radius relative to its own extent, which
    is precisely where interpolating the midpoint mass from values alone
    goes second order. See :func:`midpoint_mass`.
    """
    middle = radii[:-1] + 0.5 * spacing
    mass_middle = midpoint_mass(radii, spacing, mass, mass_slope)
    logarithm = np.empty_like(radii)
    if first_value is None:
        # Near the origin the integrand is linear in r, so the first interval
        # integrates to the integrand at the first point times half its radius.
        logarithm[0] = 0.5 * radii[0] * slope(0, False, radii[0], mass[0])
    else:
        logarithm[0] = first_value
    for index in range(len(radii) - 1):
        first = slope(index, False, radii[index], mass[index])
        second = slope(index, True, middle[index], mass_middle[index])
        fourth = slope(index + 1, False, radii[index + 1], mass[index + 1])
        logarithm[index + 1] = logarithm[index] + spacing / 6.0 * (first + 4.0 * second + fourth)
    return logarithm


def normalise_lapse(logarithm: np.ndarray, outer_mass: float, outer_radius: float) -> np.ndarray:
    """Fix the one constant: ``alpha a -> 1`` at the outermost point.

    The lapse is a gauge choice with a physical asymptote, and this is where
    the asymptote enters. One constant for the whole slice, however many
    levels it took to build.
    """
    outer = 1.0 / np.sqrt(1.0 - 2.0 * outer_mass / outer_radius)
    return np.exp(logarithm - logarithm[-1]) / outer


def solve_lapse(
    radii: np.ndarray,
    spacing: float,
    mass: np.ndarray,
    slope: Source,
    mass_slope: Source | None = None,
) -> np.ndarray:
    """A single uniform grid: :func:`accumulate_lapse` then :func:`normalise_lapse`.

    The boundary condition is that the outermost point matches Schwarzschild,
    which is what makes the lapse a gauge choice with a physical asymptote
    rather than an arbitrary scale.

    **Pass** ``mass_slope`` **whenever there is one.** The slicing
    condition's slope depends on the mass, so Simpson's midpoint sample needs
    ``m`` at the midpoint, and how that is obtained is what sets the order of
    the whole lapse -- see :func:`midpoint_mass`, which explains why the two
    ways of getting it from values alone are both wrong at the origin and
    what the derivative buys. Omitting it is second order wherever there is
    matter at the centre.

    Simpson's rule offers no protection here. A midpoint error enters every
    interval with weight ``4 dr / 6``, and summing ``1/dr`` of them preserves
    whatever order the interpolation had. The lapse converged at second order
    while the mass beside it converged at fourth, and it survived because of
    what was being measured: the convergence test watched the ADM mass, which
    the mass solve already had right, and the lapse was never measured on its
    own.
    """
    logarithm = accumulate_lapse(radii, spacing, mass, slope, mass_slope)
    return normalise_lapse(logarithm, float(mass[-1]), float(radii[-1]))


def solve_polar_metric(radii: np.ndarray, spacing: float, mass_slope: Source, lapse_slope: Source):
    """``(a, alpha, m)``. The lapse solve runs second because it needs the mass."""
    mass = solve_mass(radii, spacing, mass_slope)
    lapse = solve_lapse(radii, spacing, mass, lapse_slope, mass_slope)
    return 1.0 / np.sqrt(1.0 - 2.0 * mass / radii), lapse, mass


def mass_aspect(radii: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Misner-Sharp mass ``m(r) = (r/2)(1 - 1/a^2)``, inverted from the metric."""
    return 0.5 * radii * (1.0 - 1.0 / a**2)


__all__ = [
    "PolarSlicingBreakdown",
    "Source",
    "accumulate_lapse",
    "mass_aspect",
    "midpoint_mass",
    "normalise_lapse",
    "solve_lapse",
    "solve_linear_mass",
    "solve_mass",
    "solve_polar_metric",
]
