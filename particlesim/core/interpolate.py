"""Interpolation to cell midpoints, shared by anything integrating over a grid.

A Runge-Kutta or Simpson step needs its integrand halfway between the points
it has, and where that integrand is sampled data rather than a formula, it has
to be interpolated. The accuracy of that interpolation is the accuracy of the
whole integration: a second-order midpoint under a fourth-order rule gives a
second-order answer, because the midpoint error enters every interval with an
``O(dr)`` weight and there are ``1/dr`` intervals.

That is an easy thing to get wrong twice, which is why this lives here rather
than beside either caller. It was wrong in the polar-areal metric solve, where
the lapse quadrature averaged the mass to midpoints and converged at second
order beside a fourth-order mass solve, and wrong again and independently in
the laser wakefield solve, whose Runge-Kutta averaged the drive envelope to
its half steps and measured 2.00 where the scheme claims 4.
"""

from __future__ import annotations

import numpy as np

__all__ = ["midpoints"]


def midpoints(values: np.ndarray) -> np.ndarray:
    """Fourth-order interpolation to cell midpoints, ``len(values) - 1`` of them.

    A second-order average here would cap the whole integration at second
    order regardless of the Runge-Kutta stage count, which is the usual way
    an ostensibly fourth-order code turns out to be second order.

    The interior uses the four-point centred stencil, which is exact for a
    cubic. The two end intervals have no fourth point to reach for and use a
    one-sided quadratic, exact only for a parabola -- an ``O(dr^3)`` error at
    one point, which the ``O(dr)`` quadrature weight leaves as ``O(dr^4)``
    overall. That is harmless unless the integrand amplifies it, which is
    exactly what happens at a coordinate origin where the slope carries a
    ``1 / r^2``; see :func:`particlesim.solvers.nr.polar.midpoint_mass` for
    what to do instead there.
    """
    if len(values) < 3:
        # Two points carry no third derivative to cancel, so the average is
        # the best available and the caller is below the stencil's width.
        return 0.5 * (values[:-1] + values[1:])
    out = np.empty(len(values) - 1)
    out[1:-1] = (-values[:-3] + 9 * values[1:-2] + 9 * values[2:-1] - values[3:]) / 16.0
    out[0] = (3 * values[0] + 6 * values[1] - values[2]) / 8.0
    out[-1] = (3 * values[-1] + 6 * values[-2] - values[-3]) / 8.0
    return out
