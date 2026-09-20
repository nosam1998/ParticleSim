"""Interface reconstruction, from piecewise-constant to WENO5.

Issue #57. A finite-volume scheme stores cell averages and needs values at
the cell faces; how it builds them is what sets the order of the whole
method, so each scheme here is held to its *nominal* order rather than to a
tolerance. A smooth pulse carried at uniform velocity and pressure has an
exact solution -- pure translation -- and measuring the error against it at
two resolutions gives an order that either is the advertised one or is not.

**The right state is the mirror of the left state, by construction.** Every
scheme below is written once, as a left-biased interface value, and the
right-biased one is obtained by reconstructing the reversed field and
reversing back. That is not a shortcut: a reconstruction whose two sides
are written out separately can disagree in a way no smooth test detects,
because a smooth field is locally symmetric anyway. Doing it this way makes
the symmetry exact, and :func:`reconstruct` is checked against a reversed
field to confirm it.

**The nominal order is the spatial one, and a scheme can be prevented from
showing it.** WENO5's fifth order lives in the reconstruction alone; run
with a third-order time integrator at a fixed Courant number, the measured
order is three, because the time error is then the larger one and it does
not care how good the faces are. The measurement is right and the
conclusion drawn from it would be wrong. To see the five, the step has to
shrink faster than the cell -- ``dt ~ dx^(5/3)`` -- and the suite does
exactly that, and also records the capped value so the trap is visible
rather than avoided.

**PPM interpolates its faces to fourth order and converges at barely more
than second.** Not a bug and not a bound: the Colella-Woodward limiter
cannot tell a smooth extremum from an overshoot, so where the profile turns
over it cuts the parabola back and the reconstruction is locally first
order. On a sine the limiter touches exactly **six faces at every
resolution** -- 32 points or 512, always the two extrema and one neighbour
each -- and since that count does not grow with the grid, their share of the
error falls slowly: the advected pulse measures 1.95, 2.20, 2.30 over
successive doublings, short of three and nowhere near the four the faces
are interpolated to. The same reconstruction with the limiting removed
measures 3.99, 4.00, 4.00.

Colella and Sekora's extremum branch is what repairs it, and
``"ppm-extremum"`` carries it. The way that shows up is worth stating: on a
smooth profile it reproduces the *unlimited* faces to every digit, meaning
it correctly does nothing at all, while at a jump from 1 to 3 it still
reconstructs inside ``[1, 3]``. Both are kept, because the distance between
them is the point -- a scheme named for its high order can be delivering
second, and only a measurement says which one you have.
"""

from __future__ import annotations

import numpy as np

#: The schemes :func:`reconstruct` accepts, in increasing order of accuracy.
SCHEMES = ("constant", "minmod", "mc", "ppm", "ppm-extremum", "weno5")

#: Order each scheme reaches on a smooth field, given a time step that allows
#: it. These are *measured* on the advected pulse, not advertised: ``"ppm"``
#: interpolates its faces to fourth order and converges at second, and the
#: entry says the second because that is what it does.
NOMINAL_ORDER = {
    "constant": 1,
    "minmod": 2,
    "mc": 2,
    "ppm": 2,
    "ppm-extremum": 4,
    "weno5": 5,
}

#: Colella and Sekora's factor: how much more curved than its neighbours a
#: cell's parabola may be before the extremum branch cuts it back.
CURVATURE_TOLERANCE = 1.25

#: Ghost cells every scheme here can be evaluated with. WENO5 sets it.
GHOSTS = 3

#: Added to the WENO smoothness indicators. Small enough not to perturb the
#: weights where the field is smooth, non-zero so a constant field does not
#: divide by nothing.
WENO_EPSILON = 1e-40


def minmod(first, second):
    """The smaller slope, or zero when they disagree in sign."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    return np.where(
        first * second <= 0.0, 0.0, np.where(np.abs(first) < np.abs(second), first, second)
    )


def monotonised_central(first, second):
    """The MC limiter: the centred slope, cut back to twice either one-sided one.

    Less diffusive than :func:`minmod` at the same second order, which shows
    up as a sharper contact rather than as a different convergence rate --
    both are two, and the constant in front differs.
    """
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    centred = 0.5 * (first + second)
    limited = minmod(2.0 * first, 2.0 * second)
    return np.where(first * second <= 0.0, 0.0, minmod(centred, limited))


def _parabolic_faces(values):
    """The fourth-order interpolant at every ``i + 1/2`` face, before any limiting."""
    return (
        7.0 * (values + np.roll(values, -1)) - (np.roll(values, 1) + np.roll(values, -2))
    ) / 12.0


def _limited_curvature(own, *neighbours):
    """The smallest curvature in the stencil, or zero if they do not all agree.

    Disagreement in sign means the stencil straddles something that is not a
    smooth extremum, and the limiter then has no reason to spare the
    parabola. Agreement means it is one, and the cell is allowed the
    curvature its neighbours already have.
    """
    stacked = np.stack((own,) + neighbours)
    signs = np.sign(stacked)
    agree = np.all(signs == signs[0], axis=0) & (signs[0] != 0.0)
    magnitudes = np.stack(
        [np.abs(own)] + [CURVATURE_TOLERANCE * np.abs(other) for other in neighbours]
    )
    return np.where(agree, signs[0] * np.min(magnitudes, axis=0), 0.0)


def _monotonised_slope(values):
    """Colella and Woodward's ``delta_m``: the centred slope, cut to its neighbours.

    Used to build the face values rather than the in-cell parabola. Without
    it the fourth-order interpolant can place a face outside the two cell
    averages it sits between -- at a jump from 1 to 3 it undershoots to
    0.833 -- and no amount of in-cell limiting afterwards puts it back.
    """
    behind = values - np.roll(values, 1)
    ahead = np.roll(values, -1) - values
    centred = 0.5 * (behind + ahead)
    bounded = np.minimum(np.abs(centred), 2.0 * np.minimum(np.abs(behind), np.abs(ahead)))
    return np.where(behind * ahead > 0.0, np.sign(centred) * bounded, 0.0)


def _parabolic_right_edge(values):
    """PPM's right face of each cell, after the Colella-Woodward limiting."""
    slope = _monotonised_slope(values)
    face = values + 0.5 * (np.roll(values, -1) - values) - (np.roll(slope, -1) - slope) / 6.0
    right = face
    left = np.roll(face, 1)

    flat = (right - values) * (values - left) <= 0.0
    left = np.where(flat, values, left)
    right = np.where(flat, values, right)

    width = right - left
    offset = values - 0.5 * (left + right)
    steepened_left = (~flat) & (width * offset > width**2 / 6.0)
    left = np.where(steepened_left, 3.0 * values - 2.0 * right, left)
    steepened_right = (~flat) & (~steepened_left) & (-(width**2) / 6.0 > width * offset)
    return np.where(steepened_right, 3.0 * values - 2.0 * left, right)


def _extremum_right_edge(values):
    """The same parabola, limited by curvature instead of by monotonicity.

    Colella and Sekora's repair. A cell whose two edge offsets disagree in
    sign is at an extremum, and flattening it there is what costs PPM its
    order; so instead the parabola's curvature is compared with the
    curvature of the surrounding cell averages, and kept whenever they
    agree. Where they do not -- a discontinuity -- the curvature limit goes
    to zero and the parabola flattens exactly as before.
    """
    ahead1 = np.roll(values, -1)
    ahead2 = np.roll(values, -2)
    behind1 = np.roll(values, 1)
    behind2 = np.roll(values, 2)

    face = _parabolic_faces(values)
    overshooting = (face - values) * (ahead1 - face) < 0.0
    face = np.where(
        overshooting,
        0.5 * (values + ahead1)
        - _limited_curvature(
            3.0 * (values - 2.0 * face + ahead1),
            behind1 - 2.0 * values + ahead1,
            values - 2.0 * ahead1 + ahead2,
        )
        / 6.0,
        face,
    )

    right = face
    left = np.roll(face, 1)
    below = values - left
    above = right - values

    curvature = -12.0 * (values - 0.5 * (left + right))
    limited = _limited_curvature(
        curvature,
        behind1 - 2.0 * values + ahead1,
        behind2 - 2.0 * behind1 + values,
        values - 2.0 * ahead1 + ahead2,
    )
    flat = np.abs(curvature) < np.finfo(float).tiny
    ratio = np.where(flat, 0.0, limited / np.where(flat, 1.0, curvature))

    at_extremum = (below * above <= 0.0) | ((values - behind2) * (ahead2 - values) <= 0.0)
    return np.where(
        at_extremum,
        values + (right - values) * ratio,
        np.where(np.abs(above) >= 2.0 * np.abs(below), values + 2.0 * below, right),
    )


def _weno5(values):
    """Fifth-order WENO: three parabolas, weighted by their own smoothness."""
    back2 = np.roll(values, 2)
    back1 = np.roll(values, 1)
    ahead1 = np.roll(values, -1)
    ahead2 = np.roll(values, -2)

    candidates = (
        (2.0 * back2 - 7.0 * back1 + 11.0 * values) / 6.0,
        (-back1 + 5.0 * values + 2.0 * ahead1) / 6.0,
        (2.0 * values + 5.0 * ahead1 - ahead2) / 6.0,
    )
    smoothness = (
        13.0 / 12.0 * (back2 - 2.0 * back1 + values) ** 2
        + 0.25 * (back2 - 4.0 * back1 + 3.0 * values) ** 2,
        13.0 / 12.0 * (back1 - 2.0 * values + ahead1) ** 2 + 0.25 * (back1 - ahead1) ** 2,
        13.0 / 12.0 * (values - 2.0 * ahead1 + ahead2) ** 2
        + 0.25 * (3.0 * values - 4.0 * ahead1 + ahead2) ** 2,
    )
    weights = [
        ideal / (WENO_EPSILON + rough) ** 2
        for ideal, rough in zip((0.1, 0.6, 0.3), smoothness, strict=True)
    ]
    total = weights[0] + weights[1] + weights[2]
    return sum(w * q for w, q in zip(weights, candidates, strict=True)) / total


def _left_interface(values, scheme):
    """The value at every ``i + 1/2`` face, extrapolated from cell ``i``."""
    values = np.asarray(values, dtype=float)
    if scheme == "constant":
        return values.copy()
    if scheme in ("minmod", "mc"):
        backward = values - np.roll(values, 1)
        forward = np.roll(values, -1) - values
        limiter = minmod if scheme == "minmod" else monotonised_central
        return values + 0.5 * limiter(backward, forward)
    if scheme == "ppm":
        return _parabolic_right_edge(values)
    if scheme == "ppm-extremum":
        return _extremum_right_edge(values)
    if scheme == "weno5":
        return _weno5(values)
    raise ValueError(f"unknown reconstruction {scheme!r}; expected one of {SCHEMES}")


def reconstruct(values, scheme="minmod"):
    """``(left, right)`` at every face ``i + 1/2`` of a periodic cell array.

    ``left[i]`` is cell ``i`` extrapolated forward, ``right[i]`` is cell
    ``i + 1`` extrapolated back. The second is the first applied to the
    reversed field, so the two sides cannot drift apart.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"reconstruction is one-dimensional, got shape {values.shape}")
    left = _left_interface(values, scheme)
    mirrored = _left_interface(values[::-1], scheme)[::-1]
    return left, np.roll(mirrored, -1)


__all__ = [
    "CURVATURE_TOLERANCE",
    "GHOSTS",
    "NOMINAL_ORDER",
    "SCHEMES",
    "WENO_EPSILON",
    "minmod",
    "monotonised_central",
    "reconstruct",
]
