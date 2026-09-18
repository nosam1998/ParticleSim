"""Fixed mesh refinement: nested boxes, prolongation and restriction.

Issue #48. A puncture needs resolution where the curvature is and cannot
afford it everywhere, so the grid is a hierarchy: a coarse box covering the
domain and finer boxes inside it, each at half the spacing of its parent.

**The grids are vertex-centred, and that is what makes this cheap.**
:func:`particlesim.solvers.nr.bssn.gauge_wave` and its neighbours place
samples at ``k h`` from zero, so refining by two puts a fine sample on
*every* coarse sample and adds one midway between each pair. Two
consequences, and both are the reason to do it this way:

* restriction is injection -- the coarse value is a fine value, copied, with
  no averaging and no error at all;
* prolongation copies at the coincident points and interpolates only at the
  midpoints, where a centred Lagrange stencil of ``2p`` points is accurate
  to order ``2p`` and needs no special case anywhere.

Cell-centred grids have neither: every coarse point falls between fine
points, so restriction averages (second order, and it damps) and
prolongation interpolates everywhere.

**The stencils are periodic, so a fine box needs a buffer.** The emitted
kernels difference with ``roll``, which wraps: on a box that is a piece of
a larger domain the wrap is wrong, and each application contaminates the
outermost ``radius`` points. A Runge-Kutta step applies the operator once
per stage, so after four stages the outer ``4 * radius`` points of a fine
box are meaningless. :data:`buffer_width` is that number, and the buffer is
refilled from the parent rather than evolved -- which is what a buffer zone
is, and why it is as wide as it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: Midpoint interpolation weights, by order of accuracy.
#:
#: The centred Lagrange stencils for the value half-way between two samples:
#: two points give a mean, four give ``(-1, 9, 9, -1)/16``, six give
#: ``(3, -25, 150, 150, -25, 3)/256``. Each is exact on polynomials below
#: its order, which is the property a convergence test measures.
MIDPOINT = {
    2: (0.5, 0.5),
    4: (-1.0 / 16.0, 9.0 / 16.0, 9.0 / 16.0, -1.0 / 16.0),
    6: (3.0 / 256.0, -25.0 / 256.0, 150.0 / 256.0, 150.0 / 256.0, -25.0 / 256.0, 3.0 / 256.0),
}

#: Refinement ratio. Two, everywhere, and the module assumes it.
#:
#: Odd ratios put no fine sample on a coarse one and lose the exact
#: restriction; larger even ratios keep it but need a wider buffer for the
#: same reason a larger stencil does, and nothing here wants one.
RATIO = 2


def buffer_width(stencil_radius: int = 2, dissipation_radius: int = 3, stages: int = 4) -> int:
    """How many points at a fine box's edge a Runge-Kutta step ruins.

    Each stage differences once, and one differencing contaminates the
    outermost ``radius`` points of a box whose stencils wrap. Four stages,
    so four times the widest radius in play -- the Kreiss-Oliger operator's,
    not the derivative's, because dissipation of order ``2r`` reaches
    further than the derivative it protects.

    Twelve for the default fourth-order scheme. That is not a tuning
    parameter: narrower and the interior is wrong, wider and it is only
    wasted work.
    """
    return stages * max(int(stencil_radius), int(dissipation_radius))


def prolong_axis(coarse: np.ndarray, axis: int, order: int = 4) -> np.ndarray:
    """Refine by two along one axis: copy, then interpolate the midpoints.

    Returns an array with ``2 n`` points along ``axis``, the even ones equal
    to the input and the odd ones interpolated. The interpolation is
    periodic along the axis, which is correct on the coarse level -- it *is*
    a torus -- and on a fine box is exactly the wrap the buffer exists to
    cover.
    """
    if order not in MIDPOINT:
        raise ValueError(f"order must be one of {sorted(MIDPOINT)}, got {order}")
    weights = MIDPOINT[order]
    radius = len(weights) // 2

    moved = np.moveaxis(np.asarray(coarse), axis, 0)
    count = moved.shape[0]
    if count < len(weights):
        raise ValueError(
            f"axis {axis} has {count} points, too few for an order-{order} "
            f"midpoint stencil of {len(weights)}"
        )

    midpoints = np.zeros_like(moved)
    for offset, weight in enumerate(weights, start=1 - radius):
        midpoints = midpoints + weight * np.roll(moved, -offset, axis=0)

    out = np.empty((2 * count, *moved.shape[1:]), dtype=moved.dtype)
    out[0::2] = moved
    out[1::2] = midpoints
    return np.moveaxis(out, 0, axis)


def prolong(coarse: np.ndarray, order: int = 4, axes=None) -> np.ndarray:
    """Refine by two along every axis, one axis at a time.

    Dimension by dimension rather than with a tensor-product stencil: the
    composition of one-dimensional interpolations of order ``p`` is of order
    ``p``, and it is ``d`` passes over the array instead of ``(2p)^d``
    multiplies per point.
    """
    out = np.asarray(coarse)
    for axis in range(out.ndim) if axes is None else axes:
        out = prolong_axis(out, axis, order=order)
    return out


def restrict(fine: np.ndarray, axes=None) -> np.ndarray:
    """Coarsen by two: take every second point along every axis.

    Injection, not averaging. On vertex-centred grids the coarse sample *is*
    a fine sample, so there is nothing to average and nothing to lose;
    averaging would introduce an error where there is none and damp the
    solution on the way.
    """
    out = np.asarray(fine)
    slices = [slice(None)] * out.ndim
    for axis in range(out.ndim) if axes is None else axes:
        slices[axis] = slice(None, None, RATIO)
    return out[tuple(slices)]


@dataclass(frozen=True)
class Box:
    """A refinement box: where it sits in its parent, and how big it is.

    ``origin`` is the index of the box's lower corner *in the parent's
    index space*, and ``shape`` counts points on the parent's grid. The box
    covers parent points ``origin[i] .. origin[i] + shape[i] - 1``, and the
    level built from it has ``RATIO * shape[i]`` points at half the spacing.

    Both are in parent indices rather than coordinates on purpose. A box
    whose corner is not on a parent point has no exact restriction, which is
    the property the whole module is built on, and expressing the corner as
    an index makes that impossible to express rather than merely discouraged.
    """

    origin: tuple[int, ...]
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.origin) != len(self.shape):
            raise ValueError("origin and shape must have the same length")
        if any(value < 1 for value in self.shape):
            raise ValueError("box shape entries must be positive")

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def fine_shape(self) -> tuple[int, ...]:
        return tuple(RATIO * value for value in self.shape)

    def slices(self) -> tuple[slice, ...]:
        """Where this box sits in the parent, as an index expression."""
        pairs = zip(self.origin, self.shape, strict=True)
        return tuple(slice(start, start + count) for start, count in pairs)

    def spacing(self, parent_spacing) -> tuple[float, ...]:
        return tuple(float(value) / RATIO for value in parent_spacing)

    def contains(self, parent_shape) -> bool:
        """Does the box fit inside a parent of this shape, without wrapping?"""
        return all(
            0 <= start and start + count <= size
            for start, count, size in zip(self.origin, self.shape, parent_shape, strict=True)
        )


def extract(coarse: Any, box: Box, order: int = 4) -> np.ndarray:
    """The fine-grid values of a box, prolonged from its parent.

    Prolongs the *whole* parent and then cuts the box out, rather than
    cutting first. Interpolating a cut-out array would use its wrapped edge
    as if it were data; interpolating first uses the parent's real
    neighbours, which is what the box's edge points are entitled to.
    Wasteful on a small box and correct, and the alternative is a
    stencil-radius of quiet error exactly where the buffer is supposed to be
    trustworthy.
    """
    refined = prolong(np.asarray(coarse), order=order)
    slices = tuple(
        slice(RATIO * start, RATIO * (start + count))
        for start, count in zip(box.origin, box.shape, strict=True)
    )
    return refined[slices]


def inject(coarse: Any, fine: Any, box: Box) -> np.ndarray:
    """Copy a fine box's restriction back into its parent.

    The parent is returned changed rather than modified, because the states
    here are dictionaries of arrays that get passed around and a routine
    that mutates one of them in place is a bug waiting for the first caller
    who keeps a reference.
    """
    out = np.array(coarse, copy=True)
    out[box.slices()] = restrict(np.asarray(fine))
    return out


__all__ = [
    "MIDPOINT",
    "RATIO",
    "Box",
    "buffer_width",
    "extract",
    "inject",
    "prolong",
    "prolong_axis",
    "restrict",
]
