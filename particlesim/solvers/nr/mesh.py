"""Fixed mesh refinement: nested boxes, prolongation and restriction.

Issue #48. A puncture needs resolution where the curvature is and cannot
afford it everywhere, so the grid is a hierarchy: a coarse box covering the
domain and finer boxes inside it, each at half the spacing of its parent.

**The default grids are vertex-centred, making restriction exact.**
:func:`particlesim.solvers.nr.bssn.gauge_wave` and its neighbours place
samples at ``k h`` from zero, so refining by two puts a fine sample on
*every* coarse sample and adds one midway between each pair. Two
consequences, and both are the reason to do it this way:

* restriction is injection -- the coarse value is a fine value, copied, with
  no averaging and no error at all;
* prolongation copies at the coincident points and interpolates only at the
  midpoints, where a centred Lagrange stencil of ``2p`` points is accurate
  to order ``2p`` and needs no special case anywhere.

``centering="cell"`` instead puts point samples at cell centres. It allows
every nested grid to share a reflection centre without sampling a puncture.
These are point values, not finite-volume averages: both transfers use
Lagrange interpolation at the requested order, including restriction.

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

from particlesim.symbolic import codegen

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

# Values one quarter of a coarse spacing to the right of each sample.
# Reflect the stencil for the left child. These interpolate point values;
# they are not conservative finite-volume reconstruction coefficients.
QUARTER = {
    2: (3 / 4, 1 / 4),
    4: (-7 / 128, 105 / 128, 35 / 128, -5 / 128),
    6: (77 / 8192, -693 / 8192, 3465 / 4096, 1155 / 4096, -495 / 8192, 63 / 8192),
}


def _check_centering(centering):
    if centering not in ("vertex", "cell"):
        raise ValueError("centering must be 'vertex' or 'cell'")


def _module(backend):
    # Use the same float64 policy as the evolved fields and emitted kernels.
    return codegen._backend(backend)


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


def prolong_axis(
    coarse: Any, axis: int, order: int = 4, *, centering: str = "vertex", backend="numpy"
) -> np.ndarray:
    """Refine point samples by two along one periodic axis.

    Returns an array with ``2 n`` points along ``axis``, the even ones equal
    to the input and the odd ones interpolated. The interpolation is
    periodic along the axis, which is correct on the coarse level -- it *is*
    a torus -- and on a fine box is exactly the wrap the buffer exists to
    cover. With ``centering="cell"`` both children are interpolated, at
    offsets ``-h/4`` and ``h/4`` from their parent. ``backend="jax"`` keeps
    the operation on the device and supports tracing a complete step.
    """
    _check_centering(centering)
    if order not in MIDPOINT:
        raise ValueError(f"order must be one of {sorted(MIDPOINT)}, got {order}")
    weights = MIDPOINT[order]
    radius = len(weights) // 2

    xp = _module(backend)
    moved = xp.moveaxis(xp.asarray(coarse), axis, 0)
    count = moved.shape[0]
    if count < len(weights):
        raise ValueError(
            f"axis {axis} has {count} points, too few for an order-{order} "
            f"midpoint stencil of {len(weights)}"
        )

    if centering == "vertex":
        midpoints = xp.zeros_like(moved)
        for offset, weight in enumerate(weights, start=1 - radius):
            midpoints = midpoints + weight * xp.roll(moved, -offset, axis=0)
        left, right = moved, midpoints
    else:
        left, right = xp.zeros_like(moved), xp.zeros_like(moved)
        for offset, weight in enumerate(QUARTER[order], start=1 - radius):
            left = left + weight * xp.roll(moved, offset, axis=0)
            right = right + weight * xp.roll(moved, -offset, axis=0)
    out = xp.stack((left, right), axis=1).reshape((2 * count, *moved.shape[1:]))
    return xp.moveaxis(out.astype(moved.dtype), 0, axis)


def prolong(coarse: Any, order=4, axes=None, *, centering="vertex", backend="numpy"):
    """Refine by two along every axis, one axis at a time.

    Dimension by dimension rather than with a tensor-product stencil: the
    composition of one-dimensional interpolations of order ``p`` is of order
    ``p``, and it is ``d`` passes over the array instead of ``(2p)^d``
    multiplies per point.
    """
    out = _module(backend).asarray(coarse)
    for axis in range(out.ndim) if axes is None else axes:
        out = prolong_axis(out, axis, order=order, centering=centering, backend=backend)
    return out


def restrict(fine: Any, axes=None, *, centering="vertex", order=4, backend="numpy"):
    """Coarsen point samples by two along every selected periodic axis.

    Injection, not averaging. On vertex-centred grids the coarse sample *is*
    a fine sample, so there is nothing to average and nothing to lose;
    averaging would introduce an error where there is none and damp the
    solution on the way. For cell centres the parent is between fine
    samples, so midpoint interpolation of ``order`` replaces injection.
    """
    _check_centering(centering)
    xp = _module(backend)
    out = xp.asarray(fine)
    if centering == "cell":
        if order not in MIDPOINT:
            raise ValueError(f"order must be one of {sorted(MIDPOINT)}, got {order}")
        for axis in range(out.ndim) if axes is None else axes:
            if out.shape[axis] < order or out.shape[axis] % RATIO:
                raise ValueError(
                    "cell restriction needs an even axis at least as long as its stencil"
                )
            values = xp.zeros_like(out)
            for offset, weight in enumerate(MIDPOINT[order], start=1 - order // 2):
                values = values + weight * xp.roll(out, -offset, axis=axis)
            indices = [slice(None)] * out.ndim
            indices[axis] = slice(None, None, RATIO)
            out = values[tuple(indices)]
        return out
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

    ``centering`` records how those point samples relate. Vertex children
    start on the first parent point; cell-centred children start a quarter
    of a parent spacing below it and cover the same physical cells.
    """

    origin: tuple[int, ...]
    shape: tuple[int, ...]
    centering: str = "vertex"

    def __post_init__(self) -> None:
        _check_centering(self.centering)
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


def extract(coarse: Any, box: Box, order: int = 4, *, backend="numpy"):
    """The fine-grid values of a box, prolonged from its parent.

    A bare cut-out would be interpolated across its own wrapped edge, as if
    that were data. So the box is cut out with a margin of the midpoint
    stencil's radius, taken periodically where the box meets the parent's
    edge as the parent itself is. Every fine point in the box then sees the
    parent's real neighbours, in the same order as prolonging the whole
    parent would, and the result is that one bit for bit. With four levels
    the whole-parent version was most of a coarse step.
    """
    array = _module(backend).asarray(coarse)
    margin = len(MIDPOINT[order]) // 2
    indices = [
        np.arange(start - margin, start + count + margin) % size
        for start, count, size in zip(box.origin, box.shape, array.shape, strict=True)
    ]
    refined = prolong(
        array[np.ix_(*indices)], order=order, centering=box.centering, backend=backend
    )
    slices = tuple(slice(RATIO * margin, RATIO * (margin + count)) for count in box.shape)
    return refined[slices]


def inject(coarse: Any, fine: Any, box: Box, *, order=4, buffer=0, backend="numpy"):
    """Copy a fine box's restriction back into its parent.

    The parent is returned changed rather than modified, because the states
    here are dictionaries of arrays that get passed around and a routine
    that mutates one of them in place is a bug waiting for the first caller
    who keeps a reference. For cell centres, retain the parent's values in
    the interpolated fine buffer and wherever restriction would wrap a box
    edge. ``buffer`` counts fine points. A full-domain axis is periodic.
    """
    xp = _module(backend)
    out = xp.array(coarse, copy=True)
    restricted = restrict(fine, centering=box.centering, order=order, backend=backend)
    if box.centering == "vertex":
        target = box.slices()
    else:
        # Never wrap the restriction stencil across a nonperiodic fine edge
        # or feed interpolated buffer values back into the coarse evolution.
        # Axes spanning the entire parent are periodic and need no trimming.
        parent_slices, fine_slices = [], []
        for start, count, size in zip(box.origin, box.shape, out.shape, strict=True):
            trim = 0 if start == 0 and count == size else (buffer + order // 2) // 2
            parent_slices.append(slice(start + trim, start + count - trim))
            fine_slices.append(slice(trim, count - trim))
        target = tuple(parent_slices)
        restricted = restricted[tuple(fine_slices)]
    if backend == "jax":
        return out.at[target].set(restricted)
    out[target] = restricted
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
