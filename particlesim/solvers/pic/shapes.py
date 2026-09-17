"""Particle shape functions and the gather/scatter they define (Section 5.4).

A macro-particle is a cloud, not a point. Its shape function ``S`` spreads
its charge over neighbouring nodes, and the same shape is used to gather
fields back onto it. Using the *same* shape for both directions is what
makes the scheme free of self-force: a particle cannot push itself, because
the field it deposits is gathered back symmetrically.

Order ``p`` is the degree of the spline. Order 1 is cloud-in-cell, a
triangle two cells wide. Order 2 is the triangular-shaped cloud, a piecewise
quadratic three cells wide, which costs half again as much per particle and
buys a marked drop in grid heating.

The window convention matters for the current deposition. Charge-conserving
deposition needs the old and new shapes expressed on one common set of
indices, and since a particle may not cross more than one cell in a step,
``p + 2`` nodes always suffice. Everything here therefore returns a window
of that width, whether or not the particle moved.
"""

from __future__ import annotations

import numpy as np

SUPPORTED_ORDERS = (1, 2)


def _node_low(xi: np.ndarray, order: int) -> np.ndarray:
    """Lowest node index carrying weight for a particle at ``xi`` cells."""
    if order == 1:
        return np.floor(xi).astype(np.int64)
    return np.rint(xi).astype(np.int64) - 1


def shape(offsets: np.ndarray, order: int) -> np.ndarray:
    """B-spline of degree ``order`` evaluated at node-minus-particle offsets."""
    d = np.abs(offsets)
    if order == 1:
        return np.where(d < 1.0, 1.0 - d, 0.0)
    if order == 2:
        inner = 0.75 - d**2
        outer = 0.5 * (1.5 - d) ** 2
        return np.where(d < 0.5, inner, np.where(d < 1.5, outer, 0.0))
    raise ValueError(f"order must be one of {SUPPORTED_ORDERS}, got {order}")


def window(xi: np.ndarray, order: int, base: np.ndarray | None = None):
    """Indices and weights of one position on a ``p + 2`` node window.

    Passing ``base`` pins the window, which is how the old and the new
    position of a moving particle are put on the same indices. Without it the
    window is the natural one for ``xi``, shifted down by one node so that a
    particle moving either way stays inside it.
    """
    if order not in SUPPORTED_ORDERS:
        raise ValueError(f"order must be one of {SUPPORTED_ORDERS}, got {order}")
    xi = np.asarray(xi, dtype=float)
    if base is None:
        base = _node_low(xi, order) - 1
    width = order + 3
    indices = base[:, None] + np.arange(width)[None, :]
    weights = shape(indices - xi[:, None], order)
    return indices, weights


def common_window(xi_old: np.ndarray, xi_new: np.ndarray, order: int):
    """One window carrying both shapes, for charge-conserving deposition.

    Raises if a particle moved more than a cell in the step. That is not a
    tolerance to widen: past one cell the deposition's continuity identity no
    longer holds on this window, and the charge it fails to account for shows
    up as a growing violation of Gauss's law rather than as an error message.
    """
    jump = np.abs(xi_new - xi_old)
    if jump.size and jump.max() >= 1.0:
        raise ValueError(
            f"a particle moved {jump.max():.3f} cells in one step. Charge-"
            "conserving deposition assumes less than one, so this would break "
            "the continuity identity silently. Shorten the step"
        )
    base = np.minimum(_node_low(xi_old, order), _node_low(xi_new, order)) - 1
    indices, weights_old = window(xi_old, order, base=base)
    _, weights_new = window(xi_new, order, base=base)
    return indices, weights_old, weights_new


def gather(field: np.ndarray, positions: np.ndarray, spacing, offsets, order: int):
    """Interpolate a staggered field onto particle positions.

    ``offsets`` is the component's half-cell stagger, so the shape is
    evaluated against the component's own lattice rather than the cell
    corners. Getting this wrong shifts the force by half a cell, which looks
    like a small phase error and is in fact a systematic drift.
    """
    positions = np.atleast_2d(positions)
    ndim = field.ndim
    idx, wts = [], []
    for axis in range(ndim):
        xi = positions[:, axis] / spacing[axis] - offsets[axis]
        i, w = window(xi, order)
        idx.append(np.mod(i, field.shape[axis]))
        wts.append(w)

    if ndim == 1:
        return np.einsum("pi,pi->p", field[idx[0]], wts[0])
    values = field[idx[0][:, :, None], idx[1][:, None, :]]
    return np.einsum("pij,pi,pj->p", values, wts[0], wts[1])
