"""Uniform Cartesian grids with ghost zones and integration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

import numpy as np


@dataclass
class UniformGrid:
    """A uniform Cartesian grid.

    ``extent`` is a list of ``(lo, hi)`` per axis, ``shape`` the number of
    interior points per axis, and ``ghost`` the number of ghost cells added on
    each side. Points are cell-centred so that integration by simple sums is
    second-order accurate and no point sits exactly on a boundary.
    """

    extent: list[tuple[float, float]]
    shape: tuple[int, ...]
    ghost: int = 0
    axis_names: tuple[str, ...] = field(default=("x", "y", "z"))

    def __post_init__(self) -> None:
        if len(self.extent) != len(self.shape):
            raise ValueError("extent and shape must have the same length")
        if any(n < 1 for n in self.shape):
            raise ValueError("shape entries must be positive")
        self.axis_names = tuple(self.axis_names[: self.ndim])

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def spacing(self) -> tuple[float, ...]:
        return tuple((hi - lo) / n for (lo, hi), n in zip(self.extent, self.shape, strict=True))

    @property
    def cell_volume(self) -> float:
        return float(np.prod(self.spacing))

    @property
    def full_shape(self) -> tuple[int, ...]:
        return tuple(n + 2 * self.ghost for n in self.shape)

    def axis(self, i: int) -> np.ndarray:
        """1D coordinate array along axis ``i`` including ghost cells."""
        lo, hi = self.extent[i]
        dx = self.spacing[i]
        n = self.shape[i]
        return lo + dx * (np.arange(-self.ghost, n + self.ghost) + 0.5)

    def coords(self) -> tuple[np.ndarray, ...]:
        """Full-shape coordinate arrays (``indexing='ij'``)."""
        return tuple(np.meshgrid(*[self.axis(i) for i in range(self.ndim)], indexing="ij"))

    def interior(self, arr: np.ndarray) -> np.ndarray:
        """View of ``arr`` with ghost cells stripped from the trailing grid axes."""
        if self.ghost == 0:
            return arr
        sl = (Ellipsis,) + tuple(slice(self.ghost, -self.ghost) for _ in range(self.ndim))
        return arr[sl]

    def integrate(self, arr: np.ndarray) -> float:
        """Sum of ``arr`` over interior cells times the cell volume."""
        return float(np.sum(self.interior(arr)) * self.cell_volume)


def derivative(arr: np.ndarray, axis: int, dx: float, order: int = 4) -> np.ndarray:
    """Central finite-difference derivative along ``axis``.

    Interior points use a 2nd-, 4th-, or 6th-order centred stencil; the
    points within the stencil radius of an edge fall back to 2nd-order
    one-sided differences.
    """
    if order not in (2, 4, 6):
        raise ValueError("order must be 2, 4, or 6")
    a = np.moveaxis(arr, axis, 0)
    out = np.empty_like(a, dtype=float)
    n = a.shape[0]
    r = order // 2
    if n < 2 * r + 1:
        raise ValueError("array too short for the requested stencil")
    if order == 2:
        out[1:-1] = (a[2:] - a[:-2]) / (2 * dx)
    elif order == 4:
        out[2:-2] = (-a[4:] + 8 * a[3:-1] - 8 * a[1:-3] + a[:-4]) / (12 * dx)
    else:
        out[3:-3] = (a[6:] - 9 * a[5:-1] + 45 * a[4:-2] - 45 * a[2:-4] + 9 * a[1:-5] - a[:-6]) / (
            60 * dx
        )
    # One-sided second-order at the edges.
    for i in range(r):
        out[i] = (-3 * a[i] + 4 * a[i + 1] - a[i + 2]) / (2 * dx)
        out[n - 1 - i] = (3 * a[n - 1 - i] - 4 * a[n - 2 - i] + a[n - 3 - i]) / (2 * dx)
    return np.moveaxis(out, 0, axis)


def second_derivative(arr: np.ndarray, axis: int, dx: float, order: int = 4) -> np.ndarray:
    """Central finite-difference second derivative along ``axis``.

    A direct stencil rather than two applications of :func:`derivative`.
    Composing first derivatives is the same order but over a stencil twice
    as wide, and it squares the amplification of grid-scale noise -- which
    in an evolution shows up as the second derivatives in a Ricci tensor
    going bad before anything else does. Mixed derivatives have no such
    stencil and are composed, which is why they are the noisiest term in
    any of this.

    Points within the stencil radius of an edge fall back to a second-order
    one-sided difference.
    """
    if order not in (2, 4, 6):
        raise ValueError("order must be 2, 4, or 6")
    a = np.moveaxis(arr, axis, 0)
    out = np.empty_like(a, dtype=float)
    n = a.shape[0]
    r = order // 2
    if n < 2 * r + 1:
        raise ValueError("array too short for the requested stencil")
    if order == 2:
        out[1:-1] = (a[2:] - 2 * a[1:-1] + a[:-2]) / dx**2
    elif order == 4:
        out[2:-2] = (-a[4:] + 16 * a[3:-1] - 30 * a[2:-2] + 16 * a[1:-3] - a[:-4]) / (12 * dx**2)
    else:
        out[3:-3] = (
            2 * a[6:]
            - 27 * a[5:-1]
            + 270 * a[4:-2]
            - 490 * a[3:-3]
            + 270 * a[2:-4]
            - 27 * a[1:-5]
            + 2 * a[:-6]
        ) / (180 * dx**2)
    for i in range(r):
        out[i] = (2 * a[i] - 5 * a[i + 1] + 4 * a[i + 2] - a[i + 3]) / dx**2
        out[n - 1 - i] = (
            2 * a[n - 1 - i] - 5 * a[n - 2 - i] + 4 * a[n - 3 - i] - a[n - 4 - i]
        ) / dx**2
    return np.moveaxis(out, 0, axis)


def kreiss_oliger(u: np.ndarray, axis: int, dx: float, order: int = 4, epsilon: float = 0.1):
    """Kreiss-Oliger artificial dissipation, to be added to a right-hand side.

    Centred finite differences do not damp the shortest wavelength the grid
    can carry, so numerical noise at the Nyquist frequency accumulates until
    it swamps the solution. KO dissipation removes it while leaving the
    solution's accuracy order intact, which is why numerical relativity runs
    do not survive without it.

    For a scheme of accuracy ``order = 2r - 2``, the operator uses the
    ``2r``-th centred difference:

        Q = (-1)^(r+1) * epsilon / (2^(2r) * dx) * D^(2r)

    so that ``Q`` annihilates polynomials below degree ``2r`` (leaving the
    scheme's order untouched) and damps the Nyquist mode at rate
    ``epsilon / dx``. Both properties are pinned by tests rather than
    asserted here.

    Points within the stencil radius of a boundary get no dissipation, since
    a one-sided dissipation operator would inject exactly the error it is
    meant to remove.
    """
    if order not in (2, 4, 6):
        raise ValueError("order must be 2, 4, or 6")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    r = order // 2 + 1
    stencil = np.array([(-1) ** k * comb(2 * r, k) for k in range(2 * r + 1)], dtype=float)
    a = np.moveaxis(np.asarray(u, dtype=float), axis, 0)
    n = a.shape[0]
    if n < 2 * r + 1:
        raise ValueError("array too short for the dissipation stencil")
    out = np.zeros_like(a)
    acc = np.zeros_like(a[: n - 2 * r])
    for k, c in enumerate(stencil):
        acc = acc + c * a[k : n - 2 * r + k]
    prefactor = ((-1) ** (r + 1)) * epsilon / (2 ** (2 * r) * dx)
    out[r : n - r] = prefactor * acc
    return np.moveaxis(out, 0, axis)
