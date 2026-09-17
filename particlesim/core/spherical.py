"""Spherically symmetric 1D radial grids (design doc Section 5.2, Milestone 1).

Cells are centred, so no point sits at ``r = 0``. That is not cosmetic:
spherical evolution equations carry ``1/r`` and ``2/r`` terms, and a grid
point exactly on the origin turns them into divisions by zero. Staggering
the origin is the standard way to keep those terms finite while remaining
second-order accurate about the centre.

Optional geometric refinement concentrates resolution near the origin,
which is where collapse happens, without paying for it out at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SphericalGrid:
    """A radial grid on ``(0, r_max]`` with cell-centred points and ghost cells.

    ``refine`` > 1 grades the spacing geometrically, so the innermost cell is
    ``refine`` times finer than the outermost. ``refine = 1`` is uniform.
    """

    r_max: float
    n: int
    ghost: int = 0
    refine: float = 1.0

    def __post_init__(self) -> None:
        if self.r_max <= 0:
            raise ValueError("r_max must be positive")
        if self.n < 1:
            raise ValueError("n must be positive")
        if self.refine < 1:
            raise ValueError("refine must be >= 1 (1 means uniform)")

    @property
    def uniform(self) -> bool:
        return self.refine == 1.0

    @property
    def dr(self) -> float:
        """Uniform spacing. Only defined for an unrefined grid."""
        if not self.uniform:
            raise ValueError("a graded grid has no single spacing; use spacings()")
        return self.r_max / self.n

    def _widths(self) -> np.ndarray:
        """Cell widths, geometrically graded from fine at the centre outward."""
        if self.uniform:
            return np.full(self.n, self.r_max / self.n)
        ratio = self.refine ** (1.0 / max(self.n - 1, 1))
        w = ratio ** np.arange(self.n)
        return w * (self.r_max / w.sum())

    def spacings(self) -> np.ndarray:
        return self._widths()

    def radii(self) -> np.ndarray:
        """Cell-centred radii of the interior cells, strictly positive."""
        w = self._widths()
        edges = np.concatenate([[0.0], np.cumsum(w)])
        return 0.5 * (edges[:-1] + edges[1:])

    def edges(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(self._widths())])

    def full_radii(self) -> np.ndarray:
        """Interior radii plus ghost cells, reflected through the origin.

        Inner ghosts take negative radii, the standard trick for imposing
        parity conditions at the centre: a scalar is even across ``r = 0``
        and a radial vector component is odd, and both are expressed by
        evaluating at ``-r``.
        """
        r = self.radii()
        if self.ghost == 0:
            return r
        g = self.ghost
        inner = -r[:g][::-1]
        w = self._widths()
        outer = r[-1] + np.cumsum(w[-1] * np.ones(g))
        return np.concatenate([inner, r, outer])

    def interior(self, arr: np.ndarray) -> np.ndarray:
        if self.ghost == 0:
            return arr
        return arr[..., self.ghost : -self.ghost]

    def volume_integrate(self, f: np.ndarray) -> float:
        """``integral f * 4 pi r^2 dr`` over the interior, using cell widths."""
        r = self.radii()
        return float(np.sum(self.interior(f) * 4.0 * np.pi * r**2 * self._widths()))

    def apply_parity(self, arr: np.ndarray, even: bool = True) -> np.ndarray:
        """Fill inner ghost cells by reflection: even for scalars, odd for
        radial vector components."""
        if self.ghost == 0:
            return arr
        out = np.array(arr, dtype=float, copy=True)
        g = self.ghost
        sign = 1.0 if even else -1.0
        out[..., :g] = sign * out[..., 2 * g - 1 : g - 1 : -1]
        return out
