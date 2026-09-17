"""ADM quantities for metrics with flat spatial slices and unit lapse.

This is the fast path for warp metrics of the form
``ds² = -dt² + Σ_i (dx^i - β^i dt)²`` with time-independent shift, which
covers the Alcubierre and Natário families. With flat slices the spatial
Ricci scalar vanishes and the Hamiltonian constraint gives the Eulerian
energy density directly:

    16π ρ = K² - K_ij K^ij,   K_ij = ½ (∂_i β_j + ∂_j β_i).

The momentum constraint gives the momentum density
``8π S_i = ∂_j K^j_i - ∂_i K`` and the expansion of the Eulerian congruence
is ``θ = -K`` (York time). Sign conventions follow Baumgarte and Shapiro.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cached_property

import numpy as np
import sympy as sp

from particlesim.symbolic.curvature import lambdify_exprs


class FlatSliceADM:
    """Symbolic ADM quantities for a flat-slice, unit-lapse, static-shift metric."""

    def __init__(self, shift: Sequence[sp.Expr], coords: Sequence[sp.Symbol]):
        """``shift`` is the contravariant shift ``β^i`` (equal to ``β_i`` on flat slices)."""
        if len(shift) != len(coords):
            raise ValueError("shift needs one component per spatial coordinate")
        self.beta = [sp.sympify(b) for b in shift]
        self.x = list(coords)
        self.d = len(coords)

    @cached_property
    def extrinsic_curvature(self) -> sp.Matrix:
        d, b, x = self.d, self.beta, self.x
        return sp.Matrix(d, d, lambda i, j: (sp.diff(b[j], x[i]) + sp.diff(b[i], x[j])) / 2)

    @cached_property
    def trace_K(self) -> sp.Expr:
        return self.extrinsic_curvature.trace()

    @cached_property
    def K_squared(self) -> sp.Expr:
        K = self.extrinsic_curvature
        return sum(K[i, j] ** 2 for i in range(self.d) for j in range(self.d))

    @cached_property
    def energy_density(self) -> sp.Expr:
        return (self.trace_K**2 - self.K_squared) / (16 * sp.pi)

    @cached_property
    def momentum_density(self) -> list[sp.Expr]:
        K, Kt, x, d = self.extrinsic_curvature, self.trace_K, self.x, self.d
        return [
            (sum(sp.diff(K[j, i], x[j]) for j in range(d)) - sp.diff(Kt, x[i])) / (8 * sp.pi)
            for i in range(d)
        ]

    @cached_property
    def expansion(self) -> sp.Expr:
        return -self.trace_K

    def compile(
        self, params: dict[sp.Symbol, float] | None = None
    ) -> Callable[..., dict[str, np.ndarray]]:
        """Compile ρ, S_i, θ into a single NumPy callable over coordinate arrays."""
        exprs = [self.energy_density, self.expansion, *self.momentum_density]
        f = lambdify_exprs(exprs, self.x, params)

        def run(*coord_arrays: np.ndarray) -> dict[str, np.ndarray]:
            out = f(*coord_arrays)
            return {
                "energy_density": out[0],
                "expansion": out[1],
                "momentum_density": out[2:],
            }

        return run
