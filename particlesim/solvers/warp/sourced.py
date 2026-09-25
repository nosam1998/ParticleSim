"""Warp Mode W3, dynamical: a bubble evolved together with the matter that sources it (issue #54).

W1 asks what matter a warp metric needs. This asks the next question:
supplied with exactly that matter, does the spacetime hold together under
the Einstein equations, and what happens with less of it?

**A bubble that should not change.** In coordinates riding with the bubble,
``x' = x - v t``, Alcubierre's metric is

    ds^2 = -dt^2 + (dx' + v (1 - f(r)) dt)^2 + dy^2 + dz^2

and nothing in it depends on ``t``. The lapse is 1, the slices are flat,
the shift ``beta^x = v (1 - f)`` is static, and ``K_ij = (d_i beta_j + d_j
beta_i) / 2``. Its stress-energy, ``G_ab / 8 pi``, is computed here from the
metric symbolically. The Eulerian observers measure it as

    rho = n^a n^b T_ab,   S_i = -n^a T_ai,   S_ij = T_ij,    n^a = (1, -beta^i)

With that source, and the gauge frozen at the metric's own lapse and
shift, every BSSN rate is zero in the continuum. Discretely, the rates
converge to zero at fourth order. Without the source the bubble is not a
solution, and the rates are of order one. That makes the source the thing
under test, and the geometry the measuring instrument.

**Not enough exotic matter.** :meth:`~particlesim.solvers.nr.matter.PrescribedMatter.scaled`
supplies a fraction of the matter. The evolution then departs from the
bubble at a rate set by the shortfall. The measurements are in
``docs/benchmarks.md``.

The torus has to be large enough for ``f`` to vanish at its edge. ``f`` is
even, so its periodic extension is continuous, but its slope is not, and
the few points next to the edge carry an error that does not converge.
:func:`interior` masks them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

from particlesim.scenarios.warp.metrics import alcubierre_shape
from particlesim.solvers.nr.matter import PrescribedMatter, SourcedEvolution

_T, _X, _Y, _Z = sp.symbols("t x y z", real=True)


def comoving_alcubierre(speed: float, radius: float, sigma: float) -> sp.Matrix:
    """Alcubierre's metric in coordinates riding with the bubble: stationary."""
    r = sp.sqrt(_X**2 + _Y**2 + _Z**2)
    shift = speed * (1 - alcubierre_shape(r, sp.Float(radius), sp.Float(sigma)))
    return sp.Matrix(
        [
            [-1 + shift**2, shift, 0, 0],
            [shift, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def eulerian_stress_energy(metric: sp.Matrix):
    """``(rho, S_i, S_ij)`` of ``G_ab / 8 pi``, for a metric with unit lapse and flat slices.

    The 3+1 split needs the lapse and shift: with ``g_00 = -1 + beta_k beta^k``
    and ``g_0i = beta_i`` on flat slices, the normal is ``n^a = (1, -beta^i)``.
    """
    from particlesim.symbolic.curvature import MetricGeometry

    stress_energy = MetricGeometry(metric, [_T, _X, _Y, _Z]).einstein / (8 * sp.pi)
    shift = [metric[0, i] for i in (1, 2, 3)]
    normal = [1, -shift[0], -shift[1], -shift[2]]
    energy = sum(normal[a] * normal[b] * stress_energy[a, b] for a in range(4) for b in range(4))
    momentum = [-sum(normal[a] * stress_energy[a, i + 1] for a in range(4)) for i in range(3)]
    stress = [[stress_energy[i + 1, j + 1] for j in range(3)] for i in range(3)]
    return energy, momentum, stress


def grid(points: int, extent: float):
    """Cell centres of a periodic cube ``[-extent/2, extent/2]^3``, and the spacing."""
    spacing = extent / points
    centres = (np.arange(points) + 0.5) * spacing - extent / 2
    return np.meshgrid(centres, centres, centres, indexing="ij"), spacing


def interior(coords, extent: float, margin: float = 1.5) -> np.ndarray:
    """Points further than ``margin`` from the torus's edge."""
    limit = extent / 2 - margin
    return np.all([np.abs(c) < limit for c in coords], axis=0)


@dataclass(frozen=True)
class SourcedBubble:
    """An Alcubierre bubble of speed ``speed``, on a periodic grid, with its own source."""

    speed: float = 0.5
    radius: float = 1.5
    sigma: float = 1.0
    extent: float = 10.0
    points: int = 30

    def metric(self) -> sp.Matrix:
        return comoving_alcubierre(self.speed, self.radius, self.sigma)

    def state(self) -> dict[str, np.ndarray]:
        """The bubble as BSSN variables: ``phi = 0``, ``gammabar = 1``, ``K_ij`` from the shift."""
        (X, Y, Z), _ = grid(self.points, self.extent)
        shift = self.metric()[0, 1]
        values = sp.lambdify((_X, _Y, _Z), [shift] + [sp.diff(shift, q) for q in (_X, _Y, _Z)])
        beta, dx, dy, dz = (np.broadcast_to(v, X.shape).astype(float) for v in values(X, Y, Z))
        zero, one = np.zeros_like(X), np.ones_like(X)
        curvature = [[zero, dy / 2, dz / 2], [dy / 2, zero, zero], [dz / 2, zero, zero]]
        curvature[0][0] = dx  # K_xx = d_x beta_x; K_xy = d_y beta_x / 2, K_xz likewise
        trace = dx
        state = {"phi": zero.copy(), "trK": trace.copy(), "alpha": one.copy()}
        state.update(beta0=beta.copy(), beta1=zero.copy(), beta2=zero.copy())
        for i in range(3):
            state[f"Gt{i}"] = zero.copy()
            state[f"B{i}"] = zero.copy()
            for j in range(i, 3):
                state[f"gt{i}{j}"] = (one if i == j else zero).copy()
                state[f"At{i}{j}"] = curvature[i][j] - (trace / 3 if i == j else 0.0)
        return state

    def matter(self) -> PrescribedMatter:
        """``G_ab / 8 pi`` on the grid, as the Eulerian observers measure it."""
        (X, Y, Z), _ = grid(self.points, self.extent)
        energy, momentum, stress = eulerian_stress_energy(self.metric())
        flat = [energy, *momentum, *(stress[i][j] for i in range(3) for j in range(3))]
        values = [
            np.broadcast_to(v, X.shape).astype(float)
            for v in sp.lambdify((_X, _Y, _Z), flat)(X, Y, Z)
        ]
        return PrescribedMatter(
            (
                values[0],
                tuple(values[1:4]),
                tuple(tuple(values[4 + 3 * i : 7 + 3 * i]) for i in range(3)),
            )
        )

    def evolution(self, matter: PrescribedMatter | None = None, dissipation: float = 0.1):
        """A frozen-gauge BSSN evolution on this grid, sourced by ``matter``."""
        from particlesim.solvers.nr.bssn import Evolution

        _, spacing = grid(self.points, self.extent)
        geometry = Evolution.build(
            (spacing,) * 3,
            backend="numpy",
            slicing="frozen",
            shift_condition="frozen",
            dissipation=dissipation,
        )
        return SourcedEvolution(geometry, self.matter() if matter is None else matter)


__all__ = [
    "SourcedBubble",
    "comoving_alcubierre",
    "eulerian_stress_energy",
    "grid",
    "interior",
]
