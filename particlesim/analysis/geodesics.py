"""Geodesic integration on a symbolic metric (design doc Section 5.5).

The geodesic equation ``d²x^a/dλ² = -Γ^a_{bc} dx^b/dλ dx^c/dλ`` is
integrated with SciPy's DOP853. Christoffel symbols come from the symbolic
pipeline and are compiled once per metric.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from particlesim.symbolic.curvature import MetricGeometry, lambdify_exprs


@dataclass
class GeodesicResult:
    affine: np.ndarray  # (n,)
    x: np.ndarray  # (n, D)
    u: np.ndarray  # (n, D)
    norm: np.ndarray  # (n,) g(u, u) along the path, a conservation check
    success: bool
    message: str


class GeodesicIntegrator:
    """Integrates geodesics of ``metric`` in ``coords``."""

    def __init__(
        self,
        metric: sp.Matrix,
        coords: Sequence[sp.Symbol],
        params: dict[sp.Symbol, float] | None = None,
    ):
        self.coords = list(coords)
        self.D = len(coords)
        geom = MetricGeometry(metric, coords)
        G = geom.christoffel
        D = self.D
        self._pairs = [(b, c) for b in range(D) for c in range(b, D)]
        gamma_exprs = [G[a][b][c] for a in range(D) for (b, c) in self._pairs]
        self._gamma = lambdify_exprs(gamma_exprs, coords, params)
        self._metric = lambdify_exprs(
            [metric[a, b] for a in range(D) for b in range(D)], coords, params
        )

    def metric_at(self, x: np.ndarray) -> np.ndarray:
        return self._metric(*x).reshape(self.D, self.D)

    def christoffel_at(self, x: np.ndarray) -> np.ndarray:
        """``Γ^a_{bc}`` as a (D, D, D) array at point ``x``."""
        vals = self._gamma(*x).reshape(self.D, len(self._pairs))
        out = np.zeros((self.D, self.D, self.D))
        for k, (b, c) in enumerate(self._pairs):
            out[:, b, c] = vals[:, k]
            out[:, c, b] = vals[:, k]
        return out

    def norm(self, x: np.ndarray, u: np.ndarray) -> float:
        g = self.metric_at(x)
        return float(u @ g @ u)

    def normalize(
        self, x: np.ndarray, u_spatial: Sequence[float], kind: str = "timelike"
    ) -> np.ndarray:
        """Return a future-directed ``u`` with the given coordinate spatial components.

        ``kind`` is ``"timelike"`` (g(u,u) = -1) or ``"null"`` (g(u,u) = 0).
        Solves the quadratic in ``u^0``. Where the coordinate time is not
        timelike (``g_00 >= 0``, as inside a superluminal warp bubble) both
        roots can be future-directed and the spatial components alone do not
        fix the vector; use :meth:`from_eulerian_velocity` there.
        """
        g = self.metric_at(np.asarray(x, dtype=float))
        s = np.asarray(u_spatial, dtype=float)
        target = -1.0 if kind == "timelike" else 0.0
        a = g[0, 0]
        b = 2 * g[0, 1:] @ s
        c = s @ g[1:, 1:] @ s - target
        disc = b * b - 4 * a * c
        if a >= 0:
            raise ValueError(
                "coordinate time is not timelike here; use from_eulerian_velocity instead"
            )
        if disc < 0:
            raise ValueError(f"no real u^0 for these spatial components ({kind})")
        roots = [(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)]
        u0 = max(roots)
        if u0 <= 0:
            raise ValueError("no future-directed solution")
        return np.concatenate([[u0], s])

    def from_eulerian_velocity(
        self, x: np.ndarray, v: Sequence[float], kind: str = "timelike"
    ) -> np.ndarray:
        """Four-velocity of an observer moving at 3-velocity ``v`` relative to the
        Eulerian (normal) observer at ``x``.

        ``v`` is measured in the Eulerian observer's orthonormal frame, so
        ``|v| < 1`` for ``kind="timelike"`` and ``v`` is a direction (any
        nonzero length) for ``kind="null"``. This is unambiguous everywhere,
        including where coordinate time is spacelike.
        """
        g = self.metric_at(np.asarray(x, dtype=float))
        gamma = g[1:, 1:]
        beta_low = g[0, 1:]
        gam_inv = np.linalg.inv(gamma)
        beta_up = gam_inv @ beta_low
        alpha = np.sqrt(beta_up @ beta_low - g[0, 0])
        n = np.concatenate([[1 / alpha], -beta_up / alpha])
        L = np.linalg.cholesky(gamma)
        triad = np.linalg.inv(L.T)  # columns e_a with γ(e_a, e_b) = δ_ab
        v = np.asarray(v, dtype=float)
        e_spatial = triad @ v
        if kind == "null":
            speed = np.linalg.norm(v)
            if speed == 0:
                raise ValueError("null direction must be nonzero")
            return n + np.concatenate([[0.0], e_spatial / speed])
        speed2 = v @ v
        if speed2 >= 1:
            raise ValueError("timelike observer needs |v| < 1")
        return (n + np.concatenate([[0.0], e_spatial])) / np.sqrt(1 - speed2)

    def integrate(
        self,
        x0: Sequence[float],
        u0: Sequence[float],
        affine_max: float,
        n_out: int = 200,
        rtol: float = 1e-10,
        atol: float = 1e-12,
        stop_when=None,
    ) -> GeodesicResult:
        x0 = np.asarray(x0, dtype=float)
        u0 = np.asarray(u0, dtype=float)
        D = self.D

        def rhs(_lam, y):
            x, u = y[:D], y[D:]
            G = self.christoffel_at(x)
            if not np.isfinite(G).all():
                raise ValueError(
                    f"non-finite Christoffel symbols at x={x.tolist()}; the metric "
                    "expressions are singular there (for example r = 0 on an axis)"
                )
            acc = -np.einsum("abc,b,c->a", G, u, u)
            return np.concatenate([u, acc])

        events = None
        if stop_when is not None:

            def ev(_lam, y):
                return stop_when(y[:D], y[D:])

            ev.terminal = True
            events = [ev]

        sol = solve_ivp(
            rhs,
            (0.0, affine_max),
            np.concatenate([x0, u0]),
            method="DOP853",
            t_eval=np.linspace(0.0, affine_max, n_out),
            rtol=rtol,
            atol=atol,
            events=events,
        )
        x = sol.y[:D].T
        u = sol.y[D:].T
        norms = np.array([self.norm(xi, ui) for xi, ui in zip(x, u, strict=True)])
        return GeodesicResult(sol.t, x, u, norms, sol.success, sol.message)
