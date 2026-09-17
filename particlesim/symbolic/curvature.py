"""Curvature of a metric given in explicit coordinates.

Stage 1 of the symbolic pipeline: metric -> Christoffel -> Riemann -> Ricci
-> Einstein, plus the Eulerian (normal-observer) decomposition and
lambdification with common-subexpression elimination.

Conventions: signature (-,+,+,+); ``R^a_{bcd} = d_c Γ^a_{bd} - d_d Γ^a_{bc}
+ Γ^a_{ce} Γ^e_{bd} - Γ^a_{de} Γ^e_{bc}``; ``R_{bd} = R^a_{bad}``.
Nothing is simplified unless asked, because simplification is what makes
symbolic relativity slow; numeric evaluation goes through ``lambdify``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any

import numpy as np
import sympy as sp


class MetricGeometry:
    """Curvature tensors of a metric ``g`` in coordinates ``coords``."""

    def __init__(self, metric: sp.Matrix, coords: Sequence[sp.Symbol], simplify: bool = False):
        n = metric.shape[0]
        if metric.shape != (n, n) or len(coords) != n:
            raise ValueError("metric must be square and match the number of coordinates")
        self.g = sp.Matrix(metric)
        self.x = list(coords)
        self.n = n
        self._simp = sp.simplify if simplify else (lambda e: e)

    @cached_property
    def ginv(self) -> sp.Matrix:
        return self._simp(self.g.inv())

    @cached_property
    def christoffel(self) -> list[list[list[sp.Expr]]]:
        """``Γ^a_{bc}`` as ``christoffel[a][b][c]``."""
        n, g, gi, x = self.n, self.g, self.ginv, self.x
        dg = [[[sp.diff(g[b, c], x[a]) for c in range(n)] for b in range(n)] for a in range(n)]
        gam = [[[sp.S.Zero] * n for _ in range(n)] for _ in range(n)]
        for a in range(n):
            for b in range(n):
                for c in range(b, n):
                    s = sum(
                        gi[a, d] * (dg[b][d][c] + dg[c][d][b] - dg[d][b][c]) / 2 for d in range(n)
                    )
                    gam[a][b][c] = gam[a][c][b] = self._simp(s)
        return gam

    @cached_property
    def riemann(self) -> list[list[list[list[sp.Expr]]]]:
        """``R^a_{bcd}`` as ``riemann[a][b][c][d]``."""
        n, x, G = self.n, self.x, self.christoffel
        R = [[[[sp.S.Zero] * n for _ in range(n)] for _ in range(n)] for _ in range(n)]
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    for d in range(c + 1, n):
                        expr = sp.diff(G[a][b][d], x[c]) - sp.diff(G[a][b][c], x[d])
                        expr += sum(
                            G[a][c][e] * G[e][b][d] - G[a][d][e] * G[e][b][c] for e in range(n)
                        )
                        expr = self._simp(expr)
                        R[a][b][c][d] = expr
                        R[a][b][d][c] = -expr
        return R

    @cached_property
    def riemann_lower(self) -> list[list[list[list[sp.Expr]]]]:
        """``R_{abcd}`` with all indices down."""
        n, g, R = self.n, self.g, self.riemann
        return [
            [
                [
                    [self._simp(sum(g[a, e] * R[e][b][c][d] for e in range(n))) for d in range(n)]
                    for c in range(n)
                ]
                for b in range(n)
            ]
            for a in range(n)
        ]

    @cached_property
    def ricci(self) -> sp.Matrix:
        n, R = self.n, self.riemann
        return sp.Matrix(n, n, lambda b, d: self._simp(sum(R[a][b][a][d] for a in range(n))))

    @cached_property
    def ricci_scalar(self) -> sp.Expr:
        n, gi, Ric = self.n, self.ginv, self.ricci
        return self._simp(sum(gi[a, b] * Ric[a, b] for a in range(n) for b in range(n)))

    @cached_property
    def einstein(self) -> sp.Matrix:
        """``G_{ab}`` with lower indices."""
        return self._simp(self.ricci - self.g * self.ricci_scalar / 2)

    @cached_property
    def kretschmann(self) -> sp.Expr:
        """``R_{abcd} R^{abcd}``."""
        n, g, gi, R = self.n, self.g, self.ginv, self.riemann
        # Lower the first index, raise the last three.
        Rl = [
            [
                [
                    [sum(g[a, e] * R[e][b][c][d] for e in range(n)) for d in range(n)]
                    for c in range(n)
                ]
                for b in range(n)
            ]
            for a in range(n)
        ]
        Ru = [
            [
                [
                    [
                        sum(
                            gi[b, f] * gi[c, h] * gi[d, k] * R[a][f][h][k]
                            for f in range(n)
                            for h in range(n)
                            for k in range(n)
                        )
                        for d in range(n)
                    ]
                    for c in range(n)
                ]
                for b in range(n)
            ]
            for a in range(n)
        ]
        return self._simp(
            sum(
                Rl[a][b][c][d] * Ru[a][b][c][d]
                for a in range(n)
                for b in range(n)
                for c in range(n)
                for d in range(n)
            )
        )


class Eulerian:
    """Normal-observer decomposition of a symmetric tensor ``T_{ab}``.

    For a metric in ADM form the unit normal is ``n^a = (1/α, -β^i/α)``.
    ``energy_density = T_ab n^a n^b``, ``momentum = -T_ab n^a γ^b_i``.
    """

    def __init__(self, metric: sp.Matrix, coords: Sequence[sp.Symbol]):
        self.g = sp.Matrix(metric)
        self.x = list(coords)
        self.n = metric.shape[0]

    @cached_property
    def spatial_metric(self) -> sp.Matrix:
        return self.g[1:, 1:]

    @cached_property
    def shift_lower(self) -> sp.Matrix:
        return self.g[0, 1:].T

    @cached_property
    def shift(self) -> sp.Matrix:
        return self.spatial_metric.inv() * self.shift_lower

    @cached_property
    def lapse(self) -> sp.Expr:
        beta_sq = (self.shift_lower.T * self.shift)[0, 0]
        return sp.sqrt(beta_sq - self.g[0, 0])

    @cached_property
    def normal(self) -> sp.Matrix:
        """``n^a`` (upper index)."""
        a = self.lapse
        return sp.Matrix([1 / a] + [-b / a for b in self.shift])

    def energy_density(self, T: sp.Matrix) -> sp.Expr:
        n = self.normal
        return (n.T * T * n)[0, 0]

    def momentum_density(self, T: sp.Matrix) -> list[sp.Expr]:
        """``S_i = -T_{ab} n^a γ^b_i`` with lower spatial index."""
        n = self.normal
        d = self.n
        out = []
        for i in range(1, d):
            # γ^b_i = δ^b_i + n^b n_i ; n_i = g_ib n^b
            n_low_i = sum(self.g[i, b] * n[b] for b in range(d))
            proj = [sp.KroneckerDelta(b, i) + n[b] * n_low_i for b in range(d)]
            out.append(-sum(T[a, b] * n[a] * proj[b] for a in range(d) for b in range(d)))
        return out


def generate_source(
    exprs: Sequence[sp.Expr],
    coords: Sequence[sp.Symbol],
    params: dict[sp.Symbol, float] | None = None,
    free_params: Sequence[sp.Symbol] = (),
) -> str:
    """Generate NumPy source for a function of the coordinates returning ``exprs``.

    Parameter symbols are substituted first, then common subexpressions are
    eliminated. The source is plain text so it can be cached on disk.
    """
    subs = dict(params or {})
    # A parameter left free stays symbolic, which is what makes the emitted
    # kernel differentiable with respect to it.
    for sym in free_params:
        subs.pop(sym, None)
    exprs = [sp.sympify(e).subs(subs) for e in exprs]
    replacements, reduced = sp.cse(exprs, optimizations="basic")
    lines = [f"    {lhs} = {sp.pycode(rhs)}" for lhs, rhs in replacements]
    body = "\n".join(lines)
    outs = ", ".join(sp.pycode(e) for e in reduced)
    args = ", ".join(str(s) for s in (*coords, *free_params))
    src = (
        "def _f(" + args + "):\n"
        "    import numpy as np\n"
        + (body + "\n" if body else "")
        + "    return np.broadcast_arrays("
        + outs
        + (", " if len(reduced) == 1 else "")
        + ")\n"
    )
    # sympy's pycode uses math.* names; map them onto numpy for array evaluation.
    return src.replace("math.", "np.")


def compile_source(
    src: str, coords: Sequence[sp.Symbol], n_params: int = 0
) -> Callable[..., np.ndarray]:
    """Compile source from ``generate_source`` into a broadcasting NumPy callable."""
    ns: dict = {}
    exec(src, ns)  # noqa: S102 - generated from our own SymPy expressions
    f = ns["_f"]
    n_coords = len(coords)

    def wrapped(*args):
        coord_args = args[:n_coords]
        extra = args[n_coords : n_coords + n_params]
        arrs = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in coord_args])
        out = f(*arrs, *extra)
        return np.stack([np.broadcast_to(o, arrs[0].shape) for o in out])

    return wrapped


def compile_source_jax(
    src: str, coords: Sequence[sp.Symbol], n_params: int = 0
) -> Callable[..., Any]:
    """Compile the same generated source against JAX instead of NumPy.

    The source is emitted once and executed against whichever array library
    is bound as ``np`` inside it, so the NumPy and JAX kernels are the same
    arithmetic by construction rather than by two implementations agreeing.
    That matters because a divergence between them would show up as a wrong
    gradient, which is far harder to notice than a wrong value.

    JAX is an optional dependency; this raises a clear error when it is
    missing instead of failing inside generated code.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - exercised by a skip-guarded test
        raise ImportError("JAX emission needs the jax extra: `uv sync --extra jax`") from exc

    # JAX defaults to single precision. Computing curvature in float32 would
    # violate ADR-004, and it would not look like a failure: the first
    # comparison against NumPy here matched to six digits and differed in the
    # seventh, which is exactly the size of discrepancy that gets dismissed as
    # noise while constraint residuals quietly stop converging.
    if not jax.config.read("jax_enable_x64"):
        jax.config.update("jax_enable_x64", True)

    ns: dict = {"jnp": jnp}
    # The generated source imports numpy itself; shadow that binding with
    # jax.numpy so the identical text runs on either backend.
    jax_src = src.replace("import numpy as np", "import jax.numpy as np")
    exec(jax_src, ns)  # noqa: S102 - generated from our own SymPy expressions
    f = ns["_f"]
    n_coords = len(coords)

    def wrapped(*args):
        coord_args = [jnp.asarray(a, dtype=jnp.float64) for a in args[:n_coords]]
        extra = args[n_coords : n_coords + n_params]
        arrs = jnp.broadcast_arrays(*coord_args)
        out = f(*arrs, *extra)
        return jnp.stack([jnp.broadcast_to(o, arrs[0].shape) for o in out])

    return jax.jit(wrapped)


def lambdify_exprs(
    exprs: Sequence[sp.Expr],
    coords: Sequence[sp.Symbol],
    params: dict[sp.Symbol, float] | None = None,
    free_params: Sequence[sp.Symbol] = (),
    backend: str = "numpy",
) -> Callable[..., Any]:
    """Compile a list of expressions into one function with CSE.

    The returned callable takes coordinate arrays followed by any
    ``free_params`` values, and returns an array of shape
    ``(len(exprs), *grid_shape)``. ``backend`` is ``"numpy"`` or ``"jax"``;
    the JAX kernel is jitted and differentiable in the free parameters.
    """
    src = generate_source(exprs, coords, params, free_params)
    n = len(free_params)
    if backend == "numpy":
        return compile_source(src, coords, n)
    if backend == "jax":
        return compile_source_jax(src, coords, n)
    raise ValueError(f'unknown backend {backend!r}; use "numpy" or "jax"')


def independent_riemann_indices(n: int) -> list[tuple[int, int, int, int]]:
    """Index tuples of the independent components of ``R_{abcd}``.

    Uses antisymmetry in each pair and symmetry under pair exchange:
    ``a < b``, ``c < d``, ``(a, b) <= (c, d)``. The cyclic identity is not
    used, so for n = 4 this returns 21 tuples (one more than the 20 truly
    independent ones), which keeps reconstruction trivial.
    """
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    return [(a, b, c, d) for i, (a, b) in enumerate(pairs) for (c, d) in pairs[i:]]


class RiemannEvaluator:
    """Compiles the independent ``R_{abcd}`` components and rebuilds the full tensor."""

    def __init__(
        self,
        geom: MetricGeometry,
        params: dict[sp.Symbol, float] | None = None,
    ):
        self.n = geom.n
        self.idx = independent_riemann_indices(self.n)
        Rl = geom.riemann_lower
        self._f = lambdify_exprs([Rl[a][b][c][d] for a, b, c, d in self.idx], geom.x, params)

    def at(self, x: Sequence[float]) -> np.ndarray:
        """Full ``R_{abcd}`` as an ``(n, n, n, n)`` array at a single point."""
        vals = self._f(*[np.asarray(v, dtype=float) for v in x])
        n = self.n
        R = np.zeros((n, n, n, n))
        for k, (a, b, c, d) in enumerate(self.idx):
            v = float(vals[k])
            for (i, j, s1), (k_, l, s2) in (
                ((a, b, 1.0), (c, d, 1.0)),
                ((b, a, -1.0), (c, d, 1.0)),
                ((a, b, 1.0), (d, c, -1.0)),
                ((b, a, -1.0), (d, c, -1.0)),
            ):
                R[i, j, k_, l] = s1 * s2 * v
                R[k_, l, i, j] = s1 * s2 * v
        return R

    def tidal(self, x: Sequence[float], u: Sequence[float]) -> np.ndarray:
        """Electric part of the Riemann tensor, ``E_ab = R_acbd u^c u^d``."""
        R = self.at(x)
        u = np.asarray(u, dtype=float)
        return np.einsum("acbd,c,d->ab", R, u, u)
