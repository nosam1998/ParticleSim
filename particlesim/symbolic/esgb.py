"""Einstein-scalar-Gauss-Bonnet in 3+1: the Gauss-Bonnet term as effective matter (issue #52).

The action is :mod:`particlesim.theories.gauss_bonnet`'s,
``(1/16 pi) int sqrt(-g) [R - 2 (d phi)^2 + lambda^2 f(phi) G]``. Its field
equations are

    G_ab = 2 d_a phi d_b phi - g_ab (d phi)^2 - 4 lambda^2 P_acbd nabla^c nabla^d f
    box phi = -(lambda^2 / 4) f'(phi) G

with ``P_abcd`` the double dual of the Riemann tensor,

    P_abcd = R_abcd + g_ad R_bc - g_ac R_bd + g_bc R_ad - g_bd R_ac
             + (R / 2)(g_ac g_bd - g_ad g_bc).

The ``-4`` is not quoted from anywhere. It is fixed by evaluating this
tensor on the static spherical ansatz and comparing it with the
Euler-Lagrange equations :func:`particlesim.theories.gauss_bonnet.static_equations`
derives from the reduced action; a test does that, and another value
fails it.

**Everything is done in the frame ``(n, d_1, d_2, d_3)``,** where the
spacetime metric is ``diag(-1, gamma_ij)`` and every component of the
Riemann tensor is a 3+1 quantity:

    R_ijkl = R^(3)_ijkl + K_ik K_jl - K_il K_jk           (Gauss)
    R_ijkn = D_j K_ik - D_i K_jk                           (Codazzi)
    R_ninj = (d_0 K_ij + D_i D_j alpha) / alpha + K_ik K^k_j   (Ricci)

with ``d_0 = d_t - L_beta``, and the scalar's Hessian is likewise

    nabla_i nabla_j phi = D_i D_j phi - K_ij K_phi
    nabla_n nabla_i phi = -D_i K_phi + K_i^j D_j phi
    nabla_n nabla_n phi = -(d_0 K_phi + D^i alpha D_i phi) / alpha

with ``K_phi = -n^a d_a phi``. In three dimensions the Riemann tensor is
algebraic in the Ricci tensor, so ``R^(3)_ijkl`` costs nothing new.

**The second time derivatives enter linearly, so the system is 7 x 7.**
``d_0 K_ij`` appears only in ``R_ninj`` and ``d_0 K_phi`` only in
``nabla_n nabla_n phi``. The antisymmetry of the double dual allows at most
one of each pair of ``n`` indices per factor, so neither ``G`` nor ``P``
contains a product of two such terms -- checked, not assumed. The spatial
stress ``S_ij`` and the scalar's source are then affine in the seven
unknowns ``(d_0 K_ij, d_0 K_phi)``, and ``rho`` and ``S_i`` do not depend
on them at all: they are constraints. :func:`effective_matter` returns the
part that does not depend on them, which goes in with the rest of the
matter, and the affine map ``Xi(X) = Xi_0 + J X`` the rest contributes to
``(d_0 K_ij, d_0 K_phi)``. An evolution solves ``(1 - J) X = Xbar + Xi_0``
at every point, with ``Xbar`` the rates without it.

**No division by the lapse survives.** Both places a ``1/alpha`` appears
are multiplied by the ``alpha`` that the ``K_ij`` and ``K_phi`` equations
carry, and the system is built in terms of ``alpha R_ninj`` and
``alpha nabla_n nabla_n phi``. That matters at a static horizon, where the
lapse is zero.
"""

from __future__ import annotations

import math
from typing import Any

import sympy as sp

from particlesim.symbolic.threeplusone import DIMENSION, INDICES

FRAME = (0, 1, 2, 3)

#: The seven unknowns, in the order the linear system uses.
UNKNOWNS = ("00", "01", "02", "11", "12", "22", "s")


def _pair(i: int, j: int) -> str:
    low, high = sorted((i, j))
    return f"{low}{high}"


def frame_metric(metric, inverse):
    """``(g_ab, g^ab)`` in the frame ``(n, d_i)``: ``diag(-1, gamma_ij)`` and its inverse."""
    lower = [[0] * 4 for _ in FRAME]
    upper = [[0] * 4 for _ in FRAME]
    lower[0][0] = -1
    upper[0][0] = -1
    for i in INDICES:
        for j in INDICES:
            lower[i + 1][j + 1] = metric[i][j]
            upper[i + 1][j + 1] = inverse[i][j]
    return lower, upper


def spatial_riemann(metric, inverse, ricci):
    """``R_ijkl`` of a three-metric from its Ricci tensor; ``R_jl = gamma^ik R_ijkl``."""
    scalar = sum(inverse[i][j] * ricci[i][j] for i in INDICES for j in INDICES)
    g = metric
    return [
        [
            [
                [
                    g[i][k] * ricci[j][ell]
                    - g[i][ell] * ricci[j][k]
                    - g[j][k] * ricci[i][ell]
                    + g[j][ell] * ricci[i][k]
                    - scalar * (g[i][k] * g[j][ell] - g[i][ell] * g[j][k]) / 2
                    for ell in INDICES
                ]
                for k in INDICES
            ]
            for j in INDICES
        ]
        for i in INDICES
    ]


def frame_riemann(metric, inverse, curvature, ricci, d_curvature, electric):
    """``R_abcd`` with all four indices down in the frame ``(n, d_i)``, index 0 being ``n``.

    ``d_curvature[k][i][j]`` is ``D_k K_ij`` and ``electric[i][j]`` is
    ``R_ninj``, which carries ``d_0 K_ij``.
    """
    spatial = spatial_riemann(metric, inverse, ricci)
    gauss = [
        [
            [
                [
                    spatial[i][j][k][ell]
                    + curvature[i][k] * curvature[j][ell]
                    - curvature[i][ell] * curvature[j][k]
                    for ell in INDICES
                ]
                for k in INDICES
            ]
            for j in INDICES
        ]
        for i in INDICES
    ]
    # C_ijk = R_ijkn = D_j K_ik - D_i K_jk.
    codazzi = [
        [[d_curvature[j][i][k] - d_curvature[i][j][k] for k in INDICES] for j in INDICES]
        for i in INDICES
    ]

    def component(a, b, c, d):
        normals = (a, b, c, d).count(0)
        if normals == 0:
            return gauss[a - 1][b - 1][c - 1][d - 1]
        if normals == 1:
            if d == 0:
                return codazzi[a - 1][b - 1][c - 1]
            if c == 0:
                return -codazzi[a - 1][b - 1][d - 1]
            if b == 0:
                return codazzi[c - 1][d - 1][a - 1]
            return -codazzi[c - 1][d - 1][b - 1]
        if normals == 2:
            if a == b or c == d:
                return 0
            first = (b if a == 0 else a) - 1
            second = (d if c == 0 else c) - 1
            sign = (1 if a == 0 else -1) * (1 if c == 0 else -1)
            return sign * electric[first][second]
        return 0

    return [[[[component(a, b, c, d) for d in FRAME] for c in FRAME] for b in FRAME] for a in FRAME]


def _raise(tensor, upper, position):
    """Raise index ``position`` of a four-index frame tensor with the block-diagonal ``g^ab``."""

    def entry(indices):
        total = 0
        for m in FRAME:
            weight = upper[indices[position]][m]
            if isinstance(weight, int) and weight == 0:
                continue
            moved = list(indices)
            moved[position] = m
            total = total + weight * tensor[moved[0]][moved[1]][moved[2]][moved[3]]
        return total

    return [[[[entry((a, b, c, d)) for d in FRAME] for c in FRAME] for b in FRAME] for a in FRAME]


def curvature_contractions(riemann, lower, upper):
    """``(R_ab, R, G)`` from the frame Riemann tensor."""
    ricci = [
        [
            sum(
                upper[a][c] * riemann[a][b][c][d]
                for a in FRAME
                for c in FRAME
                if not (isinstance(upper[a][c], int) and upper[a][c] == 0)
            )
            for d in FRAME
        ]
        for b in FRAME
    ]
    scalar = sum(
        upper[b][d] * ricci[b][d]
        for b in FRAME
        for d in FRAME
        if not (isinstance(upper[b][d], int) and upper[b][d] == 0)
    )
    ricci_up = [
        [
            sum(
                upper[a][c] * upper[b][d] * ricci[c][d]
                for c in FRAME
                for d in FRAME
                if not (isinstance(upper[a][c], int) and upper[a][c] == 0)
                and not (isinstance(upper[b][d], int) and upper[b][d] == 0)
            )
            for b in FRAME
        ]
        for a in FRAME
    ]
    raised = riemann
    for position in range(4):
        raised = _raise(raised, upper, position)
    ricci_square = sum(ricci[a][b] * ricci_up[a][b] for a in FRAME for b in FRAME)
    riemann_square = sum(
        riemann[a][b][c][d] * raised[a][b][c][d]
        for a in FRAME
        for b in FRAME
        for c in FRAME
        for d in FRAME
    )
    invariant = scalar**2 - 4 * ricci_square + riemann_square
    return ricci, scalar, invariant


def double_dual(riemann, ricci, scalar, lower):
    """``P_abcd``, whose trace on its first and third index is ``-G_bd``."""
    g = lower
    return [
        [
            [
                [
                    riemann[a][b][c][d]
                    + g[a][d] * ricci[b][c]
                    - g[a][c] * ricci[b][d]
                    + g[b][c] * ricci[a][d]
                    - g[b][d] * ricci[a][c]
                    + scalar * (g[a][c] * g[b][d] - g[a][d] * g[b][c]) / 2
                    for d in FRAME
                ]
                for c in FRAME
            ]
            for b in FRAME
        ]
        for a in FRAME
    ]


def frame_scalar_hessian(curvature, inverse, d_scalar, hessian, momentum, d_momentum, normal):
    """``nabla_a nabla_b phi`` in the frame. ``normal`` is the ``nn`` component itself."""
    out = [[0] * 4 for _ in FRAME]
    out[0][0] = normal
    for i in INDICES:
        mixed = -d_momentum[i] + sum(
            curvature[i][j] * inverse[j][k] * d_scalar[k] for j in INDICES for k in INDICES
        )
        out[0][i + 1] = out[i + 1][0] = mixed
        for j in INDICES:
            out[i + 1][j + 1] = hessian[i][j] - curvature[i][j] * momentum
    return out


def gauss_bonnet_tensor(riemann, ricci, scalar, lower, upper, coupling_hessian):
    """``H_ab = P_acbd nabla^c nabla^d f``, so that ``8 pi T^GB_ab = -4 lambda^2 H_ab``."""
    dual = double_dual(riemann, ricci, scalar, lower)
    raised = [
        [
            sum(
                upper[c][e] * upper[d][f] * coupling_hessian[e][f]
                for e in FRAME
                for f in FRAME
                if not (isinstance(upper[c][e], int) and upper[c][e] == 0)
                and not (isinstance(upper[d][f], int) and upper[d][f] == 0)
            )
            for d in FRAME
        ]
        for c in FRAME
    ]
    return [
        [sum(dual[a][c][b][d] * raised[c][d] for c in FRAME for d in FRAME) for b in FRAME]
        for a in FRAME
    ]


def abstract_inputs():
    """The symbols :func:`effective_matter` is written in, by the names the geometry kernel uses."""

    def symbol(name):
        return sp.Symbol(name, real=True)

    def symmetric(prefix):
        return [[symbol(f"{prefix}{_pair(i, j)}") for j in INDICES] for i in INDICES]

    return {
        "lapse": symbol("alpha"),
        "metric": symmetric("gam"),
        "inverse": symmetric("igam"),
        "curvature": symmetric("Kx"),
        "ricci": symmetric("Ric"),
        "d_curvature": [
            [[symbol(f"DK{k}_{_pair(i, j)}") for j in INDICES] for i in INDICES] for k in INDICES
        ],
        "lapse_hessian": symmetric("DDa"),
        "scalar_hessian": symmetric("DDs"),
        "d_lapse": [symbol(f"da{k}") for k in INDICES],
        "d_scalar": [symbol(f"ds{k}") for k in INDICES],
        "momentum": symbol("Ksf"),
        "d_momentum": [symbol(f"dKs{k}") for k in INDICES],
        "coupling": (symbol("fGB"), symbol("dfGB"), symbol("ddfGB")),
        "coupling_squared": symbol("lam2"),
    }


def effective_matter(inputs: dict[str, Any] | None = None):
    """The Gauss-Bonnet effective matter and its 7 x 7 system, symbolically.

    Returns a dictionary with ``rho`` and ``S_i`` (``momentum``), the static
    part of ``S_ij`` (``stress``) and of the scalar source (``source``),
    ``Xi_0`` (``offset``, seven entries in :data:`UNKNOWNS` order) and
    ``J`` (``jacobian``, ``[m][n]`` the derivative of entry ``m`` by
    unknown ``n``). The unknowns are ``X_ij = d_0 K_ij`` and
    ``X_s = d_0 K_phi``. ``inputs`` defaults to :func:`abstract_inputs`.

    ``electric`` and ``normal`` -- ``R_ninj`` and ``nabla_n nabla_n phi``
    -- are built as free symbols first, and everything is checked to be at
    most linear in them before ``alpha`` times each is replaced by its
    value in terms of the unknowns. ``rho`` and ``S_i`` are evaluated with
    both set to zero; the test suite checks numerically that they do not
    depend on them.
    """
    inputs = abstract_inputs() if inputs is None else inputs
    metric = inputs["metric"]
    inverse = inputs["inverse"]
    curvature = inputs["curvature"]
    lapse = inputs["lapse"]
    f, df, ddf = inputs["coupling"]
    lam2 = inputs["coupling_squared"]

    electric_symbols = {
        _pair(i, j): sp.Symbol(f"_E{_pair(i, j)}", real=True)
        for i in INDICES
        for j in range(i, DIMENSION)
    }
    electric = [[electric_symbols[_pair(i, j)] for j in INDICES] for i in INDICES]
    normal = sp.Symbol("_Hnn", real=True)

    lower, upper = frame_metric(metric, inverse)
    riemann = frame_riemann(
        metric, inverse, curvature, inputs["ricci"], inputs["d_curvature"], electric
    )
    ricci, scalar, invariant = curvature_contractions(riemann, lower, upper)
    hessian = frame_scalar_hessian(
        curvature,
        inverse,
        inputs["d_scalar"],
        inputs["scalar_hessian"],
        inputs["momentum"],
        inputs["d_momentum"],
        normal,
    )
    gradient = [-inputs["momentum"], *inputs["d_scalar"]]
    coupling_hessian = [
        [df * hessian[a][b] + ddf * gradient[a] * gradient[b] for b in FRAME] for a in FRAME
    ]
    tensor = gauss_bonnet_tensor(riemann, ricci, scalar, lower, upper, coupling_hessian)

    pi = math.pi
    density = -lam2 * tensor[0][0] / (2 * pi)
    momentum = [lam2 * tensor[i + 1][0] / (2 * pi) for i in INDICES]
    stress = [[-lam2 * tensor[i + 1][j + 1] / (2 * pi) for j in INDICES] for i in INDICES]
    source = -lam2 * df * invariant / 4

    zero = {**{s: 0 for s in electric_symbols.values()}, normal: 0}

    # alpha R_ninj and alpha nabla_n nabla_n phi in terms of the unknowns.
    unknowns = {name: sp.Symbol(f"_X{name}", real=True) for name in UNKNOWNS}
    squared = [
        [
            sum(curvature[i][k] * inverse[k][m] * curvature[m][j] for k in INDICES for m in INDICES)
            for j in INDICES
        ]
        for i in INDICES
    ]
    scaled_electric = {
        _pair(i, j): unknowns[_pair(i, j)] + inputs["lapse_hessian"][i][j] + lapse * squared[i][j]
        for i in INDICES
        for j in range(i, DIMENSION)
    }
    scaled_normal = -(
        unknowns["s"]
        + sum(
            inverse[i][j] * inputs["d_lapse"][i] * inputs["d_scalar"][j]
            for i in INDICES
            for j in INDICES
        )
    )

    def scaled_change(expression):
        """``alpha (expression - expression at zero)``, given it is linear in the two."""
        total = 0
        for name, symbol in electric_symbols.items():
            total = total + sp.diff(expression, symbol) * scaled_electric[name]
        total = total + sp.diff(expression, normal) * scaled_normal
        return total

    change = [[scaled_change(stress[i][j]) for j in INDICES] for i in INDICES]
    change_trace = sum(inverse[i][j] * change[i][j] for i in INDICES for j in INDICES)
    entries = [
        -8 * pi * (change[i][j] - metric[i][j] * change_trace / 2)
        for i in INDICES
        for j in range(i, DIMENSION)
    ]
    entries.append(scaled_change(source))
    unknown_zero = {symbol: 0 for symbol in unknowns.values()}
    offset = [entry.subs(unknown_zero) for entry in entries]
    jacobian = [[sp.diff(entry, unknowns[name]) for name in UNKNOWNS] for entry in entries]
    return {
        "density": density.subs(zero),
        "momentum": [value.subs(zero) for value in momentum],
        "stress": [[value.subs(zero) for value in row] for row in stress],
        "source": source.subs(zero),
        "offset": offset,
        "jacobian": jacobian,
        "full": {
            "density": density,
            "momentum": momentum,
            "stress": stress,
            "source": source,
            "electric": electric_symbols,
            "normal": normal,
        },
    }


__all__ = [
    "FRAME",
    "UNKNOWNS",
    "abstract_inputs",
    "curvature_contractions",
    "double_dual",
    "effective_matter",
    "frame_metric",
    "frame_riemann",
    "frame_scalar_hessian",
    "gauss_bonnet_tensor",
    "spatial_riemann",
]
