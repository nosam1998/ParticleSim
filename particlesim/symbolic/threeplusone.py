"""The 3+1 split: ADM algebra that runs symbolically or on a grid.

Every formula here is written once, against a small data class carrying the
slice's fields and their spatial derivatives, and it does not care whether
those are SymPy expressions or NumPy arrays. Two constructors fill that
class:

* :func:`symbolic_slice` differentiates closed-form expressions with
  ``sp.diff``;
* :func:`grid_slice` differentiates arrays with the finite-difference
  stencils in :mod:`particlesim.core.grid`.

That is not a convenience, it is the verification strategy. An exact
solution can be pushed through the *same* algebra symbolically and on a
grid, and the two answers compared: the symbolic path says whether the
formulas are right and the grid path says whether the discretisation is,
and a disagreement localises to one or the other. Writing two
implementations of the ADM equations -- one for derivation and one for
evolution, as most codes end up with -- makes that comparison impossible
and is where a factor of two lives for years.

**Conventions**, following Baumgarte and Shapiro throughout:

    ds^2 = -alpha^2 dt^2 + gamma_ij (dx^i + beta^i dt)(dx^j + beta^j dt)

    K_ij = -(1/2) Lie_n gamma_ij

    d_t gamma_ij = -2 alpha K_ij + D_i beta_j + D_j beta_i

    d_t K_ij = alpha (R_ij - 2 K_ik K^k_j + K K_ij) - D_i D_j alpha
               - 8 pi alpha (S_ij - gamma_ij (S - rho)/2)
               + beta^k d_k K_ij + K_ik d_j beta^k + K_jk d_i beta^k

    H = R + K^2 - K_ij K^ij - 16 pi rho
    M^i = D_j (K^ij - gamma^ij K) - 8 pi S^i

with ``K = gamma^ij K_ij`` and ``S = gamma^ij S_ij``. The constraints are
written as quantities that vanish on a solution, so their size is a
diagnostic rather than something to be rearranged.

Derivative index first: ``d_metric[k][i][j]`` is ``d_k gamma_ij`` and
``d_shift[k][i]`` is ``d_k beta^i``. One convention, stated once, because
the alternative is checking the transpose at every call.

The inverse metric is the explicit cofactor formula rather than a matrix
solve. It has to work on SymPy expressions and on arrays of grid values
alike, and on a grid it has to be a handful of multiplications per point
rather than a linear solve per point.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from particlesim.core.grid import derivative, second_derivative

#: Spatial dimensions. The ADM split is three-dimensional here throughout.
DIMENSION = 3
INDICES = tuple(range(DIMENSION))


def _zeros_like(sample: Any) -> Any:
    """Additive zero of the same kind as ``sample``."""
    if isinstance(sample, np.ndarray):
        return np.zeros_like(sample)
    return 0


@dataclass(frozen=True)
class Slice:
    """ADM data on one spatial slice, with the derivatives the algebra needs.

    The fields are whatever the caller's arithmetic works on: SymPy
    expressions, NumPy arrays over a grid, or JAX arrays. Nothing here
    branches on type.
    """

    lapse: Any
    shift: list[Any]
    metric: list[list[Any]]
    curvature: list[list[Any]]
    d_lapse: list[Any]
    dd_lapse: list[list[Any]]
    d_shift: list[list[Any]]
    d_metric: list[list[list[Any]]]
    dd_metric: list[list[list[list[Any]]]]
    d_curvature: list[list[list[Any]]]
    #: The coordinates the fields are expressions in, for a symbolic slice.
    #:
    #: ``None`` for a slice built from arrays. It is carried because one
    #: term in the BSSN connection equation is a spatial derivative of a
    #: quantity that is itself a spatial derivative of the state, and a
    #: symbolic slice can differentiate it where a grid slice cannot.
    coords: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.shift) != DIMENSION:
            raise ValueError(f"shift needs {DIMENSION} components, got {len(self.shift)}")
        for name in ("metric", "curvature"):
            table = getattr(self, name)
            if len(table) != DIMENSION or any(len(row) != DIMENSION for row in table):
                raise ValueError(f"{name} must be {DIMENSION}x{DIMENSION}")


def _symmetric(table: Sequence[Sequence[Any]]) -> list[list[Any]]:
    return [[table[i][j] for j in INDICES] for i in INDICES]


def symbolic_slice(lapse, shift, metric, curvature, coords) -> Slice:
    """Build a :class:`Slice` from closed-form expressions in ``coords``."""
    import sympy as sp

    if len(coords) != DIMENSION:
        raise ValueError(f"expected {DIMENSION} spatial coordinates, got {len(coords)}")
    lapse = sp.sympify(lapse)
    shift = [sp.sympify(value) for value in shift]
    metric = [[sp.sympify(metric[i][j]) for j in INDICES] for i in INDICES]
    curvature = [[sp.sympify(curvature[i][j]) for j in INDICES] for i in INDICES]

    return Slice(
        lapse=lapse,
        shift=shift,
        metric=metric,
        curvature=curvature,
        d_lapse=[sp.diff(lapse, coords[k]) for k in INDICES],
        dd_lapse=[[sp.diff(lapse, coords[k], coords[m]) for m in INDICES] for k in INDICES],
        d_shift=[[sp.diff(shift[i], coords[k]) for i in INDICES] for k in INDICES],
        d_metric=[
            [[sp.diff(metric[i][j], coords[k]) for j in INDICES] for i in INDICES] for k in INDICES
        ],
        dd_metric=[
            [
                [[sp.diff(metric[i][j], coords[k], coords[m]) for j in INDICES] for i in INDICES]
                for m in INDICES
            ]
            for k in INDICES
        ],
        d_curvature=[
            [[sp.diff(curvature[i][j], coords[k]) for j in INDICES] for i in INDICES]
            for k in INDICES
        ],
        coords=tuple(coords),
    )


def grid_slice(lapse, shift, metric, curvature, spacing, order: int = 4) -> Slice:
    """Build a :class:`Slice` from grid arrays, differencing along each axis.

    ``spacing`` is one grid spacing per axis. Pure second derivatives use
    the direct stencil and mixed ones are composed, which is the usual
    choice and the reason mixed second derivatives are the noisiest term in
    a Ricci tensor.
    """
    spacing = tuple(float(value) for value in np.atleast_1d(spacing))
    if len(spacing) == 1:
        spacing = spacing * DIMENSION
    if len(spacing) != DIMENSION:
        raise ValueError(f"expected {DIMENSION} grid spacings, got {len(spacing)}")

    def first(field, axis):
        return derivative(field, axis, spacing[axis], order=order)

    def second(field, axis, other):
        if axis == other:
            return second_derivative(field, axis, spacing[axis], order=order)
        return first(first(field, axis), other)

    metric = _symmetric(metric)
    curvature = _symmetric(curvature)
    return Slice(
        lapse=lapse,
        shift=list(shift),
        metric=metric,
        curvature=curvature,
        d_lapse=[first(lapse, k) for k in INDICES],
        dd_lapse=[[second(lapse, k, m) for m in INDICES] for k in INDICES],
        d_shift=[[first(shift[i], k) for i in INDICES] for k in INDICES],
        d_metric=[[[first(metric[i][j], k) for j in INDICES] for i in INDICES] for k in INDICES],
        dd_metric=[
            [[[second(metric[i][j], k, m) for j in INDICES] for i in INDICES] for m in INDICES]
            for k in INDICES
        ],
        d_curvature=[
            [[first(curvature[i][j], k) for j in INDICES] for i in INDICES] for k in INDICES
        ],
    )


#: Naming scheme for the symbols :func:`abstract_slice` creates.
#:
#: ``alpha``, ``beta0``, ``gamma01``, ``K01`` for the fields, with metric
#: and curvature indices sorted so that the symmetric pair shares one
#: symbol; ``d_gamma01_2`` and ``dd_gamma01_23`` for derivatives, the
#: trailing digits being the differentiation axes in order. The scheme is
#: fixed rather than configurable because the emitted kernel source is
#: meant to be read, and a reader should be able to tell what
#: ``dd_gamma01_23`` is without a legend.
FIELD_NAMES = ("alpha", "beta", "gamma", "K")


def _pair(i: int, j: int) -> str:
    low, high = sorted((i, j))
    return f"{low}{high}"


def abstract_slice(order_of_metric: int = 2):
    """A :class:`Slice` of bare symbols, for deriving expressions to compile.

    Returns ``(slice, fields, derivatives)``:

    * ``fields`` maps each independent grid field's name to its symbol --
      sixteen of them, the lapse, three shift components and six each for
      the metric and the extrinsic curvature;
    * ``derivatives`` maps each derivative symbol to the field it
      differentiates and the axes it differentiates along.

    Nothing is differentiated here. The algebra in this module runs on
    these symbols exactly as it runs on expressions or arrays, and what
    comes out is an expression tree in terms of fields and their
    derivatives -- which is what a finite-difference kernel needs, and what
    :mod:`particlesim.symbolic.codegen` substitutes stencils into. It is
    also the only way to get the *general* equations: differentiating
    closed-form expressions gives the equations for that solution, and an
    evolution needs them for any data.

    ``order_of_metric`` is how many derivatives of the metric to provide,
    two being what the Ricci tensor needs.
    """
    import sympy as sp

    fields: dict[str, Any] = {}
    derivatives: dict[Any, tuple[str, tuple[int, ...]]] = {}

    def field(name: str):
        symbol = sp.Symbol(name, real=True)
        fields[name] = symbol
        return symbol

    def first(name: str, axis: int):
        symbol = sp.Symbol(f"d_{name}_{axis}", real=True)
        derivatives[symbol] = (name, (axis,))
        return symbol

    def second(name: str, axis: int, other: int):
        low, high = sorted((axis, other))
        symbol = sp.Symbol(f"dd_{name}_{low}{high}", real=True)
        derivatives[symbol] = (name, (low, high))
        return symbol

    lapse = field("alpha")
    shift = [field(f"beta{i}") for i in INDICES]
    metric = [[field(f"gamma{_pair(i, j)}") for j in INDICES] for i in INDICES]
    curvature = [[field(f"K{_pair(i, j)}") for j in INDICES] for i in INDICES]

    slice_ = Slice(
        lapse=lapse,
        shift=shift,
        metric=metric,
        curvature=curvature,
        d_lapse=[first("alpha", k) for k in INDICES],
        dd_lapse=[[second("alpha", k, m) for m in INDICES] for k in INDICES],
        d_shift=[[first(f"beta{i}", k) for i in INDICES] for k in INDICES],
        d_metric=[
            [[first(f"gamma{_pair(i, j)}", k) for j in INDICES] for i in INDICES] for k in INDICES
        ],
        dd_metric=[
            [
                [[second(f"gamma{_pair(i, j)}", k, m) for j in INDICES] for i in INDICES]
                for m in INDICES
            ]
            for k in INDICES
        ]
        if order_of_metric >= 2
        else None,
        d_curvature=[
            [[first(f"K{_pair(i, j)}", k) for j in INDICES] for i in INDICES] for k in INDICES
        ],
    )
    return slice_, fields, derivatives


def determinant(metric) -> Any:
    """``det gamma`` by the explicit expansion."""
    g = metric
    return (
        g[0][0] * (g[1][1] * g[2][2] - g[1][2] * g[2][1])
        - g[0][1] * (g[1][0] * g[2][2] - g[1][2] * g[2][0])
        + g[0][2] * (g[1][0] * g[2][1] - g[1][1] * g[2][0])
    )


def inverse_metric(metric) -> list[list[Any]]:
    """``gamma^ij`` by cofactors, valid for expressions and for arrays alike."""
    g = metric
    det = determinant(metric)
    cofactor = [
        [
            g[1][1] * g[2][2] - g[1][2] * g[2][1],
            g[0][2] * g[2][1] - g[0][1] * g[2][2],
            g[0][1] * g[1][2] - g[0][2] * g[1][1],
        ],
        [
            g[1][2] * g[2][0] - g[1][0] * g[2][2],
            g[0][0] * g[2][2] - g[0][2] * g[2][0],
            g[0][2] * g[1][0] - g[0][0] * g[1][2],
        ],
        [
            g[1][0] * g[2][1] - g[1][1] * g[2][0],
            g[0][1] * g[2][0] - g[0][0] * g[2][1],
            g[0][0] * g[1][1] - g[0][1] * g[1][0],
        ],
    ]
    return [[cofactor[i][j] / det for j in INDICES] for i in INDICES]


def inverse_metric_derivative(inverse, d_metric) -> list[list[list[Any]]]:
    """``d_k gamma^ij = -gamma^ia gamma^jb d_k gamma_ab``.

    From differentiating ``gamma^ia gamma_aj = delta^i_j``. Using the
    identity rather than differentiating the cofactor formula keeps the
    grid path to one pass of finite differences over the metric: the
    inverse never has to be differenced, which matters because it is the
    quantity whose derivative a determinant in a denominator would make
    expensive and noisy.
    """
    return [
        [
            [
                -sum(
                    inverse[i][a] * inverse[j][b] * d_metric[k][a][b]
                    for a in INDICES
                    for b in INDICES
                )
                for j in INDICES
            ]
            for i in INDICES
        ]
        for k in INDICES
    ]


def inverse_metric_second_derivative(inverse, d_metric, dd_metric):
    """``d_k d_l gamma^ij``, indexed ``[k][l][i][j]``.

    Differentiate :func:`inverse_metric_derivative` once more:

        d_l d_k gamma^ij = -(d_l gamma^ia) gamma^jb d_k gamma_ab
                           - gamma^ia (d_l gamma^jb) d_k gamma_ab
                           - gamma^ia gamma^jb d_l d_k gamma_ab

    Same reason as the first derivative, one order up. The quantity that
    wants it is CCZ4's ``Z_i``, which is the difference between the evolved
    ``Gammahat^i`` and the ``Gammabar^i`` the metric defines; writing that
    difference as ``Gammahat^i + d_j gammabar^ij`` makes its derivative
    algebraic in ``gammabar_ij`` and its first two derivatives, where
    differentiating ``Gammabar^i`` directly would want a third.
    """
    d_inverse = inverse_metric_derivative(inverse, d_metric)
    return [
        [
            [
                [
                    -sum(
                        d_inverse[ell][i][a] * inverse[j][b] * d_metric[k][a][b]
                        + inverse[i][a] * d_inverse[ell][j][b] * d_metric[k][a][b]
                        + inverse[i][a] * inverse[j][b] * dd_metric[ell][k][a][b]
                        for a in INDICES
                        for b in INDICES
                    )
                    for j in INDICES
                ]
                for i in INDICES
            ]
            for ell in INDICES
        ]
        for k in INDICES
    ]


def christoffel(slice_: Slice, inverse=None) -> list[list[list[Any]]]:
    """``Gamma^k_ij`` of the spatial metric, indexed ``[k][i][j]``."""
    inverse = inverse_metric(slice_.metric) if inverse is None else inverse
    d = slice_.d_metric
    return [
        [
            [
                sum(inverse[k][m] * (d[i][m][j] + d[j][m][i] - d[m][i][j]) for m in INDICES) / 2
                for j in INDICES
            ]
            for i in INDICES
        ]
        for k in INDICES
    ]


def christoffel_derivative(slice_: Slice, inverse=None, d_inverse=None):
    """``d_n Gamma^k_ij``, indexed ``[n][k][i][j]``."""
    inverse = inverse_metric(slice_.metric) if inverse is None else inverse
    if d_inverse is None:
        d_inverse = inverse_metric_derivative(inverse, slice_.d_metric)
    d, dd = slice_.d_metric, slice_.dd_metric
    return [
        [
            [
                [
                    sum(
                        d_inverse[n][k][m] * (d[i][m][j] + d[j][m][i] - d[m][i][j])
                        + inverse[k][m] * (dd[n][i][m][j] + dd[n][j][m][i] - dd[n][m][i][j])
                        for m in INDICES
                    )
                    / 2
                    for j in INDICES
                ]
                for i in INDICES
            ]
            for k in INDICES
        ]
        for n in INDICES
    ]


def ricci(slice_: Slice, inverse=None) -> list[list[Any]]:
    """Spatial Ricci tensor ``R_ij`` from the Christoffels and their derivatives.

    R_ij = d_k Gamma^k_ij - d_j Gamma^k_ik
           + Gamma^k_ij Gamma^l_kl - Gamma^l_ik Gamma^k_jl
    """
    inverse = inverse_metric(slice_.metric) if inverse is None else inverse
    gamma = christoffel(slice_, inverse)
    d_gamma = christoffel_derivative(slice_, inverse)
    out = []
    for i in INDICES:
        row = []
        for j in INDICES:
            total = _zeros_like(slice_.lapse)
            for k in INDICES:
                total = total + d_gamma[k][k][i][j] - d_gamma[j][k][i][k]
                for m in INDICES:
                    total = (
                        total + gamma[k][i][j] * gamma[m][k][m] - gamma[m][i][k] * gamma[k][j][m]
                    )
            row.append(total)
        out.append(row)
    return out


def trace(inverse, tensor) -> Any:
    """``gamma^ij T_ij``."""
    total = None
    for i in INDICES:
        for j in INDICES:
            term = inverse[i][j] * tensor[i][j]
            total = term if total is None else total + term
    return total


def raise_index(inverse, tensor) -> list[list[Any]]:
    """``T^i_j = gamma^ik T_kj``."""
    return [[sum(inverse[i][k] * tensor[k][j] for k in INDICES) for j in INDICES] for i in INDICES]


def lapse_hessian(slice_: Slice, inverse=None) -> list[list[Any]]:
    """``D_i D_j alpha = d_i d_j alpha - Gamma^k_ij d_k alpha``."""
    inverse = inverse_metric(slice_.metric) if inverse is None else inverse
    gamma = christoffel(slice_, inverse)
    return [
        [
            slice_.dd_lapse[i][j] - sum(gamma[k][i][j] * slice_.d_lapse[k] for k in INDICES)
            for j in INDICES
        ]
        for i in INDICES
    ]


def shift_covariant_derivative(slice_: Slice, inverse=None) -> list[list[Any]]:
    """``D_i beta_j``, with ``beta_j = gamma_jk beta^k``.

    Expanded as ``gamma_jk D_i beta^k`` rather than by lowering first and
    differencing afterwards: the metric's derivative then appears once,
    through the Christoffels, instead of twice.
    """
    inverse = inverse_metric(slice_.metric) if inverse is None else inverse
    gamma = christoffel(slice_, inverse)
    out = []
    for i in INDICES:
        row = []
        for j in INDICES:
            total = _zeros_like(slice_.lapse)
            for k in INDICES:
                covariant = slice_.d_shift[i][k] + sum(
                    gamma[k][i][m] * slice_.shift[m] for m in INDICES
                )
                total = total + slice_.metric[j][k] * covariant
            row.append(total)
        out.append(row)
    return out


def hamiltonian_constraint(slice_: Slice, density=0.0, inverse=None) -> Any:
    """``H = R + K^2 - K_ij K^ij - 16 pi rho``, zero on a solution."""
    inverse = inverse_metric(slice_.metric) if inverse is None else inverse
    scalar = trace(inverse, ricci(slice_, inverse))
    mean = trace(inverse, slice_.curvature)
    mixed = raise_index(inverse, slice_.curvature)
    square = sum(mixed[i][j] * mixed[j][i] for i in INDICES for j in INDICES)
    return scalar + mean**2 - square - 16 * math.pi * density


def momentum_constraint(slice_: Slice, momentum=None, inverse=None) -> list[Any]:
    """``M^i = D_j (K^ij - gamma^ij K) - 8 pi S^i``, zero on a solution.

    ``momentum`` is the contravariant ``S^i``.
    """
    inverse = inverse_metric(slice_.metric) if inverse is None else inverse
    gamma = christoffel(slice_, inverse)
    d_inverse = inverse_metric_derivative(inverse, slice_.d_metric)
    mean = trace(inverse, slice_.curvature)
    d_mean = [
        sum(
            d_inverse[k][i][j] * slice_.curvature[i][j]
            + inverse[i][j] * slice_.d_curvature[k][i][j]
            for i in INDICES
            for j in INDICES
        )
        for k in INDICES
    ]

    # K^ij and its derivative, both from the identity rather than from a
    # second pass of differencing.
    upper = [
        [
            sum(
                inverse[i][a] * inverse[j][b] * slice_.curvature[a][b]
                for a in INDICES
                for b in INDICES
            )
            for j in INDICES
        ]
        for i in INDICES
    ]
    d_upper = [
        [
            [
                sum(
                    d_inverse[k][i][a] * inverse[j][b] * slice_.curvature[a][b]
                    + inverse[i][a] * d_inverse[k][j][b] * slice_.curvature[a][b]
                    + inverse[i][a] * inverse[j][b] * slice_.d_curvature[k][a][b]
                    for a in INDICES
                    for b in INDICES
                )
                for j in INDICES
            ]
            for i in INDICES
        ]
        for k in INDICES
    ]

    tensor = [[upper[a][b] - inverse[a][b] * mean for b in INDICES] for a in INDICES]
    out = []
    for i in INDICES:
        total = _zeros_like(slice_.lapse)
        for j in INDICES:
            d_tensor = d_upper[j][i][j] - (d_inverse[j][i][j] * mean + inverse[i][j] * d_mean[j])
            total = total + d_tensor
            for k in INDICES:
                total = total + gamma[i][j][k] * tensor[k][j] + gamma[j][j][k] * tensor[i][k]
        if momentum is not None:
            total = total - 8 * math.pi * momentum[i]
        out.append(total)
    return out


def adm_rhs(slice_: Slice, density=0.0, stress=None, inverse=None):
    """``(d_t gamma_ij, d_t K_ij)`` from the ADM evolution equations.

    ``stress`` is the covariant spatial stress ``S_ij``; its trace and
    ``density`` enter the curvature equation together as
    ``S_ij - gamma_ij (S - rho)/2``. In vacuum both are omitted.
    """
    inverse = inverse_metric(slice_.metric) if inverse is None else inverse
    ricci_tensor = ricci(slice_, inverse)
    hessian = lapse_hessian(slice_, inverse)
    shift_derivative = shift_covariant_derivative(slice_, inverse)
    mean = trace(inverse, slice_.curvature)
    mixed = raise_index(inverse, slice_.curvature)

    dt_metric = []
    for i in INDICES:
        row = []
        for j in INDICES:
            row.append(
                -2 * slice_.lapse * slice_.curvature[i][j]
                + shift_derivative[i][j]
                + shift_derivative[j][i]
            )
        dt_metric.append(row)

    dt_curvature = []
    for i in INDICES:
        row = []
        for j in INDICES:
            squared = sum(slice_.curvature[i][k] * mixed[k][j] for k in INDICES)
            total = (
                slice_.lapse * (ricci_tensor[i][j] - 2 * squared + mean * slice_.curvature[i][j])
                - hessian[i][j]
            )
            for k in INDICES:
                total = total + slice_.shift[k] * slice_.d_curvature[k][i][j]
                total = total + slice_.curvature[i][k] * slice_.d_shift[j][k]
                total = total + slice_.curvature[j][k] * slice_.d_shift[i][k]
            if stress is not None:
                traced = trace(inverse, stress)
                total = total - 8 * math.pi * slice_.lapse * (
                    stress[i][j] - slice_.metric[i][j] * (traced - density) / 2
                )
            row.append(total)
        dt_curvature.append(row)

    return dt_metric, dt_curvature
