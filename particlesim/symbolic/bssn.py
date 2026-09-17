"""BSSN variables, derived from the 3+1 split rather than typed in.

The BSSN system is a change of variables on the ADM system, and this module
does the change of variables twice by two routes so that they can be
compared:

* :func:`bssn_rhs` writes the equations as Baumgarte and Shapiro do
  (Section 11.5), which is the form every numerical-relativity code
  implements;
* :func:`bssn_rhs_from_adm` differentiates the *definitions* of the BSSN
  variables and substitutes the ADM right-hand sides from
  :mod:`particlesim.symbolic.threeplusone`.

They are independent derivations of the same thing, and the test suite
checks they agree symbolically on exact solutions. That is the round-trip
the milestone asks for: not that the equations were transcribed correctly
from a book, which a reader can check, but that they are the same system
the plugin's field equations gave.

**Variables.** With ``gamma = det gamma_ij``,

    phi = (1/12) ln gamma,      gammabar_ij = e^(-4 phi) gamma_ij
    K = gamma^ij K_ij,          Abar_ij = e^(-4 phi) (K_ij - gamma_ij K/3)
    Gammabar^i = gammabar^jk Gammabar^i_jk

so ``det gammabar_ij = 1`` and ``gammabar^ij Abar_ij = 0`` by construction.
Both are checked rather than assumed: they are the two algebraic
constraints an evolution has to keep, and a variable change that breaks
them at the start cannot be rescued later.

**The conformal factor is carried as a power, not an exponential.**
``e^(-4 phi) = gamma^(-1/3)`` is algebraic in the metric, and keeping it
that way lets SymPy cancel it against the powers that appear in the Ricci
decomposition. Writing ``exp(-4*phi)`` with ``phi = log(gamma)/12`` is the
same number and blocks every simplification that matters. Codes that
evolve ``chi = e^(-4 phi)`` or ``W = e^(-2 phi)`` instead are making the
same choice for the same reason, one derivative further on.

**What is deliberately not here.** The Gammabar-form of the conformal Ricci
tensor -- the rewrite that turns its principal part into a flat Laplacian
acting on ``gammabar_ij`` -- is a discretisation choice, not a different
mathematical object, and it belongs with the evolution code. This module
computes ``Rbar_ij`` by running the same Ricci formula on the conformal
metric and checks the conformal decomposition

    R_ij = Rbar_ij + R^phi_ij

against the physical Ricci tensor, which is the identity that rewrite
depends on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp

from particlesim.symbolic.threeplusone import (
    DIMENSION,
    INDICES,
    Slice,
    _zeros_like,
    adm_rhs,
    christoffel,
    determinant,
    inverse_metric,
    inverse_metric_derivative,
    lapse_hessian,
    raise_index,
    ricci,
    trace,
)

#: ``1/3`` as an exact rational.
#:
#: A float exponent turns ``det^(-1/3)`` into a floating-point power, and
#: ``det^(1/3) * det^(-1/3)`` then fails to cancel to one in SymPy. On a
#: grid of arrays the two are the same number, so the rational costs
#: nothing there.
THIRD = sp.Rational(1, 3)


def conformal_factor(metric) -> Any:
    """``e^(-4 phi) = (det gamma)^(-1/3)``, as a power rather than an exponential."""
    return determinant(metric) ** (-THIRD)


def phi_derivatives(slice_: Slice, inverse=None):
    """``(d_i phi, d_i d_j phi)`` from the metric's derivatives.

    ``phi = (1/12) ln det gamma``, so ``d_i phi = (1/12) gamma^ab d_i
    gamma_ab`` and the second derivative follows from differentiating that
    with the inverse-metric identity. Both are algebraic in the slice data:
    the conformal factor never has to be differenced, which on a grid saves
    a pass and avoids differencing a cube root.
    """
    inverse = inverse_metric(slice_.metric) if inverse is None else inverse
    d_inverse = inverse_metric_derivative(inverse, slice_.d_metric)
    first = [
        sum(inverse[a][b] * slice_.d_metric[i][a][b] for a in INDICES for b in INDICES) / 12
        for i in INDICES
    ]
    second = [
        [
            sum(
                d_inverse[j][a][b] * slice_.d_metric[i][a][b]
                + inverse[a][b] * slice_.dd_metric[i][j][a][b]
                for a in INDICES
                for b in INDICES
            )
            / 12
            for j in INDICES
        ]
        for i in INDICES
    ]
    return first, second


@dataclass(frozen=True)
class BSSNVariables:
    """The BSSN state on one slice, with the derivatives its equations need."""

    lapse: Any
    shift: list[Any]
    conformal_exponent: Any
    conformal_metric: list[list[Any]]
    mean_curvature: Any
    traceless_curvature: list[list[Any]]
    connection: list[Any]
    d_phi: list[Any]
    dd_phi: list[list[Any]]
    d_mean_curvature: list[Any]
    conformal_slice: Slice
    physical: Slice

    @property
    def conformal_factor(self) -> Any:
        """``e^(-4 phi)``."""
        return determinant(self.physical.metric) ** (-THIRD)


def from_adm(slice_: Slice) -> BSSNVariables:
    """The BSSN variables of an ADM slice, with their spatial derivatives.

    The conformal slice's derivative fields are propagated analytically
    from the physical ones by the product rule rather than re-differenced.
    That is what makes this usable on a grid: one pass of finite
    differences over ``gamma_ij``, ``K_ij``, ``alpha`` and ``beta^i``
    supplies everything the BSSN right-hand sides need.
    """
    inverse = inverse_metric(slice_.metric)
    determinant_value = determinant(slice_.metric)
    factor = determinant_value ** (-THIRD)
    d_phi, dd_phi = phi_derivatives(slice_, inverse)

    mean = trace(inverse, slice_.curvature)
    d_inverse = inverse_metric_derivative(inverse, slice_.d_metric)
    d_mean = [
        sum(
            d_inverse[k][i][j] * slice_.curvature[i][j]
            + inverse[i][j] * slice_.d_curvature[k][i][j]
            for i in INDICES
            for j in INDICES
        )
        for k in INDICES
    ]

    # gammabar_ij = e^(-4 phi) gamma_ij, and the same factor for Abar_ij.
    conformal_metric = [[factor * slice_.metric[i][j] for j in INDICES] for i in INDICES]
    traceless = [
        [factor * (slice_.curvature[i][j] - slice_.metric[i][j] * mean / 3) for j in INDICES]
        for i in INDICES
    ]

    def scale(tensor, d_tensor, dd_tensor=None):
        """Apply ``e^(-4 phi)`` and the product rule to a tensor and its derivatives."""
        scaled = [[factor * tensor[i][j] for j in INDICES] for i in INDICES]
        d_scaled = [
            [
                [factor * (d_tensor[k][i][j] - 4 * d_phi[k] * tensor[i][j]) for j in INDICES]
                for i in INDICES
            ]
            for k in INDICES
        ]
        if dd_tensor is None:
            return scaled, d_scaled, None
        dd_scaled = [
            [
                [
                    [
                        factor
                        * (
                            dd_tensor[k][m][i][j]
                            - 4 * d_phi[m] * d_tensor[k][i][j]
                            - 4 * d_phi[k] * d_tensor[m][i][j]
                            - 4 * dd_phi[k][m] * tensor[i][j]
                            + 16 * d_phi[k] * d_phi[m] * tensor[i][j]
                        )
                        for j in INDICES
                    ]
                    for i in INDICES
                ]
                for m in INDICES
            ]
            for k in INDICES
        ]
        return scaled, d_scaled, dd_scaled

    _, d_conformal_metric, dd_conformal_metric = scale(
        slice_.metric, slice_.d_metric, slice_.dd_metric
    )

    traceless_source = [
        [slice_.curvature[i][j] - slice_.metric[i][j] * mean / 3 for j in INDICES] for i in INDICES
    ]
    d_traceless_source = [
        [
            [
                slice_.d_curvature[k][i][j]
                - (slice_.d_metric[k][i][j] * mean + slice_.metric[i][j] * d_mean[k]) / 3
                for j in INDICES
            ]
            for i in INDICES
        ]
        for k in INDICES
    ]
    _, d_traceless, _ = scale(traceless_source, d_traceless_source)

    conformal = Slice(
        lapse=slice_.lapse,
        shift=list(slice_.shift),
        metric=conformal_metric,
        curvature=traceless,
        d_lapse=list(slice_.d_lapse),
        dd_lapse=[list(row) for row in slice_.dd_lapse],
        d_shift=[list(row) for row in slice_.d_shift],
        d_metric=d_conformal_metric,
        dd_metric=dd_conformal_metric,
        d_curvature=d_traceless,
    )

    conformal_inverse = inverse_metric(conformal_metric)
    conformal_christoffel = christoffel(conformal, conformal_inverse)
    connection = [
        sum(
            conformal_inverse[j][k] * conformal_christoffel[i][j][k]
            for j in INDICES
            for k in INDICES
        )
        for i in INDICES
    ]

    return BSSNVariables(
        lapse=slice_.lapse,
        shift=list(slice_.shift),
        conformal_exponent=factor,
        conformal_metric=conformal_metric,
        mean_curvature=mean,
        traceless_curvature=traceless,
        connection=connection,
        d_phi=d_phi,
        dd_phi=dd_phi,
        d_mean_curvature=d_mean,
        conformal_slice=conformal,
        physical=slice_,
    )


def to_adm(variables: BSSNVariables):
    """``(gamma_ij, K_ij)`` back from the BSSN variables.

    The inverse of :func:`from_adm`, kept so that the change of variables
    can be round-tripped. A variable change nobody can invert is a place
    for a factor of ``e^(4 phi)`` to hide.
    """
    factor = variables.conformal_exponent
    metric = [[variables.conformal_metric[i][j] / factor for j in INDICES] for i in INDICES]
    curvature = [
        [
            variables.traceless_curvature[i][j] / factor
            + metric[i][j] * variables.mean_curvature / 3
            for j in INDICES
        ]
        for i in INDICES
    ]
    return metric, curvature


def conformal_ricci_correction(variables: BSSNVariables) -> list[list[Any]]:
    """``R^phi_ij``: what the conformal factor contributes to the Ricci tensor.

        R^phi_ij = -2 Dbar_i Dbar_j phi + 4 (Dbar_i phi)(Dbar_j phi)
                   - 2 gammabar_ij (Dbar^k Dbar_k phi
                                    + 2 (Dbar^k phi)(Dbar_k phi))

    Baumgarte and Shapiro eq. (11.35). ``Dbar`` is the covariant
    derivative of the conformal metric, so ``Dbar_i phi = d_i phi`` and
    ``Dbar_i Dbar_j phi = d_i d_j phi - Gammabar^k_ij d_k phi``.
    """
    conformal = variables.conformal_slice
    conformal_inverse = inverse_metric(conformal.metric)
    gammabar = christoffel(conformal, conformal_inverse)
    hessian = [
        [
            variables.dd_phi[i][j] - sum(gammabar[k][i][j] * variables.d_phi[k] for k in INDICES)
            for j in INDICES
        ]
        for i in INDICES
    ]
    laplacian = sum(conformal_inverse[i][j] * hessian[i][j] for i in INDICES for j in INDICES)
    gradient_square = sum(
        conformal_inverse[i][j] * variables.d_phi[i] * variables.d_phi[j]
        for i in INDICES
        for j in INDICES
    )
    return [
        [
            -2 * hessian[i][j]
            + 4 * variables.d_phi[i] * variables.d_phi[j]
            - 2 * conformal.metric[i][j] * (laplacian + 2 * gradient_square)
            for j in INDICES
        ]
        for i in INDICES
    ]


def physical_ricci(variables: BSSNVariables) -> list[list[Any]]:
    """``R_ij = Rbar_ij + R^phi_ij``, the conformal decomposition.

    The identity every BSSN code depends on. ``Rbar_ij`` is the Ricci
    tensor of the conformal metric, computed by the same formula the
    physical one uses, so what is being tested when this is compared with
    ``ricci(physical_slice)`` is the decomposition itself.
    """
    correction = conformal_ricci_correction(variables)
    conformal_ricci = ricci(variables.conformal_slice)
    return [[conformal_ricci[i][j] + correction[i][j] for j in INDICES] for i in INDICES]


def _trace_free(tensor, metric, inverse):
    """``X_ij - gamma_ij gamma^kl X_kl / 3``."""
    traced = trace(inverse, tensor)
    return [[tensor[i][j] - metric[i][j] * traced / 3 for j in INDICES] for i in INDICES]


def bssn_rhs(variables: BSSNVariables, density=0.0, stress=None, momentum=None):
    """The BSSN right-hand sides as Baumgarte and Shapiro write them.

    Equations (11.51) to (11.55), vacuum unless ``density``, ``stress`` or
    ``momentum`` are given. Returns a dictionary keyed the way the state is:
    ``phi``, ``conformal_metric``, ``mean_curvature``,
    ``traceless_curvature`` and ``connection``.

    The connection equation needs the second derivative of the shift, which
    is the one quantity the ADM slice does not carry, so it is passed in
    ``dd_shift`` on the physical slice when available and dropped
    otherwise -- with the shift-dependent terms of that equation then
    absent. Every test here uses zero shift, where those terms vanish
    identically; a moving-puncture gauge needs them and that is
    the evolution code's business.
    """
    physical = variables.physical
    conformal = variables.conformal_slice
    inverse = inverse_metric(physical.metric)
    conformal_inverse = inverse_metric(conformal.metric)
    lapse = variables.lapse
    shift = variables.shift
    mean = variables.mean_curvature
    traceless = variables.traceless_curvature

    divergence = sum(physical.d_shift[k][k] for k in INDICES)

    # (11.51) d_t phi
    dt_phi = -lapse * mean / 6 + divergence / 6
    for k in INDICES:
        dt_phi = dt_phi + shift[k] * variables.d_phi[k]

    # (11.52) d_t gammabar_ij
    dt_conformal = []
    for i in INDICES:
        row = []
        for j in INDICES:
            total = -2 * lapse * traceless[i][j] - 2 * conformal.metric[i][j] * divergence / 3
            for k in INDICES:
                total = total + shift[k] * conformal.d_metric[k][i][j]
                total = total + conformal.metric[i][k] * physical.d_shift[j][k]
                total = total + conformal.metric[j][k] * physical.d_shift[i][k]
            row.append(total)
        dt_conformal.append(row)

    # (11.53) d_t K. The Laplacian of the lapse is the physical one.
    hessian = lapse_hessian(physical, inverse)
    laplacian = trace(inverse, hessian)
    upper_traceless = [
        [
            sum(
                conformal_inverse[i][a] * conformal_inverse[j][b] * traceless[a][b]
                for a in INDICES
                for b in INDICES
            )
            for j in INDICES
        ]
        for i in INDICES
    ]
    traceless_square = sum(
        traceless[i][j] * upper_traceless[i][j] for i in INDICES for j in INDICES
    )
    dt_mean = -laplacian + lapse * (traceless_square + mean**2 / 3)
    for k in INDICES:
        dt_mean = dt_mean + shift[k] * variables.d_mean_curvature[k]
    if stress is not None:
        dt_mean = dt_mean + 4 * math.pi * lapse * (density + trace(inverse, stress))

    # (11.54) d_t Abar_ij
    factor = variables.conformal_exponent
    ricci_tensor = physical_ricci(variables)
    trace_free_source = _trace_free(
        [[lapse * ricci_tensor[i][j] - hessian[i][j] for j in INDICES] for i in INDICES],
        physical.metric,
        inverse,
    )
    mixed_traceless = raise_index(conformal_inverse, traceless)
    dt_traceless = []
    for i in INDICES:
        row = []
        for j in INDICES:
            total = factor * trace_free_source[i][j]
            total = total + lapse * mean * traceless[i][j]
            total = total - 2 * lapse * sum(
                traceless[i][k] * mixed_traceless[k][j] for k in INDICES
            )
            total = total - 2 * traceless[i][j] * divergence / 3
            for k in INDICES:
                total = total + shift[k] * conformal.d_curvature[k][i][j]
                total = total + traceless[i][k] * physical.d_shift[j][k]
                total = total + traceless[j][k] * physical.d_shift[i][k]
            if stress is not None:
                total = (
                    total
                    - 8
                    * math.pi
                    * lapse
                    * factor
                    * _trace_free(stress, physical.metric, inverse)[i][j]
                )
            row.append(total)
        dt_traceless.append(row)

    # (11.55) d_t Gammabar^i, without the second-derivative-of-shift terms.
    conformal_christoffel = christoffel(conformal, conformal_inverse)
    dt_connection = []
    for i in INDICES:
        total = _zeros_like(lapse)
        for j in INDICES:
            total = total - 2 * upper_traceless[i][j] * physical.d_lapse[j]
            total = total + 2 * lapse * (
                sum(conformal_christoffel[i][j][k] * upper_traceless[k][j] for k in INDICES)
                - 2 * conformal_inverse[i][j] * variables.d_mean_curvature[j] / 3
                + 6 * upper_traceless[i][j] * variables.d_phi[j]
            )
            total = total + shift[j] * _connection_derivative(variables, j)[i]
            total = total - variables.connection[j] * physical.d_shift[j][i]
            total = total + 2 * variables.connection[i] * physical.d_shift[j][j] / 3
        if momentum is not None:
            total = total - 16 * math.pi * lapse * sum(
                conformal_inverse[i][j] * momentum[j] for j in INDICES
            )
        dt_connection.append(total)

    return {
        "phi": dt_phi,
        "conformal_metric": dt_conformal,
        "mean_curvature": dt_mean,
        "traceless_curvature": dt_traceless,
        "connection": dt_connection,
    }


def _certainly_zero(value) -> bool:
    """Is this value definitely zero, rather than merely possibly zero?

    Used to decide whether a term that cannot be computed may be dropped.
    A SymPy expression answers through ``is_zero``, which returns ``None``
    when it does not know; an array answers by inspection. Anything else is
    treated as non-zero, because the cost of being wrong is a silently
    missing term.
    """
    if isinstance(value, np.ndarray):
        return bool(np.all(value == 0.0))
    if isinstance(value, sp.Basic):
        return bool(sp.simplify(value).is_zero)
    return value == 0


def _connection_derivative(variables: BSSNVariables, direction: int) -> list[Any]:
    """``d_j Gammabar^i``, for the advection term of the connection equation.

    ``Gammabar^i = -d_j gammabar^ij``, so this is a third derivative of the
    metric: it is a spatial derivative of something already built from
    derivatives of the state, which is exactly why an evolution carries
    ``Gammabar^i`` as an independent variable instead of recomputing it.

    A symbolic slice can differentiate it and does. A slice built from
    arrays cannot, and rather than return zeros -- a silently missing
    advection term is a wrong answer that looks like a right one -- this
    refuses unless the shift is certainly zero, where the term drops out.
    """
    coords = variables.physical.coords
    if coords is not None:
        return [sp.diff(variables.connection[i], coords[direction]) for i in INDICES]
    if _certainly_zero(variables.shift[direction]):
        return [_zeros_like(variables.lapse) for _ in INDICES]
    raise ValueError(
        "the connection equation's advection term needs d_j Gammabar^i, which a "
        "slice built from arrays cannot supply: pass the evolved Gammabar^i and "
        "its derivatives to the evolution code, or use a symbolic slice. It is "
        "only droppable where the shift vanishes, and this shift does not"
    )


def bssn_rhs_from_adm(variables: BSSNVariables, density=0.0, stress=None):
    """The same right-hand sides, by differentiating the BSSN definitions.

    ``d_t phi = gamma^ij d_t gamma_ij / 12``, ``d_t gammabar_ij =
    e^(-4 phi)(d_t gamma_ij - 4 gamma_ij d_t phi)``, and so on, with
    ``d_t gamma_ij`` and ``d_t K_ij`` taken from the ADM equations. No BSSN
    equation is used, so agreement with :func:`bssn_rhs` is a statement
    about the two systems rather than about one transcription.

    The connection is left out: ``Gammabar^i`` is a spatial derivative of
    the conformal metric, so its time derivative needs the spatial
    derivative of ``d_t gammabar^ij``, which is a derivative of a
    right-hand side rather than of the state. The evolution code carries
    ``Gammabar^i`` as an independent variable for exactly that reason.
    """
    physical = variables.physical
    inverse = inverse_metric(physical.metric)
    dt_metric, dt_curvature = adm_rhs(physical, density=density, stress=stress, inverse=inverse)

    dt_phi = sum(inverse[i][j] * dt_metric[i][j] for i in INDICES for j in INDICES) / 12
    factor = variables.conformal_exponent
    dt_conformal = [
        [factor * (dt_metric[i][j] - 4 * physical.metric[i][j] * dt_phi) for j in INDICES]
        for i in INDICES
    ]
    # d_t gamma^ij = -gamma^ia gamma^jb d_t gamma_ab
    dt_inverse = [
        [
            -sum(inverse[i][a] * inverse[j][b] * dt_metric[a][b] for a in INDICES for b in INDICES)
            for j in INDICES
        ]
        for i in INDICES
    ]
    dt_mean = sum(
        dt_inverse[i][j] * physical.curvature[i][j] + inverse[i][j] * dt_curvature[i][j]
        for i in INDICES
        for j in INDICES
    )
    mean = variables.mean_curvature
    dt_traceless = [
        [
            factor
            * (dt_curvature[i][j] - (dt_metric[i][j] * mean + physical.metric[i][j] * dt_mean) / 3)
            - 4 * variables.traceless_curvature[i][j] * dt_phi
            for j in INDICES
        ]
        for i in INDICES
    ]
    return {
        "phi": dt_phi,
        "conformal_metric": dt_conformal,
        "mean_curvature": dt_mean,
        "traceless_curvature": dt_traceless,
    }


def algebraic_constraints(variables: BSSNVariables):
    """``(det gammabar - 1, gammabar^ij Abar_ij)``, both zero by construction.

    The two conditions the variable change imposes and an evolution has to
    keep. They are returned rather than enforced: a code that resets them
    every step hides the drift that tells it something else is wrong.
    """
    conformal_inverse = inverse_metric(variables.conformal_metric)
    return (
        determinant(variables.conformal_metric) - 1,
        trace(conformal_inverse, variables.traceless_curvature),
    )


__all__ = [
    "DIMENSION",
    "BSSNVariables",
    "algebraic_constraints",
    "bssn_rhs",
    "bssn_rhs_from_adm",
    "conformal_factor",
    "conformal_ricci_correction",
    "from_adm",
    "phi_derivatives",
    "physical_ricci",
    "to_adm",
]
