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


def conformal_connection_ricci(variables: BSSNVariables, d_connection=None) -> list[list[Any]]:
    """``Rbar_ij`` written with ``Gammabar^i`` as an independent field.

        Rbar_ij = -(1/2) gammabar^lm d_l d_m gammabar_ij
                  + gammabar_k(i d_j) Gammabar^k
                  + Gammabar^k Gammabar_(ij)k
                  + gammabar^lm (2 Gammabar^k_l(i Gammabar_j)km
                                 + Gammabar^k_im Gammabar_klj)

    Baumgarte and Shapiro eq. (11.42), with ``Gammabar_ijk = gammabar_il
    Gammabar^l_jk``. Algebraically this is the same tensor as
    :func:`particlesim.symbolic.threeplusone.ricci` of the conformal slice,
    *provided* ``Gammabar^i = gammabar^jk Gammabar^i_jk``, which is what
    :func:`from_adm` sets it to and what the test suite checks the two
    forms against.

    It is not the same *discretisation*, and that is the whole point. Run
    the plain formula and the second derivatives of ``gammabar_ij``
    reassemble into the ADM Ricci tensor, whose principal part makes the
    evolution system only weakly hyperbolic: on the gauge wave it grows
    like ``exp(c t / h)``, faster the finer the grid, which is what a
    convergence test sees as a run that converges at fourth order until
    the resolution is high enough to reveal it. Carrying ``Gammabar^i`` as
    an evolved variable leaves ``-(1/2) gammabar^lm d_l d_m gammabar_ij``
    as the only second-derivative term -- a flat wave operator on each
    component -- and the system becomes strongly hyperbolic.

    ``d_connection`` is ``d_j Gammabar^i``, indexed ``[j][i]``. A symbolic
    slice can differentiate the evolved field itself and does; a slice
    built from arrays has to be given it.
    """
    conformal = variables.conformal_slice
    conformal_inverse = inverse_metric(conformal.metric)
    connection_symbols = christoffel(conformal, conformal_inverse)
    if d_connection is None:
        d_connection = [_connection_derivative(variables, j) for j in INDICES]

    # Gammabar_ijk = gammabar_il Gammabar^l_jk, all indices down.
    lowered = [
        [
            [
                sum(conformal.metric[i][ell] * connection_symbols[ell][j][k] for ell in INDICES)
                for k in INDICES
            ]
            for j in INDICES
        ]
        for i in INDICES
    ]

    tensor = []
    for i in INDICES:
        row = []
        for j in INDICES:
            total = (
                -sum(
                    conformal_inverse[ell][m] * conformal.dd_metric[ell][m][i][j]
                    for ell in INDICES
                    for m in INDICES
                )
                / 2
            )
            for k in INDICES:
                total = (
                    total
                    + (
                        conformal.metric[k][i] * d_connection[j][k]
                        + conformal.metric[k][j] * d_connection[i][k]
                    )
                    / 2
                )
                total = total + variables.connection[k] * (lowered[i][j][k] + lowered[j][i][k]) / 2
            for ell in INDICES:
                for m in INDICES:
                    weight = conformal_inverse[ell][m]
                    if isinstance(weight, sp.Basic) and weight.is_zero:
                        continue
                    for k in INDICES:
                        total = total + weight * (
                            connection_symbols[k][ell][i] * lowered[j][k][m]
                            + connection_symbols[k][ell][j] * lowered[i][k][m]
                            + connection_symbols[k][i][m] * lowered[k][ell][j]
                        )
            row.append(total)
        tensor.append(row)
    return tensor


def physical_ricci(variables: BSSNVariables, d_connection=None, form: str = "metric"):
    """``R_ij = Rbar_ij + R^phi_ij``, the conformal decomposition.

    The identity every BSSN code depends on. With ``form="metric"``,
    ``Rbar_ij`` is the Ricci tensor of the conformal metric computed by the
    same formula the physical one uses, so what is being tested when this
    is compared with ``ricci(physical_slice)`` is the decomposition itself.

    With ``form="connection"`` it is :func:`conformal_connection_ricci`
    instead -- the same tensor, written so that the evolved
    ``Gammabar^i`` carries the mixed second derivatives. That is the form
    an evolution has to use; see that function for why.
    """
    if form == "metric":
        conformal_ricci = ricci(variables.conformal_slice)
    elif form == "connection":
        conformal_ricci = conformal_connection_ricci(variables, d_connection)
    else:
        raise ValueError(f"unknown Ricci form {form!r}: use 'metric' or 'connection'")
    correction = conformal_ricci_correction(variables)
    return [[conformal_ricci[i][j] + correction[i][j] for j in INDICES] for i in INDICES]


def _trace_free(tensor, metric, inverse):
    """``X_ij - gamma_ij gamma^kl X_kl / 3``."""
    traced = trace(inverse, tensor)
    return [[tensor[i][j] - metric[i][j] * traced / 3 for j in INDICES] for i in INDICES]


def bssn_rhs(
    variables: BSSNVariables,
    density=0.0,
    stress=None,
    momentum=None,
    d_connection=None,
    dd_shift=None,
    ricci_form: str = "metric",
    ricci_tensor=None,
):
    """The BSSN right-hand sides as Baumgarte and Shapiro write them.

    Equations (11.51) to (11.55), vacuum unless ``density``, ``stress`` or
    ``momentum`` are given. Returns a dictionary keyed the way the state is:
    ``phi``, ``conformal_metric``, ``mean_curvature``,
    ``traceless_curvature`` and ``connection``.

    Two terms of the connection equation need quantities an ADM slice does
    not carry: ``d_j Gammabar^i`` and the second derivative of the shift.
    Pass them as ``d_connection`` (indexed ``[j][i]``) and ``dd_shift``
    (indexed ``[k][j][i]`` for ``d_k d_j beta^i``) and the equation is
    complete. Leave them out and a symbolic slice differentiates its own
    shift and connection; a slice built from arrays cannot, and is refused
    unless the shift vanishes, because a silently missing term is a wrong
    answer that looks like a right one.

    ``ricci_form`` selects how ``Rbar_ij`` is written; an evolution wants
    ``"connection"``. See :func:`conformal_connection_ricci`.

    ``ricci_tensor`` lets a caller that has already built ``R_ij`` hand it
    in rather than have it built again. CCZ4 needs the same tensor for the
    Hamiltonian constraint, and two structurally identical trees cost twice
    as much to build even though elimination charges for them once.
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
    if ricci_tensor is None:
        ricci_tensor = physical_ricci(variables, d_connection, form=ricci_form)
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

    # (11.55) d_t Gammabar^i.
    conformal_christoffel = christoffel(conformal, conformal_inverse)
    shift_hessian = _shift_hessian(variables) if dd_shift is None else dd_shift
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
            advection = (
                d_connection[j]
                if d_connection is not None
                else _connection_derivative(variables, j)
            )
            total = total + shift[j] * advection[i]
            total = total - variables.connection[j] * physical.d_shift[j][i]
            total = total + 2 * variables.connection[i] * physical.d_shift[j][j] / 3
            for k in INDICES:
                total = total + conformal_inverse[k][i] * shift_hessian[k][j][j] / 3
                total = total + conformal_inverse[k][j] * shift_hessian[k][j][i]
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


def _shift_hessian(variables: BSSNVariables) -> list[list[list[Any]]]:
    """``d_k d_j beta^i``, indexed ``[k][j][i]``.

    The connection equation needs the shift's second derivative, and a
    :class:`~particlesim.symbolic.threeplusone.Slice` carries the shift
    differenced only once. The same split as
    :func:`_connection_derivative`: a symbolic slice can differentiate it
    again and does, a slice built from arrays has to be handed it, and
    where the shift vanishes the terms drop out so zeros are correct.
    Anything else is refused, because the terms it would silently omit --
    ``(1/3) gammabar^ki d_k d_j beta^j + gammabar^kj d_k d_j beta^i`` --
    are a wrong answer that looks like a right one.
    """
    coords = variables.physical.coords
    shift = variables.shift
    if coords is not None:
        return [
            [[sp.diff(shift[i], coords[k], coords[j]) for i in INDICES] for j in INDICES]
            for k in INDICES
        ]
    if all(_certainly_zero(value) for value in shift):
        zero = _zeros_like(variables.lapse)
        return [[[zero for _ in INDICES] for _ in INDICES] for _ in INDICES]
    raise ValueError(
        "the connection equation's second-derivative-of-shift terms need "
        "dd_shift, which a slice built from arrays cannot supply, and the "
        "shift does not vanish: pass it, or the equation is missing "
        "(1/3) gammabar^ki d_k d_j beta^j + gammabar^kj d_k d_j beta^i"
    )


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


#: The evolved BSSN variables, in the order a state vector carries them.
#:
#: Twenty-four fields: the conformal factor, the six components of the
#: conformal metric, the mean curvature, the six of its trace-free part,
#: three connection functions, and the seven gauge variables. The names are
#: the ones the emitted kernel reads, so they are short and they are fixed.
STATE_NAMES = (
    "phi",
    "gt00",
    "gt01",
    "gt02",
    "gt11",
    "gt12",
    "gt22",
    "trK",
    "At00",
    "At01",
    "At02",
    "At11",
    "At12",
    "At22",
    "Gt0",
    "Gt1",
    "Gt2",
    "alpha",
    "beta0",
    "beta1",
    "beta2",
    "B0",
    "B1",
    "B2",
)

#: Which derivatives of which state variables the equations need.
#:
#: ``1`` means first derivatives along each axis, ``2`` means first and
#: second. The second derivatives are exactly where they have to be: the
#: conformal metric and the conformal factor because the Ricci tensor needs
#: them, the lapse because of ``D_i D_j alpha``, and the shift because of
#: the two second-derivative terms in the connection equation. Nothing
#: else, and a kernel therefore differences nothing else.
DERIVATIVE_ORDERS = {
    "phi": 2,
    "gt00": 2,
    "gt01": 2,
    "gt02": 2,
    "gt11": 2,
    "gt12": 2,
    "gt22": 2,
    "trK": 1,
    "At00": 1,
    "At01": 1,
    "At02": 1,
    "At11": 1,
    "At12": 1,
    "At22": 1,
    "Gt0": 1,
    "Gt1": 1,
    "Gt2": 1,
    "alpha": 2,
    "beta0": 2,
    "beta1": 2,
    "beta2": 2,
}


def _index_pair(i: int, j: int) -> str:
    low, high = sorted((i, j))
    return f"{low}{high}"


def abstract_state():
    """Symbols for the evolved BSSN state and the derivatives it needs.

    Returns ``(state, derivatives, registry)``: the state symbols keyed by
    the names in :data:`STATE_NAMES`, the derivative symbols keyed by their
    own names, and the registry
    :func:`particlesim.symbolic.codegen.emit` needs, mapping each
    derivative symbol to the field and axes it differentiates.

    The naming matches :func:`particlesim.symbolic.threeplusone.abstract_slice`
    -- ``d_gt01_2``, ``dd_alpha_01`` -- because the two go through the same
    emitter and a reader of the generated source should not have to hold
    two conventions at once.
    """
    state = {name: sp.Symbol(name, real=True) for name in STATE_NAMES}
    derivatives: dict[str, Any] = {}
    registry: dict[Any, tuple[str, tuple[int, ...]]] = {}

    for name, order in DERIVATIVE_ORDERS.items():
        for axis in INDICES:
            symbol = sp.Symbol(f"d_{name}_{axis}", real=True)
            derivatives[symbol.name] = symbol
            registry[symbol] = (name, (axis,))
        if order >= 2:
            for axis in INDICES:
                for other in range(axis, DIMENSION):
                    symbol = sp.Symbol(f"dd_{name}_{axis}{other}", real=True)
                    derivatives[symbol.name] = symbol
                    registry[symbol] = (name, (axis, other))
    return state, derivatives, registry


def _lookup(container, name):
    value = container.get(name)
    if value is None:
        raise KeyError(f"missing {name!r}: the BSSN state needs every field and derivative")
    return value


def from_state(state, derivatives) -> BSSNVariables:
    """Build :class:`BSSNVariables` from the evolved state, not from ADM data.

    :func:`from_adm` goes the other way and is what a variable change is
    checked with; this is what an evolution needs, and the difference
    matters. ``from_adm`` *derives* the conformal metric as
    ``(det gamma)^(-1/3) gamma_ij``, which is only the evolved
    ``gammabar_ij`` while its determinant is exactly one. In a run it drifts,
    and the equations are the ones for the evolved variables, so the evolved
    ``gammabar_ij``, ``phi``, ``K``, ``Abar_ij`` and ``Gammabar^i`` are used
    as they stand.

    The physical slice is reconstructed from them algebraically --
    ``gamma_ij = e^(4 phi) gammabar_ij``, ``K_ij = e^(4 phi) Abar_ij +
    gamma_ij K/3``, and the same product rule for the derivatives -- so one
    pass of finite differences over the state supplies everything, and the
    physical metric is never differenced.
    """
    factor = (
        sp.exp(4 * _lookup(state, "phi"))
        if isinstance(_lookup(state, "phi"), sp.Basic)
        else np.exp(4 * _lookup(state, "phi"))
    )
    inverse_factor = 1 / factor

    def first(name, axis):
        return _lookup(derivatives, f"d_{name}_{axis}")

    def second(name, axis, other):
        return _lookup(derivatives, f"dd_{name}_{_index_pair(axis, other)}")

    lapse = _lookup(state, "alpha")
    shift = [_lookup(state, f"beta{i}") for i in INDICES]
    d_lapse = [first("alpha", k) for k in INDICES]
    dd_lapse = [[second("alpha", k, m) for m in INDICES] for k in INDICES]
    d_shift = [[first(f"beta{i}", k) for i in INDICES] for k in INDICES]
    dd_shift = [[[second(f"beta{i}", k, m) for i in INDICES] for m in INDICES] for k in INDICES]

    d_phi = [first("phi", k) for k in INDICES]
    dd_phi = [[second("phi", k, m) for m in INDICES] for k in INDICES]

    def tensor(prefix):
        return [[_lookup(state, f"{prefix}{_index_pair(i, j)}") for j in INDICES] for i in INDICES]

    def d_tensor(prefix):
        return [
            [[first(f"{prefix}{_index_pair(i, j)}", k) for j in INDICES] for i in INDICES]
            for k in INDICES
        ]

    def dd_tensor(prefix):
        return [
            [
                [[second(f"{prefix}{_index_pair(i, j)}", k, m) for j in INDICES] for i in INDICES]
                for m in INDICES
            ]
            for k in INDICES
        ]

    conformal_metric = tensor("gt")
    d_conformal_metric = d_tensor("gt")
    dd_conformal_metric = dd_tensor("gt")
    traceless = tensor("At")
    d_traceless = d_tensor("At")
    mean = _lookup(state, "trK")
    d_mean = [first("trK", k) for k in INDICES]
    connection = [_lookup(state, f"Gt{i}") for i in INDICES]
    d_connection = [[first(f"Gt{i}", k) for i in INDICES] for k in INDICES]

    metric = [[factor * conformal_metric[i][j] for j in INDICES] for i in INDICES]
    d_metric = [
        [
            [
                factor * (d_conformal_metric[k][i][j] + 4 * d_phi[k] * conformal_metric[i][j])
                for j in INDICES
            ]
            for i in INDICES
        ]
        for k in INDICES
    ]
    dd_metric = [
        [
            [
                [
                    factor
                    * (
                        dd_conformal_metric[k][m][i][j]
                        + 4 * d_phi[m] * d_conformal_metric[k][i][j]
                        + 4 * d_phi[k] * d_conformal_metric[m][i][j]
                        + 4 * dd_phi[k][m] * conformal_metric[i][j]
                        + 16 * d_phi[k] * d_phi[m] * conformal_metric[i][j]
                    )
                    for j in INDICES
                ]
                for i in INDICES
            ]
            for m in INDICES
        ]
        for k in INDICES
    ]
    curvature = [
        [factor * traceless[i][j] + metric[i][j] * mean / 3 for j in INDICES] for i in INDICES
    ]
    d_curvature = [
        [
            [
                factor * (d_traceless[k][i][j] + 4 * d_phi[k] * traceless[i][j])
                + (d_metric[k][i][j] * mean + metric[i][j] * d_mean[k]) / 3
                for j in INDICES
            ]
            for i in INDICES
        ]
        for k in INDICES
    ]

    physical = Slice(
        lapse=lapse,
        shift=shift,
        metric=metric,
        curvature=curvature,
        d_lapse=d_lapse,
        dd_lapse=dd_lapse,
        d_shift=d_shift,
        d_metric=d_metric,
        dd_metric=dd_metric,
        d_curvature=d_curvature,
    )
    conformal = Slice(
        lapse=lapse,
        shift=shift,
        metric=conformal_metric,
        curvature=traceless,
        d_lapse=d_lapse,
        dd_lapse=dd_lapse,
        d_shift=d_shift,
        d_metric=d_conformal_metric,
        dd_metric=dd_conformal_metric,
        d_curvature=d_traceless,
    )
    variables = BSSNVariables(
        lapse=lapse,
        shift=shift,
        conformal_exponent=inverse_factor,
        conformal_metric=conformal_metric,
        mean_curvature=mean,
        traceless_curvature=traceless,
        connection=connection,
        d_phi=d_phi,
        dd_phi=dd_phi,
        d_mean_curvature=d_mean,
        conformal_slice=conformal,
        physical=physical,
    )
    return variables, d_connection, dd_shift


#: Slicing conditions, as ``d_t alpha`` without the advection term.
#:
#: ``harmonic`` is ``-alpha^2 K``, which the gauge wave satisfies exactly
#: and which is therefore the condition that test has to be run in.
#: ``one_plus_log`` is ``-2 alpha K``, the moving-puncture standard: it is
#: what keeps a puncture's lapse from collapsing to zero on the grid.
SLICINGS = {
    "harmonic": lambda lapse, mean: -(lapse**2) * mean,
    "one_plus_log": lambda lapse, mean: -2 * lapse * mean,
    "frozen": lambda lapse, mean: 0 * lapse,
}


def gauge_rhs(
    variables: BSSNVariables,
    connection_rhs,
    state,
    slicing: str = "one_plus_log",
    shift_condition: str = "gamma_driver",
    damping: float = 2.0,
    advect: bool | str = True,
):
    """``(d_t alpha, d_t beta^i, d_t B^i)`` for the gauge.

    The Gamma-driver shift is the standard moving-puncture pair,

        d_t beta^i = (3/4) B^i,     d_t B^i = d_t Gammabar^i - eta B^i

    with ``eta`` the damping. It is driven by the *right-hand side* of the
    connection equation rather than by a fresh derivative, which is why
    that has to be passed in: recomputing it would be the same expression
    twice and a place for the two to drift apart.

    ``advect`` adds the ``beta^j d_j`` terms. They belong in a run with a
    moving shift and vanish identically when the shift does, so the gauge
    wave is unaffected either way. ``"lapse"`` advects the lapse alone:
    ``d_t alpha = beta^j d_j alpha - 2 alpha K`` with the shift and its
    driver unadvected, the combination of Campanelli et al. (2006). The
    lapse needs its advection term for 1+log to have a stationary trumpet
    at all -- without it ``d_t alpha = -2 alpha K`` keeps collapsing
    wherever ``K`` is not zero.
    """
    if advect not in (True, False, "lapse"):
        raise ValueError(f"advect must be True, False or 'lapse', not {advect!r}")
    advect_lapse = advect is True or advect == "lapse"
    advect_shift = advect is True
    if slicing not in SLICINGS:
        raise ValueError(f"unknown slicing {slicing!r}; known: {sorted(SLICINGS)}")
    if shift_condition not in ("gamma_driver", "frozen"):
        raise ValueError(f"unknown shift condition {shift_condition!r}")

    lapse = variables.lapse
    shift = variables.shift
    physical = variables.physical
    dt_lapse = SLICINGS[slicing](lapse, variables.mean_curvature)
    if advect_lapse:
        for k in INDICES:
            dt_lapse = dt_lapse + shift[k] * physical.d_lapse[k]

    if shift_condition == "frozen":
        zero = 0 * lapse
        return dt_lapse, [zero for _ in INDICES], [zero for _ in INDICES]

    driver = [_lookup(state, f"B{i}") for i in INDICES]
    dt_shift = []
    dt_driver = []
    for i in INDICES:
        velocity = 3 * driver[i] / 4
        rate = connection_rhs[i] - damping * driver[i]
        if advect_shift:
            for k in INDICES:
                velocity = velocity + shift[k] * physical.d_shift[k][i]
        dt_shift.append(velocity)
        dt_driver.append(rate)
    return dt_lapse, dt_shift, dt_driver


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
    "DERIVATIVE_ORDERS",
    "DIMENSION",
    "SLICINGS",
    "STATE_NAMES",
    "BSSNVariables",
    "abstract_state",
    "algebraic_constraints",
    "bssn_rhs",
    "bssn_rhs_from_adm",
    "conformal_connection_ricci",
    "conformal_factor",
    "conformal_ricci_correction",
    "from_adm",
    "from_state",
    "gauge_rhs",
    "phi_derivatives",
    "physical_ricci",
    "to_adm",
]
