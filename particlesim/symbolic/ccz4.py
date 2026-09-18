"""CCZ4: the Z4 system in the conformal variables, as BSSN plus its Z terms.

BSSN keeps the constraints as diagnostics: nothing in its right-hand sides
pushes a violation back towards zero, so a violation seeded by truncation
error is free to grow on whatever timescale the equations allow. Z4 changes
the system instead of the diagnostics. It replaces Einstein's equations with

    R_ab + D_a Z_b + D_b Z_a = 8 pi (T_ab - g_ab T / 2)

for a four-vector ``Z_a`` that vanishes exactly when the constraints hold,
evolves ``Z_a`` alongside the geometry, and adds damping terms that make
``Z_a = 0`` an attractor. The constraint surface stops being a place the
solution happens to start and becomes a place it is pulled back to.

**Written as BSSN plus a difference, not from scratch.** Every term here
that does not involve ``Theta`` or ``Z_i`` is taken from
:func:`particlesim.symbolic.bssn.bssn_rhs`, which the test suite has already
checked against exact solutions two independent ways. What this module adds
is the difference, and the difference is small enough to read:

* ``d_t K`` gets back the ``alpha H`` that the textbook BSSN equation
  removes by substituting the Hamiltonian constraint, because Z4 is the
  statement that the constraint is *not* assumed; then the ``Z`` terms;
* ``d_t Abar_ij`` gets the trace-free part of ``2 alpha D_(i Z_j)`` and the
  ``-2 alpha Theta Abar_ij`` that turns ``K`` into ``K - 2 Theta``;
* ``d_t Gammahat^i`` gets the gradient of ``Theta`` and a damping term;
* ``Theta`` is new, and its equation is the Hamiltonian constraint.

That last point is the one worth keeping in view. With ``Theta = 0`` and
``Z_i = 0``,

    d_t Theta = (alpha / 2) H

exactly. ``Theta`` is not an auxiliary variable that happens to be useful;
it is the Hamiltonian constraint promoted to something the system evolves,
which is why damping it damps the constraint.

**``Z_i`` is recovered, not evolved.** The evolved connection variable is

    Gammahat^i = Gammabar^i + 2 gammabar^ij Z_j

so ``Z_i`` is the gap between the evolved connection and the one the
conformal metric defines. Writing that gap as

    2 gammabar^ij Z_j = Gammahat^i + d_j gammabar^ij

rather than as ``Gammahat^i - Gammabar^i`` is what keeps it affordable:
``Gammabar^i`` is already a derivative of the metric, so differentiating it
would want a *third* derivative, while ``d_k d_j gammabar^ij`` is algebraic
in the metric and its first two derivatives through
:func:`particlesim.symbolic.threeplusone.inverse_metric_second_derivative`.
The grid path differences nothing it was not already differencing.

**What is not here.** The ``kappa_3`` parameter of the fully covariant
CCZ4, which multiplies the terms that distinguish it from the
non-covariant variant, is fixed at one. And the conformal factor is
``phi``, not ``chi`` or ``W``; codes that evolve those write several of
these coefficients differently for that reason alone, which is worth
knowing before comparing term by term with a paper.
"""

from __future__ import annotations

import math
from typing import Any

import sympy as sp

from particlesim.symbolic import bssn
from particlesim.symbolic.bssn import BSSNVariables, _connection_derivative, _trace_free
from particlesim.symbolic.threeplusone import (
    DIMENSION,
    INDICES,
    christoffel,
    hamiltonian_constraint,
    inverse_metric,
    inverse_metric_second_derivative,
    trace,
)

#: Constraint-damping strength. Zero recovers an undamped Z4, which still
#: propagates the constraints but does not pull them back.
#:
#: The useful range is set by the grid: the damping is a rate, and a rate
#: faster than the time step resolves turns the term into a stiff source.
#: ``0.1`` in units where the mass is one is the value the moving-puncture
#: literature settled on.
DAMPING = 0.1

#: ``kappa_2``, which mixes the damping of ``Theta`` into ``d_t K``. Zero is
#: the standard choice and what the damping analysis assumes.
DAMPING_MIX = 0.0

#: The CCZ4 state: the BSSN variables, plus ``Theta``.
STATE_NAMES = (*bssn.STATE_NAMES, "Theta")

#: How many derivatives of each field the equations need.
DERIVATIVE_ORDERS = {**bssn.DERIVATIVE_ORDERS, "Theta": 1}


def abstract_state():
    """Symbols for the evolved CCZ4 state and the derivatives it needs.

    The BSSN state with ``Theta`` on the end, so that a reader comparing the
    two generated kernels sees the same names in the same order and one
    extra equation rather than a renumbering.
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


def z_vector(variables: BSSNVariables, d_connection):
    """``(gammabar^ij Z_j, d_k (gammabar^ij Z_j))`` from the evolved connection.

    ``Gammahat^i = Gammabar^i + 2 gammabar^ij Z_j`` and
    ``Gammabar^i = -d_j gammabar^ij``, so the conformal raise of ``Z`` is

        gammabar^ij Z_j = (Gammahat^i + d_j gammabar^ij) / 2

    and its derivative needs only ``d_k d_j gammabar^ij``, which is
    algebraic in the metric and its first two derivatives. Returned raised
    with ``gammabar`` because that is the form both the connection equation
    and the covariant derivative want; the physical ``Z^i`` is this times
    ``e^(-4 phi)``, and the covector ``Z_i`` is ``gammabar_ij`` times it.

    ``d_connection`` is ``d_k Gammahat^i``, indexed ``[k][i]``.
    """
    conformal = variables.conformal_slice
    conformal_inverse = inverse_metric(conformal.metric)
    d_inverse = inverse_metric_second_derivative(
        conformal_inverse, conformal.d_metric, conformal.dd_metric
    )
    first_inverse = [
        [
            [
                -sum(
                    conformal_inverse[i][a] * conformal_inverse[j][b] * conformal.d_metric[k][a][b]
                    for a in INDICES
                    for b in INDICES
                )
                for j in INDICES
            ]
            for i in INDICES
        ]
        for k in INDICES
    ]

    upper = [
        (variables.connection[i] + sum(first_inverse[j][i][j] for j in INDICES)) / 2
        for i in INDICES
    ]
    d_upper = [
        [(d_connection[k][i] + sum(d_inverse[k][j][i][j] for j in INDICES)) / 2 for i in INDICES]
        for k in INDICES
    ]
    return upper, d_upper


def z_derivatives(variables: BSSNVariables, upper, d_upper):
    """``(D_i Z^i, D_(i Z_j))`` with the *physical* covariant derivative.

    ``Z_i`` is a covector on the physical slice, so its covariant derivative
    carries the physical Christoffel symbols and its trace is taken with the
    physical inverse metric -- not the conformal ones. Mixing the two is the
    easiest mistake here to make and the hardest to see, because the two
    differ by powers of ``e^(4 phi)`` that are one on flat data.

    Returns the divergence and the *symmetrised* gradient, which is the only
    part of ``D_i Z_j`` the equations use.
    """
    physical = variables.physical
    inverse = inverse_metric(physical.metric)
    connection = christoffel(physical, inverse)
    conformal = variables.conformal_slice

    # Z_i = gammabar_ij (gammabar^jk Z_k), and the product rule for its
    # derivative. The conformal metric lowers it because ``upper`` is its
    # conformal raise; the result is the physical covector either way.
    lower = [sum(conformal.metric[i][j] * upper[j] for j in INDICES) for i in INDICES]
    d_lower = [
        [
            sum(
                conformal.d_metric[k][i][j] * upper[j] + conformal.metric[i][j] * d_upper[k][j]
                for j in INDICES
            )
            for i in INDICES
        ]
        for k in INDICES
    ]

    gradient = [
        [d_lower[i][j] - sum(connection[m][i][j] * lower[m] for m in INDICES) for j in INDICES]
        for i in INDICES
    ]
    symmetric = [[(gradient[i][j] + gradient[j][i]) / 2 for j in INDICES] for i in INDICES]
    divergence = sum(inverse[i][j] * gradient[i][j] for i in INDICES for j in INDICES)
    return divergence, symmetric


def ricci_scalar(variables: BSSNVariables, d_connection=None):
    """``R``, from the same conformal decomposition ``d_t Abar_ij`` uses.

    The point is *which* expression tree, not the value. Tracing back to
    ``physical_ricci`` in the connection form means the scalar shares every
    subexpression with the tensor the traceless equation already needs, and
    common-subexpression elimination charges for it once.
    """
    inverse = inverse_metric(variables.physical.metric)
    tensor = bssn.physical_ricci(variables, d_connection, form="connection")
    return trace(inverse, tensor)


def _hamiltonian(variables: BSSNVariables, scalar, density=0.0):
    """``H = R + 2 K^2 / 3 - Abar_ij Abar^ij - 16 pi rho`` from a given ``R``."""
    conformal_inverse = inverse_metric(variables.conformal_slice.metric)
    traceless = variables.traceless_curvature
    upper = [
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
    square = sum(traceless[i][j] * upper[i][j] for i in INDICES for j in INDICES)
    return scalar + 2 * variables.mean_curvature**2 / 3 - square - 16 * math.pi * density


def ccz4_rhs(
    variables: BSSNVariables,
    theta,
    d_theta,
    density=0.0,
    stress=None,
    momentum=None,
    d_connection=None,
    dd_shift=None,
    damping: float = DAMPING,
    damping_mix: float = DAMPING_MIX,
):
    """The CCZ4 right-hand sides, keyed the way the state is.

    Returns the five BSSN keys -- ``phi``, ``conformal_metric``,
    ``mean_curvature``, ``traceless_curvature``, ``connection`` -- plus
    ``theta``. The connection key is ``d_t Gammahat^i``, not
    ``d_t Gammabar^i``: in CCZ4 the evolved connection variable carries
    ``Z_i``, and every place the BSSN equations use the evolved connection
    (the conformal Ricci tensor above all) uses ``Gammahat^i`` here.

    ``theta`` and ``d_theta`` are the field and its gradient. ``damping`` is
    ``kappa_1`` and ``damping_mix`` is ``kappa_2``; ``kappa_3`` is one.
    """
    geometry = bssn.bssn_rhs(
        variables,
        density=density,
        stress=stress,
        momentum=momentum,
        d_connection=d_connection,
        dd_shift=dd_shift,
        ricci_form="connection",
    )

    physical = variables.physical
    conformal = variables.conformal_slice
    inverse = inverse_metric(physical.metric)
    conformal_inverse = inverse_metric(conformal.metric)
    lapse = variables.lapse
    shift = variables.shift
    mean = variables.mean_curvature
    traceless = variables.traceless_curvature
    factor = variables.conformal_exponent

    if d_connection is None:
        d_connection = [_connection_derivative(variables, k) for k in INDICES]
    upper_z, d_upper_z = z_vector(variables, d_connection)
    divergence, symmetric = z_derivatives(variables, upper_z, d_upper_z)

    # H = R + K^2 - K_ij K^ij, the quantity the textbook BSSN d_t K removes
    # and Z4 puts back.
    #
    # Spelled out in the conformal variables rather than called from
    # ``hamiltonian_constraint``, which would build the Ricci tensor by the
    # *other* route -- straight from the physical metric instead of through
    # the conformal decomposition -- and leave the derivation carrying two
    # structurally different trees for the same tensor. That cost 4 598 655
    # raw operations and ten minutes of elimination, measured, before this
    # was changed. :func:`constraint_identity` is the test that the two
    # forms agree.
    constraint = _hamiltonian(variables, ricci_scalar(variables, d_connection), density)

    # (1) d_t K. The BSSN form plus alpha H is the ADM form, which is what
    # Z4 starts from because it does not assume the constraint.
    dt_mean = geometry["mean_curvature"] + lapse * constraint
    dt_mean = dt_mean + lapse * (2 * divergence - 2 * theta * mean)
    dt_mean = dt_mean - 3 * lapse * damping * (1 + damping_mix) * theta

    # (2) d_t Abar_ij. The Ricci tensor gains 2 D_(i Z_j), trace-free and
    # conformally weighted like the rest of that bracket; and ``K`` in the
    # quadratic term becomes ``K - 2 Theta``.
    extra = _trace_free(
        [[2 * lapse * symmetric[i][j] for j in INDICES] for i in INDICES],
        physical.metric,
        inverse,
    )
    dt_traceless = [
        [
            geometry["traceless_curvature"][i][j]
            + factor * extra[i][j]
            - 2 * lapse * theta * traceless[i][j]
            for j in INDICES
        ]
        for i in INDICES
    ]

    # (3) d_t Gammahat^i. The BSSN connection equation, plus the gradient of
    # Theta, the term that couples Z to the mean curvature, and damping.
    dt_connection = []
    for i in INDICES:
        total = geometry["connection"][i]
        for k in INDICES:
            gradient = lapse * d_theta[k] - theta * physical.d_lapse[k]
            total = total + 2 * conformal_inverse[k][i] * gradient
        total = total - (4 * lapse * mean / 3) * upper_z[i]
        total = total - 2 * lapse * damping * upper_z[i]
        dt_connection.append(total)

    # (4) d_t Theta: the Hamiltonian constraint, evolved.
    dt_theta = lapse * (constraint + 2 * divergence - 2 * theta * mean) / 2
    dt_theta = dt_theta - factor * sum(upper_z[i] * physical.d_lapse[i] for i in INDICES)
    dt_theta = dt_theta - lapse * damping * (2 + damping_mix) * theta
    for k in INDICES:
        dt_theta = dt_theta + shift[k] * d_theta[k]

    return {
        "phi": geometry["phi"],
        "conformal_metric": geometry["conformal_metric"],
        "mean_curvature": dt_mean,
        "traceless_curvature": dt_traceless,
        "connection": dt_connection,
        "theta": dt_theta,
    }


def constraint_identity(variables: BSSNVariables, density=0.0):
    """``(H, R + 2 K^2 / 3 - Abar_ij Abar^ij - 16 pi rho)``, which are equal.

    The rewrite :func:`ccz4_rhs` leans on. ``d_t Theta`` is written with the
    Hamiltonian constraint in it because that is the fact worth seeing, but
    the form CCZ4 is usually published in has the right-hand side spelled
    out in the conformal variables. They are the same quantity, and this
    returns both so a test can say so rather than a reader having to trust
    it.
    """
    physical = variables.physical
    inverse = inverse_metric(physical.metric)
    scalar = trace(inverse, bssn.physical_ricci(variables))
    spelled = _hamiltonian(variables, scalar, density)
    return hamiltonian_constraint(physical, density=density, inverse=inverse), spelled


__all__ = [
    "DAMPING",
    "DAMPING_MIX",
    "DERIVATIVE_ORDERS",
    "STATE_NAMES",
    "abstract_state",
    "ccz4_rhs",
    "constraint_identity",
    "z_derivatives",
    "z_vector",
]
