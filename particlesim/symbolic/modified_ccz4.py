"""Modified CCZ4: auxiliary-metric gauge, a scalar, and matter that enters linearly (issue #52).

Einstein-scalar-Gauss-Bonnet is not strongly hyperbolic in the puncture
gauge, even at weak coupling, because the gauge, the constraint violations
and the gravitational waves of General Relativity all travel on the same
light cone there, and a small higher-derivative term can split a degenerate
characteristic into a growing pair. Kovacs and Reall (2020) cured it in
harmonic gauge by making the three speeds different. Aresté Saló, Clough
and Figueras (2022, 2023) carried that to CCZ4 and the puncture gauge. Two
auxiliary inverse metrics,

    gtilde^ab = g^ab - a(x) n^a n^b,     ghat^ab = g^ab - b(x) n^a n^b,

replace ``g^ab`` in the gauge condition and in the constraint-addition term
of the Z4 equations respectively,

    G^ab + 2 Phat_c^(d ab) nabla_d Z^c - kappa_1 [2 n^(a Z^b) + kappa_2 n.Z g^ab] = 8 pi T^ab,
    Phat_c^(d ab) = delta_c^(a ghat^b)d - (1/2) delta_c^d ghat^ab.

At ``a = b = 0`` this is CCZ4. The Bianchi identity gives
``ghat^cd nabla_c nabla_d Z^a = 0`` to principal order, so a constraint
violation travels at ``1 / sqrt(1 + b)``; the gauge travels at
``1 / sqrt(1 + a)``; gravitational waves stay on the light cone.

**The 3+1 form is derived here, not transcribed.** The two papers could not
be fetched while this was written (the proxy refuses arXiv), so nothing
below is copied from them; it is the decomposition of the covariant equation
above, and the tests check it against that equation rather than against a
transcription. With ``beta_b = b / (1 + b)``, the difference from CCZ4 is

    d_t Theta     += -beta_b alpha (H/2 - kappa_1 (2 + kappa_2) Theta)
    d_t K         += (3/2) x the same
    d_t Gammahat^i += -2 beta_b alpha gammabar^ij (M_j + d_j Theta - K_jk Z^k - kappa_1 Z_j)

and nothing else: ``d_t Abar_ij`` and the metric are unchanged. ``H`` and
``M_i`` are the Hamiltonian and momentum constraints *with* their matter
terms. Three steps give it. The ``nn`` projection of the modified
constraint term is exactly ``(1 + b)`` times CCZ4's, so solving for
``d_t Theta`` divides only ``H`` and the damping by ``1 + b``; the ``ni``
projection multiplies only the time derivative of ``Z_i`` by ``1 + b``; and
the ``ij`` projection is unchanged, but trace reversal moves ``b`` times the
``nn`` part into ``d_t K_ij`` as a pure trace. Linearised on flat space the
three terms give ``(1 + b) d_t^2 = laplacian`` for ``Theta`` and for every
``Z_i``: the coefficient of the ``K`` term is the one that cancels ``d_i H``
in the ``Z_i`` equation, so it is not a free choice.

**The gauge.** The time component of the modified harmonic condition
``gtilde^bc Gamma^a_bc = 0`` is ``d_0 alpha = -alpha^2 K / (1 + a)``; its
space part is ``d_0 beta^i = alpha^2 Gamma^i / (1 + a) - alpha D^i alpha``.
:func:`modified_gauge` makes the same change to whichever slicing and
Gamma-driver is in use: the slicing rate is divided by ``1 + a``, and the
driver becomes ``(3/4) B^i / (1 + a) - (a / (1 + a)) alpha D^i alpha``.
That is the form recalled for Aresté Saló, Clough and Figueras' modified
puncture gauge; it is what the harmonic condition gives, and at ``a = 0``
it is the standard gauge exactly.

**Matter enters linearly, and outside the kernel.** :func:`matter_terms`
is the exact linear map from ``(rho, S_i, S_ij)`` and a scalar source to
the rates, including the ``b`` terms, which see ``rho`` through ``H`` and
``S_i`` through ``M_i``. Keeping it out of the generated kernel means one
derivation serves a vacuum run, a scalar and the Gauss-Bonnet effective
matter of :mod:`particlesim.symbolic.esgb`, which needs it outside because
it depends on ``d_t K_ij``. A test checks the map against the symbolic
equations with matter in them.

**The scalar** is minimally coupled, in the normalisation of
:mod:`particlesim.theories.gauss_bonnet`: ``R - 2 (d phi)^2``, so
``8 pi T_ab = 2 d_a phi d_b phi - g_ab (d phi)^2``. Its momentum is
``K_phi = -n^a d_a phi``, with

    d_t phi   = beta^k d_k phi - alpha K_phi
    d_t K_phi = beta^k d_k K_phi + alpha (K K_phi - D^i D_i phi + s) - D^i alpha D_i phi

for ``box phi = s``. The state names are ``sf`` and ``Ksf``, because
``phi`` is already the conformal exponent.
"""

from __future__ import annotations

import math
from typing import Any

import sympy as sp

from particlesim.symbolic import bssn, ccz4
from particlesim.symbolic.bssn import BSSNVariables, _connection_derivative
from particlesim.symbolic.threeplusone import (
    DIMENSION,
    INDICES,
    christoffel,
    inverse_metric,
    lapse_hessian,
    trace,
)

#: ``a(x)`` and ``b(x)`` as Aresté Saló, Clough and Figueras use them, recalled
#: rather than read (see the module docstring): constants with
#: ``0 < a < b``, so the gauge, the constraint violations and the light
#: cone are three different speeds, ``1/sqrt(1.2)``, ``1/sqrt(1.4)`` and 1.
GAUGE_FUNCTION = 0.2
CONSTRAINT_FUNCTION = 0.4

#: The evolved state: CCZ4's twenty-five, then the scalar and its momentum.
STATE_NAMES = (*ccz4.STATE_NAMES, "sf", "Ksf")

#: How many derivatives of each field the equations need.
DERIVATIVE_ORDERS = {**ccz4.DERIVATIVE_ORDERS, "sf": 2, "Ksf": 1}

#: Parameters the kernel reads as fields, so one derivation serves every value.
#:
#: ``mccz4_b`` is ``b(x)``, which may vary in space; ``kappa1`` and
#: ``kappa2`` are the damping constants. ``a(x)`` does not appear in the
#: kernel at all: it only enters the gauge, which :func:`modified_gauge`
#: assembles outside it.
PARAMETER_NAMES = ("mccz4_b", "kappa1", "kappa2")


def state_names(conformal: str = "phi") -> tuple[str, ...]:
    """:data:`STATE_NAMES` with ``conformal`` in place of ``phi``."""
    bssn._check_conformal(conformal)
    return tuple(conformal if name == "phi" else name for name in STATE_NAMES)


def abstract_state(conformal: str = "phi"):
    """Symbols for the evolved state, its derivatives, and the codegen registry."""
    names = state_names(conformal)
    state = {name: sp.Symbol(name, real=True) for name in names}
    derivatives: dict[str, Any] = {}
    registry: dict[Any, tuple[str, tuple[int, ...]]] = {}
    for name, order in DERIVATIVE_ORDERS.items():
        name = conformal if name == "phi" else name
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


def momentum_constraint(variables: BSSNVariables, momentum=None, inverse=None) -> list[Any]:
    """``M_i = D_j K^j_i - D_i K - 8 pi S_i`` in the conformal variables, index down.

    ``K^j_i = Abar^j_i + delta^j_i K / 3`` with ``Abar^j_i = gammabar^jk
    Abar_ki``, and the physical and conformal connections differ only by
    the gradient of ``phi``, which gives

        M_i = Dbar_j Abar^j_i + 6 Abar^j_i d_j phi - (2/3) d_i K - 8 pi S_i

    for trace-free ``Abar_ij``. ``momentum`` is ``S_i``, index down. No
    Gammahat appears: this is the constraint of the data, and the evolved
    connection is not the metric's once ``Z_i`` is non-zero.
    """
    conformal = variables.conformal_slice
    conformal_inverse = inverse_metric(conformal.metric) if inverse is None else inverse
    gammas = christoffel(conformal, conformal_inverse)
    traceless = variables.traceless_curvature
    d_traceless = conformal.d_curvature
    d_inverse = [
        [
            [
                -sum(
                    conformal_inverse[j][a] * conformal_inverse[k][b] * conformal.d_metric[m][a][b]
                    for a in INDICES
                    for b in INDICES
                )
                for k in INDICES
            ]
            for j in INDICES
        ]
        for m in INDICES
    ]
    mixed = [
        [sum(conformal_inverse[j][k] * traceless[k][i] for k in INDICES) for i in INDICES]
        for j in INDICES
    ]
    out = []
    for i in INDICES:
        total = 0
        for j in INDICES:
            for k in INDICES:
                # d_j Abar^j_i, then the two connection terms of Dbar_j.
                total = total + d_inverse[j][j][k] * traceless[k][i]
                total = total + conformal_inverse[j][k] * d_traceless[j][k][i]
                total = total + gammas[j][j][k] * mixed[k][i]
                total = total - gammas[k][j][i] * mixed[j][k]
            total = total + 6 * mixed[j][i] * variables.d_phi[j]
        total = total - 2 * variables.d_mean_curvature[i] / 3
        if momentum is not None:
            total = total - 8 * math.pi * momentum[i]
        out.append(total)
    return out


def scalar_stress(metric, inverse, d_scalar, momentum_scalar):
    """``(rho, S_i, S_ij)`` of the minimally coupled scalar, indices down.

    With ``8 pi T_ab = 2 d_a phi d_b phi - g_ab (d phi)^2`` and
    ``n^a d_a phi = -K_phi``:

        rho  = (K_phi^2 + D_k phi D^k phi) / 8 pi
        S_i  = K_phi d_i phi / 4 pi
        S_ij = (d_i phi d_j phi - gamma_ij (D_k phi D^k phi - K_phi^2) / 2) / 4 pi
    """
    square = sum(inverse[i][j] * d_scalar[i] * d_scalar[j] for i in INDICES for j in INDICES)
    density = (momentum_scalar**2 + square) / (8 * math.pi)
    momentum = [momentum_scalar * d_scalar[i] / (4 * math.pi) for i in INDICES]
    stress = [
        [
            (d_scalar[i] * d_scalar[j] - metric[i][j] * (square - momentum_scalar**2) / 2)
            / (4 * math.pi)
            for j in INDICES
        ]
        for i in INDICES
    ]
    return density, momentum, stress


def scalar_hessian(slice_, d_scalar, dd_scalar, inverse=None):
    """``D_i D_j phi`` of a scalar on the physical slice."""
    inverse = inverse_metric(slice_.metric) if inverse is None else inverse
    gammas = christoffel(slice_, inverse)
    return [
        [dd_scalar[i][j] - sum(gammas[k][i][j] * d_scalar[k] for k in INDICES) for j in INDICES]
        for i in INDICES
    ]


def scalar_rates(variables, momentum_scalar, d_scalar, dd_scalar, d_momentum, inverse=None):
    """``(d_t phi, d_t K_phi)`` for ``box phi = 0``; a source adds ``alpha s`` to the second."""
    physical = variables.physical
    inverse = inverse_metric(physical.metric) if inverse is None else inverse
    hessian = scalar_hessian(physical, d_scalar, dd_scalar, inverse)
    laplacian = trace(inverse, hessian)
    lapse = variables.lapse
    shift = variables.shift
    gradient = sum(
        inverse[i][j] * physical.d_lapse[i] * d_scalar[j] for i in INDICES for j in INDICES
    )
    dt_scalar = -lapse * momentum_scalar
    dt_momentum = lapse * (variables.mean_curvature * momentum_scalar - laplacian) - gradient
    for k in INDICES:
        dt_scalar = dt_scalar + shift[k] * d_scalar[k]
        dt_momentum = dt_momentum + shift[k] * d_momentum[k]
    return dt_scalar, dt_momentum, hessian


def constraint_terms(variables, parts, theta, d_theta, constraint_function, kappa1, kappa2):
    """The ``b`` terms: what ``ghat`` adds to ``d_t Theta``, ``d_t K`` and ``d_t Gammahat^i``.

    ``parts`` is the second value :func:`~particlesim.symbolic.ccz4.ccz4_parts`
    returns, with the momentum constraint added under ``"momentum"``. Every
    term is proportional to a constraint, to ``Theta`` or to ``Z_i``, so
    the ``b`` terms vanish on any solution of General Relativity that
    starts with ``Theta = Z_i = 0``, and the physics is unchanged.
    """
    lapse = variables.lapse
    factor = variables.conformal_exponent
    conformal = variables.conformal_slice
    conformal_inverse = parts["conformal_inverse"]
    weight = constraint_function / (1 + constraint_function)
    hamiltonian = parts["hamiltonian"]
    momentum = parts["momentum"]
    upper = parts["conformal_z"]
    lower = [sum(conformal.metric[i][j] * upper[j] for j in INDICES) for i in INDICES]
    curvature = variables.physical.curvature

    theta_term = -weight * lapse * (hamiltonian / 2 - kappa1 * (2 + kappa2) * theta)
    # Z^k with the physical metric is e^(-4 phi) gammabar^kl Z_l.
    bracket = [
        momentum[j]
        + d_theta[j]
        - factor * sum(curvature[j][k] * upper[k] for k in INDICES)
        - kappa1 * lower[j]
        for j in INDICES
    ]
    connection_term = [
        -2 * weight * lapse * sum(conformal_inverse[i][j] * bracket[j] for j in INDICES)
        for i in INDICES
    ]
    return {
        "theta": theta_term,
        "mean_curvature": 3 * theta_term / 2,
        "connection": connection_term,
    }


def modified_parts(
    variables: BSSNVariables,
    theta,
    d_theta,
    constraint_function,
    kappa1,
    kappa2,
    d_connection=None,
    dd_shift=None,
    density=0.0,
    stress=None,
    momentum=None,
):
    """The modified-CCZ4 rates of the geometry and ``Theta``, and the pieces behind them.

    Returns ``(rates, parts)`` with the keys of
    :func:`~particlesim.symbolic.ccz4.ccz4_rhs`. ``constraint_function`` is
    ``b``; ``kappa1`` and ``kappa2`` may be symbols. Matter is optional
    and exact here; an evolution passes none and adds
    :func:`matter_terms` instead, and a test checks the two agree.
    """
    if d_connection is None:
        d_connection = [_connection_derivative(variables, k) for k in INDICES]
    rates, parts = ccz4.ccz4_parts(
        variables,
        theta,
        d_theta,
        density=density,
        stress=stress,
        momentum=momentum,
        d_connection=d_connection,
        dd_shift=dd_shift,
        damping=kappa1,
        damping_mix=kappa2,
    )
    parts["momentum"] = momentum_constraint(variables, momentum, parts["conformal_inverse"])
    extra = constraint_terms(variables, parts, theta, d_theta, constraint_function, kappa1, kappa2)
    rates = dict(rates)
    rates["theta"] = rates["theta"] + extra["theta"]
    rates["mean_curvature"] = rates["mean_curvature"] + extra["mean_curvature"]
    rates["connection"] = [rates["connection"][i] + extra["connection"][i] for i in INDICES]
    return rates, parts


def matter_terms(
    lapse,
    conformal_exponent,
    conformal_metric,
    density,
    momentum,
    stress,
    constraint_function,
    source=None,
):
    """What ``(rho, S_i, S_ij)`` and a scalar source add to each rate, exactly.

    Linear in the matter, and the same map :func:`modified_parts` applies
    when it is given the matter itself:

        d_t K          += 4 pi alpha (S - 3 rho) + 12 pi beta_b alpha rho
        d_t Abar_ij    += -8 pi alpha e^(-4 phi) (S_ij - gamma_ij S / 3)
        d_t Theta      += -8 pi alpha rho / (1 + b)
        d_t Gammahat^i += -16 pi alpha gammabar^ij S_j / (1 + b)
        d_t K_phi      += alpha s

    with ``S = gamma^ij S_ij``, ``beta_b = b / (1 + b)`` and ``s`` the
    source of ``box phi = s``. The ``1 + b`` comes in through ``H`` and
    ``M_i`` in the ``b`` terms. Works on SymPy expressions and on arrays.
    ``conformal_exponent`` is ``e^(-4 phi)``. The Gamma-driver's ``B^i``
    takes the connection's term through the gauge, which is assembled after
    this is added.
    """
    conformal_inverse = inverse_metric(conformal_metric)
    inverse = [[conformal_exponent * conformal_inverse[i][j] for j in INDICES] for i in INDICES]
    metric = [[conformal_metric[i][j] / conformal_exponent for j in INDICES] for i in INDICES]
    traced = trace(inverse, stress)
    weight = constraint_function / (1 + constraint_function)
    pi = math.pi
    out: dict[str, Any] = {
        "mean_curvature": 4 * pi * lapse * (traced - 3 * density)
        + 12 * pi * weight * lapse * density,
        "theta": -8 * pi * lapse * density / (1 + constraint_function),
        "connection": [
            -16
            * pi
            * lapse
            * sum(conformal_inverse[i][j] * momentum[j] for j in INDICES)
            / (1 + constraint_function)
            for i in INDICES
        ],
        "traceless_curvature": [
            [
                -8 * pi * lapse * conformal_exponent * (stress[i][j] - metric[i][j] * traced / 3)
                for j in INDICES
            ]
            for i in INDICES
        ],
    }
    if source is not None:
        out["scalar_momentum"] = lapse * source
    return out


#: Slicing conditions without their advection term: BSSN's.
SLICINGS = bssn.SLICINGS


def modified_gauge(
    lapse,
    mean_curvature,
    driver,
    connection_rate,
    gauge_function,
    lapse_gradient,
    advection_lapse=None,
    advection_shift=None,
    slicing: str = "one_plus_log",
    shift_condition: str = "gamma_driver",
    damping: float = 2.0,
):
    """``(d_t alpha, d_t beta^i, d_t B^i)`` with ``gtilde`` in place of ``g``.

    ``slicing`` and ``shift_condition`` are those of
    :func:`particlesim.symbolic.bssn.gauge_rhs`. ``gauge_function`` is
    ``a``; ``lapse_gradient`` is ``D^i alpha = gamma^ij d_j alpha``. The
    slicing rate is divided by ``1 + a`` and the driver becomes

        d_t beta^i = (3/4) B^i / (1 + a) - (a / (1 + a)) alpha D^i alpha,

    the change the modified harmonic condition makes to the harmonic shift.
    ``advection_lapse`` and ``advection_shift`` are ``beta^k d_k alpha`` and
    ``beta^k d_k beta^i``, added when given. At ``a = 0`` this is
    :func:`~particlesim.symbolic.bssn.gauge_rhs` term for term.
    """
    if slicing not in SLICINGS:
        raise ValueError(f"unknown slicing {slicing!r}; known: {sorted(SLICINGS)}")
    if shift_condition not in ("gamma_driver", "frozen"):
        raise ValueError(f"unknown shift condition {shift_condition!r}")
    scale = 1 / (1 + gauge_function)
    dt_lapse = SLICINGS[slicing](lapse, mean_curvature) * scale
    if advection_lapse is not None:
        dt_lapse = dt_lapse + advection_lapse
    if shift_condition == "frozen":
        zero = 0 * lapse
        return dt_lapse, [zero for _ in INDICES], [zero for _ in INDICES]
    dt_shift = []
    dt_driver = []
    for i in INDICES:
        velocity = (3 * driver[i] / 4) * scale - gauge_function * scale * lapse * lapse_gradient[i]
        if advection_shift is not None:
            velocity = velocity + advection_shift[i]
        dt_shift.append(velocity)
        dt_driver.append(connection_rate[i] - damping * driver[i])
    return dt_lapse, dt_shift, dt_driver


def covariant_curvature_derivative(slice_, inverse=None):
    """``D_k K_ij``, indexed ``[k][i][j]``, on the physical slice."""
    inverse = inverse_metric(slice_.metric) if inverse is None else inverse
    gammas = christoffel(slice_, inverse)
    curvature = slice_.curvature
    return [
        [
            [
                slice_.d_curvature[k][i][j]
                - sum(
                    gammas[m][k][i] * curvature[m][j] + gammas[m][k][j] * curvature[i][m]
                    for m in INDICES
                )
                for j in INDICES
            ]
            for i in INDICES
        ]
        for k in INDICES
    ]


def lie_shift_curvature(slice_):
    """``L_beta K_ij = beta^k d_k K_ij + K_ik d_j beta^k + K_jk d_i beta^k``."""
    return [
        [
            sum(
                slice_.shift[k] * slice_.d_curvature[k][i][j]
                + slice_.curvature[i][k] * slice_.d_shift[j][k]
                + slice_.curvature[j][k] * slice_.d_shift[i][k]
                for k in INDICES
            )
            for j in INDICES
        ]
        for i in INDICES
    ]


__all__ = [
    "CONSTRAINT_FUNCTION",
    "DERIVATIVE_ORDERS",
    "GAUGE_FUNCTION",
    "PARAMETER_NAMES",
    "SLICINGS",
    "STATE_NAMES",
    "abstract_state",
    "constraint_terms",
    "covariant_curvature_derivative",
    "lapse_hessian",
    "lie_shift_curvature",
    "matter_terms",
    "modified_gauge",
    "modified_parts",
    "momentum_constraint",
    "scalar_hessian",
    "scalar_rates",
    "scalar_stress",
    "state_names",
]
