"""Teukolsky's linearised quadrupole wave, as BSSN initial data.

Teukolsky (1982) wrote down the even-parity ``l = 2, m = 0`` solution of the
linearised Einstein equations in closed form, in a gauge with unit lapse and
no shift:

    ds^2 = -dt^2 + (1 + A f_rr) dr^2 + 2 B f_rθ r dr dθ
           + (1 + C f1_θθ + A f2_θθ) r^2 dθ^2
           + (1 + C f1_φφ + A f2_φφ) r^2 sin^2θ dφ^2

with the angular functions

    f_rr = 2 - 3 sin^2θ    f_rθ = -3 sinθ cosθ
    f1_θθ = 3 sin^2θ       f2_θθ = -1
    f1_φφ = -3 sin^2θ      f2_φφ = 3 sin^2θ - 1

and radial ones built from a single profile ``F``:

    A = 3 [F'' / r^3 + 3 F' / r^4 + 3 F / r^5]
    B = -[F''' / r^2 + 3 F'' / r^3 + 6 F' / r^4 + 6 F / r^5]
    C = 1/4 [F'''' / r + 2 F''' / r^2 + 9 F'' / r^3 + 21 F' / r^4 + 21 F / r^5]

``F^(n) = g^(n)(t - r) - (-1)^n g^(n)(t + r)`` is an outgoing wave minus an
incoming one of the same shape, which is the choice that makes the metric
regular at the origin. With ``g(x) = a x exp(-x^2/λ^2)``, odd, the data at
``t = 0`` is time-symmetric: ``K_ij = 0`` and the whole of the wave is in the
metric. It then splits into a shell that leaves and one that falls through
the centre and follows it out, which makes it the test an outer boundary is
judged by: a wave with the angular structure of real radiation, whose
evolution in the interior is known in closed form until it reaches the edge.

**It is a solution of the linearised equations only.** The constraints fail
at second order in the amplitude, and that is the first thing to check about
any implementation of it: a sign error in one of the dozen terms above shows
up as a violation *linear* in the amplitude, and a correct one as quadratic.
The trace of ``K_ij`` vanishes at linear order too, which is why harmonic
slicing with a frozen shift keeps the evolution in Teukolsky's gauge to
``O(a^2)`` and the evolved metric can be compared with the formula directly.

**The formula cancels catastrophically at the origin.** Every term in ``A``
goes as ``a / r^4`` there and the sum is finite, so near ``r = 0`` it is
evaluated from its Taylor series instead. Collecting powers, each of ``A``,
``B`` and ``C`` is ``sum_q c_q g^(q)(t) r^(q - 5)`` -- every one of the terms
above has ``n + p = 5`` for ``F^(n) / r^p`` -- and the coefficients below
``q = 5`` vanish, which is the regularity condition written out. The
derivatives of the Gaussian profile are Hermite functions,

    g^(q)(x) = (a/2) (-1)^q λ^(1-q) H_(q+1)(x/λ) exp(-x^2/λ^2),

so neither form needs anything symbolic at run time.
"""

from __future__ import annotations

from math import factorial
from typing import Any

import numpy as np

from particlesim.solvers.nr.bssn import DIMENSION, INDICES, _grid, _module
from particlesim.symbolic.threeplusone import determinant, inverse_metric

#: The terms of ``A``, ``B`` and ``C`` as ``(n, weight)`` for ``weight F^(n) / r^(5 - n)``,
#: with the overall factors of 3, -1 and 1/4 folded in.
TERMS: dict[str, tuple[tuple[int, float], ...]] = {
    "A": ((2, 3.0), (1, 9.0), (0, 9.0)),
    "B": ((3, -1.0), (2, -3.0), (1, -6.0), (0, -6.0)),
    "C": ((4, 0.25), (3, 0.5), (2, 2.25), (1, 5.25), (0, 5.25)),
}

#: Below this fraction of the width the series is used instead of the formula.
#:
#: The formula loses ``(λ / r)^4`` to cancellation, so a factor of 256 here;
#: the series' terms fall by ``(r / λ)^2`` each, so thirty of them are far
#: more than enough.
SERIES_RADIUS = 0.25
SERIES_TERMS = 30


def profile_derivatives(x, amplitude: float, width: float, count: int) -> list[Any]:
    """``g^(q)(x)`` for ``q < count``, with ``g(x) = a x exp(-x^2/λ^2)``.

    From the Hermite recurrence ``H_(k+1) = 2u H_k - 2k H_(k-1)``, since
    ``g = -(a λ / 2) d/du exp(-u^2)`` with ``u = x / λ``.
    """
    u = np.asarray(x, dtype=float) / width
    envelope = np.exp(-(u**2))
    hermite = [np.ones_like(u), 2 * u]
    for k in range(1, count + 1):
        hermite.append(2 * u * hermite[k] - 2 * k * hermite[k - 1])
    return [
        0.5 * amplitude * (-1) ** q * width ** (1 - q) * hermite[q + 1] * envelope
        for q in range(count)
    ]


def _series_coefficient(name: str, q: int) -> float:
    """``c_q``: the coefficient of ``g^(q)(t) r^(q-5)`` in ``A``, ``B`` or ``C``.

    ``F^(n) = sum_j g^(n+j)(t) r^j / j! [(-1)^j - (-1)^n]``, so the terms
    with ``n + j = q`` contribute ``w_n [(-1)^(q-n) - (-1)^n] / (q-n)!``.
    Only odd ``q`` survive, which makes each of ``A``, ``B``, ``C`` even in
    ``r`` as a regular scalar has to be.
    """
    total = 0.0
    for n, weight in TERMS[name]:
        if n <= q:
            total += weight * ((-1) ** (q - n) - (-1) ** n) / factorial(q - n)
    return total


def radial_functions(
    time: float, radius, amplitude: float, width: float, rate: bool = False
) -> dict[str, Any]:
    """``A``, ``B`` and ``C`` at ``(t, r)``, or their time derivatives if ``rate``.

    ``d/dt F^(n) = g^(n+1)(t - r) - (-1)^n g^(n+1)(t + r)``, which is the same
    sum with every derivative of ``g`` shifted up by one, so the time
    derivative costs one more Hermite function and no new algebra.
    """
    radius = np.asarray(radius, dtype=float)
    shift = 1 if rate else 0
    near = radius < SERIES_RADIUS * width
    far_radius = np.where(near, SERIES_RADIUS * width, radius)
    behind = profile_derivatives(time - far_radius, amplitude, width, 5 + shift)
    ahead = profile_derivatives(time + far_radius, amplitude, width, 5 + shift)

    out: dict[str, Any] = {}
    centre = None
    if np.any(near):
        centre = profile_derivatives(np.float64(time), amplitude, width, 5 + SERIES_TERMS + shift)
    for name, terms in TERMS.items():
        formula = sum(
            weight * (behind[n + shift] - (-1) ** n * ahead[n + shift]) / far_radius ** (5 - n)
            for n, weight in terms
        )
        if centre is None:
            out[name] = formula
            continue
        series = sum(
            _series_coefficient(name, q) * centre[q + shift] * radius ** (q - 5)
            for q in range(5, 5 + SERIES_TERMS, 2)
        )
        out[name] = np.where(near, series, formula)
    return out


def perturbation(
    time: float, coords, amplitude: float, width: float, rate: bool = False
) -> list[list[Any]]:
    """The Cartesian perturbation ``h_ij``, or ``d_t h_ij`` if ``rate``, about the ``z`` axis.

    Written without the azimuth, which is undefined on the axis. With
    ``n = x/r``, ``w = sinθ θhat = (zx, zy, -ρ^2)/r^2`` and ``v = sinθ φhat
    = (-y, x, 0)/r``,

        h = A (2 - 3 sin^2θ) n n - 3 B cosθ (n w + w n)
            - A (δ - n n) + 3 A v v + 3 C (w w - v v)

    using ``θhat θhat + φhat φhat = δ - n n``. Traceless, as it should be:
    ``|w|^2 = |v|^2 = sin^2θ``. At the origin itself the direction is
    undefined but the limit is not, and the ``z`` axis is used for it.
    """
    x, y, z = (np.asarray(value, dtype=float) for value in coords)
    radius = np.sqrt(x**2 + y**2 + z**2)
    origin = radius == 0.0
    safe = np.where(origin, 1.0, radius)
    n = [np.where(origin, 0.0, x / safe), np.where(origin, 0.0, y / safe)]
    n.append(np.where(origin, 1.0, z / safe))
    cylinder = x**2 + y**2
    w = [z * x / safe**2, z * y / safe**2, -cylinder / safe**2]
    v = [-y / safe, x / safe, np.zeros_like(x)]
    sine2 = cylinder / safe**2
    cosine = n[2]

    functions = radial_functions(time, radius, amplitude, width, rate=rate)
    a, b, c = functions["A"], functions["B"], functions["C"]
    out = [[None] * DIMENSION for _ in INDICES]
    for i in INDICES:
        for j in range(i, DIMENSION):
            delta = 1.0 if i == j else 0.0
            value = (
                a * (2 - 3 * sine2) * n[i] * n[j]
                - 3 * b * cosine * (n[i] * w[j] + w[i] * n[j])
                - a * (delta - n[i] * n[j])
                + 3 * a * v[i] * v[j]
                + 3 * c * (w[i] * w[j] - v[i] * v[j])
            )
            out[i][j] = out[j][i] = value
    return out


def physical(
    shape=(48, 48, 48),
    amplitude: float = 1e-3,
    width: float = 1.0,
    extent=8.0,
    time: float = 0.0,
    centre=None,
) -> tuple[list[list[Any]], list[list[Any]], tuple[float, ...]]:
    """``gamma_ij`` and ``K_ij`` of the wave on the grid, as NumPy arrays.

    ``K_ij = -d_t gamma_ij / 2``, since the lapse is one and the shift zero.
    ``centre`` defaults to the middle of the box, which is a grid point
    whenever the shape is even and is left there on purpose: the metric is
    regular at the origin, and a centre that does not move with the spacing
    puts the coarse grid's points on the fine grid's, which is what a
    pointwise convergence measurement needs.
    """
    mesh, spacing = _grid(shape, extent, "numpy")
    lengths = tuple(float(value) for value in np.atleast_1d(extent))
    if len(lengths) == 1:
        lengths = lengths * DIMENSION
    centre = [value / 2 for value in lengths] if centre is None else list(centre)
    coords = [np.asarray(mesh[i]) - centre[i] for i in INDICES]
    h = perturbation(time, coords, amplitude, width)
    dh = perturbation(time, coords, amplitude, width, rate=True)
    metric = [[(1.0 if i == j else 0.0) + h[i][j] for j in INDICES] for i in INDICES]
    curvature = [[-0.5 * dh[i][j] for j in INDICES] for i in INDICES]
    return metric, curvature, spacing


def _periodic_derivative(field, axis: int, step: float, order: int):
    """The kernel's centred stencil, wrapped, so the data matches what it will see."""
    if order == 2:
        return (np.roll(field, -1, axis) - np.roll(field, 1, axis)) / (2 * step)
    if order == 4:
        return (
            -np.roll(field, -2, axis)
            + 8 * np.roll(field, -1, axis)
            - 8 * np.roll(field, 1, axis)
            + np.roll(field, 2, axis)
        ) / (12 * step)
    if order == 6:
        return (
            np.roll(field, -3, axis)
            - 9 * np.roll(field, -2, axis)
            + 45 * np.roll(field, -1, axis)
            - 45 * np.roll(field, 1, axis)
            + 9 * np.roll(field, 2, axis)
            - np.roll(field, 3, axis)
        ) / (60 * step)
    raise ValueError("order must be 2, 4, or 6")


def teukolsky_wave(
    shape=(48, 48, 48),
    amplitude: float = 1e-3,
    width: float = 1.0,
    extent=8.0,
    time: float = 0.0,
    centre=None,
    order: int = 4,
    backend: str = "jax",
) -> tuple[dict[str, Any], tuple[float, ...]]:
    """The wave in BSSN variables, with unit lapse and zero shift.

    ``phi = ln(det gamma) / 12``, ``gammabar_ij = e^(-4 phi) gamma_ij``,
    ``Abar_ij = e^(-4 phi) (K_ij - gamma_ij K / 3)``, all algebraic.

    **The connection is differenced, not derived.** ``Gammabar^i =
    -d_j gammabar^ij`` is the one variable that is a derivative of the
    others, and it is taken with the evolution's own stencil at ``order``,
    so the discrete constraint ``Gammabar^i + d_j gammabar^ij = 0`` holds
    exactly at the start rather than to truncation error. The stencil
    wraps, which is harmless as long as the box is wide enough that the
    wave's tail at the edge is negligible: at ``extent = 8`` and ``λ = 1``
    it is ``exp(-16)``.
    """
    module = _module(backend)
    metric, curvature, spacing = physical(shape, amplitude, width, extent, time, centre)
    volume = determinant(metric)
    phi = np.log(volume) / 12.0
    conformal = np.exp(-4.0 * phi)
    inverse = inverse_metric(metric)
    mean = sum(inverse[i][j] * curvature[i][j] for i in INDICES for j in INDICES)

    tilde = [[conformal * metric[i][j] for j in INDICES] for i in INDICES]
    tilde_inverse = inverse_metric(tilde)
    connection = [
        -sum(_periodic_derivative(tilde_inverse[i][j], j, spacing[j], order) for j in INDICES)
        for i in INDICES
    ]

    state: dict[str, Any] = {"phi": module.asarray(phi), "trK": module.asarray(mean)}
    state["alpha"] = module.ones_like(state["phi"])
    for i in INDICES:
        state[f"beta{i}"] = module.zeros_like(state["phi"])
        state[f"B{i}"] = module.zeros_like(state["phi"])
        state[f"Gt{i}"] = module.asarray(connection[i])
        for j in range(i, DIMENSION):
            traceless = curvature[i][j] - metric[i][j] * mean / 3.0
            state[f"gt{i}{j}"] = module.asarray(tilde[i][j])
            state[f"At{i}{j}"] = module.asarray(conformal * traceless)
    return state, spacing


__all__ = [
    "SERIES_RADIUS",
    "TERMS",
    "perturbation",
    "physical",
    "profile_derivatives",
    "radial_functions",
    "teukolsky_wave",
]
