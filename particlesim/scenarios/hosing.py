"""The electron hose instability in an ion channel (issue #36).

A relativistic electron beam in a plasma blows the plasma electrons out of
its path and travels in the channel of ions left behind. The ions focus it,
and each slice of the beam oscillates across the channel at the betatron
wavenumber ``k_b``. The channel is not rigid, though. The plasma electrons
at its edge are pulled by the beam's charge and oscillate at their own
wavenumber ``k_c``. A slice displaced sideways drags the channel's electrons
with it, and the slices behind then feel a channel centred somewhere else.
That feedback runs from the head of the beam to the tail and grows (Whittum,
Sharp, Yu, Lampe and Joyce 1991).

**The model** is the two centroids, the beam's ``y_b(xi, s)`` and the
channel's ``y_c(xi, s)``, with ``xi`` the distance behind the head and ``s``
the distance travelled:

    d^2 y_b / ds^2  = -k_b^2 (y_b - y_c)
    d^2 y_c / dxi^2 = -k_c^2 (y_c - y_b),   y_c = y_c' = 0 at the head

The channel equation is solved exactly by its Green's function,
``y_c(xi) = k_c int_0^xi sin(k_c (xi - xi')) y_b(xi') dxi'``. That leaves one
integro-differential equation for the beam, integrated in ``s`` for a
discretised beam. The quadrature is cumulative Simpson in ``xi``, so the
discretisation error falls at fourth order.

**Two exact results the tests use.**
- **Early on**, with every slice started at ``y_0`` and at rest, the
  channel behind the head is ``y_0 (1 - cos k_c xi)``. So
  ``y_b = y_0 [1 - (k_b s)^2 cos(k_c xi) / 2] + O(s^4)``.
- **Late**, a saddle point of the double Laplace transform gives the
  asymptotic growth

      y_b ~ exp[ (3 sqrt(3) / 4) (k_b s)^(2/3) (k_c xi)^(1/3) ]

  The derivation is in the docs. The exponent grows with ``xi^(1/3)``: the
  tail of a long beam is always the part that goes first.

**Growth-rate extraction.** :func:`growth_exponent` reads ``ln |y_b|`` along
the beam and fits it against the asymptotic exponent. The slope tends to
one as the exponent grows, with a correction that falls as
``1/exponent``.

**What spread does.** :class:`HoseBeam` carries the beam as macroparticles
rather than a centroid. With no spread in energy they reproduce the centroid
model exactly, since the equations are linear. A spread in betatron
wavenumber, which is what a spread in energy gives, since
``k_b ~ gamma^(-1/2)``, phase-mixes the slice. That damps the centroid's
response, and the instability slows. That is the mechanism by which an
energy chirp or spread suppresses hosing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def asymptotic_exponent(betatron: float, channel: float, xi, s):
    """``(3 sqrt(3) / 4) (k_b s)^(2/3) (k_c xi)^(1/3)``: the long-time growth of the hose."""
    xi = np.asarray(xi, dtype=float)
    return 0.75 * np.sqrt(3.0) * (betatron * s) ** (2.0 / 3.0) * (channel * xi) ** (1.0 / 3.0)


def _cumulative_simpson(values: np.ndarray, spacing: float) -> np.ndarray:
    """``int_0^xi values``, cumulative along the last axis, fourth order.

    Simpson's rule on each pair of intervals, with the odd points from the
    cubic through four neighbours. The first point is zero.
    """
    n = values.shape[-1]
    out = np.zeros_like(values)
    # Even points: composite Simpson from the start.
    pair = spacing / 3.0 * (values[..., 0:-2:2] + 4.0 * values[..., 1:-1:2] + values[..., 2::2])
    out[..., 2::2] = np.cumsum(pair, axis=-1)
    # Odd points: the even point before, plus the integral over one interval
    # from the cubic through that point and its neighbours.
    odd = np.arange(1, n, 2)
    left = odd - 1
    a = values[..., np.maximum(left - 1, 0)]
    b = values[..., left]
    c = values[..., np.minimum(left + 1, n - 1)]
    d = values[..., np.minimum(left + 2, n - 1)]
    interior = spacing / 24.0 * (-a + 13.0 * b + 13.0 * c - d)
    # At the head there is no point before; use the cubic through the first four.
    first = (
        spacing
        / 24.0
        * (9.0 * values[..., 0] + 19.0 * values[..., 1] - 5.0 * values[..., 2] + values[..., 3])
    )
    interior[..., 0] = first
    out[..., 1::2] = out[..., left] + interior
    return out


@dataclass(frozen=True)
class HoseChannel:
    """The beam and channel centroids, for a beam ``length`` long in ``slices`` slices.

    ``betatron`` is ``k_b`` and ``channel`` is ``k_c``. ``slices`` should be
    odd for the Simpson quadrature to end on an even point; it is made so.
    """

    betatron: float = 1.0
    channel: float = 10.0
    length: float = 10.0
    slices: int = 401

    @property
    def xi(self) -> np.ndarray:
        n = self.slices if self.slices % 2 == 1 else self.slices + 1
        return np.linspace(0.0, self.length, n)

    def channel_centroid(self, beam: np.ndarray) -> np.ndarray:
        """The channel's centroid for each row of ``beam``, from its Green's function.

        ``y_c(xi) = k_c int_0^xi sin(k_c (xi - xi')) y_b(xi') dxi'``, written as
        ``sin(k_c xi) int cos(k_c xi') y_b - cos(k_c xi) int sin(k_c xi') y_b`` so that
        both integrals are cumulative.
        """
        xi = self.xi
        k = self.channel
        h = xi[1] - xi[0]
        cosine, sine = np.cos(k * xi), np.sin(k * xi)
        with_cos = _cumulative_simpson(beam * cosine, h)
        with_sin = _cumulative_simpson(beam * sine, h)
        return k * (sine * with_cos - cosine * with_sin)

    def solve(self, distances, offset: float = 1.0, rtol: float = 1e-11) -> np.ndarray:
        """``y_b(xi, s)`` at each distance in ``distances``, from every slice offset and at rest."""
        from scipy.integrate import solve_ivp

        xi = self.xi
        n = len(xi)
        kb2 = self.betatron**2

        def rhs(_, y):
            position, velocity = y[:n], y[n:]
            return np.concatenate([velocity, -kb2 * (position - self.channel_centroid(position))])

        start = np.concatenate([np.full(n, float(offset)), np.zeros(n)])
        distances = np.atleast_1d(np.asarray(distances, dtype=float))
        solution = solve_ivp(
            rhs,
            (0.0, float(distances.max())),
            start,
            method="DOP853",
            t_eval=distances,
            rtol=rtol,
            atol=rtol * abs(offset) * 1e-3,
        )
        return solution.y[:n].T


def growth_exponent(xi, profile, betatron: float, channel: float, s: float, tail_fraction=0.5):
    """``(fit slope, fit intercept)`` of ``ln |y_b|`` against the asymptotic exponent.

    The profile oscillates along the beam, so its envelope is taken as the
    running maximum of ``|y_b|`` over each channel wavelength. The fit uses
    the last ``tail_fraction`` of the beam, where the exponent is largest.
    A slope of one says the growth is Whittum's.
    """
    xi = np.asarray(xi, dtype=float)
    magnitude = np.abs(np.asarray(profile, dtype=float))
    wavelength = 2.0 * np.pi / channel
    step = xi[1] - xi[0]
    window = max(1, int(round(wavelength / step)))
    envelope = np.array(
        [magnitude[max(0, i - window) : i + 1].max() for i in range(len(magnitude))]
    )
    start = int(len(xi) * (1.0 - tail_fraction))
    exponent = asymptotic_exponent(betatron, channel, xi[start:], s)
    slope, intercept = np.polyfit(exponent, np.log(envelope[start:]), 1)
    return float(slope), float(intercept)


@dataclass(frozen=True)
class HoseBeam:
    """The same instability with the beam as macroparticles, each with its own ``k_b``.

    Each slice holds ``per_slice`` particles, whose betatron wavenumbers are
    ``betatron (1 + spread u)`` for ``u`` evenly spaced over ``[-1, 1]``, a
    flat distribution of half-width ``spread``. The channel responds to each
    slice's centroid. With ``spread = 0`` every particle in a slice moves as
    one, and the centroid model is exact.
    """

    channel_model: HoseChannel
    per_slice: int = 16
    spread: float = 0.0

    def wavenumbers(self) -> np.ndarray:
        u = np.linspace(-1.0, 1.0, self.per_slice) if self.per_slice > 1 else np.zeros(1)
        return self.channel_model.betatron * (1.0 + self.spread * u)

    def solve(self, distances, offset: float = 1.0, rtol: float = 1e-10) -> np.ndarray:
        """The beam centroid ``y_b(xi, s)`` at each distance, from particles offset and at rest."""
        from scipy.integrate import solve_ivp

        model = self.channel_model
        n = len(model.xi)
        k2 = self.wavenumbers()[:, None] ** 2  # (particles, 1)
        m = self.per_slice

        def rhs(_, y):
            position = y[: m * n].reshape(m, n)
            velocity = y[m * n :].reshape(m, n)
            centroid = model.channel_centroid(position.mean(axis=0))
            return np.concatenate([velocity.ravel(), (-k2 * (position - centroid)).ravel()])

        start = np.concatenate([np.full(m * n, float(offset)), np.zeros(m * n)])
        distances = np.atleast_1d(np.asarray(distances, dtype=float))
        solution = solve_ivp(
            rhs,
            (0.0, float(distances.max())),
            start,
            method="DOP853",
            t_eval=distances,
            rtol=rtol,
            atol=rtol * abs(offset) * 1e-3,
        )
        positions = solution.y[: m * n].T.reshape(len(distances), m, n)
        return positions.mean(axis=1)


__all__ = [
    "HoseBeam",
    "HoseChannel",
    "asymptotic_exponent",
    "growth_exponent",
]
