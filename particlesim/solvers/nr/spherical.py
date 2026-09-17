"""Spherically symmetric massless scalar collapse (design doc Section 5.4, M1).

Polar-areal coordinates, following Choptuik 1993 (PRL 70, 9):

    ds^2 = -alpha(t, r)^2 dt^2 + a(t, r)^2 dr^2 + r^2 dOmega^2

with a massless scalar field reduced to first-order variables

    Phi = d(phi)/dr,    Pi = (a / alpha) d(phi)/dt

whose evolution is

    d(Phi)/dt = d(alpha Pi / a)/dr
    d(Pi)/dt  = (1 / r^2) d(r^2 alpha Phi / a)/dr

The metric functions are **not** evolved. They are recovered at every stage
by integrating the Hamiltonian constraint and the polar slicing condition
outward from the origin:

    a'/a     = (1 - a^2) / (2r) + 2 pi r (Pi^2 + Phi^2)
    alpha'/alpha = (a^2 - 1) / (2r) + 2 pi r (Pi^2 + Phi^2)

This is a constrained evolution: the constraints cannot drift because they
are solved, not monitored. The price is a radial ODE solve per stage, which
is cheap in one dimension and buys stability that a free evolution in
spherical symmetry does not give for free.

The factor 2 pi follows from the standard normalization
``S = -(1/2) integral (d phi)^2 sqrt(-g)`` with ``G_ab = 8 pi T_ab``; it is
pinned by a test comparing the ADM mass from ``a`` at the outer boundary
against the scalar field's energy integral, rather than asserted here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from particlesim.core.grid import kreiss_oliger
from particlesim.core.spherical import SphericalGrid


class PolarSlicingBreakdown(RuntimeError):
    """Raised when a trapped region forms, which polar-areal slicing cannot cover."""


def _midpoints(f: np.ndarray) -> np.ndarray:
    """Fourth-order interpolation of ``f`` to cell midpoints.

    Returns ``len(f) - 1`` values. A second-order average here would cap the
    whole metric solve at second order regardless of the Runge-Kutta stage
    count, which is the usual way an ostensibly fourth-order code turns out
    to be second order.
    """
    out = np.empty(len(f) - 1)
    out[1:-1] = (-f[:-3] + 9 * f[1:-2] + 9 * f[2:-1] - f[3:]) / 16.0
    out[0] = (3 * f[0] + 6 * f[1] - f[2]) / 8.0
    out[-1] = (3 * f[-1] + 6 * f[-2] - f[-3]) / 8.0
    return out


@dataclass(frozen=True)
class SphericalState:
    """Scalar field variables and the time they belong to."""

    t: float
    Phi: np.ndarray
    Pi: np.ndarray

    def with_fields(self, Phi: np.ndarray, Pi: np.ndarray, t: float | None = None):
        return replace(self, Phi=Phi, Pi=Pi, t=self.t if t is None else t)


def _d_dr(f: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Fourth-order centred radial derivative on a uniform grid.

    Edge points fall back to one-sided second-order differences, which the
    boundary conditions overwrite anyway.
    """
    dr = r[1] - r[0]
    out = np.empty_like(f)
    out[2:-2] = (-f[4:] + 8 * f[3:-1] - 8 * f[1:-3] + f[:-4]) / (12 * dr)
    out[0] = (-3 * f[0] + 4 * f[1] - f[2]) / (2 * dr)
    out[1] = (f[2] - f[0]) / (2 * dr)
    out[-2] = (f[-1] - f[-3]) / (2 * dr)
    out[-1] = (3 * f[-1] - 4 * f[-2] + f[-3]) / (2 * dr)
    return out


class ScalarCollapse:
    """Evolves a massless scalar field in polar-areal spherical symmetry."""

    def __init__(
        self,
        grid: SphericalGrid,
        courant: float = 0.25,
        dissipation: float = 0.02,
    ):
        if not grid.uniform:
            raise ValueError(
                "the fourth-order radial stencils assume uniform spacing; "
                "a graded grid needs a different discretisation"
            )
        self.grid = grid
        self.r = grid.radii()
        self.dr = grid.dr
        self.courant = courant
        self.dissipation = dissipation

    # --- constraints ----------------------------------------------------

    def solve_metric(self, Phi: np.ndarray, Pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Recover ``a`` and ``alpha`` by integrating outward from the origin.

        Integrating the Misner-Sharp mass rather than ``a`` directly is what
        makes the origin unproblematic. The Hamiltonian constraint written
        for ``a`` carries a ``(1 - a^2) / 2r`` term that is zero over zero at
        ``r = 0``; the equivalent equation for the mass,

            dm/dr = 2 pi r^2 (Pi^2 + Phi^2) (1 - 2m / r)

        has a source vanishing as ``r^2`` and the boundary condition
        ``m(0) = 0``, both manifestly regular. ``a`` then follows
        algebraically. Fourth-order Runge-Kutta in radius keeps the metric
        solve from capping the evolution's spatial order.
        """
        r, dr = self.r, self.dr
        density = Pi**2 + Phi**2
        # Source of the mass equation, S(r) = 2 pi r^2 (Pi^2 + Phi^2).
        S = 2.0 * np.pi * r**2 * density
        S_mid = _midpoints(S)
        r_mid = r + 0.5 * dr
        dens_mid = _midpoints(density)

        m = np.empty_like(r)
        # First half cell: the source behaves as r^2 near the origin, for
        # which the integral from 0 to r0 is exactly S(r0) r0 / 3.
        m[0] = S[0] * r[0] / 3.0

        def dm(rr: float, mm: float, ss: float) -> float:
            return ss * (1.0 - 2.0 * mm / rr)

        for i in range(len(r) - 1):
            k1 = dm(r[i], m[i], S[i])
            k2 = dm(r_mid[i], m[i] + 0.5 * dr * k1, S_mid[i])
            k3 = dm(r_mid[i], m[i] + 0.5 * dr * k2, S_mid[i])
            k4 = dm(r[i + 1], m[i] + dr * k3, S[i + 1])
            m[i + 1] = m[i] + dr / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        two_m_over_r = 2.0 * m / r
        if np.any(two_m_over_r >= 1.0):
            raise PolarSlicingBreakdown(
                f"2m/r reached {two_m_over_r.max():.4f} at r = "
                f"{float(r[int(np.argmax(two_m_over_r))]):.4f}: polar-areal "
                "coordinates do not cover a trapped region, so the evolution "
                "cannot continue past horizon formation in this slicing"
            )
        a = 1.0 / np.sqrt(1.0 - two_m_over_r)

        # d(ln alpha)/dr = m / (r^2 (1 - 2m/r)) + 2 pi r (Pi^2 + Phi^2).
        def dlog_alpha(rr, mm, dd):
            return mm / (rr**2 * (1.0 - 2.0 * mm / rr)) + 2.0 * np.pi * rr * dd

        log_alpha = np.empty_like(r)
        # Near the origin the integrand is linear in r, so the first half
        # cell integrates to integrand(r0) * r0 / 2.
        log_alpha[0] = 0.5 * r[0] * dlog_alpha(r[0], m[0], density[0])
        m_mid = 0.5 * (m[:-1] + m[1:])
        for i in range(len(r) - 1):
            k1 = dlog_alpha(r[i], m[i], density[i])
            k2 = dlog_alpha(r_mid[i], m_mid[i], dens_mid[i])
            k4 = dlog_alpha(r[i + 1], m[i + 1], density[i + 1])
            log_alpha[i + 1] = log_alpha[i] + dr / 6.0 * (k1 + 4 * k2 + k4)

        alpha = np.exp(log_alpha - log_alpha[-1]) / a[-1]
        return a, alpha

    def mass_aspect(self, a: np.ndarray) -> np.ndarray:
        """Misner-Sharp mass ``m(r) = (r / 2) (1 - 1 / a^2)``."""
        return 0.5 * self.r * (1.0 - 1.0 / a**2)

    def adm_mass(self, a: np.ndarray) -> float:
        return float(self.mass_aspect(a)[-1])

    def energy_integral(self, Phi: np.ndarray, Pi: np.ndarray) -> float:
        """``2 pi integral r^2 (Pi^2 + Phi^2) dr``, the weak-field ADM mass."""
        return float(2.0 * np.pi * np.sum(self.r**2 * (Pi**2 + Phi**2)) * self.dr)

    # --- evolution ------------------------------------------------------

    def rhs(self, state: SphericalState) -> tuple[np.ndarray, np.ndarray]:
        r = self.r
        Phi, Pi = state.Phi, state.Pi
        a, alpha = self.solve_metric(Phi, Pi)
        f = alpha / a

        dPhi = _d_dr(f * Pi, r)
        # Write the flux term as (1/r^2) d(r^2 f Phi)/dr expanded, which keeps
        # the r^2 factors from cancelling to round-off at large radius.
        dPi = _d_dr(f * Phi, r) + 2.0 * f * Phi / r

        if self.dissipation > 0:
            dPhi = dPhi + kreiss_oliger(Phi, 0, self.dr, order=4, epsilon=self.dissipation)
            dPi = dPi + kreiss_oliger(Pi, 0, self.dr, order=4, epsilon=self.dissipation)

        # Outgoing at the outer boundary: both variables fall off as 1/r on
        # an outgoing null ray, so d(f)/dt = -d(f)/dr - f/r.
        for arr, d in ((Phi, dPhi), (Pi, dPi)):
            d[-2:] = -_d_dr(arr, r)[-2:] - arr[-2:] / r[-2:]
        return dPhi, dPi

    def step(self, state: SphericalState, dt: float) -> SphericalState:
        """One classical fourth-order Runge-Kutta step."""
        k1 = self.rhs(state)
        s2 = state.with_fields(state.Phi + 0.5 * dt * k1[0], state.Pi + 0.5 * dt * k1[1])
        k2 = self.rhs(s2)
        s3 = state.with_fields(state.Phi + 0.5 * dt * k2[0], state.Pi + 0.5 * dt * k2[1])
        k3 = self.rhs(s3)
        s4 = state.with_fields(state.Phi + dt * k3[0], state.Pi + dt * k3[1])
        k4 = self.rhs(s4)
        Phi = state.Phi + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        Pi = state.Pi + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        return SphericalState(state.t + dt, Phi, Pi)

    @property
    def dt(self) -> float:
        return self.courant * self.dr

    def horizon_radius(self, a: np.ndarray, threshold: float = 0.99) -> float | None:
        """Smallest radius where ``2m/r`` exceeds ``threshold``, or ``None``.

        Polar-areal slicing is horizon-avoiding: a trapped surface never
        forms in finite coordinate time. What happens instead is that
        ``2m/r`` asymptotes to one from below while the lapse collapses, so
        this reports an approach to a horizon rather than a crossing, and
        the threshold is a convention rather than a measurement. Pair it
        with :meth:`lapse_collapsed` before calling a run a black hole.
        """
        two_m_over_r = 1.0 - 1.0 / a**2
        idx = np.nonzero(two_m_over_r > threshold)[0]
        return float(self.r[idx[0]]) if idx.size else None

    @staticmethod
    def lapse_collapsed(alpha: np.ndarray, threshold: float = 1e-2) -> bool:
        """Whether the central lapse has collapsed, the signature of collapse
        in this slicing."""
        return bool(alpha.min() < threshold)


def gaussian_pulse(
    grid: SphericalGrid, amplitude: float, r0: float, width: float, ingoing: bool = False
) -> SphericalState:
    """Time-symmetric (or ingoing) Gaussian shell in ``phi``.

    ``phi = A r^2 exp(-((r - r0)/width)^2)`` has the right parity at the
    origin, so ``Phi`` is odd and regular there.
    """
    r = grid.radii()
    phi = amplitude * r**2 * np.exp(-(((r - r0) / width) ** 2))
    Phi = _d_dr(phi, r)
    Pi = -Phi if ingoing else np.zeros_like(r)
    return SphericalState(0.0, Phi, Pi)
