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
from particlesim.core.interpolate import midpoints
from particlesim.core.spherical import SphericalGrid
from particlesim.solvers.nr.polar import (
    PolarSlicingBreakdown,
    mass_aspect,
    normalise_lapse,
    solve_linear_mass,
)

# Re-exported: the exception was defined here before the metric solve moved
# to :mod:`particlesim.solvers.nr.polar`, and callers catch it by this name.
__all__ = [
    "PolarSlicingBreakdown",
    "ScalarCollapse",
    "SphericalState",
    "gaussian_pulse",
]


@dataclass(frozen=True)
class SphericalState:
    """Scalar field variables and the time they belong to."""

    t: float
    Phi: np.ndarray
    Pi: np.ndarray

    def with_fields(self, Phi: np.ndarray, Pi: np.ndarray, t: float | None = None):
        return replace(self, Phi=Phi, Pi=Pi, t=self.t if t is None else t)


def _d_dr(f: np.ndarray, r: np.ndarray, parity: int | None = None) -> np.ndarray:
    """Fourth-order centred radial derivative on a uniform grid.

    ``parity`` is +1 for a field that is even across the origin and -1 for one
    that is odd. Given it, the innermost two points use the same fourth-order
    centred stencil as everywhere else, reading its two inner neighbours from
    the reflection ``f(-r) = parity * f(r)``. The grid is cell-centred, so
    ``r_{-1} = -r_0`` and ``r_{-2} = -r_1`` and the reflected values are exact
    rather than extrapolated.

    Without it the two points nearest the origin fall back to one-sided
    second-order differences. That is not merely a local drop in accuracy: a
    one-sided stencil has no definite parity, so it returns a small even
    component for an odd field, and the spherical flux term ``2 f Phi / r``
    divides that component by ``r ~ dr/2``. The result is an origin
    instability that stays invisible for several light-crossing times and
    then grows without bound, which is exactly the regime a collapse
    threshold search lives in.

    The outermost two points keep one-sided differences, which the outgoing
    boundary condition overwrites anyway.
    """
    dr = r[1] - r[0]
    out = np.empty_like(f)
    out[2:-2] = (-f[4:] + 8 * f[3:-1] - 8 * f[1:-3] + f[:-4]) / (12 * dr)
    if parity is None:
        out[0] = (-3 * f[0] + 4 * f[1] - f[2]) / (2 * dr)
        out[1] = (f[2] - f[0]) / (2 * dr)
    else:
        if parity not in (1, -1):
            raise ValueError("parity must be +1 (even), -1 (odd), or None")
        s = float(parity)
        # Ghosts through the origin: f[-1] = s f[0], f[-2] = s f[1].
        out[0] = (-f[2] + 8 * f[1] - 8 * s * f[0] + s * f[1]) / (12 * dr)
        out[1] = (-f[3] + 8 * f[2] - 8 * f[0] + s * f[0]) / (12 * dr)
    out[-2] = (f[-1] - f[-3]) / (2 * dr)
    out[-1] = (3 * f[-1] - 4 * f[-2] + f[-3]) / (2 * dr)
    return out


def _dissipate(f: np.ndarray, dr: float, parity: int, epsilon: float) -> np.ndarray:
    """Kreiss-Oliger dissipation that reaches the origin.

    :func:`particlesim.core.grid.kreiss_oliger` returns zero within the
    stencil radius of either end, because a one-sided dissipation operator
    injects the noise it is meant to remove. At an outer boundary that is
    the right call. At the origin it is not, because the origin is not a
    boundary: the solution continues through it with a known parity, so the
    missing neighbours are available exactly. Leaving the innermost three
    cells undissipated leaves the shortest grid wavelength undamped in the
    one place where the ``1/r`` terms amplify it.
    """
    g = 3  # stencil radius of the fourth-order operator, order // 2 + 1
    extended = np.concatenate([parity * f[g - 1 :: -1], f])
    return kreiss_oliger(extended, 0, dr, order=4, epsilon=epsilon)[g:]


class ScalarCollapse:
    """Evolves a massless scalar field in polar-areal spherical symmetry."""

    #: Default Kreiss-Oliger coefficient.
    #:
    #: **Not a stability requirement.** It was once: with the ``Pi`` equation
    #: written expanded, the flux term ``2 f Phi / r`` grew short-wavelength
    #: error at the innermost cell at a rate ``~ 1/dr`` that only damping at
    #: ``epsilon/dr`` could hold, so the coefficient had a floor near 0.1 that
    #: no refinement removed -- below it a weak pulse ended with three times
    #: the mass it started with, and a near-critical bounce was unstable at
    #: 0.2. Written conservatively the pair conserves the discrete energy (see
    #: :meth:`rhs`) and that pulse is clean with no dissipation at all.
    #:
    #: What it still does is damp what the grid cannot represent. After a
    #: bounce 0.4% below threshold at 400 cells, the remains of the last echo
    #: linger at the origin at a curvature of 37 with no dissipation, 2.0 at
    #: 0.1 and 0.11 at 0.2, against 5e-6 at 1600 cells where they are
    #: resolved. None of it makes mass. The physical peak of a strong bounce
    #: agrees at 0.05, 0.1 and 0.2 to four figures from 800 cells up.
    DEFAULT_DISSIPATION = 0.1

    def __init__(
        self,
        grid: SphericalGrid,
        courant: float = 0.25,
        dissipation: float | None = None,
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
        self.dissipation = self.DEFAULT_DISSIPATION if dissipation is None else float(dissipation)

    # --- constraints ----------------------------------------------------

    def solve_metric(self, Phi: np.ndarray, Pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Recover ``a`` and ``alpha`` from the scalar field's stress-energy.

        A massless scalar's Eulerian energy density and radial stress are
        both ``(Pi^2 + Phi^2)/(2 a^2)``, so its mass equation is
        ``dm/dr = 2 pi r^2 (Pi^2 + Phi^2)(1 - 2m/r)`` and its lapse equation
        ``d(ln alpha)/dr = m/(r^2 (1 - 2m/r)) + 2 pi r (Pi^2 + Phi^2)``.

        **The arithmetic is** :func:`~particlesim.solvers.nr.polar.solve_mass`
        **and** :func:`~particlesim.solvers.nr.polar.solve_lapse` **exactly,
        vectorised.** The mass equation is linear in ``m``, so each
        Runge-Kutta step is an affine map and the outward pass is a linear
        recurrence (:func:`~particlesim.solvers.nr.polar.solve_linear_mass`);
        once the mass is known the lapse is a quadrature and Simpson's rule
        is a cumulative sum. The midpoint mass is the same cubic Hermite, for
        the same reason. The result agrees with the loop to rounding, refuses
        where it refused, and is roughly fifteen times faster -- which
        matters because this solve was ninety per cent of every step, and a
        refinement hierarchy calls it on every level at every stage.
        """
        r, h = self.r, self.dr
        density = Pi**2 + Phi**2
        source = 2.0 * np.pi * r**2 * density
        mass = solve_linear_mass(r, h, source, midpoints(source))
        free = 1.0 - 2.0 * mass / r

        middle = r[:-1] + 0.5 * h
        derivative = source * free
        mass_mid = 0.5 * (mass[:-1] + mass[1:]) + h / 8.0 * (derivative[:-1] - derivative[1:])
        node = mass / (r**2 * free) + 2.0 * np.pi * r * density
        mid = mass_mid / (middle**2 * (1.0 - 2.0 * mass_mid / middle)) + (
            2.0 * np.pi * middle * midpoints(density)
        )
        logarithm = 0.5 * r[0] * node[0] + np.concatenate(
            [[0.0], np.cumsum(h / 6.0 * (node[:-1] + 4.0 * mid + node[1:]))]
        )
        return 1.0 / np.sqrt(free), normalise_lapse(logarithm, float(mass[-1]), float(r[-1]))

    def mass_aspect(self, a: np.ndarray) -> np.ndarray:
        """Misner-Sharp mass ``m(r) = (r / 2) (1 - 1 / a^2)``."""
        return mass_aspect(self.r, a)

    def adm_mass(self, a: np.ndarray) -> float:
        return float(self.mass_aspect(a)[-1])

    def energy_integral(self, Phi: np.ndarray, Pi: np.ndarray) -> float:
        """``2 pi integral r^2 (Pi^2 + Phi^2) dr``, the weak-field ADM mass."""
        return float(2.0 * np.pi * np.sum(self.r**2 * (Pi**2 + Phi**2)) * self.dr)

    # --- evolution ------------------------------------------------------

    def rhs(
        self,
        state: SphericalState,
        metric: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(dPhi/dt, dPi/dt)``, solving for the metric unless one is supplied.

        ``metric`` exists for refinement levels. A level's own solve gets
        ``a`` right, because the mass is local, but normalises the lapse at
        the level's own outer edge as though that edge were the asymptotic
        boundary. It is not, unless the matter happens to lie entirely
        inside it; see :mod:`particlesim.solvers.nr.subcycle`.
        """
        r = self.r
        Phi, Pi = state.Phi, state.Pi
        a, alpha = self.solve_metric(Phi, Pi) if metric is None else metric
        f = alpha / a

        # Phi is odd across the origin and Pi is even; alpha, a and so f are
        # even. The products therefore carry the parities passed here, and
        # their derivatives come out even and odd respectively, as the
        # evolution equations require.
        dPhi = _d_dr(f * Pi, r, parity=1)
        # Conservative, not expanded, and that is the whole of the origin's
        # stability. With the reflection ghosts of a cell-centred grid the
        # odd-parity stencil is exactly minus the transpose of the even one,
        # so (1/r^2) D(r^2 g) is minus the adjoint of D in the sum over r^2,
        # and with f = 1 the pair conserves sum r^2 (Phi^2 + Pi^2) exactly.
        # Expanded as D(g) + 2 g / r it is not the adjoint, and the defect is
        # a growth rate ~ 1/dr at the origin; see test_spherical_origin.py.
        # The price is accuracy at the innermost cells: the stencil's error on
        # an r^5 term is divided by r^2 ~ dr^2 / 4 there, so the first few are
        # locally second order. Measured through the ADM mass drift the scheme
        # is still fourth order, and by that measure six to ten thousand times
        # more accurate than the expanded form at the same resolution.
        dPi = _d_dr(r**2 * f * Phi, r, parity=-1) / r**2

        if self.dissipation > 0:
            dPhi = dPhi + _dissipate(Phi, self.dr, -1, self.dissipation)
            dPi = dPi + _dissipate(Pi, self.dr, 1, self.dissipation)

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

    ``ingoing`` sets ``Pi = Phi``, which is the sign that makes the shell
    travel toward the origin. Writing the flat-space system in characteristic
    variables ``w+- = Phi +- Pi`` gives

        d(w+)/dt = +d(w+)/dr + source,   d(w-)/dt = -d(w-)/dr + source

    so ``w+`` is the ingoing mode and ``w-`` the outgoing one. A purely
    ingoing pulse is ``w- = 0``, that is ``Pi = +Phi``. The opposite sign
    kills ``w+`` instead and sends the shell outward, which is the one thing
    a collapse study cannot afford: the energy leaves the grid before it can
    focus, no amplitude reaches threshold, and the run looks subcritical at
    every amplitude the initial data can express.
    """
    r = grid.radii()
    phi = amplitude * r**2 * np.exp(-(((r - r0) / width) ** 2))
    Phi = _d_dr(phi, r, parity=1)
    Pi = Phi.copy() if ingoing else np.zeros_like(r)
    return SphericalState(0.0, Phi, Pi)
