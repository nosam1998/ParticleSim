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
from particlesim.solvers.nr.polar import (
    PolarSlicingBreakdown,
    mass_aspect,
    midpoints,
    solve_polar_metric,
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
    #: Measured, not guessed. The spherical flux term ``2 f Phi / r``
    #: amplifies short-wavelength error at the innermost cell at a rate that
    #: goes as ``1 / r ~ 2 / dr``, while KO damps it at ``epsilon / dr``. The
    #: two therefore compete at a value of ``epsilon`` that does not shrink
    #: with the grid, so the coefficient has a floor no amount of refinement
    #: removes. Below it a run looks clean for a couple of light-crossing
    #: times and then grows an origin mode that *creates* ADM mass: at
    #: ``epsilon = 0.02`` a weak pulse that should disperse to nothing
    #: instead ends with three times the mass it started with. At 0.1 and at
    #: 0.2 the physical peak agrees to four figures and the late-time origin
    #: is quiet, which is what says the dissipation is removing noise rather
    #: than solution.
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

        The integration itself lives in
        :mod:`particlesim.solvers.nr.polar`, which is shared with the
        fluid; what belongs here is the *source*. A massless scalar's
        Eulerian energy density and radial stress are both
        ``(Pi^2 + Phi^2)/(2 a^2)``, so its mass equation carries a factor
        ``1 - 2m/r`` -- the source depends on the mass being integrated
        for -- while its lapse equation does not. That asymmetry is why the
        shared solver takes callables rather than arrays.
        """
        density = Pi**2 + Phi**2
        source = 2.0 * np.pi * self.r**2 * density
        source_mid = midpoints(source)
        density_mid = midpoints(density)

        def mass_slope(index, mid, radius, mass):
            value = source_mid[index] if mid else source[index]
            return value * (1.0 - 2.0 * mass / radius)

        def lapse_slope(index, mid, radius, mass):
            value = density_mid[index] if mid else density[index]
            return mass / (radius**2 * (1.0 - 2.0 * mass / radius)) + 2.0 * np.pi * radius * value

        a, alpha, _ = solve_polar_metric(self.r, self.dr, mass_slope, lapse_slope)
        return a, alpha

    def mass_aspect(self, a: np.ndarray) -> np.ndarray:
        """Misner-Sharp mass ``m(r) = (r / 2) (1 - 1 / a^2)``."""
        return mass_aspect(self.r, a)

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

        # Phi is odd across the origin and Pi is even; alpha, a and so f are
        # even. The products therefore carry the parities passed here, and
        # their derivatives come out even and odd respectively, as the
        # evolution equations require.
        dPhi = _d_dr(f * Pi, r, parity=1)
        # Write the flux term as (1/r^2) d(r^2 f Phi)/dr expanded, which keeps
        # the r^2 factors from cancelling to round-off at large radius.
        dPi = _d_dr(f * Phi, r, parity=-1) + 2.0 * f * Phi / r

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
