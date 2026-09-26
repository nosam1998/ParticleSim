"""Einstein-scalar-Gauss-Bonnet at the leading order of order reduction (issue #52).

Einstein-scalar-Gauss-Bonnet gravity has no well-posed evolution in the
gauges the 3-D solvers use, which is why ``string.eft4d.dgb`` declares
``order_reduced`` and ADR-008 refuses it a strong-field 3-D run. Order
reduction is the standard way around that: expand in the coupling, and solve
at each order equations that are well posed because the higher-derivative
terms only ever act on the solution of the order below. At leading order the
metric is General Relativity's, and the scalar obeys

    box phi = -(lambda^2 / 4) f'(phi) G

on it. That is the *decoupling limit*: the scalar feels the Gauss-Bonnet
invariant of a fixed black hole, and the hole does not feel the scalar. This
module evolves it on Schwarzschild in spherical symmetry, for any coupling
function, which is the first evolution that uses the ``order_reduced``
formulation.

**The equation.** With ``psi = r phi``, the tortoise coordinate
``r* = r + 2M ln(r/2M - 1)`` and ``G = 48 M^2 / r^6``,

    d_t^2 psi = d_r*^2 psi - (1 - 2M/r)(2M/r^3) psi
                + r (1 - 2M/r) (lambda^2/4) f'(psi/r) G

The horizon is at ``r* -> -inf``, where the potential and the source vanish
exponentially. So a wave leaving through either end obeys
``d_t psi = -+ d_r* psi`` there exactly, and the ends are characteristic. The
interior is fourth-order finite differences, classical Runge-Kutta, and
sixth-difference Kreiss-Oliger dissipation.

**Four checks, each against something computed another way.**
- **A linear coupling relaxes to the closed-form hair.** For
  ``f'(phi) = alpha`` the static solution regular at the horizon is
  ``phi = (alpha lambda^2 / 2M)(1/r + M/r^2 + 4M^2/3r^3)``.
- **Scalarization starts where the static zero mode appears.** The lowest
  eigenvalue of the linearised operator crosses zero at the ``M/lambda`` that
  :func:`~particlesim.theories.gauss_bonnet.bifurcation_points` finds by
  shooting a different equation.
- **Below that, the scalar grows at the bound state's rate.** Growth in the
  evolution matches ``sqrt(-E_0)`` from the eigenvalue problem.
- **Nonlinearly, it saturates on the static scalarized profile.** The
  evolution's late profile matches :func:`static_scalar`, found by shooting.

**What leading order leaves out** is the scalar's pull on the metric. At the
threshold the scalar is infinitesimal, so the onset is exact, and it is the
same ``M/lambda`` the fully coupled static equations give. The endpoint is
not: at ``M/lambda = 0.5`` the decoupled scalar settles at ``phi_H = 0.291``,
and the backreacted hole of the same mass has 0.406.

The measurements are in ``docs/benchmarks.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import sympy as sp
from scipy.special import lambertw

from particlesim.theories.gauss_bonnet import _PHI, GaussBonnetCoupling


def tortoise(radius, mass: float = 1.0):
    """``r* = r + 2M ln(r/2M - 1)``, outside the horizon."""
    radius = np.asarray(radius, dtype=float)
    return radius + 2 * mass * np.log(radius / (2 * mass) - 1)


def areal_radius(tortoise_coordinate, mass: float = 1.0):
    """The inverse of :func:`tortoise`: ``r = 2M (1 + W(exp(r*/2M - 1)))``."""
    x = np.asarray(tortoise_coordinate, dtype=float)
    return 2 * mass * (1 + np.real(lambertw(np.exp(x / (2 * mass) - 1))))


def gauss_bonnet_schwarzschild(radius, mass: float = 1.0):
    """``G = R_abcd R^abcd = 48 M^2 / r^6`` on Schwarzschild."""
    return 48 * mass**2 / np.asarray(radius, dtype=float) ** 6


def static_hair(radius, slope: float, coupling_squared: float, mass: float = 1.0):
    """The decoupled scalar of a linear coupling ``f'(phi) = slope``, regular at the horizon.

    ``(alpha lambda^2 / 2M)(1/r + M/r^2 + 4M^2/3r^3)``, from integrating
    ``(r^2 (1 - 2M/r) phi')' = -12 alpha lambda^2 M^2 / r^4`` once and
    requiring the horizon to be regular.
    """
    r = np.asarray(radius, dtype=float)
    return slope * coupling_squared / (2 * mass) * (1 / r + mass / r**2 + 4 * mass**2 / (3 * r**3))


def _second(values, spacing):
    out = np.zeros_like(values)
    out[2:-2] = (
        -values[4:] + 16 * values[3:-1] - 30 * values[2:-2] + 16 * values[1:-3] - values[:-4]
    ) / (12 * spacing**2)
    return out


def _first_at_ends(values, spacing):
    """Fourth-order one-sided first derivatives at the two outermost points of each end."""
    a, h = values, spacing
    return (
        (-25 * a[0] + 48 * a[1] - 36 * a[2] + 16 * a[3] - 3 * a[4]) / (12 * h),
        (-3 * a[0] - 10 * a[1] + 18 * a[2] - 6 * a[3] + a[4]) / (12 * h),
        (3 * a[-1] + 10 * a[-2] - 18 * a[-3] + 6 * a[-4] - a[-5]) / (12 * h),
        (25 * a[-1] - 48 * a[-2] + 36 * a[-3] - 16 * a[-4] + 3 * a[-5]) / (12 * h),
    )


def _dissipation(values, spacing):
    """``+ delta^6 / 64 h``: Kreiss-Oliger for a fourth-order scheme, negative definite.

    The sign is the one to check: ``(D+ D-)^3`` is negative, so the operator
    is ``+ h^5 (D+ D-)^3 / 64``. With the sign flipped it pumps the shortest
    wavelengths instead, and a free scalar on Schwarzschild grew from
    round-off to 1e+14 in 200 M.
    """
    out = np.zeros_like(values)
    v = values
    out[3:-3] = (
        v[6:] - 6 * v[5:-1] + 15 * v[4:-2] - 20 * v[3:-3] + 15 * v[2:-4] - 6 * v[1:-5] + v[:-6]
    ) / (64 * spacing)
    return out


@dataclass(frozen=True)
class DecouplingLimit:
    """The scalar of Einstein-scalar-Gauss-Bonnet on a fixed Schwarzschild hole.

    ``coupling_squared`` is ``lambda^2``, in the units ``mass`` is in. The
    grid is uniform in the tortoise coordinate from ``inner`` to ``outer``.
    """

    coupling: GaussBonnetCoupling
    coupling_squared: float
    mass: float = 1.0
    inner: float = -200.0
    outer: float = 300.0
    points: int = 2001
    dissipation: float = 0.1

    @classmethod
    def from_theory(cls, theory, mass: float = 1.0, **grid) -> DecouplingLimit:
        """For a theory plugin with a ``coupling_function`` and an ``alpha = lambda^2``."""
        return cls(theory.coupling_function, float(theory.values["alpha"]), mass, **grid)

    @property
    def tortoise(self) -> np.ndarray:
        return np.linspace(self.inner, self.outer, self.points)

    @property
    def spacing(self) -> float:
        return (self.outer - self.inner) / (self.points - 1)

    @property
    def radius(self) -> np.ndarray:
        return areal_radius(self.tortoise, self.mass)

    @cached_property
    def _background(self):
        """``(r, 1 - 2M/r, the l = 0 potential, (lambda^2/4) r (1 - 2M/r) G, f')`` on the grid."""
        r = self.radius
        lapse = 1 - 2 * self.mass / r
        weight = (self.coupling_squared / 4) * r * lapse * gauss_bonnet_schwarzschild(r, self.mass)
        slope = sp.lambdify(_PHI, sp.diff(self.coupling.expression, _PHI), "numpy")
        return r, lapse, lapse * 2 * self.mass / r**3, weight, slope

    def rates(self, psi, pi):
        """``(d_t psi, d_t pi)``, with characteristic ends and dissipation."""
        h = self.spacing
        r, _, potential, weight, slope = self._background
        source = weight * slope(psi / r)
        d_psi = pi + self.dissipation * _dissipation(psi, h)
        d_pi = _second(psi, h) - potential * psi + source + self.dissipation * _dissipation(pi, h)
        # Leaving through the horizon end, d_t = +d_r*; through the outer end, d_t = -d_r*.
        for field, rate in ((psi, d_psi), (pi, d_pi)):
            left0, left1, right1, right0 = _first_at_ends(field, h)
            rate[0], rate[1], rate[-2], rate[-1] = left0, left1, -right1, -right0
        return d_psi, d_pi

    def evolve(
        self, initial, final: float, probes=(0.0,), every: float = 1.0, courant: float = 0.25
    ):
        """Evolve ``psi = initial(r*, r)`` from rest to ``final``.

        Returns ``(times, samples, psi)``: ``samples[k]`` is ``psi`` at the
        tortoise coordinates in ``probes`` at ``times[k]``, and ``psi`` is the
        final field.
        """
        grid = self.tortoise
        psi = np.asarray(initial(grid, self.radius), dtype=float)
        pi = np.zeros_like(psi)
        steps = int(round(final / (courant * self.spacing)))
        step = final / steps
        stride = max(1, int(round(every / step)))
        index = [int(np.argmin(np.abs(grid - p))) for p in probes]
        times, samples = [0.0], [psi[index].copy()]
        for count in range(1, steps + 1):
            a1, b1 = self.rates(psi, pi)
            a2, b2 = self.rates(psi + step / 2 * a1, pi + step / 2 * b1)
            a3, b3 = self.rates(psi + step / 2 * a2, pi + step / 2 * b2)
            a4, b4 = self.rates(psi + step * a3, pi + step * b3)
            psi = psi + step / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
            pi = pi + step / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
            if count % stride == 0:
                times.append(count * step)
                samples.append(psi[index].copy())
        return np.array(times), np.array(samples), psi


def lowest_eigenvalue(
    coupling: GaussBonnetCoupling,
    coupling_squared: float,
    mass: float = 1.0,
    inner: float = -400.0,
    outer: float = 400.0,
    points: int = 8001,
) -> float:
    """``E_0`` of ``-d_r*^2 + V_eff`` for the scalar linearised about zero.

    ``V_eff = (1 - 2M/r)(2M/r^3 - (lambda^2/4) f''(0) G)``. Negative means a
    bound state, which grows as ``exp(sqrt(-E_0) t)``. The ends are Neumann,
    which is the condition the static zero mode obeys at both: ``psi`` tends
    to ``2M phi_H`` at the horizon and to a constant far away, the
    ``phi + r phi' -> 0`` of :func:`~particlesim.theories.gauss_bonnet.bifurcation_points`.
    Dirichlet ends would demand a bound state inside the box, and put the
    threshold at 0.555 instead of 0.587.
    """
    from scipy.linalg import eigh_tridiagonal

    grid = np.linspace(inner, outer, points)
    h = grid[1] - grid[0]
    r = areal_radius(grid, mass)
    lapse = 1 - 2 * mass / r
    potential = lapse * (
        2 * mass / r**3
        - (coupling_squared / 4) * coupling.curvature_at_zero * gauss_bonnet_schwarzschild(r, mass)
    )
    diagonal = 2 / h**2 + potential
    diagonal[0] = 1 / h**2 + potential[0]
    diagonal[-1] = 1 / h**2 + potential[-1]
    return float(
        eigh_tridiagonal(
            diagonal,
            -np.ones(points - 1) / h**2,
            select="i",
            select_range=(0, 0),
            eigvals_only=True,
        )[0]
    )


def growth_rate(
    coupling: GaussBonnetCoupling, coupling_squared: float, mass: float = 1.0, **grid
) -> float:
    """``sqrt(-E_0)`` if the linearised scalar has a bound state, else zero."""
    energy = lowest_eigenvalue(coupling, coupling_squared, mass, **grid)
    return float(np.sqrt(-energy)) if energy < 0 else 0.0


def threshold(coupling: GaussBonnetCoupling, **grid) -> float:
    """``M/lambda`` below which the linearised scalar has a bound state."""
    from scipy.optimize import brentq

    curvature = coupling.curvature_at_zero
    if curvature <= 0:
        raise ValueError(f"f''(0) = {curvature}: no tachyonic mass, no threshold")
    root = brentq(
        lambda c: lowest_eigenvalue(coupling, c / curvature, 1.0, **grid), 0.5, 20.0, xtol=1e-12
    )
    return float(1.0 / np.sqrt(root / curvature))


def static_scalar(
    coupling: GaussBonnetCoupling,
    coupling_squared: float,
    mass: float = 1.0,
    search=(0.005, 1.0),
    samples: int = 100,
):
    """``(phi_H, profile)`` of nonzero static decoupled scalars, found by shooting on ``phi_H``.

    ``(r^2 (1 - 2M/r) phi')' = -(lambda^2/4) f'(phi) r^2 G``, regular at the
    horizon, and ``phi + r phi' -> 0`` far away. ``profile(r)`` is ``phi``.
    Returns every root in ``search``, smallest first.
    """
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    slope = sp.lambdify(_PHI, sp.diff(coupling.expression, _PHI), "numpy")
    m = mass

    def shoot(horizon_value):
        epsilon = 1e-6
        start = 2 * m + epsilon
        # At the horizon, 2M phi' = -(lambda^2/4) f'(phi) (2M)^2 G(2M).
        derivative = (
            -(coupling_squared / 4)
            * slope(horizon_value)
            * 4
            * m**2
            * 48
            * m**2
            / ((2 * m) ** 6 * 2 * m)
        )

        def rhs(r, y):
            phi, flux = y  # flux = r^2 (1 - 2M/r) phi'
            return [
                flux / (r * r - 2 * m * r),
                -(coupling_squared / 4) * slope(phi) * 48 * m**2 / r**4,
            ]

        return solve_ivp(
            rhs,
            (start, 4000 * m),
            [horizon_value + derivative * epsilon, (start**2 - 2 * m * start) * derivative],
            rtol=1e-11,
            atol=1e-13,
            dense_output=True,
        )

    def far(horizon_value):
        solution = shoot(horizon_value)
        phi, flux = solution.y[:, -1]
        r = solution.t[-1]
        return phi + flux / (r - 2 * m)

    grid = np.linspace(*search, samples)
    values = [far(x) for x in grid]
    roots = [
        brentq(far, grid[k], grid[k + 1], xtol=1e-13)
        for k in range(samples - 1)
        if np.sign(values[k]) != np.sign(values[k + 1])
    ]
    return [(root, (lambda s: lambda r: s.sol(r)[0])(shoot(root))) for root in roots]


__all__ = [
    "DecouplingLimit",
    "areal_radius",
    "gauss_bonnet_schwarzschild",
    "growth_rate",
    "lowest_eigenvalue",
    "static_hair",
    "static_scalar",
    "threshold",
    "tortoise",
]
