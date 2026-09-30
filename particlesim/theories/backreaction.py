"""Einstein-scalar-Gauss-Bonnet at second order in the coupling: the hair's pull on the hole (#52).

:mod:`~particlesim.theories.decoupling` is leading order: the scalar moves
on a fixed Schwarzschild hole. This module is the next order, where the
metric answers.

**The counting.** For a coupling with ``f'(0) != 0``, which includes the
dilatonic ``e^(-2 phi)`` of ``string.eft4d.dgb``, the small parameter is
``lambda^2 / M^2``. The scalar is first order,

    box phi = -(lambda^2 / 4) f'(0) G

on Schwarzschild, and the metric's correction is second order, sourced by
the scalar's stress and by the Gauss-Bonnet term linear in the scalar. A
scalarization coupling, ``f'(0) = 0``, has no hair at any order of this
expansion: :class:`SecondOrder` refuses it, and says why.

**No new evolution is needed.** In spherical symmetry there are no
gravitational waves, so the metric's correction is not dynamical. In the
polar-areal gauge

    ds^2 = -e^(2 delta) (1 - 2m/r) dt^2 + dr^2 / (1 - 2m/r) + r^2 dOmega^2,
    m = M + mu

two radial equations fix it at each instant. ``m`` is the Misner-Sharp
mass, ``(r/2)(1 - grad r . grad r)``, which is geometric. With
``k = 4 M lambda^2 f'(0)``, the tortoise coordinate ``r*`` and ``N = 1 - 2M/r``:

    d_r* mu  = (r^2/2)(phi_r*^2 + phi_t^2) + (k/r^3) [r^2 phi_r*r* - (r - M) phi_r*]
    d_r delta = r (phi_r^2 + phi_t^2/N^2) + (k/r^2) (phi_rr + phi_tt/N^2)
    d_t mu   = r^2 phi_t phi_r* + k [phi_tr*/r - M phi_t/r^3]

:func:`second_order_equations` derives the first two, and the scalar's
equation, from the action reduced to ``(t, r)``. The third, the energy flux,
follows from them and the scalar's equation by the contracted Bianchi
identity. Its Gauss-Bonnet part is a total time derivative.

**The static hole, in closed form.** On the static hair, with ``mu(2M) = 0``,

    mu(r) = f'(0)^2 lambda^4 (r - 2M)(147 r^5 + 174 M r^4 + 228 M^2 r^3
            - 1624 M^3 r^2 - 3488 M^4 r - 7360 M^5) / (960 M^3 r^6)

so a hole of horizon radius ``r_H`` has mass ``r_H/2 + (49/40) f'(0)^2
lambda^4 / r_H^3``. :func:`static_mass` and :func:`static_temperature`
give it and its temperature. Two checks are independent of this derivation:

- **The full nonlinear static hole.** :func:`~particlesim.theories.gauss_bonnet.static_black_hole`
  shoots different equations, nonlinearly, from the horizon. Its mass
  approaches this one as the coupling shrinks, at the rate the counting
  says.
- **The first law at this order.** ``dM = T dS`` with Wald's entropy
  ``pi r_H^2 + 4 pi lambda^2 f(phi_H)`` holds identically. The temperature
  shift's coefficient, ``-1/60``, is a cancellation of 441 against 440, so
  the first law is a sharp test of it.

**Growing hair.** :meth:`SecondOrder.run` starts the scalar from rest on
Schwarzschild, lets the hair grow, and books the energy through both ends.
The energy between them must change only by those fluxes, which checks the
flux against the constraint. The hole ends up lighter than it started:
the Gauss-Bonnet part of the flux through the horizon is negative. The
horizon's area falls, and its Wald entropy rises.

The measurements are in ``docs/benchmarks.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import sympy as sp

from particlesim.theories.decoupling import DecouplingLimit, _first_at_ends
from particlesim.theories.gauss_bonnet import _PHI, GaussBonnetCoupling


def _radius(radius):
    """``radius`` as a float array, or unchanged if it is symbolic."""
    return radius if isinstance(radius, sp.Basic) else np.asarray(radius, dtype=float)


def second_order_equations() -> dict[str, sp.Expr]:
    """The order-reduced equations through second order, derived from the action.

    The action ``sqrt(-g) [R - 2 (d phi)^2 + lambda^2 f(phi) G]`` is reduced
    to ``(t, r)`` on ``ds^2 = -e^(2A) dt^2 + e^(2B) dr^2 + r^2 dOmega^2``
    and varied in ``A``, ``B`` and ``phi``. Then ``lambda^2 -> epsilon
    lambda^2``, ``phi -> epsilon phi1``, ``m -> M + epsilon^2 mu``,
    ``delta -> epsilon^2 delta2``, ``f -> f(0) + f'(0) phi``, and each
    equation is expanded in ``epsilon``.

    Returns sympy expressions in ``phi1(t, r)``, with symbols ``M``,
    ``lambda2`` and ``slope`` for ``f'(0)``:

    - ``"scalar"``: ``phi1_tt``, from the first-order scalar equation;
    - ``"mass"``: ``mu_r``, from the ``A`` equation at second order;
    - ``"slicing"``: ``delta2_r``, from the ``B`` equation at second order.

    The ``epsilon^0`` and ``epsilon^1`` parts of the metric equations
    vanish, and so does the ``f(0)`` Gauss-Bonnet term, a total derivative.
    Taking about 15 s, this is for tests, not for runs.
    """
    from sympy.calculus.euler import euler_equations

    from particlesim.symbolic.curvature import MetricGeometry
    from particlesim.theories.gauss_bonnet import gauss_bonnet_invariant

    t, r = sp.symbols("t r", positive=True)
    theta, varphi = sp.symbols("theta varphi")
    A, B, phi = (sp.Function(name)(t, r) for name in ("A", "B", "phi"))
    lam = sp.Symbol("lambda", positive=True)
    f = sp.Function("f")
    coords = (t, r, theta, varphi)
    metric = sp.diag(-sp.exp(2 * A), sp.exp(2 * B), r**2, r**2 * sp.sin(theta) ** 2)
    ricci_scalar = sp.simplify(MetricGeometry(metric, coords).ricci_scalar)
    invariant = gauss_bonnet_invariant(metric, coords)
    gradient = -sp.exp(-2 * A) * sp.diff(phi, t) ** 2 + sp.exp(-2 * B) * sp.diff(phi, r) ** 2
    density = sp.exp(A + B) * r**2
    lagrangian = density * (ricci_scalar - 2 * gradient + lam**2 * f(phi) * invariant)
    E_A, E_B, E_phi = (sp.simplify(e.lhs) for e in euler_equations(lagrangian, [A, B, phi], [t, r]))

    M, eps, lam2, slope, f0 = sp.symbols("M epsilon lambda2 slope f0")
    mu, delta2, phi1 = (sp.Function(name)(t, r) for name in ("mu", "delta2", "phi1"))
    lapse = 1 - 2 * (M + eps**2 * mu) / r

    def orders(expression):
        expression = expression.subs(sp.Derivative(f(phi), (phi, 2)), 0)
        expression = expression.subs(sp.Derivative(f(phi), phi), slope)
        expression = expression.subs(f(phi), f0 + slope * phi).subs(lam, sp.sqrt(eps * lam2))
        expression = expression.subs(
            {A: eps**2 * delta2 + sp.log(lapse) / 2, B: -sp.log(lapse) / 2, phi: eps * phi1}
        ).doit()
        series = sp.series(expression, eps, 0, 3).removeO()
        return [sp.simplify(series.coeff(eps, k)) for k in range(3)]

    a_orders, b_orders, phi_orders = orders(E_A), orders(E_B), orders(E_phi)
    if any(e != 0 for e in a_orders[:2] + b_orders[:2] + [phi_orders[0]]):
        raise ArithmeticError("a metric equation does not vanish below second order")
    mass = sp.solve(a_orders[2], sp.Derivative(mu, r))[0]
    slicing = sp.solve(b_orders[2].subs(sp.Derivative(mu, r), mass), sp.Derivative(delta2, r))[0]
    scalar = sp.solve(phi_orders[1], sp.Derivative(phi1, (t, 2)))[0]
    return {
        "scalar": sp.simplify(scalar),
        "mass": sp.simplify(mass),
        "slicing": sp.simplify(slicing),
    }


def hair_energy(radius, slope: float, coupling_squared: float, mass: float = 1.0):
    """``mu(r)`` on the static hair, with ``mu(2M) = 0``: the mass inside ``r`` less ``M``.

    It rises from zero at the horizon to ``49 f'(0)^2 lambda^4 / 320 M^3``
    far away, which is the static hole's mass less its horizon's.
    """
    r = _radius(radius)
    m = mass
    polynomial = (
        147 * r**5
        + 174 * m * r**4
        + 228 * m**2 * r**3
        - 1624 * m**3 * r**2
        - 3488 * m**4 * r
        - 7360 * m**5
    )
    return slope**2 * coupling_squared**2 * (r - 2 * m) * polynomial / (960 * m**3 * r**6)


def hair_mass(slope: float, coupling_squared: float, mass: float = 1.0) -> float:
    """``49 f'(0)^2 lambda^4 / 320 M^3``: :func:`hair_energy` far away."""
    return 49 * slope**2 * coupling_squared**2 / (320 * mass**3)


def lapse_correction(radius, slope: float, coupling_squared: float, mass: float = 1.0):
    """``delta(r)`` on the static hair, zero far away: ``g_tt = -e^(2 delta)(1 - 2m/r)``."""
    r = _radius(radius)
    m = mass
    polynomial = 15 * r**4 + 40 * m * r**3 + 210 * m**2 * r**2 + 384 * m**3 * r + 720 * m**4
    return -(slope**2) * coupling_squared**2 * polynomial / (120 * m**2 * r**6)


def static_mass(horizon_radius: float, slope: float, coupling_squared: float) -> float:
    """The hairy hole's mass at second order: ``r_H/2 + (49/40) f'(0)^2 lambda^4 / r_H^3``."""
    return horizon_radius / 2 + 49 * slope**2 * coupling_squared**2 / (40 * horizon_radius**3)


def static_temperature(horizon_radius: float, slope: float, coupling_squared: float) -> float:
    """``(1 / 4 pi r_H)(1 - f'(0)^2 lambda^4 / 60 r_H^4)``.

    From ``e^(delta_H) (1 - 2 mu'(r_H)) / 4 pi r_H``. The two corrections
    nearly cancel, 441 against 440 in sixtieths.
    """
    x = slope**2 * coupling_squared**2 / horizon_radius**4
    pi = sp.pi if isinstance(x, sp.Basic) else np.pi
    return (1 - x / 60) / (4 * pi * horizon_radius)


def static_horizon_scalar(horizon_radius: float, slope: float, coupling_squared: float) -> float:
    """``phi_H = (11/6) f'(0) lambda^2 / r_H^2``, the first-order hair at the horizon."""
    return 11 * slope * coupling_squared / (6 * horizon_radius**2)


def _derivative(values, spacing):
    """Fourth-order first derivative, one-sided at the two outermost points of each end."""
    out = np.empty_like(values)
    out[2:-2] = (values[:-4] - 8 * values[1:-3] + 8 * values[3:-1] - values[4:]) / (12 * spacing)
    out[0], out[1], out[-2], out[-1] = _first_at_ends(values, spacing)
    return out


@dataclass(frozen=True)
class HairGrowth:
    """What :meth:`SecondOrder.run` measured, in the hole's units.

    ``horizon_mass[k]`` is the change in the horizon's mass by ``times[k]``:
    the energy through the inner end, which is where the horizon's flux
    passes on these slices, at advanced time ``times[k] + r*_inner``.
    ``outflow[k]`` is the energy that has left through the outer end.
    ``inside[k]`` is ``mu`` at the outer end less ``mu`` at the inner one.
    ``tail`` is the static hair's energy beyond the outer end, which leaves
    through it as the hair grows and is not radiation.
    """

    times: np.ndarray
    horizon_mass: np.ndarray
    outflow: np.ndarray
    inside: np.ndarray
    horizon_scalar: np.ndarray
    tail: float
    mass: float
    slope: float
    coupling_squared: float
    psi: np.ndarray = field(repr=False)

    @property
    def radiated(self) -> float:
        """Energy carried away by the scalar: what left through the outer end, less the tail."""
        return float(self.outflow[-1] - self.tail)

    @property
    def imbalance(self) -> float:
        """The largest ``|inside + outflow + horizon_mass|``: zero if energy is conserved."""
        return float(np.max(np.abs(self.inside + self.outflow + self.horizon_mass)))

    @property
    def area_change(self) -> float:
        """``32 pi M dM_H``: the horizon's area at the end, less Schwarzschild's."""
        return float(32 * np.pi * self.mass * self.horizon_mass[-1])

    @property
    def entropy_change(self) -> float:
        """Wald's entropy at the end, less at the start: ``area/4 + 4 pi lambda^2 f'(0) phi_H``."""
        coupling = 4 * np.pi * self.coupling_squared * self.slope * self.horizon_scalar[-1]
        return float(self.area_change / 4 + coupling)


@dataclass(frozen=True)
class SecondOrder:
    """The first-order scalar on Schwarzschild, and the metric it bends at second order.

    ``slope`` is ``f'(0)``, and ``coupling_squared`` is ``lambda^2``. The
    grid is :class:`~particlesim.theories.decoupling.DecouplingLimit`'s.
    """

    slope: float
    coupling_squared: float
    mass: float = 1.0
    inner: float = -200.0
    outer: float = 300.0
    points: int = 2001
    dissipation: float = 0.1

    def __post_init__(self) -> None:
        if self.slope == 0:
            raise ValueError(
                "f'(0) = 0: the first-order scalar has no source, so this expansion "
                "gives no hair at any order. Scalarization is not perturbative in "
                "the coupling; see decoupling.DecouplingLimit for its onset"
            )

    @classmethod
    def from_theory(cls, theory, mass: float = 1.0, **grid) -> SecondOrder:
        """For a theory plugin with a ``coupling_function`` and an ``alpha = lambda^2``."""
        _, slope, _ = theory.coupling_function(0.0)
        return cls(slope, float(theory.values["alpha"]), mass, **grid)

    @cached_property
    def limit(self) -> DecouplingLimit:
        """The first-order scalar: a linear coupling ``f'(0) phi`` in the decoupling limit."""
        linear = GaussBonnetCoupling("linear", self.slope * _PHI)
        return DecouplingLimit(
            linear,
            self.coupling_squared,
            self.mass,
            self.inner,
            self.outer,
            self.points,
            self.dissipation,
        )

    @property
    def _k(self) -> float:
        return 4 * self.mass * self.coupling_squared * self.slope

    def mass_density(self, psi, pi):
        """``d_r* mu`` on the grid, from ``psi = r phi`` and ``pi = d_t psi``."""
        h, r, m = self.limit.spacing, self.limit.radius, self.mass
        d_phi = _derivative(psi / r, h)
        dd_phi = _derivative(d_phi, h)
        kinetic = r**2 / 2 * (d_phi**2 + (pi / r) ** 2)
        return kinetic + self._k / r**3 * (r**2 * dd_phi - (r - m) * d_phi)

    def mass_flux(self, psi, pi):
        """``d_t mu`` on the grid: positive where energy moves inward."""
        h, r, m = self.limit.spacing, self.limit.radius, self.mass
        phi_t = pi / r
        return r**2 * phi_t * _derivative(psi / r, h) + self._k * (
            _derivative(phi_t, h) / r - m * phi_t / r**3
        )

    def profile(self, psi, pi, horizon_mass: float = 0.0):
        """``mu`` on the grid, integrated outward from ``horizon_mass`` at the inner end."""
        from scipy.integrate import cumulative_simpson

        return horizon_mass + cumulative_simpson(
            self.mass_density(psi, pi), dx=self.limit.spacing, initial=0.0
        )

    def run(self, final: float, every: float = 10.0, courant: float = 0.25) -> HairGrowth:
        """Grow the hair from ``phi = 0`` on Schwarzschild, and book the energy.

        The fluxes through both ends are integrated in time at every step,
        by the trapezoidal rule. The energy between the ends is integrated
        in space at each sample, by Simpson's rule.
        """
        from scipy.integrate import simpson

        limit = self.limit
        psi = np.zeros(self.points)
        pi = np.zeros(self.points)
        steps = int(round(final / (courant * limit.spacing)))
        dt = final / steps
        stride = max(1, int(round(every / dt)))
        through = np.zeros(2)
        previous = self.mass_flux(psi, pi)[[0, -1]]
        rows = [(0.0, 0.0, 0.0, 0.0, 0.0)]
        r = limit.radius
        for count in range(1, steps + 1):
            psi, pi = limit.step(psi, pi, dt)
            current = self.mass_flux(psi, pi)[[0, -1]]
            through += (previous + current) / 2 * dt
            previous = current
            if count % stride == 0 or count == steps:
                inside = simpson(self.mass_density(psi, pi), dx=limit.spacing)
                rows.append((count * dt, through[0], -through[1], inside, psi[0] / r[0]))
        times, horizon, outflow, inside, scalar = (np.array(c) for c in zip(*rows, strict=True))
        far = hair_mass(self.slope, self.coupling_squared, self.mass)
        near = hair_energy(r[[0, -1]], self.slope, self.coupling_squared, self.mass)
        return HairGrowth(
            times=times,
            horizon_mass=horizon - near[0],
            outflow=outflow,
            inside=inside,
            horizon_scalar=scalar,
            tail=float(far - near[1]),
            mass=self.mass,
            slope=self.slope,
            coupling_squared=self.coupling_squared,
            psi=psi,
        )


__all__ = [
    "HairGrowth",
    "SecondOrder",
    "hair_energy",
    "hair_mass",
    "lapse_correction",
    "second_order_equations",
    "static_horizon_scalar",
    "static_mass",
    "static_temperature",
]
