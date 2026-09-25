"""Einstein-scalar-Gauss-Bonnet gravity, and its scalarized black holes (issue #52).

The action, in the normalisation of Doneva and Yazadjiev (2018), is

    S = (1/16 pi) int sqrt(-g) [ R - 2 (d phi)^2 + lambda^2 f(phi) G ]

with ``G = R^2 - 4 R_ab R^ab + R_abcd R^abcd`` the Gauss-Bonnet invariant. In
four dimensions ``G`` alone is a total derivative. Coupled to a scalar it is
not, and the scalar's equation is

    box phi = -(lambda^2 / 4) f'(phi) G

**Spontaneous scalarization.** When ``f'(0) = 0``, ``phi = 0`` solves that
equation, and every GR black hole is a solution of the theory. Linearised
about Schwarzschild, the scalar has an effective mass squared of
``-(lambda^2/4) f''(0) G = -12 lambda^2 f''(0) M^2 / r^6``. That mass is
tachyonic near the horizon, and deep enough for small enough holes to hold a
bound state. Each new static zero mode is a point where a scalarized
branch leaves Schwarzschild. :func:`bifurcation_points` finds them by
shooting. For ``f''(0) = 1`` the first three are ``M / lambda = 0.587,
0.226, 0.140``, the values Doneva and Yazadjiev report.

**The branch itself.** :func:`static_black_hole` solves the full nonlinear
equations for a static, spherically symmetric hole,

    ds^2 = -e^(2 Phi) dt^2 + e^(2 Lambda) dr^2 + r^2 dOmega^2

outward from a regular horizon. The shooting parameter is the scalar there,
and the target is ``phi -> 0`` at infinity. The equations are not typed
in. :func:`static_equations` derives them: from the action reduced to one
dimension, by Euler-Lagrange in ``(Phi, Lambda, phi)``, with ``G``
computed from the metric by :class:`~particlesim.symbolic.curvature.MetricGeometry`.
The ``Lambda`` equation is a constraint, quadratic in ``e^(2 Lambda)``. Its
derivative and the other two equations are solved for ``Lambda'``, ``Phi''``
and ``phi''``, and the generated code is cached.

**A regular horizon is not automatic.** Regularity fixes ``phi'`` at the
horizon as a root of a quadratic,

    phi'_H = (r_H / (4 lambda^2 f'_H)) (-1 + sqrt(1 - 24 lambda^4 f'_H^2 / r_H^4))

and a solution exists only while the discriminant is positive. That is why a
scalarized branch ends at a smallest hole.

**The check that does not depend on the shooting.** Black holes of this
theory obey the first law, ``dM = T dS``, with Wald's entropy

    S = pi r_H^2 + 4 pi lambda^2 f(phi_H)

which is not the area. Along a computed branch, ``M``, ``T`` and ``S`` are
three separate outputs: the mass from the far field, the temperature from
the horizon's surface gravity, and the entropy from the horizon scalar. The
first law ties them together only if the equations, the entropy formula
and the solution are all right.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache

import numpy as np
import sympy as sp

from particlesim.symbolic import cache as kernel_cache
from particlesim.theories.base import Coupling, Theory

POSITIVE = (0.0, float("inf"))
_PHI = sp.Symbol("phi")
_VERSION = "esgb-static-1"


@dataclass(frozen=True)
class GaussBonnetCoupling:
    """A coupling function ``f(phi)`` for the Gauss-Bonnet term."""

    name: str
    expression: sp.Expr
    reference: str = ""
    _numeric: Callable = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        derivatives = [self.expression, sp.diff(self.expression, _PHI)]
        derivatives.append(sp.diff(derivatives[1], _PHI))
        object.__setattr__(self, "_numeric", sp.lambdify(_PHI, derivatives, "numpy"))

    def __call__(self, phi: float) -> tuple[float, float, float]:
        """``(f, f', f'')`` at ``phi``."""
        f, df, ddf = self._numeric(phi)
        return float(f), float(df), float(ddf)

    @property
    def curvature_at_zero(self) -> float:
        """``f''(0)``, which alone sets where scalarization starts."""
        return float(sp.diff(self.expression, _PHI, 2).subs(_PHI, 0))

    @property
    def admits_general_relativity(self) -> bool:
        """``f'(0) = 0``: then ``phi = 0`` is a solution, and so is every GR spacetime."""
        return sp.simplify(sp.diff(self.expression, _PHI).subs(_PHI, 0)) == 0


SCALARIZATION = GaussBonnetCoupling(
    "scalarization",
    (1 - sp.exp(-6 * _PHI**2)) / 12,
    "Doneva and Yazadjiev, Phys. Rev. Lett. 120, 131103 (2018)",
)
QUADRATIC = GaussBonnetCoupling(
    "quadratic",
    _PHI**2 / 2,
    "Silva, Sakstein, Gualtieri, Sotiriou and Berti, Phys. Rev. Lett. 120, 131104 (2018)",
)
DILATONIC = GaussBonnetCoupling(
    "dilatonic",
    sp.exp(-2 * _PHI),
    "the heterotic string's leading alpha' correction (Metsaev and Tseytlin 1987); "
    "black holes of Kanti, Mavromatos, Rizos, Tamvakis and Winstanley (1996)",
)


def gauss_bonnet_invariant(metric: sp.Matrix, coords) -> sp.Expr:
    """``R^2 - 4 R_ab R^ab + R_abcd R^abcd`` for any metric."""
    from particlesim.symbolic.curvature import MetricGeometry

    geometry = MetricGeometry(metric, coords)
    ricci, inverse = geometry.ricci, geometry.ginv
    size = metric.shape[0]
    ricci_squared = sum(
        inverse[a, c] * inverse[b, d] * ricci[a, b] * ricci[c, d]
        for a in range(size)
        for b in range(size)
        for c in range(size)
        for d in range(size)
    )
    return sp.simplify(geometry.ricci_scalar**2 - 4 * ricci_squared + geometry.kretschmann)


def _static_symbols():
    r = sp.Symbol("r", positive=True)
    t, theta, varphi = sp.symbols("t theta varphi")
    Phi, Lam, phi = (sp.Function(name)(r) for name in ("Phi", "Lambda", "phi"))
    return r, (t, r, theta, varphi), Phi, Lam, phi


def static_equations() -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """The static, spherically symmetric field equations, from the reduced action.

    Returns ``(E_Phi, E_Lambda, E_phi)``, the Euler-Lagrange expressions of
    ``sqrt(-g) [R - 2 (d phi)^2 + lambda^2 f(phi) G]`` with the angles
    integrated out. ``f`` is left as an undetermined function. Varying a
    symmetry-reduced action is safe here, since spherical symmetry is compact
    and staticity is a hypersurface-orthogonal isometry (Palais, Fels and
    Torre).
    """
    from sympy.calculus.euler import euler_equations

    r, coords, Phi, Lam, phi = _static_symbols()
    lam = sp.Symbol("lambda", positive=True)
    f = sp.Function("f")
    metric = sp.diag(-sp.exp(2 * Phi), sp.exp(2 * Lam), r**2, r**2 * sp.sin(coords[2]) ** 2)
    from particlesim.symbolic.curvature import MetricGeometry

    ricci_scalar = sp.simplify(MetricGeometry(metric, coords).ricci_scalar)
    invariant = gauss_bonnet_invariant(metric, coords)
    density = sp.exp(Phi + Lam) * r**2
    kinetic = 2 * sp.exp(-2 * Lam) * sp.diff(phi, r) ** 2
    lagrangian = sp.simplify(density * (ricci_scalar - kinetic + lam**2 * f(phi) * invariant))
    equations = euler_equations(lagrangian, [Phi, Lam, phi], r)
    return tuple(sp.simplify(e.lhs) for e in equations)


_STATE = sp.symbols("r Lam dPhi phi dphi f df ddf lam")


def _to_state(expression: sp.Expr, highest: dict) -> sp.Expr:
    """Replace functions and derivatives by plain symbols, ``Phi`` itself by 0."""
    r_s, Lam_s, dPhi_s, phi_s, dphi_s, f_s, df_s, ddf_s, lam_s = _STATE
    r, _, Phi, Lam, phi = _static_symbols()
    f = sp.Function("f")
    expression = expression.subs(
        {
            sp.Derivative(f(phi), (phi, 2)): ddf_s,
            sp.Derivative(f(phi), phi): df_s,
        }
    )
    expression = expression.subs(
        {
            sp.Derivative(Phi, (r, 2)): highest["ddPhi"],
            sp.Derivative(phi, (r, 2)): highest["ddphi"],
            sp.Derivative(Lam, r): highest["dLam"],
            sp.Derivative(Phi, r): dPhi_s,
            sp.Derivative(phi, r): dphi_s,
        }
    )
    lam = sp.Symbol("lambda", positive=True)
    expression = expression.subs({f(phi): f_s, Lam: Lam_s, phi: phi_s, Phi: 0, lam: lam_s})
    return expression.subs(r, r_s)


def _rates_source() -> str:
    """Python source for ``rates(r, Lam, dPhi, phi, dphi, f, df, ddf, lam)``.

    Returns ``(Lambda', Phi'', phi'')`` from ``E_Phi``, ``E_phi`` and the
    derivative of the constraint ``E_Lambda``: three equations, linear in
    those three unknowns.
    """
    highest = {name: sp.Symbol(name) for name in ("dLam", "ddPhi", "ddphi")}
    E_Phi, E_Lam, E_phi = static_equations()
    r = _static_symbols()[0]
    system = [_to_state(e, highest) for e in (E_Phi, sp.diff(E_Lam, r), E_phi)]
    unknowns = [highest["dLam"], highest["ddPhi"], highest["ddphi"]]
    solution = sp.solve(system, unknowns, dict=True)[0]
    outputs = [sp.simplify(sp.together(solution[u])) for u in unknowns]
    replacements, reduced = sp.cse(outputs)
    printer = sp.printing.numpy.NumPyPrinter()
    names = ", ".join(str(s) for s in _STATE)
    lines = [f"def rates({names}):"]
    for symbol, value in replacements:
        lines.append(f"    {symbol} = {printer.doprint(value)}")
    lines.append(f"    return ({', '.join(printer.doprint(e) for e in reduced)})")
    return "\n".join(lines)


@cache
def _rates() -> Callable:
    key = kernel_cache.key_for(_VERSION, "static-rates")
    source, _ = kernel_cache.source_cached(key, _rates_source)
    namespace: dict = {"numpy": np}
    exec(compile(source, "<esgb static rates>", "exec"), namespace)  # noqa: S102 - generated here
    return namespace["rates"]


def bifurcation_points(count: int = 3, coupling: GaussBonnetCoupling = SCALARIZATION):
    """``M / lambda`` where the first ``count`` scalarized branches leave Schwarzschild.

    The static linear scalar on Schwarzschild, with ``x = r / M``, obeys

        (x^2 (1 - 2/x) phi')' + 12 c phi / x^4 = 0,    c = f''(0) lambda^2 / M^2

    A zero mode is regular at the horizon, ``phi'(2) = -3 c / 8``, and decays
    at infinity, so ``phi + x phi' -> 0``. The ``n``-th ``c`` with such a
    solution has ``n`` nodes.
    """
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    curvature = coupling.curvature_at_zero
    if curvature <= 0:
        raise ValueError(
            f"f''(0) = {curvature} for the {coupling.name} coupling: a non-positive "
            "effective mass has no bound state, so Schwarzschild does not scalarize"
        )

    def far_value(c: float) -> float:
        start = 2.0 + 1e-6
        slope = -3.0 * c / 8.0

        def rhs(x, y):
            return [y[1], (-(2 * x - 2) * y[1] - 12 * c * y[0] / x**4) / (x * x - 2 * x)]

        solution = solve_ivp(
            rhs, (start, 2000.0), [1 + slope * 1e-6, slope], rtol=1e-11, atol=1e-13
        )
        value, derivative = solution.y[:, -1]
        return value + solution.t[-1] * derivative

    found, c, step = [], 0.2, 0.25
    previous = far_value(c)
    while len(found) < count:
        nxt = c + step * (1 + c / 4)
        value = far_value(nxt)
        if np.sign(value) != np.sign(previous):
            found.append(brentq(far_value, c, nxt, xtol=1e-13))
        c, previous = nxt, value
    return [1.0 / np.sqrt(root / curvature) for root in found]


@dataclass(frozen=True)
class StaticBlackHole:
    """A static, spherically symmetric black hole of the theory, with ``lambda = 1`` units.

    ``temperature`` is the surface gravity over ``2 pi``, with ``Phi -> 0`` at
    infinity, and ``entropy`` is Wald's. ``charge`` is the scalar charge
    ``D`` in ``phi ~ D / r``.
    """

    horizon_radius: float
    horizon_scalar: float
    mass: float
    charge: float
    temperature: float
    entropy: float
    coupling: GaussBonnetCoupling = field(repr=False)
    radius: np.ndarray = field(repr=False, default=None)
    scalar: np.ndarray = field(repr=False, default=None)

    @property
    def schwarzschild_entropy(self) -> float:
        """``4 pi M^2``: a GR hole's entropy at the same mass."""
        return 4.0 * np.pi * self.mass**2


def horizon_slope(horizon_radius: float, horizon_scalar: float, coupling) -> float | None:
    """``phi'`` at a regular horizon, or ``None`` where no regular horizon exists."""
    _, df, _ = coupling(horizon_scalar)
    discriminant = 1.0 - 24.0 * df**2 / horizon_radius**4
    if discriminant < 0.0:
        return None
    if df == 0.0:
        return 0.0
    return horizon_radius / (4.0 * df) * (-1.0 + np.sqrt(discriminant))


class _Singular(ArithmeticError):
    """A shot reached a point where the rates are not finite: no regular solution there."""


@dataclass(frozen=True)
class _Profile:
    """A solution out from the horizon: ``state(r)`` is ``(Lambda, Phi', phi, phi', Phi)``."""

    radius: np.ndarray
    states: np.ndarray
    status: int
    _dense: Callable = field(repr=False)
    _horizon: float = 0.0

    def state(self, r: float) -> np.ndarray:
        return self._dense(np.log(r - self._horizon))


def _integrate(horizon_radius, horizon_scalar, coupling, epsilon=1e-5, outer=4000.0):
    """Integrate outward from ``r_H (1 + epsilon)``, in ``s = ln(r - r_H)``.

    Near the horizon ``Phi' ~ 1 / 2 (r - r_H)`` and ``Lambda ~ -ln(r - r_H) / 2``,
    which are steady rates in ``s``. A scalarized hole also has a stiff mode
    in the first few e-folds off the horizon. From ``epsilon = 1e-7`` it cost
    25,000 steps, in ``r`` or in ``s``. From ``1e-5`` it costs 1,400, and the
    mass changes by 1e-9. The start misses the regular solution at
    ``O(epsilon)``, and the far field inherits only ``O(epsilon^2)`` of that.
    """
    from scipy.integrate import solve_ivp

    slope = horizon_slope(horizon_radius, horizon_scalar, coupling)
    if slope is None:
        return None
    rates = _rates()
    start = horizon_radius * (1.0 + epsilon)
    dPhi = 1.0 / (2.0 * (start - horizon_radius))
    phi = horizon_scalar + slope * (start - horizon_radius)
    _, df, _ = coupling(phi)
    # The constraint, X^2 + b X + c = 0 in X = e^(2 Lambda); the root that is
    # Schwarzschild's when the coupling vanishes.
    b = -4.0 * dPhi * df * slope - 2.0 * dPhi * start + slope**2 * start**2 - 1.0
    c = 12.0 * dPhi * df * slope
    X = (-b + np.sqrt(b * b - 4.0 * c)) / 2.0

    def rhs(s, y):
        gap = np.exp(s)
        Lam, dPhi, phi, dphi, _ = y
        f, df, ddf = coupling(phi)
        dLam, ddPhi, ddphi = rates(horizon_radius + gap, Lam, dPhi, phi, dphi, f, df, ddf, 1.0)
        out = [gap * dLam, gap * ddPhi, gap * dphi, gap * ddphi, gap * dPhi]
        if not np.all(np.isfinite(out)):
            raise _Singular
        return out

    y0 = [0.5 * np.log(X), dPhi, phi, slope, 0.0]
    span = (np.log(start - horizon_radius), np.log(outer - horizon_radius))
    # A shot that runs into a singularity returns NaN, which the root search
    # treats as "no solution here"; the warnings on the way add nothing.
    with np.errstate(all="ignore"):
        try:
            solution = solve_ivp(
                rhs,
                span,
                y0,
                method="LSODA",
                rtol=1e-11,
                atol=1e-13,
                min_step=1e-9,  # a shot creeping into a singularity fails rather than hangs
                dense_output=True,
            )
        except _Singular:
            return None
    return _Profile(
        horizon_radius + np.exp(solution.t),
        solution.y,
        solution.status,
        solution.sol,
        horizon_radius,
    )


def _far_field(solution) -> tuple[float, float, float, float]:
    """``(phi_inf, M, D, Phi_inf)``, Richardson-extrapolated from the outer radius and half of it.

    Each estimate is off by ``O(1/r)``: the mass from ``(r/2)(1 - e^(-2 Lambda))``,
    and so on. Combining ``R`` and ``R/2`` removes the leading error.
    """
    outer = solution.radius[-1]

    def estimate(radius):
        Lam, dPhi, phi, dphi, Phi = solution.state(radius)
        mass = radius / 2.0 * (1.0 - np.exp(-2.0 * Lam))
        return np.array([phi + radius * dphi, mass, -(radius**2) * dphi, Phi + mass / radius])

    far, half = estimate(outer), estimate(outer / 2.0)
    return tuple(2.0 * far - half)


def _scalar_at_infinity(horizon_radius, horizon_scalar, coupling) -> float:
    solution = _integrate(horizon_radius, horizon_scalar, coupling)
    if solution is None or solution.status != 0:
        return np.nan
    return _far_field(solution)[0]


def _find_horizon_scalar(horizon_radius, coupling, near=None) -> float | None:
    """The ``phi_H`` that leaves no scalar at infinity: scan for a sign change, then Brent.

    Without ``near`` the scan runs from zero upward (both ways for a coupling
    that is not even) and takes the first sign change, the nodeless hole.
    With ``near``, a previous solution's ``phi_H``, it scans a small window
    around it and takes the sign change closest to it. Shots that fail,
    for instance because no regular horizon exists, are skipped.
    """
    from scipy.optimize import brentq

    if near is not None:
        grid = near + np.linspace(-0.08, 0.08, 9)
        if coupling.admits_general_relativity:
            grid = grid[grid > 0]
    elif coupling.admits_general_relativity:
        # An even coupling is symmetric under phi -> -phi, so only phi_H > 0 is scanned.
        grid = np.linspace(1e-3, 1.5, 31)
    else:
        grid = np.linspace(-1.5, 1.5, 61)
    values = np.array([_scalar_at_infinity(horizon_radius, q, coupling) for q in grid])
    changes = [
        i
        for i in range(len(grid) - 1)
        if np.isfinite(values[i] * values[i + 1]) and values[i] * values[i + 1] < 0
    ]
    if not changes:
        return None
    if near is not None:
        changes.sort(key=lambda i: abs(0.5 * (grid[i] + grid[i + 1]) - near))
    i = changes[0]
    return brentq(
        lambda q: _scalar_at_infinity(horizon_radius, q, coupling),
        grid[i],
        grid[i + 1],
        xtol=1e-13,
    )


def static_black_hole(
    horizon_radius: float,
    coupling: GaussBonnetCoupling = SCALARIZATION,
    horizon_scalar: float | None = None,
    near: float | None = None,
) -> StaticBlackHole | None:
    """The hole with horizon radius ``r_H`` (in units of ``lambda``) and no scalar at infinity.

    With ``horizon_scalar = 0`` this is Schwarzschild, when the coupling
    allows it. Otherwise the fundamental scalarized hole is found by
    :func:`_find_horizon_scalar`, from ``near`` if given. Returns ``None``
    if no such hole exists at this size.
    """
    if horizon_scalar is None:
        horizon_scalar = _find_horizon_scalar(horizon_radius, coupling, near)
        if horizon_scalar is None:
            return None
    solution = _integrate(horizon_radius, horizon_scalar, coupling)
    if solution is None:
        return None
    _, mass, charge, Phi_inf = _far_field(solution)

    def surface_gravity(r):
        Lam, dPhi, _, _, Phi = solution.state(r)
        return np.exp(Phi - Phi_inf - Lam) * dPhi

    # e^(Phi - Lambda) Phi' tends to the surface gravity linearly in r - r_H,
    # so two points near the horizon extrapolate it to O(epsilon^2).
    gap = solution.radius[0] - horizon_radius
    kappa = 2.0 * surface_gravity(horizon_radius + gap) - surface_gravity(horizon_radius + 2 * gap)
    f, _, _ = coupling(horizon_scalar)
    return StaticBlackHole(
        horizon_radius=float(horizon_radius),
        horizon_scalar=float(horizon_scalar),
        mass=float(mass),
        charge=float(charge),
        temperature=float(kappa / (2.0 * np.pi)),
        entropy=float(np.pi * horizon_radius**2 + 4.0 * np.pi * f),
        coupling=coupling,
        radius=solution.radius,
        scalar=solution.states[2],
    )


def scalarized_branch(
    radii, coupling: GaussBonnetCoupling = SCALARIZATION
) -> list[StaticBlackHole]:
    """The fundamental scalarized branch at each horizon radius, by continuation.

    Each solution's ``phi_H`` seeds the next search, so only the first
    radius pays for a full scan. The branch stops at the first radius with
    no solution, which is where a regular horizon stops existing.
    """
    holes: list[StaticBlackHole] = []
    for radius in radii:
        near = holes[-1].horizon_scalar if holes else None
        hole = static_black_hole(radius, coupling, near=near)
        if hole is None:
            break
        holes.append(hole)
    return holes


class DilatonGaussBonnet(Theory):
    """The heterotic string's leading ``alpha'`` correction: ``alpha e^(-2 phi) G``.

    The ``R - 2 (d phi)^2`` normalisation makes ``e^(-2 phi)`` the heterotic
    dilaton's coupling, with ``alpha`` absorbing ``alpha' / 4 g^2``. The
    coupling is ``alpha = lambda^2``, and General Relativity sits at
    ``alpha = 0``, where the scalar is free and decoupled.

    **Formulation: ``order_reduced``.** Einstein-scalar-Gauss-Bonnet is not
    strongly hyperbolic in the usual gauges. The known well-posed route is
    modified CCZ4 (Kovacs and Reall 2020; Areste Salo, Clough and Figueras
    2022), and it is not implemented. Declaring ``order_reduced`` means
    ADR-008 applies: :func:`~particlesim.theories.base.require_well_posed`
    refuses a strong-field 3-D run with this plugin, and says why. The
    static, spherically symmetric holes of this module need no evolution
    and are unaffected.

    ``effective_stress_energy`` returns ``G / 8 pi``: everything the scalar
    and the Gauss-Bonnet term do is attributed to effective matter, the
    convention :class:`~particlesim.theories.dilaton.Dilaton` uses.
    """

    id = "string.eft4d.dgb"
    frame = "einstein"
    formulation = "order_reduced"
    couplings = [Coupling("alpha", 0.0, units="length^2", bounds=POSITIVE)]
    provenance = (
        "Einstein-dilaton-Gauss-Bonnet, the leading alpha' correction of the heterotic "
        "string (Metsaev and Tseytlin 1987), in the normalisation of Doneva and "
        "Yazadjiev 2018; the static equations are derived here from the reduced action"
    )
    validity_statement = (
        "first order in alpha'; curvatures below 1/alpha; static solutions only, since "
        "the theory has no well-posed evolution here (formulation order_reduced)"
    )
    coupling_function = DILATONIC

    def gr_limit(self) -> dict[str, float]:
        return {"alpha": 0.0}

    def effective_stress_energy(self, einstein, metric):
        return sp.Matrix(einstein) / (8 * sp.pi)

    def observable_predictions(self) -> dict:
        alpha = self.values["alpha"]
        return {
            "gauss_bonnet_coupling": alpha,
            "coupling_function": self.coupling_function.name,
            "black_holes_have_scalar_hair": alpha > 0.0,
            "well_posed_evolution": False,
        }


__all__ = [
    "DILATONIC",
    "DilatonGaussBonnet",
    "GaussBonnetCoupling",
    "QUADRATIC",
    "SCALARIZATION",
    "StaticBlackHole",
    "bifurcation_points",
    "gauss_bonnet_invariant",
    "horizon_slope",
    "scalarized_branch",
    "static_black_hole",
    "static_equations",
]
