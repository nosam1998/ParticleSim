"""Einstein-scalar-Gauss-Bonnet in spherical symmetry, fully coupled, in the 3-D runs' gauge (#52).

:mod:`~particlesim.theories.decoupling` evolves the scalar on a fixed hole,
and :mod:`~particlesim.theories.backreaction` adds the metric's answer at
second order in the coupling. Neither is what
:mod:`particlesim.solvers.nr.modified` evolves, which is the whole theory,
nonlinearly, in a gauge of its own. This module is that problem with the
three dimensions reduced to one, solved another way, so the 3-D evolution
has something to be compared with at a coupling where the metric's answer
is not small.

**The gauge is the 3-D runs'.** Those runs hold the lapse at
Schwarzschild's static value in isotropic coordinates and the shift at
zero, so the spatial coordinates follow the normal observers and the
coordinate time is the hole's Killing time only while the metric is
Schwarzschild's. Comparing at fixed coordinate time and radius therefore
needs the same slicing and the same radial label, because the metric's
response, of order ``lambda^4``, moves both at that order, and the scalar's
own response to it is of order ``lambda^6``: the same order as the effect.
So the metric here is

    ds^2 = -alpha(rho)^2 dt^2 + e^(2a(t, rho)) d rho^2 + e^(2b(t, rho)) dOmega^2

with ``rho = ln(R / R_H)``, ``R`` the isotropic radius of the initial slice
and ``R_H = M/2`` its horizon. On Schwarzschild ``alpha = tanh(rho/2)`` and
``e^a = e^b = 2M cosh^2(rho/2)``. The inversion symmetry of the slice is
``rho -> -rho``, under which ``a``, ``b`` and ``phi`` are even and the lapse is
odd, so the horizon ``rho = 0`` is a symmetry point and only the exterior
is evolved.

**The equations come from the action, not from the 3-D code.** The action
``sqrt(-g) [R - 2 (d phi)^2 + lambda^2 f(phi) G]`` is reduced on that metric
with the lapse a function of ``(t, rho)`` too, by
:class:`~particlesim.symbolic.curvature.MetricGeometry` and
:func:`~particlesim.theories.gauss_bonnet.gauss_bonnet_invariant`, and
varied by Euler-Lagrange in ``a``, ``b``, ``phi`` and ``alpha``. The third and
fourth derivatives cancel, as they must in a Horndeski theory, and
:func:`field_equations` checks that they do. Varying ``a``, ``b`` and ``phi``
gives the evolution equations; varying the lapse gives the Hamiltonian
constraint, which is evolved past and measured. In the variables

    k_a = d_t a / alpha,   k_b = d_t b / alpha,   pi = d_t phi / alpha

(minus the mixed curvatures ``K^rho_rho``, ``K^theta_theta`` and minus
``K_phi``) no inverse power of the lapse survives, which the derivation also
checks, and the three equations are affine in ``(d_t k_a, d_t k_b, d_t pi)``.
At every point the evolution solves the 3 x 3 system for them, as the 3-D
code solves its 7 x 7 one. The matrix is symmetric, as a Lagrangian's
kinetic matrix is.

**Nothing of the 3-D code is used**: not the 3+1 split of the Gauss-Bonnet
tensor, not its normalisation, not the CCZ4 variables or their constraint
damping. What the two share is the action and the gauge.

**The numerics.** A cell-centred grid in ``rho``, fourth-order differences,
classical Runge-Kutta, sixth-difference Kreiss-Oliger dissipation. The
horizon end is filled by the reflection, with each variable's parity. The
outer end, at ``R = R_H e^outer``, is held at its initial values: nothing
the hole does reaches it in the times this is run for.

**What it is checked against** is in ``tests/unit/test_esgb_spherical.py`` and
``docs/benchmarks.md``: a static hole of
:func:`~particlesim.theories.gauss_bonnet.static_black_hole`, a third
derivation, is static here to fourth order, and with the metric held fixed
the scalar is :mod:`~particlesim.theories.decoupling`'s.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache, cached_property

import numpy as np
import sympy as sp

from particlesim.symbolic import cache as kernel_cache
from particlesim.theories.gauss_bonnet import _PHI, GaussBonnetCoupling

_VERSION = "esgb-spherical-1"

#: Evolved variables, and their parity under ``rho -> -rho``.
VARIABLES = {"a": 1, "b": 1, "phi": 1, "ka": -1, "kb": -1, "pi": -1}

#: The symbols the generated system reads, in order.
INPUTS = (
    "A", "B", "a_01", "a_02", "b_01", "b_02", "ph_01", "ph_02",
    "ka", "ka_1", "kb", "kb_1", "pi", "pi_1",
    "alpha_0", "alpha_1", "alpha_2", "f1", "f2", "lam2",
)  # fmt: skip

#: Ghost cells at each end: the Kreiss-Oliger stencil reaches three.
GHOSTS = 3


def _reduced_euler_lagrange():
    """The Euler-Lagrange expressions in ``alpha``, ``a``, ``b`` and ``phi``, as functions."""
    from sympy.calculus.euler import euler_equations

    from particlesim.symbolic.curvature import MetricGeometry
    from particlesim.theories.gauss_bonnet import gauss_bonnet_invariant

    t, rho = sp.symbols("t rho", real=True)
    theta, varphi = sp.symbols("theta varphi")
    lapse, a, b, phi = (sp.Function(name)(t, rho) for name in ("al", "a", "b", "ph"))
    f = sp.Function("f")
    coords = (t, rho, theta, varphi)
    metric = sp.diag(-(lapse**2), sp.exp(2 * a), sp.exp(2 * b), sp.exp(2 * b) * sp.sin(theta) ** 2)
    ricci_scalar = sp.simplify(MetricGeometry(metric, coords).ricci_scalar)
    invariant = gauss_bonnet_invariant(metric, coords)
    gradient = -(sp.diff(phi, t) ** 2) / lapse**2 + sp.exp(-2 * a) * sp.diff(phi, rho) ** 2
    lagrangian = (
        lapse
        * sp.exp(a + 2 * b)
        * (ricci_scalar - 2 * gradient + sp.Symbol("lam2") * f(phi) * invariant)
    )
    equations = euler_equations(lagrangian, [lapse, a, b, phi], [t, rho])
    return (t, rho), (lapse, a, b, phi), f, [e.lhs for e in equations]


def field_equations() -> dict[str, sp.Expr]:
    """The reduced field equations, in plain symbols, regular where the lapse vanishes.

    Keys ``"hamiltonian"`` (from varying the lapse), ``"a"``, ``"b"`` and
    ``"phi"``. The symbols are those of :data:`INPUTS`, plus the unknowns
    ``ka_t``, ``kb_t`` and ``pi_t``. ``A = e^a``, ``B = e^b``; ``q_ij`` is
    ``d_t^i d_rho^j q``; ``alpha_j`` is ``d_rho^j alpha``, with the lapse
    independent of time once varied; ``f1`` and ``f2`` are ``f'`` and
    ``f''``. Raises if a third derivative survives, if an inverse power of
    the lapse does, or if an equation is not affine in the unknowns. About
    15 s.
    """
    (t, rho), functions, f, raw = _reduced_euler_lagrange()
    names = ("al", "a", "b", "ph")
    by_name = dict(zip(names, functions, strict=True))
    A, B = sp.symbols("A B", positive=True)
    S = sp.Symbol
    alpha = [S(f"alpha_{j}") for j in range(5)]

    def plain(expression):
        phi = by_name["ph"]
        for order in (3, 2, 1):
            expression = expression.subs(sp.Derivative(f(phi), (phi, order)), S(f"f{order}"))
        expression = expression.subs(f(phi), S("f0"))
        mapping = {}
        for derivative in expression.atoms(sp.Derivative):
            counts = dict(derivative.variable_count)
            i, j = counts.get(t, 0), counts.get(rho, 0)
            name = derivative.expr.func.__name__
            if name == "al":
                mapping[derivative] = alpha[j] if i == 0 else sp.S.Zero
            else:
                mapping[derivative] = S(f"{name}_{i}{j}")
        expression = expression.xreplace(mapping)
        return expression.xreplace(
            {by_name["al"]: alpha[0], by_name["a"]: sp.log(A), by_name["b"]: sp.log(B),
             by_name["ph"]: S("ph_00")}
        )  # fmt: skip

    to_k = {}
    for q, k in (("a", "ka"), ("b", "kb"), ("ph", "pi")):
        to_k[S(f"{q}_10")] = alpha[0] * S(k)
        to_k[S(f"{q}_11")] = alpha[1] * S(k) + alpha[0] * S(f"{k}_1")
        to_k[S(f"{q}_20")] = alpha[0] * S(f"{k}_t")
    unknowns = [S("ka_t"), S("kb_t"), S("pi_t")]
    out = {}
    for key, expression in zip(("hamiltonian", "a", "b", "phi"), raw, strict=True):
        expression = sp.expand(plain(expression))
        third = [
            s
            for s in expression.free_symbols
            if (s.name[-2:].isdigit() and sum(map(int, s.name[-2:])) > 2)
            or s.name in ("alpha_3", "alpha_4", "f0", "f3", "ph_00")
        ]
        if third:
            raise ArithmeticError(f"{key}: {sorted(map(str, third))} survive the variation")
        expression = sp.expand(expression.xreplace(to_k))
        powers = {term.as_powers_dict().get(alpha[0], 0) for term in sp.Add.make_args(expression)}
        if min(powers) < 0:
            raise ArithmeticError(f"{key}: an inverse power of the lapse survives")
        if any(s.name.endswith(("_10", "_11", "_20")) for s in expression.free_symbols):
            raise ArithmeticError(f"{key}: a time derivative was not replaced")
        if key != "hamiltonian":
            for u in unknowns:
                for v in unknowns:
                    if sp.expand(sp.diff(expression, u, v)) != 0:
                        raise ArithmeticError(f"{key} is not affine in {u}, {v}")
        elif any(u in expression.free_symbols for u in unknowns):
            raise ArithmeticError("the Hamiltonian constraint holds a second time derivative")
        out[key] = expression
    return out


def _system_source() -> str:
    """NumPy source for ``system(*INPUTS)``: the 3 x 3 matrix, the remainder, the constraint.

    Returns ``(m00, m01, m02, m11, m12, m22, r0, r1, r2, hamiltonian)`` with
    rows and columns ordered ``(a, b, phi)`` and ``(ka_t, kb_t, pi_t)``, so
    that ``M u + r = 0``. The matrix is symmetric, and the source is only
    emitted if it is.
    """
    equations = field_equations()
    unknowns = [sp.Symbol(name) for name in ("ka_t", "kb_t", "pi_t")]
    rows = [equations[key] for key in ("a", "b", "phi")]
    matrix = [[sp.expand(sp.diff(row, u)) for u in unknowns] for row in rows]
    for i in range(3):
        for j in range(i + 1, 3):
            if sp.expand(matrix[i][j] - matrix[j][i]) != 0:
                raise ArithmeticError(f"the kinetic matrix is not symmetric at ({i}, {j})")
    rest = [sp.expand(row.subs({u: 0 for u in unknowns})) for row in rows]
    outputs = [matrix[0][0], matrix[0][1], matrix[0][2], matrix[1][1], matrix[1][2], matrix[2][2]]
    outputs += rest + [equations["hamiltonian"]]
    replacements, reduced = sp.cse(outputs)
    printer = sp.printing.numpy.NumPyPrinter()
    lines = [f"def system({', '.join(INPUTS)}):"]
    for symbol, value in replacements:
        lines.append(f"    {symbol} = {printer.doprint(value)}")
    lines.append(f"    return ({', '.join(printer.doprint(e) for e in reduced)},)")
    return "\n".join(lines)


@cache
def _system() -> Callable:
    key = kernel_cache.key_for(_VERSION, "spherical-system")
    source, _ = kernel_cache.source_cached(key, _system_source)
    namespace: dict = {"numpy": np}
    exec(compile(source, "<esgb spherical system>", "exec"), namespace)  # noqa: S102 - generated here
    return namespace["system"]


def schwarzschild_lapse(rho):
    """Schwarzschild's static lapse in ``rho``: ``tanh(rho/2)``."""
    return np.tanh(np.asarray(rho, dtype=float) / 2)


def _d1(values, h):
    return (-values[4:] + 8 * values[3:-1] - 8 * values[1:-3] + values[:-4]) / (12 * h)


def _d2(values, h):
    return (
        -values[4:] + 16 * values[3:-1] - 30 * values[2:-2] + 16 * values[1:-3] - values[:-4]
    ) / (12 * h * h)


def _sixth(values):
    v = values
    return v[6:] - 6 * v[5:-1] + 15 * v[4:-2] - 20 * v[3:-3] + 15 * v[2:-4] - 6 * v[1:-5] + v[:-6]


@dataclass(frozen=True)
class CoupledSpherical:
    """The fully coupled theory on a spherical hole, in a frozen lapse and zero shift.

    ``coupling_squared`` is ``lambda^2`` in the units ``horizon`` is in.
    ``horizon`` is the isotropic horizon radius ``R_H`` of the initial
    slice, ``M/2`` for Schwarzschild. The grid is ``points`` cells from the
    horizon to ``rho = outer``, and the last ``frozen`` of them are held at
    their initial values. ``lapse`` maps ``rho`` to ``alpha``,
    Schwarzschild's static lapse by default; its derivatives are taken on
    the grid, as the 3-D code takes them. With
    ``backreaction=False`` the metric is held fixed and only the scalar
    moves: the decoupling limit, in this gauge.
    """

    coupling: GaussBonnetCoupling
    coupling_squared: float
    horizon: float = 0.5
    outer: float = 6.0
    points: int = 1200
    dissipation: float = 0.1
    frozen: int = 4
    backreaction: bool = True
    lapse: Callable = field(default=schwarzschild_lapse, repr=False)

    @property
    def spacing(self) -> float:
        return self.outer / self.points

    @cached_property
    def rho(self) -> np.ndarray:
        """Cell centres ``(i + 1/2) h``, ghosts included, ``-GHOSTS <= i < points + GHOSTS``."""
        return (np.arange(-GHOSTS, self.points + GHOSTS) + 0.5) * self.spacing

    @property
    def radius(self) -> np.ndarray:
        """The isotropic radius ``R`` of each cell."""
        return self.horizon * np.exp(self.rho)

    @cached_property
    def _alpha(self) -> np.ndarray:
        """The lapse on every cell, ghosts included, made exactly odd at the horizon."""
        value = np.asarray(self.lapse(self.rho), dtype=float) * np.ones_like(self.rho)
        value[:GHOSTS] = -value[2 * GHOSTS - 1 : GHOSTS - 1 : -1]
        return value

    @cached_property
    def _lapse(self):
        """``(alpha, alpha', alpha'')`` on the cells the equations are evaluated on."""
        h = self.spacing
        return self._alpha[2:-2], _d1(self._alpha, h), _d2(self._alpha, h)

    @cached_property
    def _coupling(self):
        functions = [sp.diff(self.coupling.expression, _PHI, k) for k in (1, 2)]
        return sp.lambdify(_PHI, functions, "numpy")

    @property
    def time_step_limit(self) -> float:
        """``h`` over the fastest coordinate light speed, ``|alpha| e^-a``, on the initial slice."""
        state = self.schwarzschild()
        speed = np.abs(self._alpha) * np.exp(-state["a"])
        return self.spacing / float(np.max(speed))

    def schwarzschild(self, mass: float | None = None) -> dict[str, np.ndarray]:
        """Isotropic Schwarzschild, time-symmetric, with no scalar: ``e^a = e^b = 2M cosh^2``."""
        mass = 2 * self.horizon if mass is None else mass
        a = np.log(2 * mass * np.cosh(self.rho / 2) ** 2)
        zero = np.zeros_like(self.rho)
        return {"a": a, "b": a.copy(), "phi": zero, "ka": zero, "kb": zero, "pi": zero}

    def from_isotropic(self, log_conformal, scalar) -> dict[str, np.ndarray]:
        """Time-symmetric data ``psi^4 (dR^2 + R^2 dOmega^2)``; ``ln psi``, ``phi`` are of ``R``."""
        radius = self.radius
        a = np.log(radius) + 2 * np.asarray(log_conformal(radius))
        zero = np.zeros_like(radius)
        return {
            "a": a,
            "b": a.copy(),
            "phi": np.asarray(scalar(radius), dtype=float),
            "ka": zero,
            "kb": zero,
            "pi": zero,
        }

    def _fill(self, state):
        out = {}
        for name, value in state.items():
            value = np.array(value, dtype=float)
            sign = VARIABLES[name]
            value[:GHOSTS] = sign * value[2 * GHOSTS - 1 : GHOSTS - 1 : -1]
            out[name] = value
        return out

    def evaluate(self, state):
        """``(state, matrix, remainder, E_alpha)`` on the interior, with ``M u + r = 0``."""
        h = self.spacing
        s = self._fill(state)
        inner = slice(2, -2)
        alpha, d_alpha, dd_alpha = self._lapse
        df, ddf = (
            np.asarray(v, dtype=float) * np.ones_like(s["phi"][inner])
            for v in self._coupling(s["phi"][inner])
        )
        values = {
            "A": np.exp(s["a"][inner]),
            "B": np.exp(s["b"][inner]),
            "a_01": _d1(s["a"], h),
            "a_02": _d2(s["a"], h),
            "b_01": _d1(s["b"], h),
            "b_02": _d2(s["b"], h),
            "ph_01": _d1(s["phi"], h),
            "ph_02": _d2(s["phi"], h),
            "ka": s["ka"][inner],
            "ka_1": _d1(s["ka"], h),
            "kb": s["kb"][inner],
            "kb_1": _d1(s["kb"], h),
            "pi": s["pi"][inner],
            "pi_1": _d1(s["pi"], h),
            "alpha_0": alpha,
            "alpha_1": d_alpha,
            "alpha_2": dd_alpha,
            "f1": df,
            "f2": ddf,
            "lam2": self.coupling_squared * np.ones_like(alpha),
        }
        m00, m01, m02, m11, m12, m22, r0, r1, r2, hamiltonian = (
            np.asarray(v, dtype=float) * np.ones_like(alpha)
            for v in _system()(*(values[name] for name in INPUTS))
        )
        matrix = np.stack(
            [
                np.stack([m00, m01, m02], -1),
                np.stack([m01, m11, m12], -1),
                np.stack([m02, m12, m22], -1),
            ],
            -2,
        )
        return s, matrix, np.stack([r0, r1, r2], -1), hamiltonian

    def rates(self, state) -> dict[str, np.ndarray]:
        """``d_t`` of every variable, with the ghosts and the frozen outer cells at zero."""
        s, matrix, remainder, _ = self.evaluate(state)
        inner = slice(2, -2)
        alpha = self._lapse[0]
        if self.backreaction:
            solved = np.linalg.solve(matrix, -remainder[..., None])[..., 0]
            dk_a, dk_b, d_pi = solved[:, 0], solved[:, 1], solved[:, 2]
        else:
            zero = np.zeros_like(alpha)
            dk_a, dk_b, d_pi = zero, zero, -remainder[:, 2] / matrix[:, 2, 2]
        interior = {
            "a": alpha * s["ka"][inner],
            "b": alpha * s["kb"][inner],
            "phi": alpha * s["pi"][inner],
            "ka": dk_a,
            "kb": dk_b,
            "pi": d_pi,
        }
        if not self.backreaction:
            interior["a"] = interior["b"] = np.zeros_like(alpha)
        out = {}
        live = slice(GHOSTS, GHOSTS + self.points - self.frozen)
        for name, value in s.items():
            rate = np.zeros_like(value)
            full = np.zeros_like(value)
            full[inner] = interior[name]
            # Dissipation in time units: h_rho over the local light-crossing time of a cell.
            speed = np.abs(self._alpha[3:-3]) * np.exp(-s["a"][3:-3])
            full[3:-3] += self.dissipation * speed * _sixth(value) / (64 * self.spacing)
            rate[live] = full[live]
            out[name] = rate
        return out

    def hamiltonian(self, state) -> np.ndarray:
        """``E_alpha / 2A`` on the interior cells, ``A = e^a``: zero on a solution.

        Normalised so that its Schwarzschild part is the ``1`` of ``2A``,
        the curvature of the spheres.
        """
        s, _, _, hamiltonian = self.evaluate(state)
        return hamiltonian / (2 * np.exp(s["a"][2:-2]))

    def step(self, state, dt: float):
        k1 = self.rates(state)
        k2 = self.rates({n: state[n] + dt / 2 * k1[n] for n in state})
        k3 = self.rates({n: state[n] + dt / 2 * k2[n] for n in state})
        k4 = self.rates({n: state[n] + dt * k3[n] for n in state})
        return {n: state[n] + dt / 6 * (k1[n] + 2 * k2[n] + 2 * k3[n] + k4[n]) for n in state}

    def sample(self, values, radii) -> np.ndarray:
        """``values`` at isotropic radii ``radii``, by a cubic spline in ``rho``."""
        from scipy.interpolate import CubicSpline

        rho = np.log(np.asarray(radii, dtype=float) / self.horizon)
        return CubicSpline(self.rho, values)(rho)

    def evolve(self, state, final: float, radii=(), every: float = 0.25, courant: float = 0.25):
        """Evolve to ``final``; returns ``(times, phi at radii, final state)``.

        ``radii`` are isotropic radii; ``phi`` is sampled every ``every``.
        """
        dt_max = courant * self.time_step_limit
        stride_time = min(every, final)
        per = max(1, int(np.ceil(stride_time / dt_max)))
        dt = stride_time / per
        steps = int(round(final / dt))
        current = {n: np.array(v, dtype=float) for n, v in state.items()}
        times, samples = [0.0], [self.sample(current["phi"], radii)]
        for index in range(1, steps + 1):
            current = self.step(current, dt)
            if index % per == 0 or index == steps:
                times.append(index * dt)
                samples.append(self.sample(current["phi"], radii))
        return np.array(times), np.array(samples), current


__all__ = [
    "GHOSTS",
    "INPUTS",
    "CoupledSpherical",
    "VARIABLES",
    "field_equations",
    "schwarzschild_lapse",
]
