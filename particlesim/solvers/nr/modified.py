"""``nr.modified``: modified CCZ4 with a scalar and Einstein-scalar-Gauss-Bonnet (issue #52).

The well-posed 3-D route for ``string.eft4d.dgb`` that design doc Section
5.4 names: CCZ4 with the auxiliary metrics of Kovacs and Reall, in the
puncture-gauge form of Aresté Saló, Clough and Figueras, a minimally
coupled scalar, and the Gauss-Bonnet term as effective matter that depends
on ``d_t K_ij`` and ``d_t K_phi`` and is solved for at every point.

**Three generated kernels and some arithmetic.** All of it is derived by
the symbolic pipeline; none of the equations are written in this module.

* The *geometry kernel* (:func:`rhs_kernel`) is
  :func:`particlesim.symbolic.modified_ccz4.modified_parts` with the scalar's
  wave equation: the vacuum rates, three ingredients the gauge is assembled
  from, and the 3+1 tensors the Gauss-Bonnet term is built from. ``b``,
  ``kappa_1`` and ``kappa_2`` are fields it reads, so one derivation serves
  every value of them, including ``b`` varying in space.
* The *Gauss-Bonnet kernel* (:func:`gauss_bonnet_kernel`) is
  :func:`particlesim.symbolic.esgb.effective_matter`: the part of the
  effective stress-energy that does not depend on ``d_t K_ij`` and
  ``d_t K_phi``, and the 7 x 7 linear system for those.
* :func:`particlesim.symbolic.modified_ccz4.matter_terms` adds the matter
  -- the scalar's and the Gauss-Bonnet term's -- to the vacuum rates; a
  batched ``solve`` gives ``d_t K_ij`` and ``d_t K_phi``; and
  :func:`particlesim.symbolic.modified_ccz4.modified_gauge` assembles the
  gauge with ``a``.

At ``a = b = 0`` with no scalar this is CCZ4, which a test checks on random
data against :mod:`particlesim.solvers.nr.ccz4`'s kernel to round-off.

Everything around it -- RK4, Kreiss-Oliger dissipation, upwinding, the
algebraic projection -- is :class:`particlesim.solvers.nr.bssn.Evolution`'s,
which runs this through :class:`ModifiedKernel` exactly as it runs a
generated kernel. The measurements are in ``docs/benchmarks.md``, under
"Modified CCZ4".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

from particlesim.solvers.nr import bssn as bssn_solver
from particlesim.symbolic import cache, codegen
from particlesim.symbolic import modified_ccz4 as symbolic
from particlesim.symbolic.bssn import from_state
from particlesim.symbolic.threeplusone import DIMENSION, INDICES, lapse_hessian

#: Bumped when the generated equations change, so stale cache entries miss.
KERNEL_VERSION = "mccz4-1"

#: The same for the Gauss-Bonnet kernel, which is derived separately.
GAUSS_BONNET_VERSION = "esgb-2"

_RHS_CACHE: dict[tuple, codegen.Kernel] = {}

#: Geometry rates the kernel returns, by the state name they belong to.
GEOMETRY_NAMES = (
    "trK",
    "Theta",
    "sf",
    "Ksf",
    *(f"Gt{i}" for i in INDICES),
    *(f"gt{i}{j}" for i in INDICES for j in range(i, DIMENSION)),
    *(f"At{i}{j}" for i in INDICES for j in range(i, DIMENSION)),
)

#: Gauge ingredients: ``beta^k d_k alpha``, ``beta^k d_k beta^i``, ``D^i alpha``.
GAUGE_INGREDIENTS = (
    "adv_alpha",
    *(f"adv_beta{i}" for i in INDICES),
    *(f"grad_alpha{i}" for i in INDICES),
)


def _pairs():
    return [(i, j) for i in INDICES for j in range(i, DIMENSION)]


#: The 3+1 tensors the Gauss-Bonnet kernel reads, as the geometry kernel names them.
INTERMEDIATE_NAMES = (
    *(f"gam{i}{j}" for i, j in _pairs()),
    *(f"igam{i}{j}" for i, j in _pairs()),
    *(f"Kx{i}{j}" for i, j in _pairs()),
    *(f"Ric{i}{j}" for i, j in _pairs()),
    *(f"DK{k}_{i}{j}" for k in INDICES for i, j in _pairs()),
    *(f"DDa{i}{j}" for i, j in _pairs()),
    *(f"DDs{i}{j}" for i, j in _pairs()),
    *(f"LbK{i}{j}" for i, j in _pairs()),
    "advKsf",
    *(f"da{k}" for k in INDICES),
    *(f"ds{k}" for k in INDICES),
    *(f"dKs{k}" for k in INDICES),
)


def rhs_expressions(conformal: str = "phi"):
    """The geometry kernel's outputs, symbolically, and the derivative registry.

    The conformal rate is keyed ``phi`` or ``W``. Every other key is in
    :data:`GEOMETRY_NAMES`, :data:`GAUGE_INGREDIENTS` or
    :data:`INTERMEDIATE_NAMES`. The gauge variables themselves have no
    entry: their rates are assembled outside, with ``a``.
    """
    state, derivatives, registry = symbolic.abstract_state(conformal)
    variables, d_connection, dd_shift = from_state(state, derivatives)
    theta = state["Theta"]
    d_theta = [derivatives[f"d_Theta_{k}"] for k in INDICES]
    constraint_function, kappa1, kappa2 = (
        sp.Symbol(name, real=True) for name in symbolic.PARAMETER_NAMES
    )
    rates, parts = symbolic.modified_parts(
        variables,
        theta,
        d_theta,
        constraint_function,
        kappa1,
        kappa2,
        d_connection=d_connection,
        dd_shift=dd_shift,
    )
    physical = variables.physical
    inverse = parts["inverse"]

    def first(name, k):
        return derivatives[f"d_{name}_{k}"]

    d_scalar = [first("sf", k) for k in INDICES]
    dd_scalar = [[derivatives[f"dd_sf_{min(i, j)}{max(i, j)}"] for j in INDICES] for i in INDICES]
    d_momentum = [first("Ksf", k) for k in INDICES]
    dt_scalar, dt_momentum, scalar_hessian = symbolic.scalar_rates(
        variables, state["Ksf"], d_scalar, dd_scalar, d_momentum, inverse
    )

    expressions: dict[str, Any] = {}
    if conformal == "W":
        expressions["W"] = -2 * state["W"] * rates["phi"]
    else:
        expressions["phi"] = rates["phi"]
    expressions["trK"] = rates["mean_curvature"]
    expressions["Theta"] = rates["theta"]
    expressions["sf"] = dt_scalar
    expressions["Ksf"] = dt_momentum
    for i in INDICES:
        expressions[f"Gt{i}"] = rates["connection"][i]
        for j in range(i, DIMENSION):
            expressions[f"gt{i}{j}"] = rates["conformal_metric"][i][j]
            expressions[f"At{i}{j}"] = rates["traceless_curvature"][i][j]

    shift = variables.shift
    d_lapse = physical.d_lapse
    expressions["adv_alpha"] = sum(shift[k] * d_lapse[k] for k in INDICES)
    for i in INDICES:
        expressions[f"adv_beta{i}"] = sum(shift[k] * physical.d_shift[k][i] for k in INDICES)
        expressions[f"grad_alpha{i}"] = sum(inverse[i][k] * d_lapse[k] for k in INDICES)

    covariant = symbolic.covariant_curvature_derivative(physical, inverse)
    lapse_second = lapse_hessian(physical, inverse)
    lie = symbolic.lie_shift_curvature(physical)
    for i, j in _pairs():
        expressions[f"gam{i}{j}"] = physical.metric[i][j]
        expressions[f"igam{i}{j}"] = inverse[i][j]
        expressions[f"Kx{i}{j}"] = physical.curvature[i][j]
        expressions[f"Ric{i}{j}"] = parts["ricci"][i][j]
        for k in INDICES:
            expressions[f"DK{k}_{i}{j}"] = covariant[k][i][j]
        expressions[f"DDa{i}{j}"] = lapse_second[i][j]
        expressions[f"DDs{i}{j}"] = scalar_hessian[i][j]
        expressions[f"LbK{i}{j}"] = lie[i][j]
    expressions["advKsf"] = sum(shift[k] * d_momentum[k] for k in INDICES)
    for k in INDICES:
        expressions[f"da{k}"] = d_lapse[k]
        expressions[f"ds{k}"] = d_scalar[k]
        expressions[f"dKs{k}"] = d_momentum[k]
    return expressions, registry


def rhs_kernel(
    order: int = 4,
    backend: str = "jax",
    conformal: str = "phi",
    jit: bool = True,
    use_cache: bool = True,
) -> codegen.Kernel:
    """The compiled geometry kernel, from cache when possible."""
    signature = (KERNEL_VERSION, "geometry", order, backend, conformal, jit)
    if signature in _RHS_CACHE:
        return _RHS_CACHE[signature]

    def build() -> str:
        expressions, registry = rhs_expressions(conformal)
        return codegen.build_source(
            expressions, registry, order=order, backend=backend, name="mccz4_rhs"
        )

    key = cache.key_for(*signature) if use_cache else None
    source, _ = cache.source_cached(key, build)
    kernel = codegen.from_source(source, backend=backend, jit=jit)
    _RHS_CACHE[signature] = kernel
    return kernel


_GB_CACHE: dict[tuple, codegen.Kernel] = {}


def gauss_bonnet_expressions():
    """The Gauss-Bonnet kernel's outputs: static matter, ``Xi_0`` and ``J``, by name."""
    from particlesim.symbolic import esgb

    matter = esgb.effective_matter()
    expressions: dict[str, Any] = {"rho_gb": matter["density"], "src_gb": matter["source"]}
    for i in INDICES:
        expressions[f"S_gb{i}"] = matter["momentum"][i]
    for i, j in _pairs():
        expressions[f"Sxx_gb{i}{j}"] = matter["stress"][i][j]
    for m, name in enumerate(esgb.UNKNOWNS):
        expressions[f"Xi{name}"] = matter["offset"][m]
        for n, other in enumerate(esgb.UNKNOWNS):
            expressions[f"J{name}_{other}"] = matter["jacobian"][m][n]
    leftover = {
        str(symbol)
        for value in expressions.values()
        for symbol in sp.sympify(value).free_symbols
        if str(symbol).startswith("_")
    }
    if leftover:
        raise RuntimeError(
            f"the Gauss-Bonnet kernel would read internal symbols {sorted(leftover)}"
        )
    return expressions


def gauss_bonnet_kernel(
    backend: str = "jax", jit: bool = True, use_cache: bool = True
) -> codegen.Kernel:
    """The compiled Gauss-Bonnet kernel: pure algebra on the geometry kernel's tensors."""
    signature = (GAUSS_BONNET_VERSION, backend, jit)
    if signature in _GB_CACHE:
        return _GB_CACHE[signature]

    def build() -> str:
        return codegen.build_source(gauss_bonnet_expressions(), {}, backend=backend, name="esgb")

    key = cache.key_for(*signature) if use_cache else None
    source, _ = cache.source_cached(key, build)
    kernel = codegen.from_source(source, backend=backend, jit=jit)
    _GB_CACHE[signature] = kernel
    return kernel


def _pair_value(container, prefix, i, j):
    return container[f"{prefix}{min(i, j)}{max(i, j)}"]


def _array_module(value):
    """NumPy or ``jax.numpy``, whichever ``value`` belongs to."""
    return bssn_solver._module("jax" if "jax" in type(value).__module__ else "numpy")


@dataclass(frozen=True)
class ModifiedKernel:
    """The generated kernels and the arithmetic around them, called as a kernel is.

    :class:`~particlesim.solvers.nr.bssn.Evolution` calls
    ``kernel(state, spacing)`` and adds dissipation and upwinding to what
    comes back; this returns the rate of every evolved variable.

    ``gauge_function`` and ``constraint_function`` are ``a`` and ``b``:
    numbers, or callables of the state returning arrays, for profiles that
    vary in space. ``coupling`` is a
    :class:`~particlesim.theories.gauss_bonnet.GaussBonnetCoupling` and
    ``coupling_squared`` is ``lambda^2``; with ``coupling=None`` the scalar
    is minimally coupled and the Gauss-Bonnet kernel is never called.

    ``theta_in_slicing`` makes the slicing read ``K - 2 Theta`` instead of
    ``K``, as Alic et al.'s CCZ4 and Aresté Saló, Clough and Figueras write
    it. :mod:`particlesim.solvers.nr.ccz4` reads ``K``, so reproducing it
    needs ``False``.
    """

    geometry: codegen.Kernel = field(repr=False)
    gauss_bonnet: codegen.Kernel | None = field(default=None, repr=False)
    conformal: str = "phi"
    gauge_function: Any = symbolic.GAUGE_FUNCTION
    constraint_function: Any = symbolic.CONSTRAINT_FUNCTION
    kappa1: float = 0.1
    kappa2: float = 0.0
    slicing: str = "one_plus_log"
    shift_condition: str = "gamma_driver"
    gauge_damping: float = 2.0
    advect: bool | str = True
    coupling: Any = None
    coupling_squared: float = 0.0
    theta_in_slicing: bool = False

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.geometry.outputs

    def summary(self) -> dict[str, Any]:
        out = {"geometry": self.geometry.summary()}
        if self.gauss_bonnet is not None:
            out["gauss_bonnet"] = self.gauss_bonnet.summary()
        return out

    @staticmethod
    def _value(parameter, state):
        return parameter(state) if callable(parameter) else parameter

    def coupling_values(self, scalar):
        """``(f, f', f'')`` of the coupling at ``scalar``, in the array module of ``scalar``."""
        from particlesim.theories.gauss_bonnet import _PHI

        module = _array_module(scalar)
        functions = [sp.diff(self.coupling.expression, _PHI, k) for k in range(3)]
        printer = "jax" if module.__name__.startswith("jax") else "numpy"
        values = sp.lambdify(_PHI, functions, printer)(scalar)
        return [module.asarray(value) * module.ones_like(scalar) for value in values]

    def __call__(self, fields: Mapping[str, Any], spacing: Sequence[float]):
        state = dict(fields)
        b = self._value(self.constraint_function, state)
        a = self._value(self.gauge_function, state)
        raw = self.geometry(
            {**state, "mccz4_b": b, "kappa1": self.kappa1, "kappa2": self.kappa2}, spacing
        )
        rates = {name: raw[name] for name in (self.conformal, *GEOMETRY_NAMES)}
        rates = self.matter_rates(state, raw, rates, b)
        return {**rates, **self.gauge_rates(state, raw, rates, a)}

    def shrink(self, state):
        """``e^(-4 phi)``, from whichever conformal variable the state carries."""
        if self.conformal == "W":
            return state["W"] ** 2
        return _array_module(state["alpha"]).exp(-4 * state["phi"])

    def matter(self, state, raw):
        """``(rho, S_i, S_ij, source, gauss_bonnet_outputs)`` of the scalar and the coupling."""
        metric = [[_pair_value(raw, "gam", i, j) for j in INDICES] for i in INDICES]
        inverse = [[_pair_value(raw, "igam", i, j) for j in INDICES] for i in INDICES]
        d_scalar = [raw[f"ds{k}"] for k in INDICES]
        density, momentum, stress = symbolic.scalar_stress(metric, inverse, d_scalar, state["Ksf"])
        source = None
        extra = None
        if self.coupling is not None and self.coupling_squared != 0.0:
            f, df, ddf = self.coupling_values(state["sf"])
            extra = self.gauss_bonnet(
                {
                    **raw,
                    "alpha": state["alpha"],
                    "Ksf": state["Ksf"],
                    "fGB": f,
                    "dfGB": df,
                    "ddfGB": ddf,
                    "lam2": self.coupling_squared,
                },
                (),
            )
            density = density + extra["rho_gb"]
            momentum = [momentum[i] + extra[f"S_gb{i}"] for i in INDICES]
            stress = [
                [stress[i][j] + _pair_value(extra, "Sxx_gb", i, j) for j in INDICES]
                for i in INDICES
            ]
            source = extra["src_gb"]
        return density, momentum, stress, source, extra

    def matter_rates(self, state, raw, rates, b):
        """The vacuum rates with all the matter added, the Gauss-Bonnet solve included."""
        shrink = self.shrink(state)
        density, momentum, stress, source, extra = self.matter(state, raw)
        conformal_metric = [[_pair_value(state, "gt", i, j) for j in INDICES] for i in INDICES]
        terms = symbolic.matter_terms(
            state["alpha"], shrink, conformal_metric, density, momentum, stress, b, source
        )
        out = dict(rates)
        out["trK"] = out["trK"] + terms["mean_curvature"]
        out["Theta"] = out["Theta"] + terms["theta"]
        if "scalar_momentum" in terms:
            out["Ksf"] = out["Ksf"] + terms["scalar_momentum"]
        for i in INDICES:
            out[f"Gt{i}"] = out[f"Gt{i}"] + terms["connection"][i]
            for j in range(i, DIMENSION):
                out[f"At{i}{j}"] = out[f"At{i}{j}"] + terms["traceless_curvature"][i][j]
        if extra is not None:
            out = self._second_derivatives(state, raw, out, extra, shrink)
        return out

    def curvature_rate(self, state, rates, shrink):
        """``d_t K_ij`` from the rates of ``phi`` (or ``W``), ``gammabar``, ``K`` and ``Abar``."""
        if self.conformal == "W":
            growth = -2 * rates["W"] / state["W"]  # d_t ln e^(4 phi)
        else:
            growth = 4 * rates["phi"]
        out = {}
        for i, j in _pairs():
            inner = state[f"At{i}{j}"] + state[f"gt{i}{j}"] * state["trK"] / 3
            d_inner = (
                rates[f"At{i}{j}"]
                + (rates[f"gt{i}{j}"] * state["trK"] + state[f"gt{i}{j}"] * rates["trK"]) / 3
            )
            out[f"{i}{j}"] = (growth * inner + d_inner) / shrink
        return out

    def _second_derivatives(self, state, raw, rates, extra, shrink):
        """Solve ``(1 - J) X = Xbar + Xi_0``; put ``X - Xbar`` into ``K``, ``Abar``, ``K_phi``."""
        from particlesim.symbolic.esgb import UNKNOWNS

        module = _array_module(state["alpha"])
        curvature_rate = self.curvature_rate(state, rates, shrink)
        bar = [curvature_rate[name] - raw[f"LbK{name}"] for name in UNKNOWNS[:-1]]
        bar.append(rates["Ksf"] - raw["advKsf"])
        size = len(UNKNOWNS)
        shape = state["alpha"].shape

        def full(value):
            return module.broadcast_to(module.asarray(value), shape)

        right = module.stack([full(bar[m] + extra[f"Xi{n}"]) for m, n in enumerate(UNKNOWNS)])
        matrix = module.stack(
            [
                module.stack(
                    [
                        full((1.0 if m == n else 0.0) - extra[f"J{name}_{other}"])
                        for n, other in enumerate(UNKNOWNS)
                    ]
                )
                for m, name in enumerate(UNKNOWNS)
            ]
        )
        flat_matrix = module.moveaxis(matrix.reshape(size, size, -1), -1, 0)
        flat_right = module.moveaxis(right.reshape(size, -1), -1, 0)
        solved = module.linalg.solve(flat_matrix, flat_right[..., None])[..., 0]
        solution = module.moveaxis(solved, 0, -1).reshape(size, *shape)
        change = {name: solution[m] - bar[m] for m, name in enumerate(UNKNOWNS)}

        metric = [[_pair_value(raw, "gam", i, j) for j in INDICES] for i in INDICES]
        inverse = [[_pair_value(raw, "igam", i, j) for j in INDICES] for i in INDICES]
        tensor = [[change[f"{min(i, j)}{max(i, j)}"] for j in INDICES] for i in INDICES]
        traced = sum(inverse[i][j] * tensor[i][j] for i in INDICES for j in INDICES)
        out = dict(rates)
        out["trK"] = out["trK"] + traced
        out["Ksf"] = out["Ksf"] + change["s"]
        for i, j in _pairs():
            out[f"At{i}{j}"] = out[f"At{i}{j}"] + shrink * (
                tensor[i][j] - metric[i][j] * traced / 3
            )
        return out

    def gauge_rates(self, state, raw, rates, a):
        """``(alpha, beta^i, B^i)`` rates from the modified gauge, with ``a``."""
        advect_lapse = self.advect is True or self.advect == "lapse"
        advect_shift = self.advect is True
        driver = [state.get(f"B{i}", 0 * state["alpha"]) for i in INDICES]
        mean = state["trK"] - 2 * state["Theta"] if self.theta_in_slicing else state["trK"]
        dt_lapse, dt_shift, dt_driver = symbolic.modified_gauge(
            state["alpha"],
            mean,
            driver,
            [rates[f"Gt{i}"] for i in INDICES],
            a,
            [raw[f"grad_alpha{i}"] for i in INDICES],
            advection_lapse=raw["adv_alpha"] if advect_lapse else None,
            advection_shift=[raw[f"adv_beta{i}"] for i in INDICES] if advect_shift else None,
            slicing=self.slicing,
            shift_condition=self.shift_condition,
            damping=self.gauge_damping,
        )
        out = {"alpha": dt_lapse}
        for i in INDICES:
            out[f"beta{i}"] = dt_shift[i]
            out[f"B{i}"] = dt_driver[i]
        return out


@dataclass(frozen=True)
class ModifiedEvolution(bssn_solver.Evolution):
    """:class:`~particlesim.solvers.nr.bssn.Evolution`, without one ``jit`` around everything.

    The parent compiles its whole right-hand side as one XLA module. For
    this one -- two generated kernels, the matter, a batched 7 x 7 solve and
    the gauge -- that module took 759 s to compile at 40^3 and then ran at
    84 s a call, against 2.3 s when each generated kernel is compiled on its
    own and the arithmetic between them is dispatched as it stands. The
    kernels are already compiled; only the glue is not.
    """

    def right_hand_side(self, state):
        return self._rates(dict(state))


def admit_theory(theory, dimensions: int = 3) -> None:
    """ADR-008 for this solver: what ``nr.modified`` evolves, and what it refuses.

    A plugin at its GR limit is GR. A plugin declaring ``modified_ccz4`` is
    admitted when it is one this module implements -- a Gauss-Bonnet coupling
    with a ``coupling_function`` and ``alpha = lambda^2``. Anything else goes
    through :func:`particlesim.solvers.nr.bssn.admit_theory`, which refuses
    ``order_reduced`` with :class:`~particlesim.theories.base.IllPosedRun`.
    """
    if theory is None:
        return
    limit = theory.gr_limit()
    if all(theory.values.get(name) == value for name, value in limit.items()):
        return
    if theory.formulation == "modified_ccz4" and hasattr(theory, "coupling_function"):
        return
    bssn_solver.admit_theory(theory, dimensions=dimensions)


def build(
    spacing,
    order: int = 4,
    backend: str = "jax",
    conformal: str = "phi",
    gauge_function=symbolic.GAUGE_FUNCTION,
    constraint_function=symbolic.CONSTRAINT_FUNCTION,
    damping: float = 0.1,
    damping_mix: float = 0.0,
    slicing: str = "one_plus_log",
    shift_condition: str = "gamma_driver",
    gauge_damping: float = 2.0,
    dissipation: float = bssn_solver.DISSIPATION,
    courant: float = bssn_solver.COURANT,
    jit: bool = True,
    enforce: bool = True,
    advect: bool | str = True,
    upwind: bool = False,
    theory=None,
    coupling=None,
    coupling_squared: float = 0.0,
    theta_in_slicing: bool = False,
) -> bssn_solver.Evolution:
    """An :class:`~particlesim.solvers.nr.bssn.Evolution` running modified CCZ4.

    ``theory`` is checked by :func:`admit_theory` before anything is built
    and, when it is a Gauss-Bonnet plugin, supplies ``coupling`` and
    ``coupling_squared`` (``alpha`` of ``string.eft4d.dgb`` is
    ``lambda^2``). ``damping`` and ``damping_mix`` are ``kappa_1`` and
    ``kappa_2``. ``theta_in_slicing`` is :class:`ModifiedKernel`'s.
    """
    if theory is not None:
        admit_theory(theory, dimensions=len(spacing))
        if hasattr(theory, "coupling_function"):
            coupling = theory.coupling_function
            coupling_squared = float(theory.values["alpha"])
    needs_coupling = coupling is not None and coupling_squared != 0.0
    kernel = ModifiedKernel(
        geometry=rhs_kernel(order=order, backend=backend, conformal=conformal, jit=jit),
        gauss_bonnet=gauss_bonnet_kernel(backend=backend, jit=jit) if needs_coupling else None,
        conformal=conformal,
        gauge_function=gauge_function,
        constraint_function=constraint_function,
        kappa1=damping,
        kappa2=damping_mix,
        slicing=slicing,
        shift_condition=shift_condition,
        gauge_damping=gauge_damping,
        advect=advect,
        coupling=coupling if needs_coupling else None,
        coupling_squared=coupling_squared if needs_coupling else 0.0,
        theta_in_slicing=theta_in_slicing,
    )
    return ModifiedEvolution(
        spacing=tuple(float(value) for value in spacing),
        order=order,
        backend=backend,
        dissipation=dissipation,
        courant=courant,
        kernel=kernel,
        constraint_kernel=bssn_solver.constraint_kernel(order=order, backend=backend, jit=jit),
        slicing=slicing,
        shift_condition=shift_condition,
        damping=gauge_damping,
        enforce=enforce,
        upwind=upwind,
        advect=advect,
    )


def with_scalar(state, scalar=None, momentum=None, backend: str = "jax"):
    """CCZ4 or BSSN data with a scalar and its momentum, zero unless given; ``Theta = 0``."""
    module = bssn_solver._module(backend)
    out = dict(state)
    zero = module.zeros_like(state["alpha"])
    out.setdefault("Theta", zero)
    out["sf"] = zero if scalar is None else module.asarray(scalar)
    out["Ksf"] = zero if momentum is None else module.asarray(momentum)
    return out


def scalar_gauge_wave(
    shape=(32, 8, 8),
    amplitude: float = 0.1,
    scalar_amplitude: float = 1e-6,
    time: float = 0.0,
    extent: float = 1.0,
    backend: str = "jax",
):
    """A gauge wave with a scalar travelling the other way through it: exact for a test field.

    The gauge wave is ``H (-dt^2 + dx^2) + dy^2 + dz^2``, conformally flat in
    the ``(t, x)`` plane, and the wave operator of a field that depends on
    ``t`` and ``x`` alone is ``(-d_t^2 + d_x^2) / H`` there. So
    ``phi = A sin(2 pi (x + t))`` solves ``box phi = 0`` on it exactly, with
    ``K_phi = -d_t phi / alpha`` and ``alpha = sqrt(H)``: a closed form that
    the scalar's coupling to a lapse and a metric that both vary has to
    reproduce. Its stress-energy is second order in ``A``, and at ``1e-6``
    it is below the truncation error.
    """
    module = bssn_solver._module(backend)
    state, spacing = bssn_solver.gauge_wave(
        shape=shape, amplitude=amplitude, time=time, extent=extent, backend=backend
    )
    mesh, _ = bssn_solver._grid(shape, extent, backend)
    period = float(np.atleast_1d(extent)[0])
    argument = 2 * np.pi * (mesh[0] / period + time)
    scalar = scalar_amplitude * module.sin(argument)
    rate = scalar_amplitude * 2 * np.pi * module.cos(argument)
    return with_scalar(state, scalar, -rate / state["alpha"], backend=backend), spacing


def static_schwarzschild(shape, extent, mass: float = 1.0, backend: str = "jax"):
    """Isotropic Schwarzschild with its static lapse, for a frozen gauge: time is Killing time.

    ``psi = 1 + M/2R`` and ``alpha = (1 - M/2R)/(1 + M/2R)``, zero shift,
    ``K_ij = 0``, no scalar. Frozen, this lapse makes the coordinate time
    Schwarzschild's ``t``, so a field on the hole can be compared with a
    one-dimensional evolution in the same time without tracking anything.
    Inside the horizon the lapse is taken smoothly to zero near ``R = 0``,
    the other asymptotic end, by ``1 - exp(-(R / 0.35 M)^8)``: that region
    is outside the exterior's domain of dependence, and with the static
    lapse, ``-1`` there, the puncture's truncation error runs away. At the
    horizon the taper differs from one by ``3e-8``. Returns
    ``(state, spacing, R)``.
    """
    module = bssn_solver._module(backend)
    mesh, spacing = bssn_solver._grid(shape, extent, "numpy")
    lengths = tuple(float(v) for v in np.atleast_1d(extent))
    if len(lengths) == 1:
        lengths = lengths * DIMENSION
    centre = [lengths[i] / 2 + spacing[i] / 2 for i in INDICES]
    radius = np.sqrt(sum((np.asarray(mesh[i]) - centre[i]) ** 2 for i in INDICES))
    psi = 1 + mass / (2 * radius)
    lapse = (1 - mass / (2 * radius)) / psi * (1 - np.exp(-((radius / (0.35 * mass)) ** 8)))
    zero = np.zeros_like(radius)
    state: dict[str, Any] = {"phi": np.log(psi), "alpha": lapse, "trK": zero, "Theta": zero}
    state["sf"] = zero
    state["Ksf"] = zero
    for i in INDICES:
        state[f"beta{i}"] = zero
        state[f"B{i}"] = zero
        state[f"Gt{i}"] = zero
        for j in range(i, DIMENSION):
            state[f"gt{i}{j}"] = np.full_like(radius, 1.0 if i == j else 0.0)
            state[f"At{i}{j}"] = zero
    return {k: module.asarray(v) for k, v in state.items()}, spacing, radius


@dataclass(frozen=True)
class IsotropicHole:
    """A static hole of :func:`~particlesim.theories.gauss_bonnet.static_black_hole` in 3-D.

    The static slice through the bifurcation sphere is two copies of the
    exterior glued at the horizon. In the isotropic radius ``R``, with
    ``e^(2 Lambda) dr^2 + r^2 dOmega^2 = psi^4 (dR^2 + R^2 dOmega^2)``, it is
    symmetric under ``R -> R_H^2 / R``: ``psi`` and ``phi`` are even in
    ``rho = ln(R / R_H)``, and the static lapse is odd, so it changes sign at
    the horizon and the slice is regular there. Built by integrating the
    static equations outward from the horizon with the proper distance and
    ``rho`` carried along, ``d rho / d ell = 1 / r``, and fitting splines in
    ``rho`` to the profile and its mirror image.

    ``lapse``, ``scalar`` and ``log_conformal`` (``ln psi``) take ``R``.
    Lengths are in units of ``lambda``; ``R -> r`` far away.
    """

    horizon_radius: float
    horizon_scalar: float
    mass: float
    isotropic_horizon: float
    coupling: Any = field(repr=False)
    _splines: tuple = field(repr=False, default=())

    @classmethod
    def solve(cls, horizon_radius: float, coupling, horizon_scalar=None, epsilon: float = 1e-7):
        from scipy.integrate import solve_ivp
        from scipy.interpolate import CubicSpline

        from particlesim.theories import gauss_bonnet as gb

        if horizon_scalar is None:
            hole = gb.static_black_hole(horizon_radius, coupling)
            if hole is None:
                raise ValueError(f"no static hole of radius {horizon_radius} for {coupling.name}")
            horizon_scalar = hole.horizon_scalar
        slope = gb.horizon_slope(horizon_radius, horizon_scalar, coupling)
        if slope is None:
            raise ValueError("no regular horizon there")
        rates = gb._rates()
        start = horizon_radius * (1.0 + epsilon)
        gap0 = start - horizon_radius
        dPhi = 1.0 / (2.0 * gap0)
        phi = horizon_scalar + slope * gap0
        _, df, _ = coupling(phi)
        b = -4.0 * dPhi * df * slope - 2.0 * dPhi * start + slope**2 * start**2 - 1.0
        c = 12.0 * dPhi * df * slope
        X = (-b + np.sqrt(b * b - 4.0 * c)) / 2.0
        # Proper distance from the horizon to the start: e^Lambda ~ g / sqrt(x).
        ell0 = 2.0 * np.sqrt(X) * gap0
        y0 = [0.5 * np.log(X), dPhi, phi, slope, 0.0, ell0, ell0 / horizon_radius]

        def rhs(s, y):
            gap = np.exp(s)
            Lam, dPhi, phi, dphi, _, _, _ = y
            f, df, ddf = coupling(phi)
            dLam, ddPhi, ddphi = rates(horizon_radius + gap, Lam, dPhi, phi, dphi, f, df, ddf, 1.0)
            stretch = gap * np.exp(Lam)
            return [
                gap * dLam,
                gap * ddPhi,
                gap * dphi,
                gap * ddphi,
                gap * dPhi,
                stretch,
                stretch / (horizon_radius + gap),
            ]

        outer = 4000.0
        span = (np.log(gap0), np.log(outer - horizon_radius))
        solution = solve_ivp(
            rhs, span, y0, method="LSODA", rtol=1e-12, atol=1e-14, dense_output=True
        )
        if solution.status != 0:
            raise RuntimeError(f"static integration failed: {solution.message}")
        s = np.linspace(span[0], span[1], 40001)
        Lam, _, scalar, _, Phi, _, rho = solution.sol(s)
        r = horizon_radius + np.exp(s)
        mass = r[-1] / 2.0 * (1.0 - np.exp(-2.0 * Lam[-1]))
        # Far away Phi + M/r and ln R - ln r + M/r tend to their limits.
        Phi_inf = Phi[-1] + mass / r[-1]
        log_isotropic_horizon = np.log(r[-1]) - mass / r[-1] - rho[-1]
        lapse = np.exp(Phi - Phi_inf)
        log_r = np.log(r)
        mirror = np.concatenate([-rho[::-1], rho])
        splines = (
            CubicSpline(mirror, np.concatenate([-lapse[::-1], lapse])),
            CubicSpline(mirror, np.concatenate([scalar[::-1], scalar])),
            CubicSpline(mirror, np.concatenate([log_r[::-1], log_r])),
        )
        return cls(
            horizon_radius=float(horizon_radius),
            horizon_scalar=float(horizon_scalar),
            mass=float(mass),
            isotropic_horizon=float(np.exp(log_isotropic_horizon)),
            coupling=coupling,
            _splines=splines,
        )

    def _rho(self, radius):
        return np.log(np.asarray(radius, dtype=float) / self.isotropic_horizon)

    def lapse(self, radius):
        return self._splines[0](self._rho(radius))

    def scalar(self, radius):
        return self._splines[1](self._rho(radius))

    def log_conformal(self, radius):
        """``ln psi = (ln r - ln R) / 2``."""
        rho = self._rho(radius)
        return 0.5 * (self._splines[2](rho) - np.log(self.isotropic_horizon) - rho)

    def state(self, shape, extent, backend: str = "jax"):
        """The hole at the centre of a periodic box, half a cell off every grid point.

        Static, time-symmetric data: ``K_ij = 0``, ``K_phi = 0``, zero shift,
        ``gammabar_ij = delta_ij``, ``Gammabar^i = 0``. Everything is exact
        except at the box faces, where the periodic stencils wrap, and near
        ``R = 0``, the other asymptotic end.
        """
        module = bssn_solver._module(backend)
        mesh, spacing = bssn_solver._grid(shape, extent, "numpy")
        lengths = tuple(float(v) for v in np.atleast_1d(extent))
        if len(lengths) == 1:
            lengths = lengths * DIMENSION
        centre = [lengths[i] / 2 + spacing[i] / 2 for i in INDICES]
        radius = np.sqrt(sum((np.asarray(mesh[i]) - centre[i]) ** 2 for i in INDICES))
        state: dict[str, Any] = {
            "phi": self.log_conformal(radius),
            "alpha": self.lapse(radius),
            "trK": np.zeros_like(radius),
            "Theta": np.zeros_like(radius),
            "sf": self.scalar(radius),
            "Ksf": np.zeros_like(radius),
        }
        for i in INDICES:
            state[f"beta{i}"] = np.zeros_like(radius)
            state[f"B{i}"] = np.zeros_like(radius)
            state[f"Gt{i}"] = np.zeros_like(radius)
            for j in range(i, DIMENSION):
                state[f"gt{i}{j}"] = np.full_like(radius, 1.0 if i == j else 0.0)
                state[f"At{i}{j}"] = np.zeros_like(radius)
        return {k: module.asarray(v) for k, v in state.items()}, spacing, radius


def matter_constraints(evolution, state):
    """``(H, M_i)`` norms with the scalar's and the coupling's matter in them.

    ``rho`` and ``S_i`` of the effective matter depend on the slice only,
    so the constraints of a coupled run are as well defined as in vacuum.
    The vacuum parts come from the ADM constraint kernel, as for
    :func:`~particlesim.solvers.nr.bssn.constraints`.
    """
    kernel = evolution.kernel
    spacing = tuple(evolution.spacing)
    vacuum = evolution.constraint_kernel(
        bssn_solver.physical_slice_arrays(state, evolution.backend), spacing
    )
    b = kernel._value(kernel.constraint_function, state)
    raw = kernel.geometry(
        {**state, "mccz4_b": b, "kappa1": kernel.kappa1, "kappa2": kernel.kappa2}, spacing
    )
    density, momentum, _, _, _ = kernel.matter(state, raw)
    inverse = [[_pair_value(raw, "igam", i, j) for j in INDICES] for i in INDICES]
    hamiltonian = np.asarray(vacuum["hamiltonian"]) - 16 * np.pi * np.asarray(density)
    components = [
        np.asarray(vacuum[f"momentum{i}"])
        - 8 * np.pi * np.asarray(sum(inverse[i][j] * momentum[j] for j in INDICES))
        for i in INDICES
    ]
    return {
        "hamiltonian": float(np.sqrt(np.mean(hamiltonian**2))),
        "momentum": float(np.sqrt(sum(np.mean(c**2) for c in components))),
        "hamiltonian_field": hamiltonian,
    }


__all__ = [
    "GAUGE_INGREDIENTS",
    "GEOMETRY_NAMES",
    "INTERMEDIATE_NAMES",
    "GAUSS_BONNET_VERSION",
    "KERNEL_VERSION",
    "IsotropicHole",
    "ModifiedEvolution",
    "ModifiedKernel",
    "admit_theory",
    "build",
    "gauss_bonnet_expressions",
    "gauss_bonnet_kernel",
    "matter_constraints",
    "rhs_expressions",
    "rhs_kernel",
    "scalar_gauge_wave",
    "static_schwarzschild",
    "with_scalar",
]
