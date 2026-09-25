"""CCZ4 evolution: the BSSN integrator, driven by the Z4 right-hand sides.

Everything around the equations is shared with
:mod:`particlesim.solvers.nr.bssn` -- the same ``Evolution``, the same RK4,
the same Kreiss-Oliger dissipation, the same algebraic projection, the same
constraint diagnostics. What changes is the kernel: twenty-five equations
from :mod:`particlesim.symbolic.ccz4` instead of twenty-four from
:mod:`particlesim.symbolic.bssn`, with ``Theta`` on the end.

That sharing is the point rather than a convenience. The two systems differ
only in their right-hand sides, so a difference in what a run does is a
difference the equations made, and not the integrator, the gauge or the
diagnostics quietly differing too.

**``Theta`` is left out of the algebraic projection and the constraint
norms, deliberately.** ``Evolution.project`` restores ``det gammabar = 1``
and ``gammabar^ij Abar_ij = 0``, which are statements about the conformal
variables and say nothing about ``Theta``; and ``constraints()`` measures
the Hamiltonian and momentum constraints of the *physical* slice the state
stands for, which is the quantity with a meaning independent of which
system is being evolved. Folding ``Theta`` into either would make the
damping look better than it is by changing what is measured, when the
whole claim is that damping changes what the solution does.
"""

from __future__ import annotations

from typing import Any

from particlesim.solvers.nr import bssn as bssn_solver
from particlesim.solvers.nr.bssn import (
    COURANT,
    DISSIPATION,
    Evolution,
    admit_theory,
    constraint_kernel,
)
from particlesim.symbolic import cache, codegen
from particlesim.symbolic import ccz4 as symbolic_ccz4
from particlesim.symbolic.bssn import from_state
from particlesim.symbolic.threeplusone import DIMENSION, INDICES

#: Bumped when the generated equations change, so stale cache entries miss.
KERNEL_VERSION = "ccz4-3"

_RHS_CACHE: dict[tuple, codegen.Kernel] = {}


def rhs_expressions(
    slicing: str = "one_plus_log",
    shift_condition: str = "gamma_driver",
    gauge_damping: float = 2.0,
    damping: float = symbolic_ccz4.DAMPING,
    damping_mix: float = symbolic_ccz4.DAMPING_MIX,
    advect: bool = True,
):
    """The twenty-five right-hand sides, symbolically, keyed by state name.

    ``from_state`` is BSSN's: the CCZ4 state is the BSSN state plus
    ``Theta``, and the twenty-four it shares mean the same things. The
    evolved connection is ``Gammahat^i``, which is what the conformal Ricci
    tensor uses -- the same substitution that makes BSSN strongly
    hyperbolic, with ``Z_i`` now riding along inside it.
    """
    state, derivatives, registry = symbolic_ccz4.abstract_state()
    variables, d_connection, dd_shift = from_state(state, derivatives)
    theta = state["Theta"]
    d_theta = [derivatives[f"d_Theta_{k}"] for k in INDICES]

    geometry = symbolic_ccz4.ccz4_rhs(
        variables,
        theta,
        d_theta,
        d_connection=d_connection,
        dd_shift=dd_shift,
        damping=damping,
        damping_mix=damping_mix,
    )
    dt_lapse, dt_shift, dt_driver = symbolic_ccz4.bssn.gauge_rhs(
        variables,
        geometry["connection"],
        state,
        slicing=slicing,
        shift_condition=shift_condition,
        damping=gauge_damping,
        advect=advect,
    )

    expressions: dict[str, Any] = {
        "phi": geometry["phi"],
        "trK": geometry["mean_curvature"],
        "Theta": geometry["theta"],
    }
    for i in INDICES:
        expressions[f"Gt{i}"] = geometry["connection"][i]
        expressions[f"beta{i}"] = dt_shift[i]
        expressions[f"B{i}"] = dt_driver[i]
        for j in range(i, DIMENSION):
            expressions[f"gt{i}{j}"] = geometry["conformal_metric"][i][j]
            expressions[f"At{i}{j}"] = geometry["traceless_curvature"][i][j]
    expressions["alpha"] = dt_lapse
    return expressions, registry


def rhs_kernel(
    order: int = 4,
    backend: str = "jax",
    slicing: str = "one_plus_log",
    shift_condition: str = "gamma_driver",
    gauge_damping: float = 2.0,
    damping: float = symbolic_ccz4.DAMPING,
    damping_mix: float = symbolic_ccz4.DAMPING_MIX,
    advect: bool = True,
    jit: bool = True,
    use_cache: bool = True,
) -> codegen.Kernel:
    """The compiled CCZ4 right-hand side, from cache when possible.

    The damping constants are baked into the generated source rather than
    passed at call time, because they multiply terms that
    common-subexpression elimination can then fold away entirely when they
    are zero -- which is what makes the undamped comparison run the same
    speed as the damped one instead of carrying dead arithmetic. The cost
    is a kernel per value, and they are cached separately.
    """
    signature = (
        KERNEL_VERSION,
        order,
        backend,
        slicing,
        shift_condition,
        gauge_damping,
        damping,
        damping_mix,
        advect,
        jit,
    )
    if signature in _RHS_CACHE:
        return _RHS_CACHE[signature]

    def build() -> str:
        expressions, registry = rhs_expressions(
            slicing, shift_condition, gauge_damping, damping, damping_mix, advect
        )
        return codegen.build_source(
            expressions, registry, order=order, backend=backend, name="ccz4_rhs"
        )

    key = cache.key_for(*signature) if use_cache else None
    source, _ = cache.source_cached(key, build)
    kernel = codegen.from_source(source, backend=backend, jit=jit)
    _RHS_CACHE[signature] = kernel
    return kernel


def build(
    spacing,
    order: int = 4,
    backend: str = "jax",
    slicing: str = "one_plus_log",
    shift_condition: str = "gamma_driver",
    gauge_damping: float = 2.0,
    damping: float = symbolic_ccz4.DAMPING,
    damping_mix: float = symbolic_ccz4.DAMPING_MIX,
    dissipation: float = DISSIPATION,
    courant: float = COURANT,
    jit: bool = True,
    enforce: bool = True,
    theory=None,
) -> Evolution:
    """An :class:`~particlesim.solvers.nr.bssn.Evolution` running CCZ4.

    ``theory``, if given, is checked by
    :func:`~particlesim.solvers.nr.bssn.admit_theory` before anything is built.
    """
    admit_theory(theory, dimensions=len(spacing))
    return Evolution(
        spacing=tuple(float(value) for value in spacing),
        order=order,
        backend=backend,
        dissipation=dissipation,
        courant=courant,
        kernel=rhs_kernel(
            order=order,
            backend=backend,
            slicing=slicing,
            shift_condition=shift_condition,
            gauge_damping=gauge_damping,
            damping=damping,
            damping_mix=damping_mix,
            jit=jit,
        ),
        constraint_kernel=constraint_kernel(order=order, backend=backend, jit=jit),
        slicing=slicing,
        shift_condition=shift_condition,
        damping=gauge_damping,
        enforce=enforce,
    )


def with_theta(state, backend: str = "jax"):
    """BSSN initial data as CCZ4 initial data, with ``Theta = 0``.

    Every exact solution is on the constraint surface, so ``Theta`` and
    ``Z_i`` start at zero for all of them -- which is also why a gauge wave
    cannot tell the two systems apart, and why the damping has to be
    measured on data deliberately pushed off that surface.
    """
    module = bssn_solver._module(backend)
    out = dict(state)
    out["Theta"] = module.zeros_like(state["phi"])
    return out


def gauge_wave(*args, **kwargs):
    """:func:`particlesim.solvers.nr.bssn.gauge_wave` with ``Theta = 0``."""
    state, spacing = bssn_solver.gauge_wave(*args, **kwargs)
    return with_theta(state, kwargs.get("backend", "jax")), spacing


def brill_lindquist(*args, **kwargs):
    """:func:`particlesim.solvers.nr.bssn.brill_lindquist` with ``Theta = 0``."""
    state, spacing = bssn_solver.brill_lindquist(*args, **kwargs)
    return with_theta(state, kwargs.get("backend", "jax")), spacing


__all__ = [
    "KERNEL_VERSION",
    "brill_lindquist",
    "build",
    "gauge_wave",
    "rhs_expressions",
    "rhs_kernel",
    "with_theta",
]
