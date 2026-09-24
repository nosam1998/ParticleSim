"""BSSN evolution in JAX, from the symbolic pipeline (Section 5.4, M4).

The right-hand side is not written here. It is derived by
:mod:`particlesim.symbolic.bssn` from the 3+1 split, reduced by
common-subexpression elimination and emitted as a kernel by
:mod:`particlesim.symbolic.codegen`; this module is the integrator, the
gauge, the initial data and the diagnostics around it. That division is the
point of the pipeline: the equations being evolved are the ones the
symbolic tests checked against exact solutions, not a transcription of
them.

What the derivation costs and what it buys:

* the twenty-four right-hand sides are **1 451 185 operations** as written
  and **3 017** after elimination, a factor of 481;
* the elimination takes about three and a half minutes of SymPy, so the
  generated source is cached on disk by
  :func:`particlesim.symbolic.cache.source_cached` and a second run pays a
  file read;
* the kernel is ``jax.jit``-compiled, and at 32^3 that is worth seven times
  the NumPy path, almost all of which is the temporaries XLA fuses.

**The conformal Ricci tensor is written with the evolved ``Gammabar^i``**
(``ricci_form="connection"``), which is what makes the system strongly
hyperbolic. Computing ``Rbar_ij`` from the conformal metric instead is the
same tensor and does not work: the gauge wave then grows like
``exp(c t / h)``, converging at fourth order until the grid is fine enough
to show it. See :func:`particlesim.symbolic.bssn.conformal_connection_ricci`
and ``docs/benchmarks.md``.

**Periodic, on purpose.** The emitted stencils are ``roll``-based, so the
domain is a torus. Every test an evolution scheme is first judged by is
periodic -- a gauge wave, a Teukolsky wave -- and boundary conditions are a
separate piece of work with their own failure modes. A puncture on a torus
is not a black hole in an asymptotically flat spacetime, and
:func:`brill_lindquist` says so rather than pretending the difference is
small.

**Kreiss-Oliger dissipation is applied outside the kernel.** It is linear
in the state and needs no elimination, so putting it in the kernel would
add seventy-two stencils to a kernel that already has a hundred and
twenty-nine, for arithmetic that is three lines here. Centred differences
do not damp the shortest wavelength the grid carries, and without it a run
of this kind fails from grid-scale noise rather than from anything
physical. It is not a substitute for hyperbolicity: tripling it changed the
growth rate of the weakly hyperbolic form by a quarter and stopped nothing.

**The algebraic constraints are projected after each step** by
:meth:`Evolution.project`, and measured by :func:`constraints`, which never
enforces them. Both halves are needed: nothing in the right-hand side pulls
a discrete run back onto ``det gammabar = 1``, and a routine that enforced
and reported would report zero for a drift it was creating.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import comb
from typing import Any

import numpy as np

from particlesim.symbolic import bssn as symbolic_bssn
from particlesim.symbolic import cache, codegen
from particlesim.symbolic.threeplusone import (
    DIMENSION,
    INDICES,
    abstract_slice,
    determinant,
    hamiltonian_constraint,
    inverse_metric,
    momentum_constraint,
    trace,
)

#: Bumped when the generated equations change, so stale cache entries are missed.
KERNEL_VERSION = "bssn-2"

#: Courant factor. RK4 with fourth-order centred differences is stable well
#: above this; 0.25 is the usual working value and leaves room for the
#: gauge speeds a moving-puncture run reaches.
COURANT = 0.25

#: Kreiss-Oliger dissipation strength, in the convention of
#: :func:`particlesim.core.grid.kreiss_oliger`.
DISSIPATION = 0.1


def _module(backend: str):
    """The array module, with ``jax_enable_x64`` already on for JAX.

    Delegated to the emitter's own backend resolver rather than importing
    ``jax.numpy`` here, because the flag has to be set *before* the first
    array is made and initial data is built before any kernel is. Do it
    the other way and ``gauge_wave`` returns float32: the run still looks
    plausible, and ``det gammabar - 1`` sits at 1.2e-07 -- which is
    2^-23, and was found that way.
    """
    return codegen._backend(backend)


# --- the emitted kernels ------------------------------------------------


#: Compiled dissipation operators, keyed by the evolution they belong to.
#:
#: ``Evolution`` is a frozen dataclass and so hashable, which makes it its
#: own cache key. Compiling inside the call instead would retrace every
#: time and spend more in XLA than in the run.
_DISSIPATORS: dict[Any, Callable] = {}

#: Compiled upwinding corrections, keyed the same way.
_UPWINDERS: dict[Any, Callable] = {}

#: The variables with an advection term ``beta^k d_k f`` in their equation.
#:
#: Everything but the Gamma-driver's ``B^i``, whose equation here has none.
#: Theta is CCZ4's and is advected there; a state without it simply does not
#: carry it.
UNADVECTED = frozenset(f"B{i}" for i in INDICES)

#: The fourth-order lopsided first derivative minus the centred one, for a
#: shift pointing along ``+k``: offsets ``-2 .. 3``.
#:
#: The lopsided stencil ``(-3, -10, 18, -6, 1) / 12`` on offsets ``-1 .. 3``
#: less the centred ``(1, -8, 0, 8, -1) / 12`` on ``-2 .. 2`` is a single
#: fifth difference, ``(-1, 5, -10, 10, -5, 1) / 12``. So upwinding the
#: advection terms *is* the centred scheme plus an ``h^4`` dissipation aimed
#: along the shift, which is why it can be added outside the kernel exactly
#: as Kreiss-Oliger is. For a shift along ``-k`` the stencil is mirrored.
UPWIND_CORRECTION = (
    (-2, -1 / 12),
    (-1, 5 / 12),
    (0, -10 / 12),
    (1, 10 / 12),
    (2, -5 / 12),
    (3, 1 / 12),
)

_RHS_CACHE: dict[tuple, codegen.Kernel] = {}
_CONSTRAINT_CACHE: dict[tuple, codegen.Kernel] = {}


def rhs_expressions(
    slicing: str = "one_plus_log",
    shift_condition: str = "gamma_driver",
    damping: float = 2.0,
    advect: bool | str = True,
):
    """The twenty-four right-hand sides, symbolically, keyed by state name.

    Derived rather than typed: :func:`particlesim.symbolic.bssn.from_state`
    rebuilds the physical slice from the evolved variables and
    :func:`particlesim.symbolic.bssn.bssn_rhs` supplies the seventeen
    geometric equations, with the gauge on top.
    """
    state, derivatives, registry = symbolic_bssn.abstract_state()
    variables, d_connection, dd_shift = symbolic_bssn.from_state(state, derivatives)
    geometry = symbolic_bssn.bssn_rhs(
        variables,
        d_connection=d_connection,
        dd_shift=dd_shift,
        ricci_form="connection",
    )
    dt_lapse, dt_shift, dt_driver = symbolic_bssn.gauge_rhs(
        variables,
        geometry["connection"],
        state,
        slicing=slicing,
        shift_condition=shift_condition,
        damping=damping,
        advect=advect,
    )

    expressions: dict[str, Any] = {"phi": geometry["phi"], "trK": geometry["mean_curvature"]}
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
    damping: float = 2.0,
    advect: bool | str = True,
    jit: bool = True,
    use_cache: bool = True,
) -> codegen.Kernel:
    """The compiled right-hand side, from cache when possible.

    Three layers, because the elimination is a minute and the compile is
    microseconds: an in-process dictionary, then the on-disk source cache,
    then the derivation.
    """
    signature = (KERNEL_VERSION, order, backend, slicing, shift_condition, damping, advect, jit)
    if signature in _RHS_CACHE:
        return _RHS_CACHE[signature]

    def build() -> str:
        expressions, registry = rhs_expressions(slicing, shift_condition, damping, advect)
        return codegen.build_source(
            expressions, registry, order=order, backend=backend, name="bssn_rhs"
        )

    key = cache.key_for(*signature) if use_cache else None
    source, _ = cache.source_cached(key, build)
    kernel = codegen.from_source(source, backend=backend, jit=jit)
    _RHS_CACHE[signature] = kernel
    return kernel


def constraint_kernel(
    order: int = 4, backend: str = "jax", jit: bool = True, use_cache: bool = True
) -> codegen.Kernel:
    """The Hamiltonian and momentum constraints, as an ADM kernel.

    Built from :func:`particlesim.symbolic.threeplusone.abstract_slice`
    rather than from the BSSN state, and fed the physical ``gamma_ij`` and
    ``K_ij`` reconstructed from that state. Two reasons. It is the same
    artifact any ADM data can be checked with, so it is worth having
    separately; and it measures the constraint violation of the physical
    data the state stands for, which is the quantity that has a meaning
    independent of the variables being evolved.
    """
    signature = (KERNEL_VERSION, "constraints", order, backend, jit)
    if signature in _CONSTRAINT_CACHE:
        return _CONSTRAINT_CACHE[signature]

    def build() -> str:
        slice_, _, registry = abstract_slice()
        expressions = {"hamiltonian": hamiltonian_constraint(slice_)}
        momentum = momentum_constraint(slice_)
        for i in INDICES:
            expressions[f"momentum{i}"] = momentum[i]
        return codegen.build_source(
            expressions, registry, order=order, backend=backend, name="adm_constraints"
        )

    key = cache.key_for(*signature) if use_cache else None
    source, _ = cache.source_cached(key, build)
    kernel = codegen.from_source(source, backend=backend, jit=jit)
    _CONSTRAINT_CACHE[signature] = kernel
    return kernel


# --- initial data -------------------------------------------------------


def _grid(shape, extent, backend: str):
    module = _module(backend)
    shape = tuple(int(value) for value in shape)
    extent = tuple(float(value) for value in np.atleast_1d(extent))
    if len(extent) == 1:
        extent = extent * len(shape)
    axes = [
        np.linspace(0.0, length, count, endpoint=False)
        for count, length in zip(shape, extent, strict=True)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    spacing = tuple(length / count for count, length in zip(shape, extent, strict=True))
    return [module.asarray(m) for m in mesh], spacing


def gauge_wave(
    shape=(64, 8, 8),
    amplitude: float = 0.1,
    time: float = 0.0,
    direction=(1.0, 0.0, 0.0),
    extent=1.0,
    backend: str = "jax",
) -> tuple[dict[str, Any], tuple[float, ...]]:
    """Exact gauge-wave data in BSSN variables, at any time.

    ``ds^2 = H(-dt^2 + dl^2) + (transverse)`` with ``H = 1 - A sin(2 pi (n.x
    - t))`` and ``n`` a unit vector: flat spacetime in a wavy gauge, so both
    constraints vanish identically and the whole solution is known in closed
    form. Every BSSN variable follows analytically, including the connection

        Gammabar^i = (2/3) H' H^(-5/3) n^i

    which is worth having in closed form rather than differencing, because
    it is the one evolved variable that is a derivative of the others and
    initialising it with a finite difference would seed exactly the error
    the test is trying to measure.

    A diagonal ``direction`` is the more searching test: it makes all six
    components of the conformal metric non-trivial and exercises the mixed
    second derivatives, which are the noisiest term in the Ricci tensor.
    Its period has to fit the box, so the direction is normalised but its
    *components* must be commensurate with the extent -- ``(1,1,1)`` on a
    cube is fine, ``(1, 0.3, 0)`` is not periodic and the run will show it.
    """
    module = _module(backend)
    mesh, spacing = _grid(shape, extent, backend)
    normal = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        raise ValueError("direction must be a non-zero vector")
    period = float(np.atleast_1d(extent)[0])
    # The phase runs over n.x with the *unnormalised* direction, so that a
    # (1,1,1) wave has one period along each axis and stays periodic.
    phase = sum(normal[i] * mesh[i] for i in INDICES) / period - time
    unit = normal / norm
    argument = 2 * np.pi * phase
    profile = 1 - amplitude * module.sin(argument)
    # d/ds along the unit direction, and d/dt, of the profile.
    wavenumber = 2 * np.pi * norm / period
    along = -amplitude * wavenumber * module.cos(argument)
    in_time = amplitude * 2 * np.pi * module.cos(argument)

    lapse = module.sqrt(profile)
    state: dict[str, Any] = {}
    state["alpha"] = lapse
    for i in INDICES:
        state[f"beta{i}"] = module.zeros_like(profile)
        state[f"B{i}"] = module.zeros_like(profile)

    conformal_factor = profile ** (-1 / 3)
    state["phi"] = module.log(profile) / 12
    mean = -in_time / (2 * profile ** (3 / 2))
    state["trK"] = mean
    for i in INDICES:
        state[f"Gt{i}"] = (2 / 3) * along * profile ** (-5 / 3) * unit[i]
        for j in range(i, DIMENSION):
            delta = 1.0 if i == j else 0.0
            metric = delta + (profile - 1) * unit[i] * unit[j]
            curvature = -in_time * unit[i] * unit[j] / (2 * lapse)
            state[f"gt{i}{j}"] = conformal_factor * metric
            state[f"At{i}{j}"] = conformal_factor * (curvature - metric * mean / 3)
    return state, spacing


def brill_lindquist(
    shape=(48, 48, 48),
    punctures=((0.0, (0.0, 0.0, 0.0)),),
    extent=8.0,
    backend: str = "jax",
) -> tuple[dict[str, Any], tuple[float, ...]]:
    """Time-symmetric puncture data: ``psi = 1 + sum m_a/(2 |x - x_a|)``.

    Conformally flat, ``K_ij = 0``, so both constraints are satisfied
    exactly except at the punctures themselves, where the data is singular
    and the grid has to miss them -- which is what staggering the grid off
    the origin is for.

    **The punctures sit between grid points, half a cell off centre.** The
    grid runs from zero with ``endpoint=False``, so the box centre is a grid
    point whenever the shape is even -- and ``psi`` is infinite there, which
    is not an approximation that degrades gracefully: at ``48^3`` on a box
    of 8 the centre is exactly a sample and the whole run is ``NaN`` from
    the first constraint evaluation. Shifting the punctures by half a cell
    puts the nearest sample at ``sqrt(3)/2`` of a cell, which is what every
    moving-puncture code does and why it is called puncture *evolution*. A
    puncture that still lands on a sample is refused rather than returned.

    **On a torus this is not a black hole.** The data is periodic here, so
    what is being set up is an infinite lattice of punctures rather than an
    isolated one, and the difference is not small at this box size. It is
    useful for exercising a puncture gauge and useless for a mass
    measurement, and issue #48's mesh refinement with an outer boundary is
    what changes that.
    """
    module = _module(backend)
    mesh, spacing = _grid(shape, extent, backend)
    lengths = tuple(float(value) for value in np.atleast_1d(extent))
    if len(lengths) == 1:
        lengths = lengths * DIMENSION
    centre = [lengths[i] / 2 + spacing[i] / 2 for i in INDICES]
    conformal = module.ones_like(mesh[0])
    for mass, position in punctures:
        offset = [mesh[i] - (centre[i] + position[i]) for i in INDICES]
        radius = module.sqrt(sum(value**2 for value in offset))
        closest = float(np.min(np.asarray(radius)))
        if closest <= 0.0:
            raise ValueError(
                f"a puncture at {tuple(position)} lands on a grid point: psi is "
                "infinite there and the whole run is NaN. Move it off the "
                "lattice, or change the shape"
            )
        conformal = conformal + float(mass) / (2 * radius)

    state: dict[str, Any] = {"phi": module.log(conformal), "trK": module.zeros_like(conformal)}
    state["alpha"] = conformal ** (-2)
    for i in INDICES:
        state[f"beta{i}"] = module.zeros_like(conformal)
        state[f"B{i}"] = module.zeros_like(conformal)
        state[f"Gt{i}"] = module.zeros_like(conformal)
        for j in range(i, DIMENSION):
            state[f"gt{i}{j}"] = module.full_like(conformal, 1.0 if i == j else 0.0)
            state[f"At{i}{j}"] = module.zeros_like(conformal)
    return state, spacing


# --- diagnostics --------------------------------------------------------


def physical_slice_arrays(state: Mapping[str, Any], backend: str = "jax") -> dict[str, Any]:
    """``gamma_ij`` and ``K_ij`` as the constraint kernel's input fields.

    Algebraic in the state: ``gamma_ij = e^(4 phi) gammabar_ij`` and
    ``K_ij = e^(4 phi) Abar_ij + gamma_ij K/3``. No derivatives, so this
    costs a handful of array multiplications and the kernel does the
    differencing.
    """
    module = _module(backend)
    factor = module.exp(4 * state["phi"])
    fields: dict[str, Any] = {"alpha": state["alpha"]}
    for i in INDICES:
        fields[f"beta{i}"] = state[f"beta{i}"]
    for i in INDICES:
        for j in range(i, DIMENSION):
            metric = factor * state[f"gt{i}{j}"]
            fields[f"gamma{i}{j}"] = metric
            fields[f"K{i}{j}"] = factor * state[f"At{i}{j}"] + metric * state["trK"] / 3
    return fields


@dataclass(frozen=True)
class Constraints:
    """Constraint violations on one slice, as norms rather than fields."""

    hamiltonian: float
    momentum: float
    determinant: float
    trace: float

    def summary(self) -> dict[str, float]:
        return {
            "hamiltonian": self.hamiltonian,
            "momentum": self.momentum,
            "determinant": self.determinant,
            "trace": self.trace,
        }


def _norm(values, module) -> float:
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def constraints(
    state: Mapping[str, Any],
    spacing: Sequence[float],
    order: int = 4,
    backend: str = "jax",
    kernel: codegen.Kernel | None = None,
) -> Constraints:
    """L2 norms of the Hamiltonian and momentum constraints, and the algebraic pair.

    The algebraic constraints -- ``det gammabar = 1`` and
    ``gammabar^ij Abar_ij = 0`` -- are measured here and restored
    elsewhere, by :meth:`Evolution.project`. Keeping the measurement out of
    the projection is the point: a code that only resets them reports zero
    for a drift it is creating, and the drift is the cheapest signal that
    something upstream is wrong.
    """
    module = _module(backend)
    kernel = constraint_kernel(order=order, backend=backend) if kernel is None else kernel
    fields = physical_slice_arrays(state, backend)
    out = kernel(fields, tuple(spacing))
    momentum = np.sqrt(sum(np.mean(np.asarray(out[f"momentum{i}"]) ** 2) for i in INDICES))

    metric = [[state[f"gt{min(i, j)}{max(i, j)}"] for j in INDICES] for i in INDICES]
    traceless = [[state[f"At{min(i, j)}{max(i, j)}"] for j in INDICES] for i in INDICES]
    return Constraints(
        hamiltonian=_norm(out["hamiltonian"], module),
        momentum=float(momentum),
        determinant=_norm(determinant(metric) - 1.0, module),
        trace=_norm(trace(inverse_metric(metric), traceless), module),
    )


# --- the integrator -----------------------------------------------------


def dissipation_operator(order: int = 4):
    """Kreiss-Oliger stencil weights for a scheme of the given accuracy.

    ``Q = (-1)^(r+1) epsilon / (2^(2r) h) D^(2r)`` with ``r = order/2 + 1``,
    which annihilates polynomials below degree ``2r`` and so leaves the
    scheme's order alone while damping the Nyquist mode at ``epsilon/h``
    *per axis* -- a mode that alternates along all three is damped at
    ``3 epsilon/h``, and one constant along two of them at ``epsilon/h``.
    The same operator as :func:`particlesim.core.grid.kreiss_oliger`, in
    periodic form: there is no boundary to fall back from.
    """
    if order not in (2, 4, 6):
        raise ValueError("order must be 2, 4, or 6")
    radius = order // 2 + 1
    weights = [(-1) ** k * comb(2 * radius, k) for k in range(2 * radius + 1)]
    sign = (-1) ** (radius + 1)
    scale = sign / 2 ** (2 * radius)
    return radius, [scale * float(w) for w in weights]


@dataclass(frozen=True)
class Evolution:
    """A BSSN evolution on a periodic grid."""

    spacing: tuple[float, ...]
    order: int = 4
    backend: str = "jax"
    dissipation: float = DISSIPATION
    courant: float = COURANT
    kernel: codegen.Kernel = field(default=None, repr=False)
    constraint_kernel: codegen.Kernel = field(default=None, repr=False)
    slicing: str = "one_plus_log"
    shift_condition: str = "gamma_driver"
    damping: float = 2.0
    enforce: bool = True
    upwind: bool = False
    advect: bool | str = True

    @classmethod
    def build(
        cls,
        spacing,
        order: int = 4,
        backend: str = "jax",
        slicing: str = "one_plus_log",
        shift_condition: str = "gamma_driver",
        damping: float = 2.0,
        dissipation: float = DISSIPATION,
        courant: float = COURANT,
        jit: bool = True,
        enforce: bool = True,
        upwind: bool = False,
        advect: bool | str = True,
    ) -> Evolution:
        kernel = rhs_kernel(
            order=order,
            backend=backend,
            slicing=slicing,
            shift_condition=shift_condition,
            damping=damping,
            advect=advect,
            jit=jit,
        )
        return cls(
            spacing=tuple(float(value) for value in spacing),
            order=order,
            backend=backend,
            dissipation=dissipation,
            courant=courant,
            kernel=kernel,
            constraint_kernel=constraint_kernel(order=order, backend=backend, jit=jit),
            slicing=slicing,
            shift_condition=shift_condition,
            damping=damping,
            enforce=enforce,
            upwind=upwind,
            advect=advect,
        )

    @property
    def time_step(self) -> float:
        return self.courant * min(self.spacing)

    def _dissipator(self):
        """The dissipation operator, compiled once per evolution.

        Applied to all twenty-four variables at once on a stacked array:
        the operator is the same stencil for each of them, so doing it
        variable by variable is twenty-four times the array operations for
        the same arithmetic.
        """
        cached = _DISSIPATORS.get(self)
        if cached is not None:
            return cached
        module = _module(self.backend)
        radius, weights = dissipation_operator(self.order)
        spacing = self.spacing
        strength = self.dissipation

        def apply(stacked):
            total = None
            for axis, step in enumerate(spacing):
                stencil = None
                for offset, weight in enumerate(weights, start=-radius):
                    term = weight * module.roll(stacked, -offset, axis=axis + 1)
                    stencil = term if stencil is None else stencil + term
                contribution = stencil / step
                total = contribution if total is None else total + contribution
            return strength * total

        if self.backend == "jax":
            import jax

            apply = jax.jit(apply)
        _DISSIPATORS[self] = apply
        return apply

    def _upwinder(self):
        """The upwinding correction, compiled once per evolution.

        ``sum_k beta^k (D_k^lopsided f - D_k^centred f)`` for every advected
        variable, with the lopsided stencil leaning the way the shift points.
        The kernel's advection terms are ``beta^k`` times a centred
        derivative, and linear in it, so adding this turns each of them into
        ``beta^k`` times the lopsided one and changes nothing else.
        """
        cached = _UPWINDERS.get(self)
        if cached is not None:
            return cached
        if self.order != 4:
            raise ValueError("upwinded advection is implemented for order 4 only")
        module = _module(self.backend)
        spacing = self.spacing

        def apply(stacked, shift):
            total = None
            for axis, step in enumerate(spacing):
                ahead = None
                behind = None
                for offset, weight in UPWIND_CORRECTION:
                    term = weight * module.roll(stacked, -offset, axis=axis + 1)
                    ahead = term if ahead is None else ahead + term
                    # The mirror image: offset -> -offset, weight -> -weight.
                    term = -weight * module.roll(stacked, offset, axis=axis + 1)
                    behind = term if behind is None else behind + term
                velocity = shift[axis]
                chosen = module.where(velocity > 0, ahead, behind)
                contribution = velocity * chosen / step
                total = contribution if total is None else total + contribution
            return total

        if self.backend == "jax":
            import jax

            apply = jax.jit(apply)
        _UPWINDERS[self] = apply
        return apply

    def right_hand_side(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """The kernel's output plus dissipation and upwinding, one entry per variable."""
        rates = dict(self.kernel(dict(state), self.spacing))
        module = _module(self.backend)
        if self.dissipation > 0.0:
            names = list(state)
            damped = self._dissipator()(module.stack([state[name] for name in names]))
            rates = {name: rates[name] + damped[index] for index, name in enumerate(names)}
        if self.upwind:
            shift_names = {f"beta{i}" for i in INDICES}
            if self.advect is True:
                gauge = set()
            elif self.advect == "lapse":
                gauge = shift_names
            else:
                gauge = {"alpha", *shift_names}
            advected = [name for name in state if name not in UNADVECTED | gauge]
            shift = module.stack([state[f"beta{i}"] for i in INDICES])
            corrections = self._upwinder()(module.stack([state[name] for name in advected]), shift)
            for index, name in enumerate(advected):
                rates[name] = rates[name] + corrections[index]
                # The Gamma-driver's B^i is driven by d_t Gammabar^i itself, so
                # it is given the rate Gammabar^i actually has, upwinding and
                # all -- which is also what codes that upwind inside the kernel
                # do. With the gauge advected it made no difference to the
                # outcome (a two-level puncture failed either way, at 50 M with
                # it and 90 M without); with the gauge unadvected B stays near
                # 2e-3 and the two agree to three figures.
                if name.startswith("Gt") and self.shift_condition == "gamma_driver":
                    driver = f"B{name[2:]}"
                    if driver in rates:
                        rates[driver] = rates[driver] + corrections[index]
        return rates

    def _raw_step(self, state, step):
        names = list(state)

        def advance(base, rates, factor):
            return {name: base[name] + factor * rates[name] for name in names}

        first = self.right_hand_side(state)
        second = self.right_hand_side(advance(state, first, step / 2))
        third = self.right_hand_side(advance(state, second, step / 2))
        fourth = self.right_hand_side(advance(state, third, step))
        return {
            name: state[name]
            + (step / 6) * (first[name] + 2 * second[name] + 2 * third[name] + fourth[name])
            for name in names
        }

    def project(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Put the state back on ``det gammabar = 1`` and ``gammabar^ij Abar_ij = 0``.

        The two algebraic constraints hold identically at ``t = 0`` and are
        preserved by the continuum equations, so nothing in the
        right-hand side pushes the evolution back onto them. Discretely
        they drift, and the drift feeds the equations quantities that are
        not the variables they are equations for -- the conformal metric's
        determinant enters through ``e^(4 phi)`` twice over. Projecting
        after each step,

            gammabar_ij -> (det gammabar)^(-1/3) gammabar_ij,
            Abar_ij     -> Abar_ij - gammabar_ij (gammabar^kl Abar_kl) / 3,

        costs two determinants and is what every production code does.
        """
        metric = [[state[f"gt{min(i, j)}{max(i, j)}"] for j in INDICES] for i in INDICES]
        scale = determinant(metric) ** (-1.0 / 3.0)
        metric = [[scale * metric[i][j] for j in INDICES] for i in INDICES]
        traceless = [[state[f"At{min(i, j)}{max(i, j)}"] for j in INDICES] for i in INDICES]
        traced = trace(inverse_metric(metric), traceless)
        projected = dict(state)
        for i in INDICES:
            for j in range(i, DIMENSION):
                projected[f"gt{i}{j}"] = metric[i][j]
                projected[f"At{i}{j}"] = traceless[i][j] - metric[i][j] * traced / 3.0
        return projected

    def step(self, state: Mapping[str, Any], time_step: float | None = None):
        """One classical fourth-order Runge-Kutta step.

        The right-hand side is compiled and the four-stage combination is
        not, which is the opposite of what one would try first. Compiling
        the whole step means unrolling four copies of a kernel with four
        hundred and forty-five temporaries and a hundred and fourteen
        stencils into one graph, and XLA's fusion pass does not finish on
        it: measured here, the first call had not returned after ten
        minutes. Compiling the right-hand side alone takes seconds, and
        what is left outside is ninety-six array operations per step.
        """
        step = self.time_step if time_step is None else float(time_step)
        advanced = self._raw_step(dict(state), step)
        return self.project(advanced) if self.enforce else advanced

    def run(
        self,
        state: Mapping[str, Any],
        steps: int,
        time_step: float | None = None,
        sample_every: int = 0,
        monitor: Callable[[int, float, Mapping[str, Any]], None] | None = None,
    ) -> History:
        """Integrate ``steps`` steps, recording constraints as it goes."""
        step = self.time_step if time_step is None else float(time_step)
        sample_every = steps if sample_every <= 0 else sample_every
        current = dict(state)
        times = [0.0]
        records = [self.constraints(current)]
        for index in range(1, steps + 1):
            current = self.step(current, step)
            if index % sample_every == 0 or index == steps:
                times.append(index * step)
                records.append(self.constraints(current))
            if monitor is not None:
                monitor(index, index * step, current)
        return History(times=times, constraints=records, state=current, time_step=step)

    def constraints(self, state: Mapping[str, Any]) -> Constraints:
        return constraints(
            state,
            self.spacing,
            order=self.order,
            backend=self.backend,
            kernel=self.constraint_kernel,
        )


@dataclass(frozen=True)
class History:
    """What a run did."""

    times: list[float]
    constraints: list[Constraints]
    state: dict[str, Any]
    time_step: float

    @property
    def final(self) -> Constraints:
        return self.constraints[-1]

    @property
    def growth(self) -> float:
        """Final Hamiltonian norm over the initial one."""
        first = self.constraints[0].hamiltonian
        if first == 0.0:
            return float("inf") if self.final.hamiltonian > 0 else 1.0
        return self.final.hamiltonian / first

    def summary(self) -> dict[str, Any]:
        return {
            "duration": self.times[-1],
            "time_step": self.time_step,
            "samples": len(self.times),
            "hamiltonian": self.final.hamiltonian,
            "momentum": self.final.momentum,
            "determinant": self.final.determinant,
            "trace": self.final.trace,
            "growth": self.growth,
        }


def solution_error(state: Mapping[str, Any], exact: Mapping[str, Any]) -> float:
    """Largest L2 difference over the state's variables."""
    worst = 0.0
    for name, value in exact.items():
        difference = np.asarray(state[name]) - np.asarray(value)
        worst = max(worst, float(np.sqrt(np.mean(difference**2))))
    return worst


__all__ = [
    "COURANT",
    "DISSIPATION",
    "KERNEL_VERSION",
    "UNADVECTED",
    "UPWIND_CORRECTION",
    "Constraints",
    "Evolution",
    "History",
    "brill_lindquist",
    "constraint_kernel",
    "constraints",
    "dissipation_operator",
    "gauge_wave",
    "physical_slice_arrays",
    "rhs_expressions",
    "rhs_kernel",
    "solution_error",
]
