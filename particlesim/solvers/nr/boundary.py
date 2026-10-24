"""A radiative outer boundary, so that an isolated body can be isolated.

Issue #132. The emitted kernels difference with ``roll``, so the domain is a
torus: right for a gauge wave or a Teukolsky wave, which are periodic, and
wrong for anything asymptotically flat. A single puncture on a torus is an
infinite lattice of punctures, which is why ``brill_lindquist`` says so in
its own docstring and why its Hamiltonian constraint is bounded rather than
small.

**No change to the emitted kernels, and that is the point.** The trick is
the one fixed mesh refinement already needed. Let the periodic kernel
compute everywhere, including a zone at the outer edge where its wrap is
wrong, and then *override the rates in that zone* with the boundary
condition before the stage is combined. The wrapped values never reach the
interior because the zone is at least a stencil radius wide and its rates
are discarded; the interior sees only values the boundary condition
produced. The alternative -- emitting one-sided stencils near an edge --
means a second kernel and a region-split right-hand side, for a boundary
that is an approximation either way.

**The condition.** Each variable is taken to behave at large radius as an
outgoing wave on a constant background,

    f = f_0 + u(r - v t) / r

which differentiates to the Sommerfeld condition

    d_t f = -v (x^i / r) d_i f - v (f - f_0) / r

The second term is what distinguishes this from a plain outgoing-wave
condition and what makes it work at finite radius: without it a field
falling off as ``1/r`` is reflected at the amplitude of its own falloff.

**What this is not.** It is not constraint-preserving. The condition is
applied variable by variable to quantities that are not characteristic
variables of the system, so it has no reason to respect the constraints and
in general does not. :class:`ConstraintPreserving`, below, is the condition
that does: the zone evolves Einstein's equations there too, and only the
incoming modes are damped.

**What that costs turned out to be negative, which was not the guess.** The
expectation was that a non-constraint-preserving boundary would make the
Hamiltonian constraint worse. Measured against the same run with periodic
boundaries, at 1.25 crossing times:

    n     periodic |H|   radiative |H|   ratio
    32      1.297e-03      3.806e-05      0.03
    48      1.224e-03      1.938e-05      0.02
    64      1.243e-03      1.088e-05      0.01

Thirty to a hundred times *better*, because the violation leaves with the
pulse instead of recirculating forever. The periodic numbers do not converge
at all -- 1.297, 1.224, 1.243 -- since whatever the pulse deposits stays in
the domain; the radiative ones converged at order 1.7 to 2.0, which was
read as Sommerfeld being a second-order condition.

**It was the stencil.** The derivative the condition takes fell back to
second-order one-sided differences at the two outermost points, which are
the points the condition exists for. :func:`edge_derivative` keeps fourth
order there. That halves the reflection of a Teukolsky wave, and it lowers
the pulse's constraint above to 9.9e-06, 5.4e-06 and 4.4e-06. That is four
times lower at 32 points, and what remains is the bump's own violation
parked at the centre by the frozen shift, not anything the boundary does.

**A Teukolsky wave leaves, and Sommerfeld's floor shows at 96 points.**
Measured against the closed form in a cube inside a box of 12, with the
zone from ``r = 5``, what is left after the wave has gone is 1/1750 of its
start at 96 points. The error that comes back is at the truncation error
through 72 points: 0.81 and 1.02 times it. At 96 points it is 2.3 times,
because it stops converging near 1e-02 -- what the condition's own
reflection should look like, since it is exact only for the ``1/r`` part
and a quadrupole wave at this radius still has ``1/r^2`` and ``1/r^3``
parts. ``|H|`` after the reflection is 3.3e-02 at 72 points
and 3.0e-02 at 96, the continuum cost of a condition that is not
constraint-preserving. See :mod:`particlesim.solvers.nr.teukolsky` and
``docs/benchmarks.md``.

**Bayliss and Turkel's second condition removes that floor.**
:class:`SecondOrder` also annihilates the ``1/r^2`` part. On the same wave,
what comes back after the reflection is 3.2e-02, 1.35e-02 and 6.0e-03 at 48,
72 and 96 points, converging at order 2.8 where Sommerfeld's stalled at 1.2.
``|H|`` after the reflection converges again too, 7.4e-03 at 96 points
against Sommerfeld's 3.0e-02. Both still converge more slowly than the
interior's 3.9, so the ratio to truncation grows, from 0.91 to 1.26 between
72 and 96 points, but slowly.

**A constraint-preserving layer.** :class:`ConstraintPreserving` keeps
the zone but not the override. Inside a rim two points deep, the zone
evolves the interior's own equations, so the constraints propagate there as
they do inside, and only two terms are added: an outgoing radiation
condition on the lapse, a choice of slicing the constraints do not see, and
a relaxation of the transverse-traceless incoming radiation towards the
value Bayliss and Turkel's second condition gives it, carried by a surface
auxiliary that uses only derivatives tangential to spheres. The rim keeps
:class:`SecondOrder`. Damping the constraint modes as well was tried and is
unstable in BSSN; see its docstring and ``docs/benchmarks.md``.

**The cost that is real is arithmetic.** :meth:`Radiative.rates` takes three
bounded-domain derivatives per variable per stage over the *whole* array,
which is seventy-two array passes a stage for BSSN and dominates the run at
64^3. Restricting it to a slab is the obvious fix and is not done here.
"""

from __future__ import annotations

import dataclasses
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import sympy as sp

from particlesim.solvers.nr.bssn import (
    DIMENSION,
    INDICES,
    Evolution,
    _module,
    dissipation_operator,
)
from particlesim.solvers.nr.stepping import operations
from particlesim.symbolic import codegen

#: What each evolved variable tends to at large radius, in vacuum.
#:
#: Minkowski in the BSSN variables: a unit lapse, a flat conformal metric, no
#: shift and no curvature. A variable whose background is wrong here is
#: reflected at the amplitude of the difference, which is the loudest way to
#: get a boundary condition wrong and the easiest to check.
ASYMPTOTIC: dict[str, float] = {
    "alpha": 1.0,
    "phi": 0.0,
    "W": 1.0,
    "trK": 0.0,
    "Theta": 0.0,
    **{f"beta{i}": 0.0 for i in INDICES},
    **{f"B{i}": 0.0 for i in INDICES},
    **{f"Gt{i}": 0.0 for i in INDICES},
    **{f"gt{i}{j}": (1.0 if i == j else 0.0) for i in INDICES for j in range(i, DIMENSION)},
    **{f"At{i}{j}": 0.0 for i in INDICES for j in range(i, DIMENSION)},
}


@lru_cache(maxsize=12)
def _edge_kernel(axis: int, order: int):
    symbol = sp.Symbol("d_edge_field")
    return codegen.emit(
        {"derivative": symbol},
        {symbol: ("edge_field", (axis,))},
        order=order,
        nonperiodic_axes=(axis,),
    )


def edge_derivative(field, axis: int, step: float, order: int = 4) -> np.ndarray:
    """A bounded-domain derivative that keeps its order at the edge.

    :func:`particlesim.core.grid.derivative` falls back to second-order
    one-sided differences at the outermost points, which is harmless for a
    diagnostic and is not harmless here: those points are where the
    condition acts, and a second-order stencil there made the reflection of
    a Teukolsky wave twice what the fourth-order one gives. The one-sided
    stencils below are fourth order,

        f'_0 = (-25 f_0 + 48 f_1 - 36 f_2 + 16 f_3 - 3 f_4) / 12h
        f'_1 = (-3 f_0 - 10 f_1 + 18 f_2 - 6 f_3 + f_4) / 12h

    and mirrored with the sign flipped at the far edge. Both lean on the
    interior, which is upwind for a wave leaving the domain. At sixth order
    all three edge points use seven-point stencils, preserving sixth order.
    The operators are shared with the explicitly bounded codegen path.
    """
    field = np.asarray(field)
    if not -field.ndim <= axis < field.ndim:
        raise ValueError(f"axis {axis} is outside an array with {field.ndim} dimensions")
    axis %= field.ndim
    return _edge_kernel(axis, order)({"edge_field": field}, (step,) * field.ndim)["derivative"]


def _stacked_edge_derivative(module, stacked, axis: int, step: float):
    """:func:`edge_derivative` at fourth order on every field of a stack at once.

    ``stacked`` has the variables first, so ``axis`` is counted after them.
    Written with slices and a concatenation rather than assignment, so that
    JAX can compile it.
    """
    a = module.moveaxis(stacked, axis + 1, 1)
    centred = (-a[:, 4:] + 8 * a[:, 3:-1] - 8 * a[:, 1:-3] + a[:, :-4]) / (12 * step)
    first = (-25 * a[:, 0] + 48 * a[:, 1] - 36 * a[:, 2] + 16 * a[:, 3] - 3 * a[:, 4]) / (12 * step)
    second = (-3 * a[:, 0] - 10 * a[:, 1] + 18 * a[:, 2] - 6 * a[:, 3] + a[:, 4]) / (12 * step)
    before = (3 * a[:, -1] + 10 * a[:, -2] - 18 * a[:, -3] + 6 * a[:, -4] - a[:, -5]) / (12 * step)
    last = (25 * a[:, -1] - 48 * a[:, -2] + 36 * a[:, -3] - 16 * a[:, -4] + 3 * a[:, -5]) / (
        12 * step
    )
    parts = [first[:, None], second[:, None], centred, before[:, None], last[:, None]]
    return module.moveaxis(module.concatenate(parts, axis=1), 1, axis + 1)


#: Compiled condition kernels, per boundary and set of variables. Held weakly,
#: so a boundary that is dropped takes its kernels with it; the kernels
#: themselves hold arrays, not the boundary.
_STACKED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


#: Asymptotic speeds of the gauge variables, by slicing condition.
#:
#: The lapse and the trace of the extrinsic curvature form the slicing's
#: own subsystem, whose speed is ``sqrt(f(alpha) alpha)`` for
#: ``d_t alpha = -alpha^2 f(alpha) K``: ``sqrt(2)`` far out for 1+log
#: (``f = 2/alpha``), one for harmonic slicing (``f = 1``). Everything else
#: leaves at the speed of light.
GAUGE_SPEEDS: dict[str, dict[str, float]] = {
    "one_plus_log": {"alpha": float(np.sqrt(2.0)), "trK": float(np.sqrt(2.0))},
    "harmonic": {"alpha": 1.0, "trK": 1.0},
}


@dataclass(frozen=True, eq=False)
class Radiative:
    """The Sommerfeld condition on a slab at the outer edge of the grid.

    ``axes`` are the axes that have an outer boundary; the rest stay
    periodic, which is what makes a one-dimensional test possible. ``width``
    is the zone in points, and has to be at least the stencil radius so the
    wrapped rates the kernel produces there are all discarded.

    ``speeds`` overrides ``speed`` variable by variable. Not every field
    leaves at the speed of light: with 1+log slicing the lapse and ``K``
    carry gauge pulses at ``sqrt(2 alpha)``, which is ``sqrt(2)`` far out,
    and a condition that waits for them at speed one lets them pile up. On a
    two-level puncture the lapse in the zone drifted from its background by
    0.10, 0.19, 0.38, 0.84 at ``t`` = 10, 40, 100, 150 M, and by 165 M the
    hole had dissolved from the outside in. :data:`GAUGE_SPEEDS` has the
    values for each slicing.

    ``eq=False`` so that instances hash by identity. The class holds
    coordinate arrays, which are not hashable, and the evolutions it is
    attached to are used as cache keys.

    ``background`` optionally supplies time-independent reference fields.
    The radiation condition then acts on their departures, including the
    radial derivative. This permits a known stationary Schwarzschild
    exterior at a finite boundary. It changes only the outer condition;
    the interior still evolves the full equations. It is an approximate
    radiation condition, not a constraint-preserving characteristic one.
    """

    coords: tuple[Any, ...]
    spacing: tuple[float, ...]
    axes: tuple[int, ...]
    width: int
    speed: float = 1.0
    order: int = 4
    backend: str = "jax"
    speeds: Mapping[str, float] | None = None
    background: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        radius = self.order // 2
        if self.width < radius:
            raise ValueError(
                f"a boundary zone of {self.width} points is narrower than the "
                f"stencil radius {radius}: the kernel's wrapped rates would "
                "reach the interior"
            )
        if self.background is not None:
            shape = np.shape(self.coords[0])
            for name, value in self.background.items():
                if np.shape(value) not in ((), shape):
                    raise ValueError(
                        f"background {name!r} must be scalar or have grid shape {shape}"
                    )

    def background_value(self, name):
        value = ASYMPTOTIC.get(name, 0.0)
        return value if self.background is None else self.background.get(name, value)

    @property
    def radius(self):
        """``r`` measured from the coordinate origin of ``coords``."""
        return np.sqrt(sum(np.asarray(value) ** 2 for value in self.coords))

    def mask(self) -> np.ndarray:
        """True on the points the condition owns.

        The outer ``width`` points along every bounded axis, and nothing
        along a periodic one. A corner belongs to the zone as much as a face
        does, which is why this is a mask rather than a list of slabs: the
        condition is applied pointwise and the direction ``x^i / r`` already
        knows which way is out.
        """
        shape = np.asarray(self.coords[0]).shape
        out = np.zeros(shape, dtype=bool)
        for axis in self.axes:
            index = np.arange(shape[axis])
            edge = (index < self.width) | (index >= shape[axis] - self.width)
            out |= edge.reshape([-1 if k == axis else 1 for k in range(len(shape))])
        return out

    @property
    def stacked(self) -> bool:
        """Whether the compiled, all-variables-at-once path applies: JAX at fourth order."""
        return self.backend == "jax" and self.order == 4

    def _compiled(self, names: tuple[str, ...]):
        """``(radial, sommerfeld)`` for this boundary and these variables, compiled once.

        ``radial(stack)`` is ``(x^i/r) d_i`` of every field in ``stack``, with
        :func:`edge_derivative`'s stencils. ``sommerfeld(stack)`` is the
        condition's rate for each. Both run on the device, where the
        variable-by-variable NumPy version cost more than the kernel: 0.24 s
        of Sommerfeld against 0.50 s of BSSN at 64^3, and with the second
        condition more than half the right-hand side.
        """
        cache = _STACKED.setdefault(self, {})
        if names in cache:
            return cache[names]
        import jax

        jnp = _module("jax")

        radius = np.maximum(self.radius, min(self.spacing))
        direction = jnp.asarray(
            np.stack([np.asarray(self.coords[axis]) / radius for axis in range(DIMENSION)])
        )
        inverse = jnp.asarray(1.0 / radius)
        spacing = tuple(self.spacing)
        column = (len(names),) + (1,) * DIMENSION
        background = jnp.asarray([ASYMPTOTIC.get(name, 0.0) for name in names]).reshape(column)
        reference = None
        if self.background is not None:
            reference = jnp.stack(
                [
                    jnp.broadcast_to(jnp.asarray(self.background_value(n)), radius.shape)
                    for n in names
                ]
            )
        table = self.speeds or {}
        speeds = [table.get(name, self.speed) for name in names]
        speed = jnp.asarray(speeds).reshape(column)

        def radial(stack):
            return sum(
                direction[axis] * _stacked_edge_derivative(jnp, stack, axis, spacing[axis])
                for axis in range(DIMENSION)
            )

        def sommerfeld(stack):
            if reference is not None:
                departure = stack - reference
                return -speed * (radial(departure) + departure * inverse)
            return -speed * (radial(stack) + (stack - background) * inverse)

        compiled = (jax.jit(radial), jax.jit(sommerfeld), speed, inverse)
        cache[names] = compiled
        return compiled

    def rates(self, state) -> dict[str, Any]:
        """``d_t f`` from the Sommerfeld condition, everywhere.

        Computed on the whole array and used only in the zone, because the
        derivative it needs is the *bounded-domain* one --
        :func:`edge_derivative`, which uses one-sided stencils at an edge
        rather than wrapping. That is the one place in this file where not
        wrapping matters, and slicing a slab out first would put the wrap
        back.

        Three derivatives per variable per call is the cost. With JAX at
        fourth order they are taken for every variable at once on the device
        (:meth:`_compiled`), which cut this call from 0.24 s to 0.09 s at 64^3.
        A slab would save less than it seems: with a zone a unit deep the six
        faces are most of a 96^3 grid.
        """
        if self.stacked:
            import jax.numpy as jnp

            names = tuple(state)
            _, sommerfeld, _, _ = self._compiled(names)
            rates = sommerfeld(jnp.stack([jnp.asarray(state[name]) for name in names]))
            return {name: rates[index] for index, name in enumerate(names)}
        radius = np.maximum(self.radius, min(self.spacing))
        out = {}
        for name, value in state.items():
            field = np.asarray(value)
            background = self.background_value(name)
            differentiated = field if self.background is None else field - background
            gradient = sum(
                np.asarray(self.coords[axis])
                / radius
                * edge_derivative(differentiated, axis, self.spacing[axis], order=self.order)
                for axis in range(DIMENSION)
            )
            speed = self.speed if self.speeds is None else self.speeds.get(name, self.speed)
            out[name] = -speed * (gradient + (field - background) / radius)
        return out

    def apply(self, rates, state) -> dict[str, Any]:
        """Replace the kernel's rates with the condition's, inside the zone."""
        module = _module(self.backend)
        zone = self.mask()
        boundary = self.rates(state)
        return {
            name: module.where(zone, module.asarray(boundary[name]), value)
            for name, value in rates.items()
        }


@dataclass(frozen=True, eq=False)
class Bounded:
    """An evolution with a radiative outer boundary.

    A wrapper rather than a field on :class:`Evolution` for the same reason
    :class:`~particlesim.solvers.nr.refined.Hierarchy` is one: the boundary
    has to act at every Runge-Kutta stage, not once a step, so the stage loop
    has to be here. Overriding once a step would leave stages two to four
    integrating the wrapped rates -- the same mistake that cost fixed mesh
    refinement an order, measured at 8.5 against 16.
    """

    evolution: Evolution
    boundary: Radiative

    def right_hand_side(self, state) -> dict[str, Any]:
        return self.boundary.apply(self.evolution.right_hand_side(state), state)

    @property
    def time_step(self) -> float:
        return self.evolution.time_step

    # What a refinement hierarchy asks of its coarse level, so that a bounded
    # evolution can be one: the outer boundary belongs to the coarsest level
    # and to no other.
    @property
    def spacing(self) -> tuple[float, ...]:
        return self.evolution.spacing

    @property
    def order(self) -> int:
        return self.evolution.order

    @property
    def backend(self) -> str:
        return self.evolution.backend

    @property
    def enforce(self) -> bool:
        return self.evolution.enforce

    def project(self, state) -> dict[str, Any]:
        return self.evolution.project(state)

    def step(self, state, time_step: float | None = None):
        """One classical fourth-order step, the boundary applied at each stage."""
        step = self.time_step if time_step is None else float(time_step)
        advance, finish, _ = operations(self.backend)
        base = dict(state)

        first = self.right_hand_side(base)
        second = self.right_hand_side(advance(base, first, step / 2))
        third = self.right_hand_side(advance(base, second, step / 2))
        fourth = self.right_hand_side(advance(base, third, step))

        out = finish(base, first, second, third, fourth, step)
        return self.evolution.project(out) if self.evolution.enforce else out

    def run(self, state, steps: int, time_step: float | None = None, sample=None):
        """Integrate, optionally recording ``sample(state)`` at every step."""
        step = self.time_step if time_step is None else float(time_step)
        current = dict(state)
        history = [] if sample is None else [sample(current)]
        for _ in range(steps):
            current = self.step(current, step)
            if sample is not None:
                history.append(sample(current))
        return current, history


#: The prefix of the auxiliary fields :class:`SecondOrder` carries in the state. Not
#: ``B2``, which would read as the Gamma-driver field of the same name.
AUXILIARY = "aux:"


@dataclass(frozen=True, eq=False)
class SecondOrder(Bounded):
    """Bayliss and Turkel's second condition on the zone: exact for ``1/r`` and ``1/r^2``.

    Sommerfeld's condition is ``B1 u = 0`` with
    ``B1 = d_t + c d_r + c/r`` and ``u = f - f_0``. It annihilates an outgoing
    ``a(t - r)/r`` exactly, and leaves ``-c b(t - r)/r^3`` of a ``b(t - r)/r^2``
    term. Near a quadrupole wave that residual is what comes back. Bayliss and
    Turkel (1980) apply a second factor,

        (d_t + c d_r + 3c/r)(d_t + c d_r + c/r) u = 0

    which annihilates both terms. It is second order in time, so it is carried
    as Sommerfeld plus an auxiliary field ``v = B1 u``. That field obeys
    ``d_t v = -c (d_r v + 3 v / r)`` in the zone, and ``d_t u = -c (d_r u + u/r) + v``.

    Inside the zone ``v`` is evolved, and starts at zero, which is right when
    nothing is leaving at ``t = 0``. Outside it ``v`` is what the interior's own
    rates make of ``B1 u``: the kernel's ``d_t f`` less Sommerfeld's. The
    zone's radial derivative of ``v`` then reaches into a field consistent
    with the evolution, as ``d_r f`` does.

    The state carries one auxiliary field per evolved variable, named with the
    prefix :data:`AUXILIARY`. :meth:`start` adds them, and :meth:`step` adds
    any that are missing. They are not projected, and not seen by the kernel.
    """

    def start(self, state) -> dict[str, Any]:
        """``state`` with its auxiliary fields, zero where they are missing."""
        out = dict(state)
        for name in state:
            if not name.startswith(AUXILIARY) and AUXILIARY + name not in out:
                out[AUXILIARY + name] = np.zeros(np.asarray(state[name]).shape)
        return out

    def project(self, state) -> dict[str, Any]:
        """The evolution's projection on the geometry; the auxiliary fields pass through."""
        geometry, auxiliary = self._split(state)
        return {**self.evolution.project(geometry), **auxiliary}

    def _split(self, state):
        geometry = {k: v for k, v in state.items() if not k.startswith(AUXILIARY)}
        return geometry, {k: v for k, v in state.items() if k.startswith(AUXILIARY)}

    def _radial(self, field):
        b = self.boundary
        radius = np.maximum(b.radius, min(b.spacing))
        gradient = sum(
            np.asarray(b.coords[axis])
            / radius
            * edge_derivative(field, axis, b.spacing[axis], order=b.order)
            for axis in range(DIMENSION)
        )
        return gradient, radius

    def _stacked_rates(self, geometry, auxiliary, rates) -> dict[str, Any]:
        """The same rates as below, for every variable at once on the device."""
        import jax
        import jax.numpy as jnp

        b = self.boundary
        names = tuple(sorted(geometry))
        radial, sommerfeld, speed, inverse = b._compiled(names)
        cache = _STACKED.setdefault(b, {})
        key = (names, "second")
        if key not in cache:
            zone = jnp.asarray(b.mask())

            def combine(geometry, auxiliary, rates):
                fields = jnp.stack([geometry[n] for n in names])
                kernel = jnp.stack([rates[n] for n in names])
                carried_zone = jnp.stack([auxiliary[AUXILIARY + n] for n in names])
                condition = sommerfeld(fields)
                carried = jnp.where(zone, carried_zone, kernel - condition)
                own = jnp.where(zone, condition + carried, kernel)
                moved = -speed * (radial(carried) + 3.0 * carried * inverse)
                moved = jnp.where(zone, moved, 0.0)
                out = {name: own[index] for index, name in enumerate(names)}
                out.update({AUXILIARY + name: moved[index] for index, name in enumerate(names)})
                return out

            cache[key] = jax.jit(combine)
        return cache[key](geometry, auxiliary, rates)

    def right_hand_side(self, state) -> dict[str, Any]:
        geometry, auxiliary = self._split(state)
        rates = self.evolution.right_hand_side(geometry)
        if self.boundary.stacked:
            return self._stacked_rates(geometry, auxiliary, rates)
        zone = self.boundary.mask()
        b = self.boundary
        out = {}
        for name, value in geometry.items():
            field = np.asarray(value)
            speed = b.speed if b.speeds is None else b.speeds.get(name, b.speed)
            departure = field - b.background_value(name)
            gradient, radius = self._radial(field if b.background is None else departure)
            sommerfeld = -speed * (gradient + departure / radius)
            kernel = np.asarray(rates[name])
            carried = np.where(zone, np.asarray(auxiliary[AUXILIARY + name]), kernel - sommerfeld)
            out[name] = np.where(zone, sommerfeld + carried, kernel)
            carried_gradient, _ = self._radial(carried)
            out[AUXILIARY + name] = np.where(
                zone, -speed * (carried_gradient + 3.0 * carried / radius), 0.0
            )
        return out

    def step(self, state, time_step: float | None = None):
        """One classical fourth-order step, both conditions applied at each stage."""
        step = self.time_step if time_step is None else float(time_step)
        advance, finish, _ = operations(self.backend)
        base = self.start(state)
        first = self.right_hand_side(base)
        second = self.right_hand_side(advance(base, first, step / 2))
        third = self.right_hand_side(advance(base, second, step / 2))
        fourth = self.right_hand_side(advance(base, third, step))
        out = finish(base, first, second, third, fourth, step)
        return self.project(out) if self.evolution.enforce else out


# --- a constraint-preserving absorbing layer ---------------------------------


def angular_laplacian(module, stacked, coords, spacing):
    """``r^2`` times the Laplacian on the unit sphere, of every field in ``stacked``.

    ``sum_k L_k L_k f`` with ``L_k = eps_kab x_a d_b``, the rotation
    generators: only derivatives tangential to spheres about the origin,
    which is what a surface formulation is allowed to use. Taken with
    :func:`edge_derivative`'s stencils, so it does not wrap.
    """

    def rotate(field, a, b):
        return coords[a] * _stacked_edge_derivative(module, field, b, spacing[b]) - coords[
            b
        ] * _stacked_edge_derivative(module, field, a, spacing[a])

    return sum(rotate(rotate(stacked, a, b), a, b) for a, b in ((1, 2), (2, 0), (0, 1)))


#: The conformal-metric components whose incoming radiation the layer relaxes.
METRIC = tuple(f"gt{i}{j}" for i in INDICES for j in range(i, DIMENSION))

#: The prefix of the rim's carried fields, distinct from the layer's surface auxiliaries.
RIM = AUXILIARY + AUXILIARY


@dataclass(frozen=True, eq=False)
class ConstraintPreserving(Bounded):
    """Einstein's equations in the zone, with only the incoming radiation changed there.

    Sommerfeld's condition and :class:`SecondOrder` replace the rates in the
    whole zone by an outgoing-wave equation for every variable separately,
    which is why neither respects the constraints: the components of
    ``gammabar_ij`` and ``Abar_ij`` along ``x^i / r``, and ``Gammabar^i``,
    are tied together by the constraints and get conditions that are not.

    Here the zone is split. Its outer ``rim`` points, the only ones where the
    periodic kernel's stencil wraps, keep :class:`SecondOrder`'s treatment.
    Everywhere inside that -- the *layer*, from the zone's inner face to the
    rim -- the evolution is the interior's: the same kernel, the same
    dissipation, so the constraints propagate there as they do inside and
    nothing in the layer can put the solution off the constraint surface.
    Two things are added in the layer, for the two kinds of incoming mode
    that are not constraints in BSSN linearised about flat space with a
    frozen shift:

    * **the radiative modes.** ``w_ij = B1 gammabar_ij``, with Einstein's own
      ``d_t gammabar_ij`` in it, is the incoming characteristic variable of
      each component: zero for a wave ``f(t - r)/r`` and ``-b/r^3`` for a
      ``b(t - r)/r^2`` part. Its part transverse and traceless with respect
      to ``x^i / r`` -- the radiation -- is relaxed at the rate
      ``strength`` towards the value Bayliss and Turkel's second condition
      gives it, through ``d_t Abar_ij``. That value is a surface auxiliary,

          d_t v = c^2 Delta_S u / (2 r^2) - c v / r

      which is ``(d_t + c d_r + 3c/r) v = 0`` with the radial derivative
      eliminated by the wave equation, so it is evolved at each point from
      derivatives tangential to spheres and never read from the interior.
      This is the only term that changes the physics, and on an outgoing
      wave it is as large as the wave's failure of that condition.
    * **the gauge mode.** The lapse obeys an outgoing radiation condition
      in the layer instead of its slicing condition, with the same surface
      auxiliary. That is a choice of slicing, which the constraints do not
      see.

    **The constraint modes are left alone, and that is measured, not
    assumed.** Damping the connection constraint ``G^i = Gammabar^i +
    d_j gammabar^ij`` was the obvious third term, and it is unstable: it
    pulls ``Gammabar^i`` towards the metric's own connection, which is
    ADM's principal part. Along ``x^i / r`` the longitudinal sector then has
    ``(lambda + sigma)(lambda^2 + k^2) = (4/3) sigma k^2``, a positive real
    root for every ``sigma > 0``, about ``k / sqrt(3)`` for ``sigma >> k``. On
    the Teukolsky wave at 48 points it grew threefold every half unit.

    The rim's reflection is an incoming mode at the layer's outer face, and
    the layer absorbs its radiative part on the way in. A narrower rim than
    the dissipation's stencil is allowed because the dissipation here is
    switched off, per axis, within its radius of the edge, as
    :func:`particlesim.core.grid.kreiss_oliger` does.
    """

    strength: float = 20.0
    rim: int = 2

    def __post_init__(self) -> None:
        radius = self.evolution.order // 2
        if self.rim < radius:
            raise ValueError(
                f"a rim of {self.rim} points is narrower than the stencil radius {radius}: "
                "the kernel's wrapped rates would reach the layer"
            )
        if self.boundary.width <= self.rim:
            raise ValueError(
                f"a zone of {self.boundary.width} points leaves no layer inside a rim of {self.rim}"
            )
        if not self.boundary.stacked:
            raise ValueError("the constraint-preserving layer is implemented for JAX at order 4")

    def auxiliary_names(self, state) -> tuple[str, ...]:
        """The layer's surface auxiliaries and the rim's carried fields."""
        geometry = [name for name in state if not name.startswith(AUXILIARY)]
        surface = tuple(AUXILIARY + name for name in (*METRIC, "alpha") if name in geometry)
        return surface + tuple(RIM + name for name in geometry)

    def start(self, state) -> dict[str, Any]:
        """``state`` with its auxiliary fields, zero where they are missing."""
        out = dict(state)
        shape = np.asarray(state["alpha"]).shape
        for name in self.auxiliary_names(state):
            if name not in out:
                out[name] = np.zeros(shape)
        return out

    def project(self, state) -> dict[str, Any]:
        geometry, auxiliary = SecondOrder._split(self, state)
        return {**self.evolution.project(geometry), **auxiliary}

    def gauge_speed(self) -> float:
        b = self.boundary
        table = b.speeds or GAUGE_SPEEDS.get(self.evolution.slicing, {})
        return float(table.get("alpha", b.speed))

    def masks(self):
        """``(layer, rim)``: the zone less its outer ``rim`` points, and those points."""
        b = self.boundary
        rim = dataclasses.replace(b, width=self.rim).mask()
        return b.mask() & ~rim, rim

    def _undissipated(self) -> Evolution:
        cache = _STACKED.setdefault(self.boundary, {})
        key = ("undissipated", id(self.evolution))
        if key not in cache:
            cache[key] = dataclasses.replace(self.evolution, dissipation=0.0)
        return cache[key]

    def _compiled(self, names: tuple[str, ...]):
        b = self.boundary
        cache = _STACKED.setdefault(b, {})
        key = ("constraint-preserving", names, self.strength, self.rim)
        if key in cache:
            return cache[key]
        import jax

        jnp = _module("jax")
        layer_np, rim_np = self.masks()
        layer, rim = jnp.asarray(layer_np), jnp.asarray(rim_np)
        rim_boundary = dataclasses.replace(b, width=self.rim)
        radial, sommerfeld, speed, inverse = rim_boundary._compiled(names)
        radius = np.maximum(b.radius, min(b.spacing))
        direction = [jnp.asarray(np.asarray(b.coords[axis]) / radius) for axis in INDICES]
        coords = [jnp.asarray(np.asarray(b.coords[axis])) for axis in INDICES]
        spacing = tuple(b.spacing)
        light, gauge, sigma = float(b.speed), self.gauge_speed(), float(self.strength)
        projector = [
            [(1.0 if i == j else 0.0) - direction[i] * direction[j] for j in INDICES]
            for i in INDICES
        ]

        # Kreiss-Oliger dissipation as the evolution applies it, but per axis
        # and switched off within its radius of a bounded edge.
        ko_radius, ko_weights = dissipation_operator(self.evolution.order)
        shape = np.asarray(b.coords[0]).shape
        allowed = []
        for axis in INDICES:
            index = np.arange(shape[axis])
            keep = np.ones(shape[axis], dtype=bool)
            if axis in b.axes:
                keep = (index >= ko_radius) & (index < shape[axis] - ko_radius)
            allowed.append(jnp.asarray(keep.reshape([-1 if k == axis else 1 for k in INDICES])))

        def dissipation(stack):
            total = 0.0
            for axis in INDICES:
                stencil = sum(
                    weight * jnp.roll(stack, -offset, axis=axis + 1)
                    for offset, weight in enumerate(ko_weights, start=-ko_radius)
                )
                total = total + jnp.where(allowed[axis], stencil / spacing[axis], 0.0)
            return self.evolution.dissipation * total

        def surface(departure, carried, c):
            # d_t v = c^2 Delta_S u / (2 r^2) - c v / r, in the layer only.
            moved = 0.5 * c**2 * angular_laplacian(jnp, departure, coords, spacing)
            return jnp.where(layer, moved * inverse**2 - c * carried * inverse, 0.0)

        def combine(geometry, auxiliary, einstein):
            stack = jnp.stack([geometry[n] for n in names])
            rates = jnp.stack([jnp.broadcast_to(einstein[n], stack.shape[1:]) for n in names])
            if self.evolution.dissipation > 0.0:
                rates = rates + dissipation(stack)
            rate = {name: rates[index] for index, name in enumerate(names)}
            out = {}

            # The radiative modes, through Abar_ij, in the layer.
            flat = jnp.asarray([1.0 if n[2] == n[3] else 0.0 for n in METRIC])
            metric = jnp.stack([geometry[n] for n in METRIC]) - flat.reshape((-1, 1, 1, 1))
            incoming = jnp.stack([einstein[n] for n in METRIC]) + light * (
                radial(metric) + metric * inverse
            )
            carried = jnp.stack([auxiliary[AUXILIARY + n] for n in METRIC])
            mismatch = dict(zip(METRIC, incoming - carried, strict=True))
            sandwiched = [
                [
                    sum(
                        projector[i][a] * projector[j][c] * mismatch[f"gt{min(a, c)}{max(a, c)}"]
                        for a in INDICES
                        for c in INDICES
                    )
                    for j in INDICES
                ]
                for i in INDICES
            ]
            traced = sum(sandwiched[a][a] for a in INDICES)
            relax = 0.5 * sigma / geometry["alpha"]
            for i in INDICES:
                for j in range(i, DIMENSION):
                    transverse = sandwiched[i][j] - 0.5 * projector[i][j] * traced
                    rate[f"At{i}{j}"] = rate[f"At{i}{j}"] + jnp.where(layer, relax * transverse, 0)
            moved = surface(metric, carried, light)
            out.update({AUXILIARY + n: moved[k] for k, n in enumerate(METRIC)})

            # The gauge mode: the lapse radiates out, a choice of slicing.
            lapse = (geometry["alpha"] - 1.0)[None]
            held = auxiliary[AUXILIARY + "alpha"][None]
            condition = -gauge * (radial(lapse) + lapse * inverse) + held
            rate["alpha"] = jnp.where(layer, condition[0], rate["alpha"])
            out[AUXILIARY + "alpha"] = surface(lapse, held, gauge)[0]

            # The rim: SecondOrder, fed by the layer's rates.
            dynamics = jnp.stack([rate[n] for n in names])
            held = jnp.stack([auxiliary[RIM + n] for n in names])
            outgoing = sommerfeld(stack)
            carried = jnp.where(rim, held, dynamics - outgoing)
            final = jnp.where(rim, outgoing + carried, dynamics)
            moved = jnp.where(rim, -speed * (radial(carried) + 3.0 * carried * inverse), 0.0)
            out.update({name: final[k] for k, name in enumerate(names)})
            out.update({RIM + name: moved[k] for k, name in enumerate(names)})
            return out

        compiled = jax.jit(combine)
        cache[key] = compiled
        return compiled

    def right_hand_side(self, state) -> dict[str, Any]:
        geometry, auxiliary = SecondOrder._split(self, state)
        names = tuple(sorted(geometry))
        einstein = self._undissipated().right_hand_side(geometry)
        return self._compiled(names)(geometry, auxiliary, einstein)

    def step(self, state, time_step: float | None = None):
        """One classical fourth-order step, the layer and the rim applied at each stage."""
        step = self.time_step if time_step is None else float(time_step)
        advance, finish, _ = operations(self.backend)
        base = self.start(state)
        first = self.right_hand_side(base)
        second = self.right_hand_side(advance(base, first, step / 2))
        third = self.right_hand_side(advance(base, second, step / 2))
        fourth = self.right_hand_side(advance(base, third, step))
        out = finish(base, first, second, third, fourth, step)
        return self.project(out) if self.evolution.enforce else out


def interior(array, width: int, axes) -> np.ndarray:
    """The physical region, with the boundary zone cut off.

    The zone's values are the boundary condition's, not the evolution's, so
    a norm that includes them measures the condition rather than the
    solution -- the same distinction the refinement buffer needed, and the
    same reason for keeping it explicit.
    """
    out = np.asarray(array)
    slices = [slice(None)] * out.ndim
    for axis in axes:
        slices[axis] = slice(width, out.shape[axis] - width)
    return out[tuple(slices)]


__all__ = [
    "ASYMPTOTIC",
    "AUXILIARY",
    "GAUGE_SPEEDS",
    "METRIC",
    "RIM",
    "Bounded",
    "ConstraintPreserving",
    "Radiative",
    "SecondOrder",
    "angular_laplacian",
    "edge_derivative",
    "interior",
]
