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
in general does not. Constraint-preserving boundary conditions are a harder
problem and are not attempted here.

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

**The cost that is real is arithmetic.** :meth:`Radiative.rates` takes three
bounded-domain derivatives per variable per stage over the *whole* array,
which is seventy-two array passes a stage for BSSN and dominates the run at
64^3. Restricting it to a slab is the obvious fix and is not done here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from particlesim.core.grid import derivative
from particlesim.solvers.nr.bssn import DIMENSION, INDICES, Evolution, _module

#: What each evolved variable tends to at large radius, in vacuum.
#:
#: Minkowski in the BSSN variables: a unit lapse, a flat conformal metric, no
#: shift and no curvature. A variable whose background is wrong here is
#: reflected at the amplitude of the difference, which is the loudest way to
#: get a boundary condition wrong and the easiest to check.
ASYMPTOTIC: dict[str, float] = {
    "alpha": 1.0,
    "phi": 0.0,
    "trK": 0.0,
    "Theta": 0.0,
    **{f"beta{i}": 0.0 for i in INDICES},
    **{f"B{i}": 0.0 for i in INDICES},
    **{f"Gt{i}": 0.0 for i in INDICES},
    **{f"gt{i}{j}": (1.0 if i == j else 0.0) for i in INDICES for j in range(i, DIMENSION)},
    **{f"At{i}{j}": 0.0 for i in INDICES for j in range(i, DIMENSION)},
}


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
    the third point from the edge takes the centred fourth-order stencil,
    so the edge is fourth order either way.
    """
    out = derivative(np.asarray(field), axis, step, order=order)
    if order == 2:
        return out
    a = np.moveaxis(np.asarray(field), axis, 0)
    o = np.moveaxis(out, axis, 0)
    o[0] = (-25 * a[0] + 48 * a[1] - 36 * a[2] + 16 * a[3] - 3 * a[4]) / (12 * step)
    o[1] = (-3 * a[0] - 10 * a[1] + 18 * a[2] - 6 * a[3] + a[4]) / (12 * step)
    o[-1] = (25 * a[-1] - 48 * a[-2] + 36 * a[-3] - 16 * a[-4] + 3 * a[-5]) / (12 * step)
    o[-2] = (3 * a[-1] + 10 * a[-2] - 18 * a[-3] + 6 * a[-4] - a[-5]) / (12 * step)
    if order == 6:
        o[2] = (-a[4] + 8 * a[3] - 8 * a[1] + a[0]) / (12 * step)
        o[-3] = (a[-5] - 8 * a[-4] + 8 * a[-2] - a[-1]) / (12 * step)
    return out


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
    """

    coords: tuple[Any, ...]
    spacing: tuple[float, ...]
    axes: tuple[int, ...]
    width: int
    speed: float = 1.0
    order: int = 4
    backend: str = "jax"
    speeds: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        radius = self.order // 2
        if self.width < radius:
            raise ValueError(
                f"a boundary zone of {self.width} points is narrower than the "
                f"stencil radius {radius}: the kernel's wrapped rates would "
                "reach the interior"
            )

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

    def rates(self, state) -> dict[str, Any]:
        """``d_t f`` from the Sommerfeld condition, everywhere.

        Computed on the whole array and used only in the zone, because the
        derivative it needs is the *bounded-domain* one --
        :func:`edge_derivative`, which uses one-sided stencils at an edge
        rather than wrapping. That is the one place in this file where not
        wrapping matters, and slicing a slab out first would put the wrap
        back.

        Three derivatives per variable per call is the cost, and it is the
        reason this is worth restricting to a slab once it carries a
        production run.
        """
        radius = np.maximum(self.radius, min(self.spacing))
        out = {}
        for name, value in state.items():
            field = np.asarray(value)
            background = ASYMPTOTIC.get(name, 0.0)
            gradient = sum(
                np.asarray(self.coords[axis])
                / radius
                * edge_derivative(field, axis, self.spacing[axis], order=self.order)
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
        names = list(state)
        base = dict(state)

        first = self.right_hand_side(base)
        second = self.right_hand_side({n: base[n] + (step / 2) * first[n] for n in names})
        third = self.right_hand_side({n: base[n] + (step / 2) * second[n] for n in names})
        fourth = self.right_hand_side({n: base[n] + step * third[n] for n in names})

        out = {
            n: base[n] + (step / 6) * (first[n] + 2 * second[n] + 2 * third[n] + fourth[n])
            for n in names
        }
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

    def right_hand_side(self, state) -> dict[str, Any]:
        geometry, auxiliary = self._split(state)
        rates = self.evolution.right_hand_side(geometry)
        zone = self.boundary.mask()
        b = self.boundary
        out = {}
        for name, value in geometry.items():
            field = np.asarray(value)
            speed = b.speed if b.speeds is None else b.speeds.get(name, b.speed)
            gradient, radius = self._radial(field)
            sommerfeld = -speed * (gradient + (field - ASYMPTOTIC.get(name, 0.0)) / radius)
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
        base = self.start(state)
        names = list(base)
        first = self.right_hand_side(base)
        second = self.right_hand_side({n: base[n] + (step / 2) * first[n] for n in names})
        third = self.right_hand_side({n: base[n] + (step / 2) * second[n] for n in names})
        fourth = self.right_hand_side({n: base[n] + step * third[n] for n in names})
        out = {
            n: base[n] + (step / 6) * (first[n] + 2 * second[n] + 2 * third[n] + fourth[n])
            for n in names
        }
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
    "Bounded",
    "Radiative",
    "SecondOrder",
    "edge_derivative",
    "interior",
]
