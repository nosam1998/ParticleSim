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

**What this is not.** It is not constraint-preserving. The Sommerfeld
condition is applied variable by variable to quantities that are not
characteristic variables of the system, so it injects constraint violation
at the boundary at the level of its own error. Constraint-preserving
boundary conditions are a harder problem and are not attempted here; what
is measured instead is that a pulse leaves and does not come back, and how
much constraint violation the boundary costs.
"""

from __future__ import annotations

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


@dataclass(frozen=True, eq=False)
class Radiative:
    """The Sommerfeld condition on a slab at the outer edge of the grid.

    ``axes`` are the axes that have an outer boundary; the rest stay
    periodic, which is what makes a one-dimensional test possible. ``width``
    is the zone in points, and has to be at least the stencil radius so the
    wrapped rates the kernel produces there are all discarded.

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
        :func:`particlesim.core.grid.derivative`, which falls back to a
        one-sided stencil at an edge rather than wrapping. That is the one
        place in this file where not wrapping matters, and slicing a slab out
        first would put the wrap back.

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
                * derivative(field, axis, self.spacing[axis], order=self.order)
                for axis in range(DIMENSION)
            )
            out[name] = -self.speed * (gradient + (field - background) / radius)
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


__all__ = ["ASYMPTOTIC", "Bounded", "Radiative", "interior"]
