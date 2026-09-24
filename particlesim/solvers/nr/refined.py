"""A two-level refinement hierarchy: Berger-Oliger subcycling with buffers.

The coarse level is the periodic domain the uniform solvers already evolve.
The fine level is one :class:`~particlesim.solvers.nr.mesh.Box` inside it at
half the spacing, taking two steps for every coarse one because the Courant
condition ties the time step to the spacing.

**One coarse step, in order.** Advance the coarse level; take two fine steps;
refill the fine level's buffer from the coarse level after each; restrict the
fine interior back onto the coarse points it covers. The restriction is last
because the coarse solution inside the box is the fine one's, injected, and
the coarse level's own value there is only a placeholder.

**The buffer is filled by Hermite interpolation in time, not linear, and at
every Runge-Kutta stage rather than once a step.** The fine level needs
coarse data between coarse steps. Linear interpolation between the two ends
is second-order accurate and would cap the whole scheme there however good
the spatial stencils are; Hermite cubic through the values *and the rates* at
both ends is fourth-order, and costs one extra right-hand-side evaluation
per coarse step on the cheap level.

Filling once a step is not enough either, and the gauge wave says so. A
frozen buffer holds values correct at the step's start while stages two to
four want later times, and that time error marches inward three points per
stage -- clear of a twelve-point buffer and into the interior. Measured that
way the scheme converged at third order, ratios 8.5 and 8.9 against the 16
fourth order owes. Each stage is filled at its own time instead, which is
five prolongations per coarse step after caching the five distinct times a
pair of fine steps asks for.

**Axes the box spans need no buffer at all.** A box covering a whole
periodic axis *is* periodic along it, so the wrap the stencils do is the
right one and there is nothing to fill. Refining in one direction only --
the useful case for a plane wave, and the cheapest thing to test on -- then
buffers one axis instead of three.

**The operators run on the host.** ``prolong`` and ``restrict`` are NumPy,
so a JAX state makes one round trip per coarse step for the arrays that get
buffered. That is a handful of transfers against eight kernel launches, and
it is not free: making them device-resident is worth doing before this
carries a production run, and is not done here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from particlesim.solvers.nr import mesh
from particlesim.solvers.nr.bssn import Evolution, dissipation_operator
from particlesim.solvers.nr.mesh import RATIO, Box


def _as_backend(array, backend: str):
    if backend == "numpy":
        return np.asarray(array)
    import jax.numpy as jnp

    return jnp.asarray(array)


def hermite(start, start_rate, end, end_rate, theta: float, step: float):
    """Cubic Hermite interpolation between two states, one field at a time.

    ``theta`` runs from zero at ``start`` to one at ``end``. The basis is the
    usual one,

        h00 = 2 t^3 - 3 t^2 + 1,   h10 = t^3 - 2 t^2 + t
        h01 = -2 t^3 + 3 t^2,      h11 = t^3 - t^2

    with the rates multiplied by the step so that ``start_rate`` is a time
    derivative rather than an increment. Fourth-order accurate, which is the
    point: it matches the spatial stencils instead of throttling them.
    """
    t2 = theta * theta
    t3 = t2 * theta
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + theta
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return {
        name: h00 * start[name]
        + h10 * step * start_rate[name]
        + h01 * end[name]
        + h11 * step * end_rate[name]
        for name in start
    }


@dataclass(frozen=True)
class Hierarchy:
    """A coarse evolution, a fine one, and the box that relates them."""

    coarse: Evolution
    fine: Evolution
    box: Box
    parent_shape: tuple[int, ...]
    interpolation: int = 4
    buffer: int = 0

    @classmethod
    def build(
        cls,
        coarse: Evolution,
        box: Box,
        parent_shape,
        interpolation: int = 4,
        buffer: int | None = None,
    ) -> Hierarchy:
        """The fine level from the coarse one: same kernel, half the spacing.

        The emitted kernels take the spacing as an argument and difference
        with ``roll``, so they do not know or care how big the grid is. One
        derivation serves both levels, which is why refinement costs no
        symbolic work at all.

        ``parent_shape`` is the coarse grid's shape, which the box is indexed
        against. It is needed to tell a box that spans an axis from one that
        merely happens to be the same size, and a box that does not fit is
        refused here rather than producing a silently wrapped fine level.

        ``buffer`` overrides :func:`~particlesim.solvers.nr.mesh.buffer_width`.
        That width assumes the wrap's damage compounds over four stages, which
        it no longer does now the buffer is refilled at every stage: only one
        stencil radius is ever wrong at once. On the two-level gauge wave the
        error in a fixed window is *lower* with a narrower buffer, because
        more of the box is evolved at the fine spacing instead of
        interpolated from the coarse one:

            buffer   n = 32     n = 64     n = 128    ratios
            12       1.43e-4    1.04e-5    6.39e-7    13.8, 16.2
            6        8.45e-5    6.34e-6    4.34e-7    13.3, 14.6
            3        4.14e-5    4.49e-6    3.26e-7     9.2, 13.8

        The default stays at twelve so that measurements already made with it
        stand; a buffer narrower than the widest stencil is refused.
        """
        parent_shape = tuple(int(value) for value in parent_shape)
        if not box.contains(parent_shape):
            raise ValueError(
                f"box at {box.origin} of shape {box.shape} does not fit inside a "
                f"parent of shape {parent_shape}: a box that runs off the edge "
                "would be filled from the wrap rather than from its parent"
            )
        radius, _ = dissipation_operator(coarse.order)
        widest = max(coarse.order // 2, radius)
        if buffer is None:
            width = mesh.buffer_width(
                stencil_radius=coarse.order // 2, dissipation_radius=radius, stages=4
            )
        elif int(buffer) < widest:
            raise ValueError(
                f"a buffer of {buffer} points is narrower than the widest stencil "
                f"radius {widest}: the wrapped rates would reach the interior"
            )
        else:
            width = int(buffer)
        fine = replace(coarse, spacing=box.spacing(coarse.spacing))
        return cls(
            coarse=coarse,
            fine=fine,
            box=box,
            parent_shape=parent_shape,
            interpolation=interpolation,
            buffer=width,
        )

    @property
    def buffered_axes(self) -> tuple[int, ...]:
        """Which axes need their edges filled from the parent.

        Only the ones the box does not span. An axis it spans is periodic on
        the fine level exactly as it is on the coarse one, so the wrap the
        stencils do is the right one and there is nothing to fill.
        """
        pairs = zip(self.box.origin, self.box.shape, strict=True)
        return tuple(
            axis
            for axis, (origin, count) in enumerate(pairs)
            if not (origin == 0 and count == self.parent_shape[axis])
        )

    def refine(self, coarse_state: Mapping[str, Any]) -> dict[str, Any]:
        """The fine level's initial state, prolonged out of the coarse one.

        Exact initial data evaluated directly on the fine grid is better
        where it exists, and is what the tests use for the gauge wave. This
        is what a hierarchy does when it does not have that.
        """
        return {
            name: _as_backend(
                mesh.extract(np.asarray(value), self.box, self.interpolation),
                self.coarse.backend,
            )
            for name, value in coarse_state.items()
        }

    def _fill_buffer(self, fine_state, boundary) -> dict[str, Any]:
        """Overwrite the fine box's edge with the parent's values.

        The buffer is as wide as four Runge-Kutta stages reach, so what is
        overwritten is exactly what the wrapped stencils ruined and nothing
        that was computed correctly.
        """
        axes = self.buffered_axes
        if not axes:
            return dict(fine_state)
        out = {}
        for name, value in fine_state.items():
            array = np.array(np.asarray(value), copy=True)
            source = np.asarray(boundary[name])
            for axis in axes:
                low = [slice(None)] * array.ndim
                high = [slice(None)] * array.ndim
                low[axis] = slice(0, self.buffer)
                high[axis] = slice(array.shape[axis] - self.buffer, array.shape[axis])
                array[tuple(low)] = source[tuple(low)]
                array[tuple(high)] = source[tuple(high)]
            out[name] = _as_backend(array, self.coarse.backend)
        return out

    def _restrict_into(self, coarse_state, fine_state) -> dict[str, Any]:
        """Inject the fine interior onto the coarse points inside the box."""
        return {
            name: _as_backend(
                mesh.inject(np.asarray(value), np.asarray(fine_state[name]), self.box),
                self.coarse.backend,
            )
            for name, value in coarse_state.items()
        }

    def _fine_step(self, state, step: float, boundary_at, start: float, span: float):
        """One fine Runge-Kutta step, with the buffer refilled at each stage.

        Refilling once per step is not enough, and the gauge wave says so.
        A frozen buffer holds values correct at the step's *start* while
        stages two to four want ``t + dt/2`` and ``t + dt``; that time error
        lives in the buffer and marches inward three points per stage, so it
        clears the twelve-point buffer and reaches the interior. Measured
        that way the scheme converged at **third** order -- ratios 8.5 and
        8.9 where fourth order owes 16 -- which is exactly what one
        order-lost looks like.

        Filling at each stage's own time costs four prolongations per fine
        step instead of one. ``boundary_at`` is cached on the five distinct
        times a pair of fine steps asks for, so it is five per coarse step
        rather than eight.
        """
        names = list(state)

        def stage(values, theta):
            filled = self._fill_buffer(values, boundary_at(theta))
            return filled, self.fine.right_hand_side(filled)

        half = span / 2
        base, first = stage(dict(state), start)
        _, second = stage({n: base[n] + (step / 2) * first[n] for n in names}, start + half)
        _, third = stage({n: base[n] + (step / 2) * second[n] for n in names}, start + half)
        _, fourth = stage({n: base[n] + step * third[n] for n in names}, start + span)

        out = {
            n: base[n] + (step / 6) * (first[n] + 2 * second[n] + 2 * third[n] + fourth[n])
            for n in names
        }
        out = self._fill_buffer(out, boundary_at(start + span))
        return self.fine.project(out) if self.fine.enforce else out

    def step(self, coarse_state, fine_state, time_step: float | None = None):
        """One coarse step and the two fine steps inside it."""
        step = self.coarse.time_step if time_step is None else float(time_step)
        before = dict(coarse_state)
        rate_before = self.coarse.right_hand_side(before)
        after = self.coarse.step(before, step)
        rate_after = self.coarse.right_hand_side(after)

        cache: dict[float, dict[str, Any]] = {}

        def boundary_at(theta: float):
            key = round(float(theta), 12)
            if key not in cache:
                level = hermite(before, rate_before, after, rate_after, key, step)
                cache[key] = {
                    name: mesh.extract(np.asarray(value), self.box, self.interpolation)
                    for name, value in level.items()
                }
            return cache[key]

        span = 1.0 / RATIO
        current = dict(fine_state)
        for sub in range(RATIO):
            current = self._fine_step(current, step / RATIO, boundary_at, sub * span, span)

        return self._restrict_into(after, current), current

    def run(self, coarse_state, fine_state, steps: int, time_step: float | None = None):
        """Integrate ``steps`` coarse steps, and twice that many fine ones."""
        step = self.coarse.time_step if time_step is None else float(time_step)
        coarse, fine = dict(coarse_state), dict(fine_state)
        for _ in range(steps):
            coarse, fine = self.step(coarse, fine, step)
        return coarse, fine


def constraint_norms(state, spacing, width: int, axes, order: int = 4, backend: str = "jax"):
    """``(|H|, |M|)`` over a fine box's interior, with the buffer excluded.

    :func:`particlesim.solvers.nr.bssn.constraints` norms the whole array,
    which on a refinement box is the wrong region twice over: the buffer
    holds prolonged parent values rather than evolved ones, and the
    constraint stencils wrap at the box edge exactly as the evolution's do.
    Measured over the whole box the Hamiltonian constraint sat at 1e-02 and
    did not converge at all -- ratios 0.9 and 1.4 -- because the edge
    dominated the norm and the edge is not a solution of anything.

    So the fields are computed and *then* the buffer is cut off, which is
    the region the fine level is actually responsible for.
    """
    from particlesim.solvers.nr.bssn import (
        INDICES,
        constraint_kernel,
        physical_slice_arrays,
    )

    kernel = constraint_kernel(order=order, backend=backend)
    fields = physical_slice_arrays(state, backend)
    out = kernel(fields, tuple(spacing))

    def norm(values):
        return float(np.sqrt(np.mean(interior(values, width, axes) ** 2)))

    momentum = np.sqrt(sum(norm(out[f"momentum{i}"]) ** 2 for i in INDICES))
    return norm(out["hamiltonian"]), float(momentum)


def interior(array, width: int, axes) -> np.ndarray:
    """A fine box with its buffer removed, for comparing against the truth.

    The buffer holds the parent's values, not the fine level's, so including
    it in an error norm measures the interpolation rather than the evolution.
    """
    out = np.asarray(array)
    slices = [slice(None)] * out.ndim
    for axis in axes:
        slices[axis] = slice(width, out.shape[axis] - width)
    return out[tuple(slices)]


__all__ = ["Hierarchy", "constraint_norms", "hermite", "interior"]
