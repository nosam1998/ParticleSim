"""An octant domain: one eighth of a reflection-symmetric problem, evolved.

:func:`~particlesim.solvers.nr.symmetry.project` restricts a full grid to
the three coordinate reflections by group averaging, and pays for the full
grid. This evolves only the octant ``x, y, z > 0`` of each cell-centred
level, plus :data:`GHOSTS` cells below each symmetry plane, as an octant
code with symmetry faces does (e.g. Imbiriba et al., gr-qc/0403048,
section IV). It is the same physical restriction, so the same caveat
applies: it projects out every asymmetric perturbation, and says nothing
about stability against them.

**How the ghosts are filled.** The kernels difference with ``roll``. On a
level's low side the wrapped values are wrong, so every right-hand side
first overwrites the ghost layer with the mirror image of the first
:data:`GHOSTS` interior cells, with each component's parity
(:func:`~particlesim.solvers.nr.symmetry.parity`), and then reflects its
own rates the same way. Runge-Kutta combinations of reflected states and
rates stay reflected, so the ghost layer is the mirror image at every
stage without being reset anywhere else.

**Cell centres make the mirror exact.** No sample lies on a symmetry plane:
the first interior cell is at ``h/2`` and its mirror at ``-h/2``. Four
ghosts cover the widest stencil in play (the Kreiss-Oliger and lopsided
advection stencils reach three cells) and leave two parent cells below
each box's origin for the fourth-order prolongation stencil, so nothing
wraps across a symmetry plane.

**What changes in the hierarchy.** Only the high side of each level is an
edge. The coarsest level's radiation zone (:class:`OctantRadiative`) and
each fine level's buffer (:class:`OctantHierarchy`) are applied there and
nowhere else. Restriction runs right up to the symmetry plane, because the
mirrored fine cells are valid data; only the parent's own ghost cells are
excluded, and the next right-hand side refills them.

An octant of ``n`` cells across has ``(n/2 + 4)^3`` points against ``n^3``:
a quarter at ``n = 32``, a fifth at 48, approaching an eighth. An octant run
agrees with the reflection-symmetric full-grid run to rounding
(``tests/unit/test_octant.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from particlesim.solvers.nr import mesh
from particlesim.solvers.nr.boundary import Bounded, Radiative, SecondOrder
from particlesim.solvers.nr.bssn import Evolution, _module
from particlesim.solvers.nr.refined import Hierarchy, Nested
from particlesim.solvers.nr.symmetry import parity

#: Ghost cells below each symmetry plane, on every level.
GHOSTS = 4


def _mirror(value, sign, axis, ghosts, xp):
    source = [slice(None)] * value.ndim
    source[axis] = slice(ghosts, 2 * ghosts)
    target = [slice(None)] * value.ndim
    target[axis] = slice(0, ghosts)
    image = sign * xp.flip(value[tuple(source)], axis=axis)
    if xp is np:
        value = np.array(value, copy=True)
        value[tuple(target)] = image
        return value
    return value.at[tuple(target)].set(image)


def fill_ghosts(state, ghosts: int = GHOSTS, xp=np) -> dict[str, Any]:
    """Overwrite each axis's first ``ghosts`` cells with the mirror of the next ``ghosts``."""
    out = {}
    for name, value in state.items():
        value = xp.asarray(value)
        if value.ndim != 3 or min(value.shape) < 2 * ghosts:
            raise ValueError(f"{name!r} is too small for {ghosts} reflection ghosts")
        for axis, sign in enumerate(parity(name)):
            value = _mirror(value, sign, axis, ghosts, xp)
        out[name] = value
    return out


@lru_cache(maxsize=8)
def _filler(ghosts: int, backend: str):
    if backend == "numpy":
        return lambda state: fill_ghosts(state, ghosts, np)
    import jax

    xp = _module("jax")
    return jax.jit(lambda state: fill_ghosts(state, ghosts, xp))


def reflect(state, ghosts: int = GHOSTS, backend: str = "jax") -> dict[str, Any]:
    """:func:`fill_ghosts`, compiled once for JAX."""
    return _filler(int(ghosts), backend)(dict(state))


def unfold(state, ghosts: int = GHOSTS) -> dict[str, np.ndarray]:
    """The full reflection-symmetric grid an octant state stands for, as NumPy arrays."""
    out = {}
    for name, value in state.items():
        value = np.asarray(value)
        for axis, sign in enumerate(parity(name)):
            inner = [slice(None)] * 3
            inner[axis] = slice(ghosts, None)
            half = value[tuple(inner)]
            value = np.concatenate((sign * np.flip(half, axis=axis), half), axis=axis)
        out[name] = value
    return out


def fold(state, ghosts: int = GHOSTS, xp=np) -> dict[str, Any]:
    """The octant of a full even grid, with its ghost layer: the inverse of :func:`unfold`."""
    out = {}
    for name, value in state.items():
        value = xp.asarray(value)
        if any(size % 2 for size in value.shape):
            raise ValueError("an octant needs three even cell-centred axes")
        out[name] = value[tuple(slice(size // 2 - ghosts, None) for size in value.shape)]
    return out


@dataclass(frozen=True)
class OctantEvolution(Evolution):
    """An evolution whose right-hand side reflects its input and output across the low faces."""

    ghosts: int = GHOSTS

    def right_hand_side(self, state):
        state = reflect(state, self.ghosts, self.backend)
        return reflect(super().right_hand_side(state), self.ghosts, self.backend)


@dataclass(frozen=True, eq=False)
class OctantRadiative(Radiative):
    """The radiation zone on the high faces only: the low faces are symmetry planes."""

    def mask(self) -> np.ndarray:
        shape = np.asarray(self.coords[0]).shape
        out = np.zeros(shape, dtype=bool)
        for axis in self.axes:
            edge = np.arange(shape[axis]) >= shape[axis] - self.width
            out |= edge.reshape([-1 if k == axis else 1 for k in range(len(shape))])
        return out


class _Reflected:
    """Reflect a bounded evolution's rates, auxiliary fields included.

    The state's ghosts are already mirror images: every operation that
    makes a state combines reflected states and rates, or reflects its
    result. The evolution inside reflects its input again regardless.
    """

    def right_hand_side(self, state):
        ghosts, backend = self.evolution.ghosts, self.evolution.backend
        return reflect(super().right_hand_side(state), ghosts, backend)


@dataclass(frozen=True, eq=False)
class OctantBounded(_Reflected, Bounded):
    """:class:`~particlesim.solvers.nr.boundary.Bounded` on an octant."""


@dataclass(frozen=True, eq=False)
class OctantSecondOrder(_Reflected, SecondOrder):
    """:class:`~particlesim.solvers.nr.boundary.SecondOrder` on an octant."""


@lru_cache(maxsize=64)
def _high_mask(shape, width):
    mask = np.zeros(shape, dtype=bool)
    for axis in range(len(shape)):
        index = [slice(None)] * len(shape)
        index[axis] = slice(shape[axis] - width, shape[axis])
        mask[tuple(index)] = True
    return mask


def _inject(coarse, fine, box, order, buffer, ghosts, backend):
    restricted = mesh.restrict(fine, centering="cell", order=order, backend=backend)
    low, high = ghosts // 2, (buffer + order // 2) // 2
    target = tuple(
        slice(start + low, start + count - high)
        for start, count in zip(box.origin, box.shape, strict=True)
    )
    restricted = restricted[tuple(slice(low, count - high) for count in box.shape)]
    if backend == "numpy":
        out = np.array(coarse, copy=True)
        out[target] = restricted
        return out
    return coarse.at[target].set(restricted)


@lru_cache(maxsize=64)
def _jax_octant(box, order, buffer, ghosts, shape):
    import jax
    import jax.numpy as jnp

    mask = _high_mask(shape, buffer)
    inject_field = jax.vmap(
        lambda coarse, fine: _inject(coarse, fine, box, order, buffer, ghosts, "jax")
    )

    def inject(coarse, fine):
        names = tuple(fine)
        changed = inject_field(
            jnp.stack([coarse[name] for name in names]),
            jnp.stack([fine[name] for name in names]),
        )
        return {**coarse, **{name: changed[i] for i, name in enumerate(names)}}

    def fill(values, source):
        return {name: jnp.where(mask, source[name], value) for name, value in values.items()}

    return jax.jit(inject), jax.jit(fill)


@dataclass(frozen=True)
class OctantHierarchy(Hierarchy):
    """A parent and child octant: buffer on the high faces, restriction up to the planes.

    The box is indexed in the parent's octant array. Its origin is
    ``ghosts / 2`` along every axis, so that its fine cells start
    ``ghosts`` cells below the symmetry planes, as its parent's do.
    """

    ghosts: int = GHOSTS

    def __post_init__(self) -> None:
        if self.box.centering != "cell":
            raise ValueError("an octant hierarchy needs cell-centred boxes")
        if any(origin != self.ghosts // 2 for origin in self.box.origin):
            raise ValueError(
                f"an octant box starts {self.ghosts // 2} parent cells into the ghost layer"
            )

    def _fill_buffer(self, fine_state, boundary) -> dict[str, Any]:
        if self.coarse.backend == "jax":
            shape = next(iter(fine_state.values())).shape
            _, fill = _jax_octant(
                self.box, self.interpolation, self.buffer, self.ghosts, tuple(shape)
            )
            return fill(fine_state, boundary)
        return {
            name: np.where(_high_mask(np.shape(value), self.buffer), boundary[name], value)
            for name, value in fine_state.items()
        }

    def _restrict_into(self, coarse_state, fine_state) -> dict[str, Any]:
        """Restrict, then refill the parent's ghosts from what was restricted."""
        backend = self.coarse.backend
        if backend == "jax":
            shape = tuple(next(iter(fine_state.values())).shape)
            inject, _ = _jax_octant(self.box, self.interpolation, self.buffer, self.ghosts, shape)
            return reflect(inject(coarse_state, fine_state), self.ghosts, backend)
        restricted = {
            name: _inject(
                value,
                np.asarray(fine_state[name]),
                self.box,
                self.interpolation,
                self.buffer,
                self.ghosts,
                "numpy",
            )
            if name in fine_state
            else value
            for name, value in coarse_state.items()
        }
        return reflect(restricted, self.ghosts, backend)


@dataclass(frozen=True)
class OctantNested(Nested):
    """:class:`~particlesim.solvers.nr.refined.Nested`, each box clear of its parent's high edge.

    The low faces are symmetry planes rather than edges, so only the high
    side has a buffer to stay clear of.
    """

    def __post_init__(self) -> None:
        for outer, inner in zip(self.pairs, self.pairs[1:], strict=False):
            if tuple(inner.parent_shape) != outer.box.fine_shape:
                raise ValueError("level shapes do not chain")
            margin = len(mesh.MIDPOINT[inner.interpolation]) // 2
            for axis in range(len(inner.parent_shape)):
                above = inner.parent_shape[axis] - inner.box.origin[axis] - inner.box.shape[axis]
                if above - margin < outer.buffer:
                    raise ValueError(
                        f"the box comes within {outer.buffer} points of its parent's high "
                        f"edge along axis {axis}, where the parent is interpolated"
                    )


__all__ = [
    "GHOSTS",
    "OctantBounded",
    "OctantEvolution",
    "OctantHierarchy",
    "OctantNested",
    "OctantRadiative",
    "OctantSecondOrder",
    "fill_ghosts",
    "fold",
    "reflect",
    "unfold",
]
