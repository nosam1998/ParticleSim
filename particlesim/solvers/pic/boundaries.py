"""Boundary conditions for the Yee solver (design doc Section 5.4).

Three are provided and they differ in what they do to a wave that reaches
the edge: :class:`Periodic` returns it on the other side, :class:`Conducting`
reflects it, and :class:`PerfectlyMatchedLayer` absorbs it.

A boundary here is not a post-processing step applied to the fields. It is
the difference operator itself, because that is where the distinction lives:
a periodic difference wraps, a conducting one does not, and an absorbing one
is a differently stretched coordinate. Bolting absorption on afterwards, by
tapering the fields in a sponge layer, reflects at the taper's own edge and
is what a perfectly matched layer exists to avoid.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Boundary(Protocol):
    """What the solver asks of a boundary condition."""

    def attach(self, solver) -> None: ...

    def forward(self, f: np.ndarray, axis: int, delta: float, key: str) -> np.ndarray: ...

    def backward(self, f: np.ndarray, axis: int, delta: float, key: str) -> np.ndarray: ...

    def after_magnetic(self, B): ...

    def after_electric(self, D): ...


def _roll_forward(f: np.ndarray, axis: int, delta: float) -> np.ndarray:
    return (np.roll(f, -1, axis=axis) - f) / delta


def _roll_backward(f: np.ndarray, axis: int, delta: float) -> np.ndarray:
    return (f - np.roll(f, 1, axis=axis)) / delta


def _open_forward(f: np.ndarray, axis: int, delta: float) -> np.ndarray:
    """Forward difference with nothing beyond the last cell."""
    shifted = np.roll(f, -1, axis=axis)
    index = [slice(None)] * f.ndim
    index[axis] = -1
    shifted[tuple(index)] = 0.0
    return (shifted - f) / delta


def _open_backward(f: np.ndarray, axis: int, delta: float) -> np.ndarray:
    """Backward difference with nothing before the first cell."""
    shifted = np.roll(f, 1, axis=axis)
    index = [slice(None)] * f.ndim
    index[axis] = 0
    shifted[tuple(index)] = 0.0
    return (f - shifted) / delta


class Periodic:
    """Wrap every axis. The domain is a torus."""

    def attach(self, solver) -> None:
        self.solver = solver

    def forward(self, f, axis, delta, key):
        return _roll_forward(f, axis, delta)

    def backward(self, f, axis, delta, key):
        return _roll_backward(f, axis, delta)

    def after_magnetic(self, B):
        return B

    def after_electric(self, D):
        return D


class Conducting:
    """A perfect electric conductor on every face.

    Tangential ``E`` vanishes on the wall, which is the whole of the
    condition; the normal ``B`` vanishing follows from it rather than being
    imposed. On the Yee lattice the tangential components are exactly those
    sitting at integer index along the axis in question, so the condition is
    local and needs no interpolation.

    A cavity bounded this way supports standing waves at the discrete
    frequencies of the box, which is the cheapest check that the condition is
    the one intended and not a slightly lossy approximation to it.
    """

    def attach(self, solver) -> None:
        self.solver = solver
        self.ndim = solver.grid.ndim

    def forward(self, f, axis, delta, key):
        return _open_forward(f, axis, delta)

    def backward(self, f, axis, delta, key):
        return _open_backward(f, axis, delta)

    def after_magnetic(self, B):
        return B

    def after_electric(self, D):
        out = list(D)
        # Along axis d, the components at integer index are the two
        # tangential ones. Zero them on both walls.
        for axis in range(self.ndim):
            for comp in range(3):
                if comp == axis:
                    continue  # normal component sits at a half index
                arr = np.array(out[comp], copy=True)
                lo = [slice(None)] * arr.ndim
                lo[axis] = 0
                arr[tuple(lo)] = 0.0
                hi = [slice(None)] * arr.ndim
                hi[axis] = -1
                arr[tuple(hi)] = 0.0
                out[comp] = arr
        return tuple(out)


class PerfectlyMatchedLayer:
    """Convolutional PML, unsplit, with ``kappa = 1`` and ``alpha = 0``.

    The coordinate is stretched into the complex plane inside the layer, so a
    wave entering it decays without seeing an impedance step at the
    interface. The unsplit form keeps that from touching the field variables:
    the whole of the absorption lives in the difference operator, as one
    auxiliary array per derivative term obeying

        psi <- b psi + (b - 1) d(f),      b = exp(-sigma dt)

    with the operator returning ``d(f) + psi``. Berenger's original
    formulation splits each field into two, which works but makes the state
    depend on the boundary condition and does not survive contact with a
    nonlinear medium. Nothing here does either.

    The conductivity is graded as ``sigma_max (depth / thickness)^order``
    rather than stepped, because a jump in ``sigma`` is itself an impedance
    step and reflects. ``sigma_max`` follows from the target reflection:

        sigma_max = -(order + 1) ln(R0) / (2 * thickness * dx)

    A layer is backed by a conductor, so whatever survives the round trip
    comes back; ``thickness`` of eight to sixteen cells is the usual range.
    """

    def __init__(
        self,
        thickness: int = 12,
        order: float = 3.0,
        reflection: float = 1e-6,
        axes: tuple[int, ...] | None = None,
    ):
        if thickness < 1:
            raise ValueError("thickness must be at least one cell")
        if not 0.0 < reflection < 1.0:
            raise ValueError("reflection must lie strictly between zero and one")
        self.thickness = int(thickness)
        self.order = float(order)
        self.reflection = float(reflection)
        self.axes = axes

    def attach(self, solver) -> None:
        self.solver = solver
        grid = solver.grid
        self.dt = solver.dt
        self.shape = grid.shape
        active = tuple(range(grid.ndim)) if self.axes is None else self.axes
        for axis in active:
            if grid.shape[axis] < 2 * self.thickness + 2:
                raise ValueError(
                    f"axis {axis} has {grid.shape[axis]} cells, too few for two "
                    f"layers of {self.thickness} plus an interior. Either thin the "
                    "layer or lengthen the axis"
                )
        self.active = active
        self._b: dict[tuple[int, bool], np.ndarray] = {}
        for axis in active:
            dx = grid.spacing[axis]
            for at_half in (False, True):
                sigma = self._profile(grid.shape[axis], dx, at_half)
                b = np.exp(-sigma * self.dt)
                self._b[(axis, at_half)] = self._broadcast(b, axis, grid.ndim)
        self._psi: dict[str, np.ndarray] = {}

    def _profile(self, n: int, dx: float, at_half: bool) -> np.ndarray:
        """``sigma`` along one axis, at integer or half-integer positions."""
        length = self.thickness * dx
        sigma_max = -(self.order + 1.0) * np.log(self.reflection) / (2.0 * length)
        position = (np.arange(n) + (0.5 if at_half else 0.0)) * dx
        upper = n * dx
        depth = np.maximum(length - position, position - (upper - length))
        depth = np.clip(depth, 0.0, length)
        return sigma_max * (depth / length) ** self.order

    @staticmethod
    def _broadcast(profile: np.ndarray, axis: int, ndim: int) -> np.ndarray:
        shape = [1] * ndim
        shape[axis] = profile.size
        return profile.reshape(shape)

    def _stretch(self, derivative: np.ndarray, axis: int, at_half: bool, key: str):
        if axis not in self.active:
            return derivative
        b = self._b[(axis, at_half)]
        psi = self._psi.get(key)
        if psi is None:
            psi = np.zeros_like(derivative)
        psi = b * psi + (b - 1.0) * derivative
        self._psi[key] = psi
        return derivative + psi

    def forward(self, f, axis, delta, key):
        # A forward difference lands on a half-integer position along `axis`.
        return self._stretch(_open_forward(f, axis, delta), axis, True, "f" + key)

    def backward(self, f, axis, delta, key):
        return self._stretch(_open_backward(f, axis, delta), axis, False, "b" + key)

    def after_magnetic(self, B):
        return B

    def after_electric(self, D):
        return D

    def reset(self) -> None:
        """Forget the convolution history, for reusing a layer across runs."""
        self._psi.clear()
