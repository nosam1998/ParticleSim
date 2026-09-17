"""Charge-conserving current deposition (design doc Section 5.4, Milestone 2).

The problem this solves: a particle-in-cell code evolves ``E`` from
``dE/dt = curl B - J`` and never enforces ``div E = rho``. If the deposited
current does not satisfy the discrete continuity equation exactly, the
mismatch accumulates as a spurious space charge that grows without bound,
and the usual remedy is a Poisson solve every few steps -- expensive, and it
treats the symptom.

Esirkepov 2001 (Comput. Phys. Comm. 135, 144) removes the cause. Rather than
depositing ``q w v S``, it deposits the current implied by the change in the
charge distribution, so that

    (rho^{n+1} - rho^n) / dt + div J = 0

holds identically on the grid, for any shape order and any trajectory inside
one cell. Since the Yee solver's ``div curl`` vanishes identically too, the
combination conserves ``div D - rho`` to round-off rather than approximately,
and no correction step is needed at all.

In one dimension the construction is a running sum:

    J_x(i+1/2) = J_x(i-1/2) - (q w / dt) [S^1(i) - S^0(i)]

which telescopes to zero outside the particle's window because both shapes
sum to one. In two dimensions the change in the product shape is split as

    S^1_x S^1_y - S^0_x S^0_y = W_x + W_y
    W_x(i,j) = dS_x(i) [S^0_y(j) + dS_y(j)/2]
    W_y(i,j) = dS_y(j) [S^0_x(i) + dS_x(i)/2]

which is an identity, not an approximation, and each half drives its own
component's running sum.

Components with no gradient in the simulated plane -- ``J_z`` always, and
``J_y`` in one dimension -- are unconstrained by continuity and are
deposited directly from the velocity, averaged over the two shapes.
"""

from __future__ import annotations

import numpy as np

from particlesim.solvers.pic.particles import Species
from particlesim.solvers.pic.shapes import common_window, window


def _scatter_1d(indices: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    flat = np.mod(indices, n).ravel()
    return np.bincount(flat, weights=values.ravel().astype(np.float64), minlength=n)


def _scatter_2d(ix: np.ndarray, iy: np.ndarray, values: np.ndarray, shape) -> np.ndarray:
    nx, ny = shape
    flat = (np.mod(ix, nx)[:, :, None] * ny + np.mod(iy, ny)[:, None, :]).ravel()
    out = np.bincount(flat, weights=values.ravel().astype(np.float64), minlength=nx * ny)
    return out.reshape(nx, ny)


def deposit_charge(grid, species: Species, order: int = 1) -> np.ndarray:
    """``rho`` at cell corners, the positions ``div D`` is evaluated at.

    Accumulation is in double precision whatever the particles are stored
    in, because the conservation identity is a statement about exact sums.
    """
    spacing = grid.spacing
    cell = float(np.prod(spacing))
    charge = (species.charge * species.weight).astype(np.float64)

    ix, sx = window(species.position[:, 0] / spacing[0], order)
    if grid.ndim == 1:
        return _scatter_1d(ix, charge[:, None] * sx, grid.shape[0]) / cell
    iy, sy = window(species.position[:, 1] / spacing[1], order)
    values = charge[:, None, None] * sx[:, :, None] * sy[:, None, :]
    return _scatter_2d(ix, iy, values, grid.shape) / cell


def esirkepov_current(
    grid,
    species: Species,
    previous_position: np.ndarray,
    dt: float,
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Current on the Yee lattice from a step that moved the particles.

    ``previous_position`` is where they were before the step and
    ``species.position`` is where they are now; ``species.velocity`` supplies
    the components continuity does not fix. The result is laid out exactly as
    the solver's ``D``, so ``J_x`` sits at ``(i+1/2, j)`` and ``J_y`` at
    ``(i, j+1/2)``.

    Positions must be the unwrapped ones. A particle that crossed a periodic
    boundary between the two arrays looks, to this function, like a particle
    that crossed the whole box in one step, and the guard in
    :func:`~particlesim.solvers.pic.shapes.common_window` will say so.
    """
    spacing = grid.spacing
    cell = float(np.prod(spacing))
    charge = (species.charge * species.weight).astype(np.float64)
    velocity = species.velocity

    ix, s0x, s1x = common_window(
        previous_position[:, 0] / spacing[0], species.position[:, 0] / spacing[0], order
    )
    dsx = s1x - s0x

    if grid.ndim == 1:
        nx = grid.shape[0]
        # J_x(i+1/2) = J_x(i-1/2) - (q w / dt) dS(i); the running sum's last
        # entry is zero because both shapes sum to one, so it is dropped.
        running = np.cumsum(dsx, axis=1)[:, :-1]
        Jx = _scatter_1d(ix[:, :-1], -(charge / dt)[:, None] * running, nx)
        mean_shape = 0.5 * (s0x + s1x)
        Jy = _scatter_1d(ix, (charge * velocity[:, 1])[:, None] * mean_shape, nx) / cell
        Jz = _scatter_1d(ix, (charge * velocity[:, 2])[:, None] * mean_shape, nx) / cell
        return Jx, Jy, Jz

    iy, s0y, s1y = common_window(
        previous_position[:, 1] / spacing[1], species.position[:, 1] / spacing[1], order
    )
    dsy = s1y - s0y

    Wx = dsx[:, :, None] * (s0y[:, None, :] + 0.5 * dsy[:, None, :])
    Wy = dsy[:, None, :] * (s0x[:, :, None] + 0.5 * dsx[:, :, None])

    scale_x = -(charge / (spacing[1] * dt))[:, None, None]
    scale_y = -(charge / (spacing[0] * dt))[:, None, None]
    Jx = _scatter_2d(ix[:, :-1], iy, (scale_x * np.cumsum(Wx, axis=1))[:, :-1, :], grid.shape)
    Jy = _scatter_2d(ix, iy[:, :-1], (scale_y * np.cumsum(Wy, axis=2))[:, :, :-1], grid.shape)

    mean = 0.5 * (s0x[:, :, None] * s0y[:, None, :] + s1x[:, :, None] * s1y[:, None, :])
    Jz = _scatter_2d(ix, iy, (charge * velocity[:, 2])[:, None, None] * mean, grid.shape) / cell
    return Jx, Jy, Jz
