"""The conservative update: method of lines over faces the Riemann solver sets.

Issue #57. Cell averages in, flux differences out, third-order strong
stability-preserving Runge-Kutta in between. Everything interesting has
already happened by the time control reaches here -- the reconstruction
chose the face values and the Riemann solver turned them into fluxes -- so
this module's job is to not spoil either, and the two ways it could are both
measured rather than assumed.

**Conservation is exact, and that is a property of the form rather than of
the scheme.** The update subtracts neighbouring fluxes, so on a periodic
grid every interior face is added once and subtracted once and the total
cancels before any physics is consulted. It therefore holds for any
reconstruction, any Riemann solver, any Courant number, and at any stage of
the Runge-Kutta -- and it holds through a shock, where nothing else does.
The total rest mass drifts by ``2e-14`` over a shock tube, on every
reconstruction and every Riemann solver alike, which is the summation's
round-off and not the scheme's error.

**A measured order can belong to the time integrator instead.** Run at a
fixed Courant number, the step shrinks in proportion to the cell, so the
third-order time error falls like ``dx^3`` and eventually caps everything
above it. WENO5 then measures 4.73, 4.25, 3.61 over successive doublings --
falling towards three, and at no point obviously wrong. Refining the step
as ``dx^(5/3)`` instead gives 5.00, 5.00, 5.00. The suite does both, and
keeps the capped number, because a convergence test that quietly reports
the smaller of two orders is the same failure as an autocorrelation time
truncated by its window: confident, stable, and about the wrong thing.

**The primitives are recovered once per stage, and the floors are relative.**
An absolute atmosphere would be a different physical problem at a different
overall density scale; :data:`ATMOSPHERE` multiplies the largest value on
the grid instead, so a problem and the same problem rescaled behave
identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from particlesim.core.grid import UniformGrid
from particlesim.solvers.hydro.reconstruct import GHOSTS, SCHEMES, reconstruct
from particlesim.solvers.hydro.riemann import SOLVERS, riemann_flux
from particlesim.solvers.hydro.srhd import (
    ATMOSPHERE,
    GammaLaw,
    characteristic_speeds,
    conserved_to_primitive,
    primitive_to_conserved,
)

#: Largest velocity a reconstruction may produce. Past it the Lorentz factor
#: is not representable and the recovery below has nothing left to recover.
VELOCITY_CEILING = 1.0 - 1e-12

#: The Runge-Kutta this module steps with, as ``(old weight, new weight)``
#: per stage. Shu-Osher's third-order strong stability-preserving form: each
#: stage is a convex combination of forward Euler steps, so whatever bound a
#: limiter enforces on one step survives all three.
SSP_RK3 = ((0.0, 1.0), (0.75, 0.25), (1.0 / 3.0, 2.0 / 3.0))


def grid_for(size: float = 1.0, points: int = 128, origin: float = 0.0) -> UniformGrid:
    """A one-dimensional cell-centred grid, as the rest of the package builds them."""
    return UniformGrid(extent=[(origin, origin + size)], shape=(points,), ghost=0)


@dataclass
class RelativisticHydro:
    """A one-dimensional Valencia solver: reconstruct, solve, difference, step."""

    grid: UniformGrid
    eos: GammaLaw = field(default_factory=GammaLaw)
    reconstruction: str = "minmod"
    solver: str = "hllc"
    boundary: str = "periodic"
    courant: float = 0.4
    recovery_tolerance: float = 1e-13

    def __post_init__(self) -> None:
        if self.grid.ndim != 1:
            raise ValueError(f"this solver is one-dimensional, got {self.grid.ndim} axes")
        if self.reconstruction not in SCHEMES:
            raise ValueError(
                f"unknown reconstruction {self.reconstruction!r}; expected one of {SCHEMES}"
            )
        if self.solver not in SOLVERS:
            raise ValueError(f"unknown Riemann solver {self.solver!r}; expected one of {SOLVERS}")
        if self.boundary not in ("periodic", "outflow"):
            raise ValueError(
                f"unknown boundary {self.boundary!r}; expected 'periodic' or 'outflow'"
            )
        if not 0.0 < self.courant <= 1.0:
            raise ValueError(f"the Courant number must lie in (0, 1], got {self.courant}")

    @property
    def spacing(self) -> float:
        return float(self.grid.spacing[0])

    @property
    def centres(self) -> np.ndarray:
        return self.grid.axis(0)

    def conserved(self, density, velocity, pressure) -> np.ndarray:
        """Pack ``(rho, v, p)`` into the evolved ``(D, S, tau)``."""
        return np.stack(primitive_to_conserved(density, velocity, pressure, self.eos))

    def primitives(self, state) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return conserved_to_primitive(
            state[0], state[1], state[2], self.eos, self.recovery_tolerance
        )

    def _pad(self, values: np.ndarray) -> np.ndarray:
        if self.boundary == "periodic":
            return np.concatenate([values[-GHOSTS:], values, values[:GHOSTS]])
        return np.concatenate([np.full(GHOSTS, values[0]), values, np.full(GHOSTS, values[-1])])

    def _faces(self, density, velocity, pressure):
        """Reconstructed states on both sides of every face bounding an interior cell."""
        count = self.grid.shape[0]
        window = slice(GHOSTS - 1, count + GHOSTS)
        sides = ([], [])
        for values in (density, velocity, pressure):
            low, high = reconstruct(self._pad(values), self.reconstruction)
            sides[0].append(low[window])
            sides[1].append(high[window])
        floors = (
            ATMOSPHERE * float(np.max(density)),
            None,
            ATMOSPHERE * float(np.max(pressure)),
        )
        cleaned = []
        for side in sides:
            cleaned.append(
                (
                    np.maximum(side[0], floors[0]),
                    np.clip(side[1], -VELOCITY_CEILING, VELOCITY_CEILING),
                    np.maximum(side[2], floors[2]),
                )
            )
        return cleaned[0], cleaned[1]

    def rhs(self, state: np.ndarray) -> np.ndarray:
        """``-dF/dx``, as a difference of the fluxes at the two bounding faces."""
        density, velocity, pressure = self.primitives(state)
        left, right = self._faces(density, velocity, pressure)
        fluxes = riemann_flux(left, right, self.eos, self.solver)
        return -(fluxes[:, 1:] - fluxes[:, :-1]) / self.spacing

    def time_step(self, state: np.ndarray) -> float:
        """Courant number over the fastest characteristic anywhere on the grid."""
        density, velocity, pressure = self.primitives(state)
        speeds = characteristic_speeds(density, velocity, pressure, self.eos)
        fastest = float(np.max([np.max(np.abs(s)) for s in speeds]))
        return self.courant * self.spacing / fastest

    def step(self, state: np.ndarray, step_size: float) -> np.ndarray:
        """One SSP-RK3 step. Three convex combinations of forward Euler."""
        updated = state
        for old, new in SSP_RK3:
            updated = old * state + new * (updated + step_size * self.rhs(updated))
        return updated

    def run(self, state: np.ndarray, duration: float, steps: int | None = None):
        """Advance to ``duration``; fixed steps if given, Courant-limited if not.

        A fixed count is what a convergence study needs -- the time error has
        to be refinable independently of the cell, or the measured order is
        the smaller of the two and says nothing about the reconstruction.
        """
        if duration < 0.0:
            raise ValueError(f"the duration must not be negative, got {duration}")
        if steps is not None:
            if steps < 1:
                raise ValueError(f"a fixed-step run needs at least one step, got {steps}")
            size = duration / steps
            for _ in range(steps):
                state = self.step(state, size)
            return state
        elapsed = 0.0
        while elapsed < duration:
            size = min(self.time_step(state), duration - elapsed)
            state = self.step(state, size)
            elapsed += size
        return state

    def totals(self, state: np.ndarray) -> np.ndarray:
        """``(D, S, tau)`` integrated over the grid: what a periodic run must keep."""
        return np.sum(np.asarray(state), axis=1) * self.spacing


def riemann_initial_data(solver: RelativisticHydro, left, right, interface: float = 0.5):
    """A discontinuity at ``interface``, as the evolved variables."""
    centres = solver.centres
    picked = [np.where(centres < interface, lo, hi) for lo, hi in zip(left, right, strict=True)]
    return solver.conserved(*picked)


def advected_pulse(solver: RelativisticHydro, time=0.0, velocity=0.5, pressure=1.0, amplitude=0.2):
    """A density pulse at uniform velocity and pressure, as exact cell averages.

    Uniform ``v`` and ``p`` make the momentum and energy equations advect
    the density and nothing else, so the exact solution at any time is the
    initial profile translated by ``v t``. That is what lets a convergence
    study report an order rather than a difference between two resolutions.

    The averages are exact rather than sampled, and they have to be. A
    finite-volume scheme evolves cell averages, so seeding it with values at
    the cell centres is an ``O(dx^2)`` error in the initial data and caps
    every measured order at two -- WENO5 and minmod would then agree, and
    the conclusion drawn would be that the reconstruction does not matter.
    Here the closed form exists because all three conserved variables are
    *affine* in the density at fixed ``v`` and ``p``, so averaging the
    density averages them: the enthalpy contributes ``rho h = rho + Gamma p
    / (Gamma - 1)``, whose second term is a constant.
    """
    origin, extent = solver.grid.extent[0]
    span = extent - origin
    wavenumber = 2.0 * np.pi / span
    phase = wavenumber * ((solver.centres - origin - velocity * time) % span)
    half = 0.5 * wavenumber * solver.spacing
    averaged = 1.0 + amplitude * np.sin(phase) * np.sin(half) / half
    return solver.conserved(averaged, velocity, pressure)


def smooth_pulse(solver: RelativisticHydro, velocity=0.5, pressure=1.0, amplitude=0.2):
    """:func:`advected_pulse` at ``t = 0``: the initial data of the smooth test."""
    return advected_pulse(solver, 0.0, velocity, pressure, amplitude)


__all__ = [
    "SSP_RK3",
    "VELOCITY_CEILING",
    "RelativisticHydro",
    "advected_pulse",
    "grid_for",
    "riemann_initial_data",
    "smooth_pulse",
]
