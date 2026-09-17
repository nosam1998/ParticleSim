"""One particle-in-cell cycle (design doc Section 5.4, Milestone 2).

Ties the Yee solver, the pushers and the charge-conserving deposition into
the loop they are meant to run in, and gets two orderings right that are
easy to get wrong.

**The magnetic field is gathered at the integer step, not the half step.**
``B`` lives at half-integer times, so gathering it directly would give the
pusher a field half a step stale and make the scheme first order in time
however good the pusher is. The cycle evaluates ``curl E`` once and uses it
twice: to carry ``B`` half a step forward for the gather, and to carry it
the full step for the field update. Once, because a perfectly matched layer
keeps convolution history per call, so computing the curl twice would apply
the layer twice in one step.

**The current is deposited from unwrapped positions.** A particle that
crosses a periodic boundary between the two arrays looks like one that
crossed the whole box, which would break the deposition's continuity
identity. Positions are therefore wrapped after the current is built, never
before.

What the combination buys is that ``div D - rho`` is conserved to round-off
and not merely to the order of the scheme. Gauss's law is never solved for
and never corrected; it cannot drift, because the discrete ``div curl`` is
identically zero and the deposited current satisfies discrete continuity
identically.
"""

from __future__ import annotations

import numpy as np

from particlesim.solvers.pic.deposition import deposit_charge, esirkepov_current
from particlesim.solvers.pic.particles import Species, push_momentum
from particlesim.solvers.pic.shapes import gather
from particlesim.solvers.pic.yee import B_OFFSETS, E_OFFSETS, Fields, YeeSolver


def gather_fields(solver: YeeSolver, fields: Fields, species: Species, order: int):
    """``E`` and ``B`` at the particle positions, each on its own sublattice."""
    spacing = solver.grid.spacing
    E = np.stack(
        [
            gather(f, species.position, spacing, E_OFFSETS[c], order)
            for f, c in zip(solver.medium.electric(fields.D, fields.B), "xyz", strict=True)
        ],
        axis=-1,
    )
    B = np.stack(
        [
            gather(f, species.position, spacing, B_OFFSETS[c], order)
            for f, c in zip(solver.medium.magnetic(fields.D, fields.B), "xyz", strict=True)
        ],
        axis=-1,
    )
    return E, B


def advance(
    solver: YeeSolver,
    fields: Fields,
    species: Species,
    order: int = 1,
    scheme: str = "boris",
    wrap: bool = True,
) -> tuple[Fields, Species]:
    """One full cycle: gather, push, deposit, update the fields."""
    dt = solver.dt
    grid = solver.grid

    # curl E once, used for the half step that centres the gather and for
    # the full step that carries B to n + 1/2.
    curl = solver.curl_E(solver.medium.electric(fields.D, fields.B))
    half = fields.with_B(
        solver.boundary.after_magnetic(
            tuple(b - 0.5 * dt * c for b, c in zip(fields.B, curl, strict=True))
        )
    )
    E_at, B_at = gather_fields(solver, half, species, order)

    species = push_momentum(species, E_at, B_at, dt, scheme=scheme)
    previous = np.array(species.position, dtype=float, copy=True)
    moved = species.with_position(species.position + dt * species.velocity[:, : species.ndim])

    current = esirkepov_current(grid, moved, previous, dt, order)

    fields = fields.with_B(
        solver.boundary.after_magnetic(
            tuple(b - dt * c for b, c in zip(fields.B, curl, strict=True))
        )
    )
    fields = solver.advance_electric(fields, current)

    if wrap:
        moved = moved.with_position(np.mod(moved.position, np.asarray(grid.extent)))
    return fields, moved


def gauss_residual(
    solver: YeeSolver, fields: Fields, species: Species, order: int = 1
) -> np.ndarray:
    """``div D - rho``, the quantity a charge-conserving cycle holds fixed.

    Not ``div D - rho`` being zero: that depends on the initial data, and
    starting from zero fields with charge present makes it non-zero from the
    first instant. What the cycle guarantees is that it does not *change*.
    """
    return solver.divergence_D(fields) - deposit_charge(solver.grid, species, order)
