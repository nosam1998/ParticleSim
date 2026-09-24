"""A single puncture on two levels, with a radiative outer boundary.

Issue #48 asks for a Schwarzschild puncture stable to ``t = 1000 M`` with two
levels. The refinement (:mod:`particlesim.solvers.nr.refined`) and the outer
boundary (:mod:`particlesim.solvers.nr.boundary`) exist separately; this puts
them together around Brill-Lindquist data and the moving-puncture gauge.

**The puncture sits a quarter of a coarse cell off the grid.**
:func:`~particlesim.solvers.nr.bssn.brill_lindquist` staggers by half a cell,
which is right for one grid and wrong for two: half a coarse cell is a fine
grid point, and ``psi`` is infinite there. A quarter of a coarse cell is half
a fine one, so the nearest sample on either level is ``sqrt(3)/4`` of a coarse
cell away.

**Both levels get exact data.** Prolonging the coarse level into the box
interpolates across ``1/r``, which is exactly what fourth-order interpolation
cannot do, so the fine level is set up from the closed form on its own grid.

**Upwinded advection, and why.** With centred advection the run fails before
``t = 10 M`` at a fine spacing of ``M/4``. At the grid point nearest the
puncture the lapse collapses, ``K`` climbs and ``Abar_xx`` goes from 1.4 to
3.4 in a quarter of ``M``, then ``NaN``. The same run at ``M/2`` survives to
``t = 20 M``, which is the signature of a grid-scale instability the finer grid
resolves. Lopsided advection stencils (``Evolution(upwind=True)``) are the
standard cure: the same ``M/4`` run then passes ``t = 20 M``. Five times the
dissipation also gets past ``t = 10 M``.

**The gauge is not advected, and that is the difference between 90 M and
the long run.** With the lapse and shift equations carrying their
``beta^k d_k`` terms and the driver ``B^i`` not, the coordinates drift: the
conformal metric along each axis reaches 5.8, 7.6, 11 at ``t`` = 60, 70, 80 M
near ``r = 2.4 M`` while the ``Gammabar`` constraint stays small, and the run
fails at 90 M. That mixes two published forms of the Gamma-driver.
The original one advects neither -- ``d_t alpha = -2 alpha K``, ``d_t beta =
3B/4``, ``d_t B = d_t Gammabar - eta B`` -- which a puncture that does not
move loses nothing by, and in it the same run reads 1.38 for the conformal
metric at 60 M, with ``B`` near ``2e-3`` instead of 0.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from particlesim.solvers.nr import bssn, mesh
from particlesim.solvers.nr.boundary import GAUGE_SPEEDS, Bounded, Radiative, interior
from particlesim.solvers.nr.bssn import DIMENSION, INDICES, _module
from particlesim.solvers.nr.refined import Hierarchy


def puncture_state(coords, position, mass: float = 1.0, backend: str = "jax") -> dict[str, Any]:
    """Brill-Lindquist data for one puncture, on any grid.

    ``psi = 1 + m / 2r``, ``phi = ln psi``, the lapse pre-collapsed to
    ``psi^-2``, everything else flat or zero. A sample exactly on the puncture
    is refused rather than returned as an infinity.
    """
    module = _module(backend)
    radius = np.sqrt(sum((np.asarray(c) - p) ** 2 for c, p in zip(coords, position, strict=True)))
    if float(np.min(radius)) <= 0.0:
        raise ValueError("a grid point lands on the puncture: psi is infinite there")
    conformal = 1.0 + mass / (2 * radius)
    zero = np.zeros_like(conformal)
    state: dict[str, Any] = {"phi": np.log(conformal), "trK": zero, "alpha": conformal**-2}
    for i in INDICES:
        state[f"beta{i}"] = zero
        state[f"B{i}"] = zero
        state[f"Gt{i}"] = zero
        for j in range(i, DIMENSION):
            state[f"gt{i}{j}"] = np.full_like(conformal, 1.0 if i == j else 0.0)
            state[f"At{i}{j}"] = zero
    return {name: module.asarray(value) for name, value in state.items()}


@dataclass(frozen=True, eq=False)
class TwoLevelPuncture:
    """A puncture, a refinement box around it, and a radiative edge on the coarse level."""

    hierarchy: Hierarchy
    position: tuple[float, ...]
    coarse_axis: np.ndarray
    fine_axis: np.ndarray
    zone: int
    mass: float = 1.0

    @classmethod
    def build(
        cls,
        n: int = 32,
        extent: float = 16.0,
        box: int = 24,
        zone: float = 2.0,
        mass: float = 1.0,
        upwind: bool = True,
        dissipation: float = bssn.DISSIPATION,
        buffer: int = 6,
        advect: bool = False,
        backend: str = "jax",
    ) -> tuple[TwoLevelPuncture, dict[str, Any], dict[str, Any]]:
        """The setup and its initial coarse and fine states.

        ``n`` coarse points across a box of side ``extent`` (in units of
        ``M``), a fine box ``box`` coarse points wide at the centre, and a
        radiative zone ``zone`` deep at the coarse level's edge.

        ``buffer`` is six fine points rather than the hierarchy's default
        twelve. With twelve, a 48-point fine box keeps only ``+-3 M`` of its
        own, and a run at ``M/4`` grew a constraint violation from
        ``t = 40 M`` at the edge of that region -- the conformal metric
        reaching 5.7 at ``r = 2.4 M``, fed by a coarse level that cannot
        resolve the field there. Six keeps ``+-4.5 M`` at the same cost.

        ``advect`` is off by default: see the module docstring for what the
        advected gauge does over a hundred ``M``.
        """
        spacing = extent / n
        axis = np.arange(n) * spacing
        position = (n // 2 * spacing + spacing / 4,) * DIMENSION
        origin = n // 2 - box // 2
        region = mesh.Box(origin=(origin,) * DIMENSION, shape=(box,) * DIMENSION)
        fine_axis = origin * spacing + np.arange(mesh.RATIO * box) * spacing / mesh.RATIO

        evolution = bssn.Evolution.build(
            (spacing,) * DIMENSION,
            backend=backend,
            dissipation=dissipation,
            upwind=upwind,
            advect=advect,
        )
        width = max(3, int(round(zone / spacing)))
        coarse_mesh = np.meshgrid(axis, axis, axis, indexing="ij")
        relative = tuple(m - p for m, p in zip(coarse_mesh, position, strict=True))
        boundary = Radiative(
            coords=relative,
            spacing=(spacing,) * DIMENSION,
            axes=tuple(INDICES),
            width=width,
            backend=backend,
            speeds=GAUGE_SPEEDS[evolution.slicing],
        )
        plain = Hierarchy.build(evolution, region, (n,) * DIMENSION, buffer=buffer)
        hierarchy = Hierarchy(
            coarse=Bounded(evolution, boundary),
            fine=plain.fine,
            box=region,
            parent_shape=(n,) * DIMENSION,
            interpolation=plain.interpolation,
            buffer=plain.buffer,
        )
        setup = cls(
            hierarchy=hierarchy,
            position=position,
            coarse_axis=axis,
            fine_axis=fine_axis,
            zone=width,
            mass=mass,
        )
        coarse = puncture_state(coarse_mesh, position, mass, backend)
        fine_mesh = np.meshgrid(fine_axis, fine_axis, fine_axis, indexing="ij")
        fine = puncture_state(fine_mesh, position, mass, backend)
        return setup, coarse, fine

    @property
    def time_step(self) -> float:
        return self.hierarchy.coarse.time_step

    def step(self, coarse, fine, time_step: float | None = None):
        return self.hierarchy.step(coarse, fine, time_step)

    def _radius(self, axis) -> np.ndarray:
        grid = np.meshgrid(axis, axis, axis, indexing="ij")
        return np.sqrt(sum((g - p) ** 2 for g, p in zip(grid, self.position, strict=True)))

    def diagnostics(self, coarse, fine, excise: float = 2.0) -> dict[str, float]:
        """What says whether the run is healthy, as plain numbers.

        The Hamiltonian constraint is measured outside ``r = excise``: inside
        the horizon the data is singular and the stencils are differencing
        ``1/r`` at a fraction of a cell, so what they report there is how
        close the nearest sample is, not whether the evolution is right. On
        the fine level it is taken inside the buffer; on the coarse level
        outside the box and inside the radiative zone, so each level is
        measured where it and nothing else is responsible.
        """
        kernel = bssn.constraint_kernel(order=self.hierarchy.fine.order)
        buffer = self.hierarchy.buffer
        inner = (slice(buffer, -buffer),) * DIMENSION

        fine_h = np.asarray(
            kernel(bssn.physical_slice_arrays(fine), self.hierarchy.fine.spacing)["hamiltonian"]
        )[inner]
        fine_r = self._radius(self.fine_axis)[inner]
        outside = fine_r > excise

        coarse_h = np.asarray(
            kernel(bssn.physical_slice_arrays(coarse), self.hierarchy.coarse.spacing)["hamiltonian"]
        )
        covered = np.zeros(coarse_h.shape, dtype=bool)
        covered[self.hierarchy.box.slices()] = True
        axes = tuple(INDICES)
        coarse_h = interior(coarse_h, self.zone, axes)
        covered = interior(covered, self.zone, axes)

        lapse = np.asarray(fine["alpha"])
        finite = all(np.all(np.isfinite(np.asarray(value))) for value in fine.values())
        finite = finite and all(np.all(np.isfinite(np.asarray(v))) for v in coarse.values())
        return {
            "lapse_min": float(lapse.min()),
            # A puncture has phi = ln psi >> 1 beside it. A run whose hole has
            # dissolved stays finite -- this is the number that says so.
            "phi_max": float(np.max(np.asarray(fine["phi"]))),
            "hamiltonian_fine": float(np.sqrt(np.mean(fine_h[outside] ** 2))),
            "hamiltonian_coarse": float(np.sqrt(np.mean(coarse_h[~covered] ** 2))),
            "shift_max": float(max(np.max(np.abs(np.asarray(fine[f"beta{i}"]))) for i in INDICES)),
            "finite": bool(finite),
        }


__all__ = ["TwoLevelPuncture", "puncture_state"]
