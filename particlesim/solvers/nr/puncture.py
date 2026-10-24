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

``advect=True`` is now the other published form, ``d_0`` throughout, with
``B^i`` advected too (:meth:`~particlesim.solvers.nr.bssn.Evolution.right_hand_side`).
On current code the mix was NaN by 50 M. The ``d_0`` form reads 2.03 at
60 M and 3.18 at 80 M for the conformal metric, keeps ``B`` under 0.08 to
85 M, and is NaN by 100 M. That is twice as long, and still not a stable
advected puncture, so the unadvected gauge stays the default here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from particlesim.solvers.nr import bssn, mesh
from particlesim.solvers.nr.boundary import (
    AUXILIARY,
    GAUGE_SPEEDS,
    Bounded,
    Radiative,
    SecondOrder,
    interior,
)
from particlesim.solvers.nr.bssn import DIMENSION, INDICES, _module
from particlesim.solvers.nr.refined import Hierarchy, Nested


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


def brill_lindquist_state(coords, positions, masses, backend: str = "jax") -> dict[str, Any]:
    """Brill-Lindquist data for several punctures at rest, on any grid.

    ``psi = 1 + sum_a m_a / (2 |x - x_a|)`` with the conformal metric flat and
    the extrinsic curvature zero, which solves both constraints exactly
    (Brill and Lindquist 1963). Its ADM mass is ``sum_a m_a``
    (:func:`brill_lindquist_adm_mass`). The lapse starts at ``psi^-2``, the
    shift and ``B^i`` at zero, as for :func:`puncture_state`. A sample on
    a puncture is refused.
    """
    if len(positions) != len(masses) or not masses:
        raise ValueError("give one mass per puncture")
    module = _module(backend)
    conformal = 1.0
    for position, mass in zip(positions, masses, strict=True):
        radius = np.sqrt(
            sum((np.asarray(c) - p) ** 2 for c, p in zip(coords, position, strict=True))
        )
        if float(np.min(radius)) <= 0.0:
            raise ValueError("a grid point lands on a puncture: psi is infinite there")
        conformal = conformal + mass / (2 * radius)
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


def brill_lindquist_adm_mass(masses) -> float:
    """The ADM mass of Brill-Lindquist data: the bare masses add, ``M = sum_a m_a``.

    At large ``r``, ``psi -> 1 + (sum_a m_a)/(2r)``, and ``psi = 1 + M/2r`` is
    Schwarzschild's isotropic form for mass ``M``. Each puncture's own
    asymptotic end has mass ``m_a (1 + sum_{b != a} m_b / (2 r_ab))``, which is
    not this.
    """
    return float(sum(masses))


def trumpet_isotropic_radius(areal, mass: float = 1.0):
    """The isotropic radius of the maximal trumpet, in closed form.

    Baumgarte and Naculich (2007): the stationary maximal slice of
    Schwarzschild, ``K = 0``, reaches down to the areal radius ``R = 3M/2`` and
    no further. In isotropic coordinates that throat sits at ``r = 0``.
    """
    R, m = np.asarray(areal, dtype=float), mass
    a = (2 * R + m + np.sqrt(4 * R**2 + 4 * m * R + 3 * m**2)) / 4
    b = ((4 + 3 * np.sqrt(2)) * (2 * R - 3 * m)) / (
        8 * R + 6 * m + 3 * np.sqrt(8 * R**2 + 8 * m * R + 6 * m**2)
    )
    return a * b ** (1 / np.sqrt(2))


def trumpet_areal_radius(isotropic, mass: float = 1.0):
    """``R(r)``, inverting :func:`trumpet_isotropic_radius` by bisection in ``ln(R - 3M/2)``."""
    r = np.asarray(isotropic, dtype=float)
    throat = 1.5 * mass
    low = np.full(r.shape, np.log(1e-300))
    high = np.full(r.shape, np.log(1e6 * mass + float(np.max(r))))
    for _ in range(200):
        middle = 0.5 * (low + high)
        above = trumpet_isotropic_radius(throat + np.exp(middle), mass) > r
        high = np.where(above, middle, high)
        low = np.where(above, low, middle)
    return throat + np.exp(0.5 * (low + high))


def trumpet_state(coords, position, mass: float = 1.0, backend: str = "jax") -> dict[str, Any]:
    """Schwarzschild's maximal trumpet with its stationary lapse and shift.

    With ``C = 3 sqrt(3) M^2 / 4`` and ``f = (1 - 2M/R + C^2/R^4)^(1/2)``:
    - ``psi^4 = (R/r)^2`` and the conformal metric flat;
    - ``K = 0`` and ``Abar_ij = (C/R^3)(delta_ij - 3 n_i n_j)``;
    - the lapse ``f`` and the shift ``beta^i = C x^i / R^3``;
    - ``Gammabar^i = 0`` and ``B^i = 0``.

    This is what Brill-Lindquist data with the moving-puncture gauge
    settles to, but here it is present from the start. It is an exact
    stationary solution of the unadvected gauge, ``d_t alpha = -2 alpha K``
    and ``d_t beta = 3B/4``: ``K`` stays zero, ``Gammabar^i`` stays constant,
    and ``B`` stays zero. The kernel's right-hand side on it is truncation
    error, which falls by 13 to 16 per halving of the spacing at
    ``2 <= r <= 6 M``. There is nothing left for the gauge to do, and so no
    pulse for the outer boundary to mishandle.

    That removes the start-up pulse but not the failure. On two levels at
    ``M/4`` the zone's lapse moved seventy times less by 10 M, but the whole
    slice then drifted from the hole outward. The hole went between 210 and
    220 M, against 185 to 190 M from Brill-Lindquist data. The measurements
    are in ``docs/benchmarks.md``.
    """
    module = _module(backend)
    relative = [np.asarray(c) - p for c, p in zip(coords, position, strict=True)]
    radius = np.sqrt(sum(r**2 for r in relative))
    if float(np.min(radius)) <= 0.0:
        raise ValueError("a grid point lands on the puncture: the throat is there")
    areal = trumpet_areal_radius(radius, mass)
    c = 3 * np.sqrt(3) / 4 * mass**2
    lapse = np.sqrt(np.maximum(1 - 2 * mass / areal + c**2 / areal**4, 0.0))
    zero = np.zeros_like(radius)
    state: dict[str, Any] = {"phi": 0.5 * np.log(areal / radius), "trK": zero, "alpha": lapse}
    for i in INDICES:
        state[f"beta{i}"] = c * relative[i] / areal**3
        state[f"B{i}"] = zero
        state[f"Gt{i}"] = zero
        for j in range(i, DIMENSION):
            delta = 1.0 if i == j else 0.0
            state[f"gt{i}{j}"] = np.full_like(radius, delta)
            state[f"At{i}{j}"] = c / areal**3 * (delta - 3 * relative[i] * relative[j] / radius**2)
    return {name: module.asarray(value) for name, value in state.items()}


def perturbed_puncture_state(
    coords,
    position,
    amplitude: float,
    width: float = 1.0,
    time: float = -7.0,
    mass: float = 1.0,
    spacing: float | None = None,
    backend: str = "jax",
) -> dict[str, Any]:
    """A puncture with a Teukolsky ``l = 2`` wave around it, for a ringdown (issue #136).

    ``gamma_ij = psi^4 (delta_ij + h_ij)`` and ``K_ij = -psi^4 dh_ij/dt / 2``,
    with ``psi = 1 + m/2r`` and ``h`` the flat-space wave of
    :mod:`~particlesim.solvers.nr.teukolsky` at ``time``. At the default
    ``time = -7`` the wave is an ingoing shell near ``r = 7``, so it barely
    overlaps the hole when it starts. The superposition is not a solution of
    the constraints. What it violates is the cross term between ``h`` and
    the curvature of ``psi``, a few percent of the wave at the shell, which
    is enough to excite the hole and small enough to read a frequency
    through. ``Gammabar^i`` is differenced with ``spacing``.
    """
    from particlesim.core.grid import derivative
    from particlesim.solvers.nr import teukolsky
    from particlesim.symbolic.threeplusone import determinant, inverse_metric

    module = _module(backend)
    relative = [np.asarray(c) - p for c, p in zip(coords, position, strict=True)]
    radius = np.sqrt(sum(r**2 for r in relative))
    if float(np.min(radius)) <= 0.0:
        raise ValueError("a grid point lands on the puncture: psi is infinite there")
    if spacing is None:
        axis = np.asarray(coords[0])[:, 0, 0]
        spacing = float(axis[1] - axis[0])
    psi = 1.0 + mass / (2 * radius)
    h = teukolsky.perturbation(time, relative, amplitude, width)
    rate = teukolsky.perturbation(time, relative, amplitude, width, rate=True)
    metric = [[psi**4 * ((1.0 if i == j else 0.0) + h[i][j]) for j in INDICES] for i in INDICES]
    curvature = [[-0.5 * psi**4 * rate[i][j] for j in INDICES] for i in INDICES]
    phi = np.log(determinant(metric)) / 12.0
    conformal = np.exp(-4.0 * phi)
    inverse = inverse_metric(metric)
    trace = sum(inverse[i][j] * curvature[i][j] for i in INDICES for j in INDICES)
    tilde = [[conformal * metric[i][j] for j in INDICES] for i in INDICES]
    tilde_inverse = inverse_metric(tilde)
    state: dict[str, Any] = {"phi": phi, "trK": trace, "alpha": psi**-2}
    zero = np.zeros_like(psi)
    for i in INDICES:
        state[f"beta{i}"] = zero
        state[f"B{i}"] = zero
        state[f"Gt{i}"] = -sum(derivative(tilde_inverse[i][j], j, spacing) for j in INDICES)
        for j in range(i, DIMENSION):
            state[f"gt{i}{j}"] = tilde[i][j]
            state[f"At{i}{j}"] = conformal * (curvature[i][j] - metric[i][j] * trace / 3)
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
        advect: bool | str = False,
        backend: str = "jax",
        second_order: bool = False,
        data: str = "brill_lindquist",
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
        advected gauge does over a hundred ``M``. Advecting the lapse alone
        (``"lapse"``, Campanelli et al.'s combination) was measured too, and
        lasted less long: the hole dissolved by 125 M against 150 M.

        ``second_order`` puts Bayliss and Turkel's condition on the coarse
        edge in place of Sommerfeld's
        (:class:`~particlesim.solvers.nr.boundary.SecondOrder`), and the
        coarse state then carries its auxiliary fields. With ``n = 36`` over
        18 M it keeps the coarse constraint two to thirty times lower
        and the hole to about 187 M instead of 160 M. The lapse in the zone
        still drifts, so it delays the failure rather than removing it.
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
        edge = (SecondOrder if second_order else Bounded)(evolution, boundary)
        hierarchy = Hierarchy(
            coarse=edge,
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
        initial = {"brill_lindquist": puncture_state, "trumpet": trumpet_state}[data]
        coarse = initial(coarse_mesh, position, mass, backend)
        if second_order:
            coarse = edge.start(coarse)
        fine_mesh = np.meshgrid(fine_axis, fine_axis, fine_axis, indexing="ij")
        fine = initial(fine_mesh, position, mass, backend)
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


def staggered_offset(levels: int) -> int:
    """How far off the coarse grid to put a puncture, in half the finest spacing.

    Every level's points sit on the finest level's lattice, so any whole
    number of finest spacings lands on one of them and ``psi`` is infinite
    there. An odd number of half-spacings avoids them all. Among those, this
    picks the one whose nearest sample is furthest away on the level that
    has it worst, as a fraction of that level's spacing. For two levels that
    is one half-spacing, the quarter coarse cell :class:`TwoLevelPuncture`
    uses. For four it is five, which leaves ``5/16``, ``3/8``, ``1/4`` and
    ``1/2`` of a cell from the coarsest level to the finest.

    **It is the wrong thing to maximise for many levels**, and
    :meth:`NestedPuncture.build` no longer uses it by default. For five
    levels it is eleven: 0.69 M off the boxes' common centre at a finest
    spacing of ``M/8``. In the ringdown run the hole drifted back toward
    that centre, 0.13 M by 90 M and 0.33 M by 120 M. The coarse levels are
    overwritten by restriction wherever the puncture is, so how close it
    comes to their points matters less than keeping it centred. The build
    now puts it one half-spacing off; see ``docs/benchmarks.md``.
    """
    if levels < 1:
        raise ValueError(f"{levels} levels")
    finest = levels - 1

    def worst(multiple: int) -> float:
        distances = []
        for level in range(levels):
            fraction = (multiple * 2**level / 2 ** (finest + 1)) % 1.0
            distances.append(min(fraction, 1.0 - fraction))
        return min(distances)

    return max(range(1, 2 ** (finest + 1), 2), key=lambda m: (worst(m), -m))


@dataclass(frozen=True, eq=False)
class NestedPuncture:
    """A puncture inside any number of nested boxes, with a radiative edge on the coarsest.

    Every level is ``n`` points across when ``box`` is ``n / 2``. Each box is
    centred in its parent and refines it by two, so ``levels`` of them
    resolve the hole at ``extent / n / 2^(levels - 1)``: four levels of 48
    points over 24 M reach ``M/16`` with 442,368 points, where a single
    grid at that spacing would need 56.6 million. Like
    :class:`TwoLevelPuncture`, every level is given the closed-form data on
    its own grid rather than an interpolant of ``1/r``.
    """

    nested: Nested
    position: tuple[float, ...]
    axes: tuple[np.ndarray, ...]
    zone: int
    mass: float = 1.0
    reflection_symmetry: bool = False
    ghosts: int = 0
    full_axes: tuple[np.ndarray, ...] | None = None
    full_boxes: tuple[mesh.Box, ...] | None = None
    punctures: tuple[tuple[float, tuple[float, ...]], ...] | None = None

    @classmethod
    def build(
        cls,
        n: int = 48,
        extent: float = 24.0,
        levels: int = 4,
        box: int | None = None,
        zone: float = 2.0,
        mass: float = 1.0,
        upwind: bool = True,
        dissipation: float = bssn.DISSIPATION,
        buffer: int = 6,
        advect: bool | str = False,
        backend: str = "jax",
        second_order: bool = True,
        data: str = "brill_lindquist",
        wave: dict[str, float] | None = None,
        conformal: str = "phi",
        formulation: str = "bssn",
        constraint_damping: float | None = None,
        offset: int | None = None,
        centering: str = "vertex",
        boundary_background: str = "flat",
        reflection_symmetry: bool = False,
        octant: bool = False,
        punctures: tuple[tuple[float, tuple[float, ...]], ...] | None = None,
        courant: float = bssn.COURANT,
    ) -> tuple[NestedPuncture, list[dict[str, Any]]]:
        """The setup and one initial state per level, coarsest first.

        ``box`` is how many points of its parent each box covers, ``n / 2``
        by default. ``wave`` puts an ingoing Teukolsky shell around the hole
        (:func:`perturbed_puncture_state`, with these keyword arguments)
        instead of ``data``. Everything else is as in
        :meth:`TwoLevelPuncture.build`, except that the second-order boundary
        is on by default.

        ``conformal="W"`` evolves ``W = e^(-2 phi)``, which vanishes at the
        puncture like the distance to it, in place of ``phi``, which diverges
        like its logarithm (:func:`~particlesim.solvers.nr.bssn.with_conformal`).
        On trumpet data at ``M/8`` it halves the horizon's mass error at 45 M.

        ``formulation="ccz4"`` evolves CCZ4 instead, with ``Theta`` starting
        at zero and ``constraint_damping`` as its ``kappa_1`` (the module's
        default if not given). It carries ``phi`` only. Measured on trumpet
        data at ``M/8`` with the unadvected gauge, it is not an improvement:
        the lapse goes negative by 30 M. See ``docs/benchmarks.md``.

        ``offset`` is how far the puncture sits from the boxes' common centre
        along each axis, in half the finest spacing. It must be odd, so that
        no level samples it, and one keeps the hole as close to centred as
        that allows. :func:`staggered_offset` gives the multiple that keeps it
        furthest from every level's points instead, which for five levels is
        eleven, 0.69 M off centre at ``M/8``.

        ``centering="cell"`` uses point samples at cell centres, with an
        even ``n``. Its default ``offset=0`` puts the hole at the reflection
        centre of every level without sampling the singular point. Both
        transfers remain fourth order. Vertex centering keeps its existing
        default of one finest half-spacing; explicit offsets are still
        checked against the actual initial-data grids.

        ``boundary_background="trumpet"`` applies the outer radiation
        condition to deviations from the initial stationary trumpet. This
        is available only for unperturbed trumpet initial data, and changes
        no interior evolution rates. The default retains the asymptotically
        flat reference used by the existing boundary benchmarks.

        ``reflection_symmetry=True`` restricts the experiment to the three
        coordinate reflections, as an octant domain would. Tensor-aware
        group averaging removes roundoff asymmetries after each coarse
        step; it does not hold any field at its initial value. It requires
        a centred cell grid and unperturbed single-hole initial data.

        ``octant=True`` evolves only the octant ``x, y, z > 0`` of each
        level about the common centre, with reflection ghosts below each
        symmetry plane (:mod:`~particlesim.solvers.nr.octant`). It is the
        same restriction as ``reflection_symmetry`` at a quarter to a fifth
        of the points. The states are octant arrays; :meth:`full_states` unfolds
        them, and :meth:`diagnostics` measures the unfolded grid.

        ``punctures`` replaces the single hole with Brill-Lindquist data for
        several, as ``(mass, offset)`` pairs with each offset from the
        boxes' common centre (:func:`brill_lindquist_state`). The lapse
        starts at ``psi^-2`` and the shift at zero, so ``data`` must be
        ``"brill_lindquist"``. On an octant the set must be symmetric under
        the three reflections, e.g. an equal pair on the ``z`` axis.

        ``courant`` is the coarsest level's Courant factor; every finer level
        takes two steps for each of its parent's, so it is the same on all.
        """
        if octant and (
            centering != "cell"
            or (offset or 0) != 0
            or wave is not None
            or reflection_symmetry
            or formulation != "bssn"
        ):
            raise ValueError(
                "an octant requires centred cell grids, BSSN and reflection-symmetric data"
            )
        if punctures is not None:
            if data != "brill_lindquist" or wave is not None or boundary_background != "flat":
                raise ValueError("several punctures take Brill-Lindquist data and a flat edge")
            punctures = tuple((float(m), tuple(float(c) for c in where)) for m, where in punctures)
            if octant:
                mirrors = [(1 - 2 * a, 1 - 2 * b, 1 - 2 * c) for a, b, c in np.ndindex(2, 2, 2)]
                images = {
                    (m, tuple(sign * c for sign, c in zip(signs, where, strict=True)))
                    for m, where in punctures
                    for signs in mirrors
                }
                if images != set(punctures):
                    raise ValueError(
                        "the punctures are not symmetric under the octant's reflections"
                    )
        if formulation not in ("bssn", "ccz4"):
            raise ValueError(f"formulation {formulation!r} is not 'bssn' or 'ccz4'")
        if formulation == "ccz4" and conformal != "phi":
            raise ValueError("the CCZ4 kernel carries phi only")
        if levels < 2:
            raise ValueError("a nested puncture needs at least two levels")
        if boundary_background not in ("flat", "trumpet"):
            raise ValueError("boundary_background must be 'flat' or 'trumpet'")
        if boundary_background == "trumpet" and (data != "trumpet" or wave is not None):
            raise ValueError(
                "a trumpet boundary background requires unperturbed trumpet initial data"
            )
        if boundary_background == "trumpet" and advect is not False:
            raise ValueError(
                "the stationary trumpet boundary background needs the unadvected gauge"
            )
        if centering not in ("vertex", "cell"):
            raise ValueError("centering must be 'vertex' or 'cell'")
        if centering == "cell" and n % 2:
            raise ValueError("a symmetric cell-centred puncture needs an even grid size")
        offset = (0 if centering == "cell" else 1) if offset is None else offset
        if reflection_symmetry and (centering != "cell" or offset != 0 or wave is not None):
            raise ValueError("reflection symmetry requires centred cell grids and single-hole data")
        if centering == "vertex" and offset % 2 == 0:
            raise ValueError(
                f"an offset of {offset} half-spacings puts the puncture on a grid point: "
                "it must be an odd number"
            )
        box = n // 2 if box is None else int(box)
        if box % 2:
            raise ValueError(f"a box of {box} points has no centre point to share with its parent")
        spacing = extent / n
        finest = spacing / mesh.RATIO ** (levels - 1)
        position = ((n // 2) * spacing + offset * finest / 2,) * DIMENSION

        axes = [(np.arange(n) + (0.5 if centering == "cell" else 0)) * spacing]
        sizes = [n]
        regions = []
        for _ in range(1, levels):
            origin = sizes[-1] // 2 - box // 2
            regions.append(
                mesh.Box(
                    origin=(origin,) * DIMENSION,
                    shape=(box,) * DIMENSION,
                    centering=centering,
                )
            )
            step = (axes[-1][1] - axes[-1][0]) / mesh.RATIO
            axes.append(
                axes[-1][origin]
                + (np.arange(mesh.RATIO * box) - (0.5 if centering == "cell" else 0)) * step
            )
            sizes.append(mesh.RATIO * box)

        ghosts = 0
        full_axes = full_boxes = None
        if octant:
            from particlesim.solvers.nr import octant as octants

            ghosts = octants.GHOSTS
            # The box's origin, its extent and the prolongation stencil's reach.
            if ghosts // 2 + (box // 2 + ghosts // 2) + 2 > n // 2 + ghosts:
                raise ValueError(f"a box of {box} points leaves no octant parent around it")
            full_axes, full_boxes = tuple(axes), tuple(regions)
            axes = [axis[axis.size // 2 - ghosts :] for axis in axes]
            regions = [
                mesh.Box(
                    origin=(ghosts // 2,) * DIMENSION,
                    shape=(box // 2 + ghosts // 2,) * DIMENSION,
                    centering="cell",
                )
                for _ in regions
            ]
            sizes = [axes[0].size] + [axis.size for axis in axes[1:]]

        if formulation == "ccz4":
            from particlesim.solvers.nr import ccz4

            extra = {} if constraint_damping is None else {"damping": constraint_damping}
            evolution = ccz4.build(
                (spacing,) * DIMENSION,
                backend=backend,
                dissipation=dissipation,
                upwind=upwind,
                advect=advect,
                **extra,
            )
        else:
            evolution = (octants.OctantEvolution if octant else bssn.Evolution).build(
                (spacing,) * DIMENSION,
                backend=backend,
                courant=courant,
                dissipation=dissipation,
                upwind=upwind,
                advect=advect,
                conformal=conformal,
            )
        width = max(3, int(round(zone / spacing)))
        coarse_mesh = np.meshgrid(axes[0], axes[0], axes[0], indexing="ij")
        relative = tuple(m - p for m, p in zip(coarse_mesh, position, strict=True))
        reference = None
        if boundary_background == "trumpet":
            reference = bssn.with_conformal(
                trumpet_state(coarse_mesh, position, mass, backend), conformal, backend
            )
        if octant:
            classes = (octants.OctantSecondOrder, octants.OctantBounded, octants.OctantRadiative)
        else:
            classes = (SecondOrder, Bounded, Radiative)
        edge = (classes[0] if second_order else classes[1])(
            evolution,
            classes[2](
                coords=relative,
                spacing=(spacing,) * DIMENSION,
                axes=tuple(INDICES),
                width=width,
                backend=backend,
                speeds=GAUGE_SPEEDS[evolution.slicing],
                background=reference,
            ),
        )
        pairs = []
        parent = evolution
        pairing = octants.OctantHierarchy if octant else Hierarchy
        for region, size in zip(regions, sizes, strict=False):
            pair = pairing.build(parent, region, (size,) * DIMENSION, buffer=buffer)
            pairs.append(pair)
            parent = pair.fine
        pairs[0] = replace(pairs[0], coarse=edge)
        setup = cls(
            nested=(octants.OctantNested if octant else Nested)(tuple(pairs)),
            position=position,
            axes=tuple(axes),
            zone=width,
            mass=mass,
            reflection_symmetry=reflection_symmetry,
            ghosts=ghosts,
            full_axes=full_axes,
            full_boxes=full_boxes,
            punctures=punctures,
        )

        states = []
        for axis in axes:
            grid = np.meshgrid(axis, axis, axis, indexing="ij")
            if wave is not None:
                state = perturbed_puncture_state(
                    grid,
                    position,
                    mass=mass,
                    spacing=float(axis[1] - axis[0]),
                    backend=backend,
                    **wave,
                )
            elif punctures is not None:
                state = brill_lindquist_state(
                    grid,
                    [
                        tuple(p + c for p, c in zip(position, where, strict=True))
                        for _, where in punctures
                    ],
                    [m for m, _ in punctures],
                    backend,
                )
            else:
                initial = {"brill_lindquist": puncture_state, "trumpet": trumpet_state}[data]
                state = initial(grid, position, mass, backend)
            state = bssn.with_conformal(state, conformal, backend)
            if formulation == "ccz4":
                from particlesim.solvers.nr import ccz4

                state = ccz4.with_theta(state, backend)
            states.append(state)
        if second_order:
            states[0] = edge.start(states[0])
        if reflection_symmetry:
            from particlesim.solvers.nr.symmetry import project

            states = [project(state, backend=backend) for state in states]
        if octant:
            states = [octants.reflect(state, ghosts, backend) for state in states]
        return setup, states

    @property
    def levels(self) -> int:
        return self.nested.depth

    @property
    def time_step(self) -> float:
        return self.nested.time_step

    def spacing(self, level: int) -> float:
        return float(self.axes[level][1] - self.axes[level][0])

    def full_states(self, states) -> list[dict[str, Any]]:
        """The states on whole grids: unfolded from an octant, otherwise as they are."""
        if not self.ghosts:
            return list(states)
        from particlesim.solvers.nr.octant import unfold

        return [unfold(state, self.ghosts) for state in states]

    def full_axis(self, level: int) -> np.ndarray:
        """The level's whole coordinate axis, the octant's unfolded one included."""
        return self.axes[level] if self.full_axes is None else self.full_axes[level]

    def full_box(self, level: int) -> mesh.Box:
        """Where level ``level + 1`` sits in the whole grid of level ``level``."""
        if self.full_boxes is None:
            return self.nested.pairs[level].box
        return self.full_boxes[level]

    def step(self, states, time_step: float | None = None) -> list:
        advanced = self.nested.step(states, time_step)
        if self.reflection_symmetry:
            from particlesim.solvers.nr.symmetry import project

            backend = self.nested.pairs[0].coarse.backend
            advanced = [project(state, backend=backend) for state in advanced]
        return advanced

    def diagnostics(self, states, excise: float = 1.0, split: float = 2.0) -> dict[str, Any]:
        """What says whether the run is healthy, as plain numbers.

        The Hamiltonian constraint is given per level, each measured where
        that level and nothing else is responsible: inside its own buffer,
        outside the next box, outside ``r = excise``, and on the coarsest
        level inside the radiative zone. A level with nothing left to
        measure reports ``nan``. ``hamiltonian_inner`` and
        ``hamiltonian_outer`` split the same regions at ``r = split``. An
        octant's states are unfolded first, so every number is the whole
        grid's.
        """
        states = self.full_states(states)
        kernel = bssn.constraint_kernel(order=self.nested.pairs[0].fine.order)
        buffer = self.nested.pairs[0].buffer
        axes = tuple(INDICES)
        norms, inner, outer = [], [], []

        def rms(values, where):
            return float(np.sqrt(np.mean(values[where] ** 2))) if where.any() else float("nan")

        for level, state in enumerate(states):
            fields = {k: v for k, v in state.items() if not k.startswith(AUXILIARY)}
            h = np.asarray(
                kernel(bssn.physical_slice_arrays(fields), (self.spacing(level),) * 3)[
                    "hamiltonian"
                ]
            )
            grid = np.meshgrid(*(self.full_axis(level),) * DIMENSION, indexing="ij")
            radius = np.sqrt(sum((g - p) ** 2 for g, p in zip(grid, self.position, strict=True)))
            keep = radius > excise
            if level + 1 < self.levels:
                keep[self.full_box(level).slices()] = False
            width = self.zone if level == 0 else buffer
            h, keep = interior(h, width, axes), interior(keep, width, axes)
            radius = interior(radius, width, axes)
            norms.append(rms(h, keep))
            inner.append(rms(h, keep & (radius <= split)))
            outer.append(rms(h, keep & (radius > split)))
        finest = states[-1]
        finite = all(
            np.all(np.isfinite(np.asarray(value))) for state in states for value in state.values()
        )
        return {
            "lapse_min": float(np.min(np.asarray(finest["alpha"]))),
            "phi_max": float(np.max(np.asarray(bssn.conformal_exponent(finest)))),
            "hamiltonian": norms,
            "hamiltonian_inner": inner,
            "hamiltonian_outer": outer,
            "shift_max": float(
                max(np.max(np.abs(np.asarray(finest[f"beta{i}"]))) for i in INDICES)
            ),
            "finite": bool(finite),
        }


__all__ = [
    "NestedPuncture",
    "brill_lindquist_adm_mass",
    "brill_lindquist_state",
    "TwoLevelPuncture",
    "perturbed_puncture_state",
    "puncture_state",
    "staggered_offset",
    "trumpet_areal_radius",
    "trumpet_isotropic_radius",
    "trumpet_state",
]
