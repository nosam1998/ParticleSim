"""Nested refinement levels for the spherical solver, and the constraint across them.

Issue #111. The critical solution of scalar collapse is discretely
self-similar with echoing period ``Delta = 3.44`` in the logarithm of scale,
so each successive echo lives on a region ``exp(3.44) = 31`` times smaller
than the last. A uniform grid carries a fixed dynamic range and runs out
after two; what is needed is resolution that follows the solution inward.

**The refinement does not have to search for where to refine.** The critical
solution collapses to the origin, always, so the hierarchy is a set of
nested shells centred there and the only adaptive question is *how deep*.
That is why the levels below are created and retired in time rather than
moved in space, and it is the whole of what makes this affordable: a level
resolving a scale costs ``2^k`` more per coarse step and lives ``2^-k`` as
long, so each level costs about the same as the last and a deep hierarchy
is linear in its depth rather than exponential.

**The constraint solve is the part that does not carry over, and it is local
for a reason specific to this system.** The Misner-Sharp mass integrates
outward from the origin, so ``m(r)`` depends only on the matter inside
``r`` -- all of which lives on that level or a finer one. A level's mass
solve therefore needs nothing from its parent except where the integration
had reached. The lapse needs one global constant, but ``d(ln alpha)/dr`` is
local too, so the levels accumulate ``ln alpha`` and the constant is fixed
once at the outer boundary. Neither quantity is ever interpolated across a
level boundary, so neither can lose accuracy there.

**What is easy to get wrong there is the gap.** Between the outermost point
of one level and the innermost point of the next lies a stretch that belongs
to neither, about three quarters of a coarse cell wide. Crossing it with a
single Euler step is a local ``O(dr^2)`` error at one point, and that alone
takes the whole metric from fourth order to second.

**And it has to be crossed for the mass and the lapse together.** The
slicing condition's slope is ``(m + 4 pi r^3 S)/(r(r - 2m))``, so carrying
``ln alpha`` across needs to know how ``m`` varies along the way. Crossing
them separately, with the mass held at its end value while the lapse
crosses, is wrong by ``O(dm/dr * gap^2)`` -- second order, and exactly zero
wherever there is no matter left to accumulate.

That second mistake is the instructive one, because it measures as correct.
With the level boundary placed at ``r = 5`` in a shell centred on ``r = 3``,
where ``dm/dr`` has fallen to nothing, the split version converges at a
clean 4.00 and looks finished. Move the boundary to the shell's own peak --
which is where a refinement boundary will actually sit, since refinement
follows the solution -- and it is second order. Both placements are measured
now, and the coupled Runge-Kutta crossing holds fourth order at each:

===========  ==========  ==========  ==========
boundary     Euler gap   split       coupled
===========  ==========  ==========  ==========
``r = 5``    2.0 - 3.8   4.00        4.00
``r = 3``    2.00        ~2          3.6 - 3.8
``r = 2``    2.0         --          4.00
===========  ==========  ==========  ==========

The ``r = 3`` column settles near 3.6 rather than 4 and stays there rather
than drifting, because that is where the crossing has to reconstruct the
field from the coarse level over the widest gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from particlesim.core.interpolate import midpoints
from particlesim.core.spherical import SphericalGrid
from particlesim.solvers.nr.polar import (
    accumulate_lapse,
    normalise_lapse,
    solve_mass,
)

#: Refinement ratio in space, and in time when the levels subcycle.
RATIO = 2


@dataclass
class Level:
    """One uniform shell of the hierarchy, covering ``[0, grid.r_max]``."""

    grid: SphericalGrid
    Phi: np.ndarray
    Pi: np.ndarray

    def __post_init__(self) -> None:
        if not self.grid.uniform:
            raise ValueError("a refinement level is a uniform grid; grade the hierarchy instead")
        if self.Phi.shape != (self.grid.n,) or self.Pi.shape != (self.grid.n,):
            raise ValueError(
                f"the fields must have one value per cell ({self.grid.n}), got "
                f"{self.Phi.shape} and {self.Pi.shape}"
            )

    @property
    def radii(self) -> np.ndarray:
        return self.grid.radii()

    @property
    def spacing(self) -> float:
        return float(self.grid.dr)

    @property
    def outer(self) -> float:
        """The level's outer *face*, which is where the next one takes over."""
        return float(self.grid.r_max)


def refine(level: Level, extent: float) -> Level:
    """A child covering ``[0, extent]`` at half the spacing, by interpolation.

    The extent is snapped outward to a whole number of parent cells so the
    two grids nest exactly: each parent cell is the union of two child
    cells, which is what makes restriction an average and prolongation a
    interpolation rather than a resampling.
    """
    if extent <= 0.0 or extent >= level.outer:
        raise ValueError(
            f"a child must cover a strict sub-interval of its parent, got {extent} "
            f"against {level.outer}"
        )
    cells = max(1, int(np.ceil(extent / level.spacing)))
    child = SphericalGrid(r_max=cells * level.spacing, n=RATIO * cells)
    return Level(child, prolong(level, child), prolong(level, child, field="Pi"))


def _interpolate(source: Level, radii: np.ndarray, values: np.ndarray, odd: bool) -> np.ndarray:
    """Fourth-order interpolation onto ``radii``, with parity across the origin.

    The origin is not a boundary, it is the centre of a sphere, so the
    stencil reaches across it with the sign the field's rank demands. Doing
    anything else there is what turns the ``2 f Phi / r`` term of the
    evolution into an amplifier: it grows wrong-parity error as ``1/r``, and
    the finest level is the one nearest the origin.
    """
    sign = -1.0 if odd else 1.0
    extended_r = np.concatenate([-source.radii[:2][::-1], source.radii, source.radii[-2:]])
    extended_v = np.concatenate([sign * values[:2][::-1], values, np.full(2, values[-1])])
    out = np.empty_like(radii)
    for index, radius in enumerate(radii):
        centre = int(np.searchsorted(extended_r, radius))
        start = min(max(centre - 2, 0), len(extended_r) - 4)
        window = slice(start, start + 4)
        out[index] = _lagrange(extended_r[window], extended_v[window], radius)
    return out


def _lagrange(nodes: np.ndarray, values: np.ndarray, at: float) -> float:
    total = 0.0
    for index in range(len(nodes)):
        weight = 1.0
        for other in range(len(nodes)):
            if other != index:
                weight *= (at - nodes[other]) / (nodes[index] - nodes[other])
        total += weight * values[index]
    return float(total)


def prolong(parent: Level, grid: SphericalGrid, field: str = "Phi") -> np.ndarray:
    """Parent values onto a child grid, fourth order, parity-aware.

    ``Phi`` is odd across the origin and ``Pi`` is even, which the stencil
    has to know: interpolating an odd field as though it were even puts a
    cusp at ``r = 0`` on the level that can least afford one.
    """
    values = parent.Phi if field == "Phi" else parent.Pi
    return _interpolate(parent, grid.radii(), values, odd=field == "Phi")


def restrict(child: Level, parent: Level) -> tuple[np.ndarray, np.ndarray]:
    """Child values averaged back onto the parent cells they cover.

    Two child cells make one parent cell exactly, so this is an average and
    not an interpolation.

    **It is second-order accurate, and that is a statement about what the
    stored numbers mean.** Averaging the two children is exactly the parent
    cell average if the values are read as cell averages, and this solver
    reads them as point values at cell centres -- the radial derivatives,
    the parity reflection and the midpoint interpolation all do. Against a
    point value the average is off by ``dr^2 f'' / 32``: exact for a field
    linear in ``r``, and falling by four per refinement for anything else,
    measured at 5.7e-2, 1.4e-2, 3.6e-3 on ``r^3``.

    Nothing in the metric solve restricts, so nothing here is capped by it
    today: each level integrates its own matter and the levels are chained
    by value, never by interpolation. It is the evolution that will restrict,
    and a second-order restriction inside a fourth-order evolution caps the
    evolution -- the same way a second-order midpoint capped the lapse. The
    fix when that lands is to restrict with the fourth-order rule rather
    than to reinterpret the values.
    """
    covered = parent.radii < child.outer
    count = int(np.count_nonzero(covered))
    if RATIO * count > child.grid.n:
        raise ValueError(
            f"the child covers {child.grid.n} cells but the parent expects {RATIO * count}; "
            "the two grids are not nested"
        )
    updated = []
    for values, whole in ((child.Phi, parent.Phi), (child.Pi, parent.Pi)):
        merged = np.array(whole, dtype=float)
        merged[covered] = values[: RATIO * count].reshape(count, RATIO).mean(axis=1)
        updated.append(merged)
    return updated[0], updated[1]


@dataclass
class Hierarchy:
    """Nested levels, coarsest first, each covering ``[0, r_max]`` of its own."""

    levels: list[Level]

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("a hierarchy needs at least one level")
        for coarse, fine in zip(self.levels[:-1], self.levels[1:], strict=True):
            if fine.outer >= coarse.outer:
                raise ValueError(
                    f"level extents must decrease inward, got {fine.outer} inside {coarse.outer}"
                )
            if abs(fine.spacing * RATIO - coarse.spacing) > 1e-12 * coarse.spacing:
                raise ValueError(
                    f"levels refine by {RATIO}:1, got spacings {coarse.spacing} and {fine.spacing}"
                )

    @property
    def depth(self) -> int:
        return len(self.levels)

    def owned(self, index: int) -> np.ndarray:
        """The cells of level ``index`` that no finer level covers."""
        radii = self.levels[index].radii
        inner = self.levels[index + 1].outer if index + 1 < self.depth else 0.0
        return radii > inner

    def composite_radii(self) -> np.ndarray:
        """Every owned cell, finest first, in increasing radius."""
        pieces = [self.levels[k].radii[self.owned(k)] for k in range(self.depth - 1, -1, -1)]
        return np.concatenate(pieces)

    def solve_metric(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """``(a, alpha)`` on every level, from one outward pass through all of them.

        The pass runs finest first because that is the direction the mass
        accumulates. Each level is handed where the integration reached and
        crosses the gap with a Runge-Kutta step of its own, so no
        interpolation happens at a level boundary and the fourth order
        survives it.
        """
        masses: list[np.ndarray] = [np.empty(0)] * self.depth
        logarithms: list[np.ndarray] = [np.empty(0)] * self.depth
        reached_mass: float | None = None
        reached_log: float | None = None
        reached_radius = 0.0

        for index in range(self.depth - 1, -1, -1):
            level = self.levels[index]
            owned = self.owned(index)
            radii = level.radii[owned]
            density = level.Pi[owned] ** 2 + level.Phi[owned] ** 2
            source = 2.0 * np.pi * radii**2 * density
            source_mid = midpoints(source)
            density_mid = midpoints(density)

            def mass_slope(i, mid, rr, mm, source=source, source_mid=source_mid):
                value = source_mid[i] if mid else source[i]
                return value * (1.0 - 2.0 * mm / rr)

            def lapse_slope(i, mid, rr, mm, density=density, density_mid=density_mid):
                value = density_mid[i] if mid else density[i]
                return mm / (rr**2 * (1.0 - 2.0 * mm / rr)) + 2.0 * np.pi * rr * value

            if reached_mass is None:
                first_mass = first_log = None
            else:
                first_mass, first_log = _cross_gap(
                    level, reached_radius, reached_mass, reached_log, float(radii[0])
                )
            mass = solve_mass(radii, level.spacing, mass_slope, first_value=first_mass)
            logarithm = accumulate_lapse(
                radii, level.spacing, mass, lapse_slope, mass_slope, first_value=first_log
            )
            masses[index] = mass
            logarithms[index] = logarithm
            reached_mass, reached_log, reached_radius = mass[-1], logarithm[-1], radii[-1]

        outer_mass = masses[0][-1]
        outer_radius = self.levels[0].radii[-1]
        a_levels: list[np.ndarray] = []
        alpha_levels: list[np.ndarray] = []
        for index in range(self.depth):
            mass = masses[index]
            radii = self.levels[index].radii[self.owned(index)]
            a_levels.append(1.0 / np.sqrt(1.0 - 2.0 * mass / radii))
            alpha_levels.append(
                normalise_lapse(
                    np.concatenate([logarithms[index], [logarithms[0][-1]]]),
                    outer_mass,
                    outer_radius,
                )[:-1]
            )
        return a_levels, alpha_levels

    def composite_metric(self) -> tuple[np.ndarray, np.ndarray]:
        """``(a, alpha)`` on :meth:`composite_radii`, finest first."""
        a_levels, alpha_levels = self.solve_metric()
        order = range(self.depth - 1, -1, -1)
        return (
            np.concatenate([a_levels[k] for k in order]),
            np.concatenate([alpha_levels[k] for k in order]),
        )


def _cross_gap(
    level: Level,
    inner_radius: float,
    inner_mass: float,
    inner_log: float,
    outer_radius: float,
) -> tuple[float, float]:
    """Carry ``(m, ln alpha)`` across the stretch that belongs to no level.

    Between the outermost point of one level and the innermost point of the
    next lies a gap about three quarters of a coarse cell wide. It has to be
    integrated, and it has to be integrated **as a pair**: the slicing
    condition's slope is ``(m + 4 pi r^3 S)/(r(r - 2m))``, so the lapse
    cannot be carried across without knowing how the mass varies along the
    way.

    Splitting them and holding the mass at its end value while the lapse
    crosses is wrong by ``O(dm/dr * gap^2)``, which is second order -- and
    invisible wherever ``dm/dr`` is small. Putting the level boundary out in
    a shell's tail, where there is nothing left to accumulate, measures a
    clean fourth order from that split version. Putting it at the shell's
    own peak, which is where a refinement boundary will actually sit, does
    not. This is the same failure as crossing the gap with an Euler step,
    one level of subtlety further in.

    The fields come from the level about to be integrated, which is the
    accurate side: the gap is half a cell of *this* level wide and several of
    the finer one's, so this level's stencil spans it with room to spare.
    """

    def field_at(radius: float) -> float:
        phi = _interpolate(level, np.array([radius]), level.Phi, odd=True)[0]
        pi = _interpolate(level, np.array([radius]), level.Pi, odd=False)[0]
        return float(pi**2 + phi**2)

    def slopes(radius: float, mass: float) -> tuple[float, float]:
        density = field_at(radius)
        d_mass = 2.0 * np.pi * radius**2 * density * (1.0 - 2.0 * mass / radius)
        d_log = mass / (radius**2 * (1.0 - 2.0 * mass / radius)) + 2.0 * np.pi * radius * density
        return d_mass, d_log

    step = outer_radius - inner_radius
    if step <= 0.0:
        return inner_mass, inner_log

    first = slopes(inner_radius, inner_mass)
    second = slopes(inner_radius + 0.5 * step, inner_mass + 0.5 * step * first[0])
    third = slopes(inner_radius + 0.5 * step, inner_mass + 0.5 * step * second[0])
    fourth = slopes(outer_radius, inner_mass + step * third[0])
    weighted = [
        step / 6.0 * (first[k] + 2.0 * second[k] + 2.0 * third[k] + fourth[k]) for k in (0, 1)
    ]
    return inner_mass + weighted[0], inner_log + weighted[1]


__all__ = ["RATIO", "Hierarchy", "Level", "prolong", "refine", "restrict"]
