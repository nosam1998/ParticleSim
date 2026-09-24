"""The refined spherical solver behind the uniform one's interface.

Issue #111. :mod:`particlesim.analysis.critical_collapse` searches for the
threshold through a ``make_sim`` callable and asks the object it gets for
five things: ``grid`` and ``r`` to lay out data, ``dt`` and ``step`` to
evolve it, and ``solve_metric`` to classify it. :class:`AdaptiveCollapse`
answers all five from a :class:`~particlesim.solvers.nr.subcycle.Subcycler`,
so the search, the bisection and the scaling fit run on a refined hierarchy
without knowing it is one.

**The initial data is laid out on the base grid.** A pulse family builds its
state from ``sim.grid``, which is the coarsest level's, and the hierarchy is
seeded from that state on the first step. Levels below are created by the
regridding as the solution asks for them, so the initial slice does not need
to know how deep the run will go.

**By default the regridding follows the focusing, not the infall.** The
curvature trigger refines where the curvature radius is short, and an
ingoing shell is weak until it nears the origin: the thin shell of the
threshold search gets its first level at ``t = 4.4``, about a unit of time
before it bounces, and makes the trip in on the base grid. That costs 7.5e-4
in the bounce's peak curvature at 400 cells. Two ways to buy it back, both
measured in ``docs/benchmarks.md``: ``seed_extent`` keeps one level over
``[0, seed_extent]`` from the start (7e-6), and ``tolerance`` turns on the
Richardson estimate, which sees the infall's error and refines for it
(4.9e-4, and a deeper hierarchy through the bounce). The estimate compares a
level with its parent, so it needs a seed level or the curvature trigger to
create the first.

**What comes back from a step is the composite.** Every cell that no finer
level covers, finest first, which is the solution at the best resolution the
hierarchy has anywhere. ``r`` follows it, and changes length whenever a
level is created or retired: anything that masks by radius has to recompute
the mask after each step.

**The peak curvature is sampled at the finest level's steps.** The search
records the largest ``|R|`` it sees after each step, and one step here is a
step of the base grid -- ``2^k`` steps of the finest level. The peak lasts
about one curvature time, which near threshold falls below a base step, and
read only from the composite it is missed by however far the nearest base
step landed from it: by nothing to four figures at ``1 - p/p* = 1e-4``, by
0.3% at 2e-6 and by 3.7% at 4e-7, growing as the threshold nears. So the
finest level reports every value it passes through; see
:meth:`watch_ricci`.

**The metric is the hierarchy's.** A level's own lapse is normalised at that
level's edge, which is wrong wherever there is matter outside it; the
composite metric is solved across all the levels at once and normalised at
the outer boundary. ``solve_metric`` therefore does not solve anything from
the arrays it is given. It returns the metric of the state the last step
produced, and refuses any other.
"""

from __future__ import annotations

import numpy as np

from particlesim.analysis.spherical_diagnostics import ricci_scalar
from particlesim.core.spherical import SphericalGrid
from particlesim.solvers.nr.hierarchy import Hierarchy, Level, refine
from particlesim.solvers.nr.spherical import ScalarCollapse, SphericalState
from particlesim.solvers.nr.subcycle import Subcycler


class AdaptiveCollapse:
    """A :class:`Subcycler` with :class:`ScalarCollapse`'s interface.

    ``level_cells`` is the size of every level the regridding creates; the
    first covers ``level_cells`` of the base grid's half-cells, and each one
    after that the inner half of its parent. ``cells_per_radius`` and
    ``tolerance`` are the two refinement triggers -- see
    :meth:`Subcycler._regrid`.
    """

    def __init__(
        self,
        grid: SphericalGrid,
        courant: float = 0.25,
        dissipation: float | None = None,
        cells_per_radius: float | None = 16.0,
        tolerance: float | None = None,
        level_cells: int = 160,
        max_depth: int = 24,
        seed_extent: float | None = None,
        record_centre: bool = False,
    ):
        self.grid = grid
        self.base = ScalarCollapse(grid, courant=courant, dissipation=dissipation)
        self.courant, self.dissipation = courant, dissipation
        self.cells_per_radius, self.level_cells = cells_per_radius, level_cells
        self.tolerance = tolerance
        self.max_depth, self.seed_extent = max_depth, seed_extent
        self.subcycler: Subcycler | None = None
        self.r = self.base.r
        self._latest: tuple[SphericalState, np.ndarray, np.ndarray] | None = None
        self._watch_radius: float | None = None
        self._between = 0.0
        #: ``(t, r, alpha, a, Pi)`` at the finest level's innermost cell after
        #: every one of its steps, when ``record_centre``: what the echoing is
        #: read from, see :func:`~particlesim.analysis.critical_collapse.echo_period`.
        self.record_centre = record_centre
        self.centre: list[tuple[float, float, float, float, float]] = []

    @property
    def dt(self) -> float:
        return self.base.dt

    @property
    def depth(self) -> int:
        return 1 if self.subcycler is None else self.subcycler.depth

    def step(self, state: SphericalState, dt: float) -> SphericalState:
        if self.subcycler is None:
            if state.Phi.shape != self.base.r.shape:
                raise ValueError("the first state must be laid out on the base grid")
            levels = [Level(self.grid, state.Phi, state.Pi)]
            if self.seed_extent is not None:
                levels.append(refine(levels[0], self.seed_extent))
            self.subcycler = Subcycler(
                Hierarchy(levels),
                courant=self.courant,
                dissipation=self.dissipation,
                tolerance=self.tolerance,
                cells_per_radius=self.cells_per_radius,
                level_cells=self.level_cells,
                max_depth=self.max_depth,
                on_step=self._observe,
            )
        elif self._latest is None or state is not self._latest[0]:
            raise ValueError(
                "an adaptive run holds its own state; step it from the one it last returned"
            )
        self._between = 0.0
        self.subcycler.step(dt)
        r, Phi, Pi, a, alpha = self.subcycler.composite()
        after = SphericalState(self.subcycler.t, Phi, Pi)
        self.r, self._latest = r, (after, a, alpha)
        return after

    def watch_ricci(self, inner_radius: float) -> None:
        """Track the largest ``|R|`` inside ``inner_radius`` on the finest level,
        at every one of its steps; :meth:`ricci_between_steps` reports it."""
        self._watch_radius = inner_radius

    def ricci_between_steps(self) -> float:
        """The largest ``|R|`` the finest level passed through during the last
        :meth:`step`, inside the watched radius; zero when not watching."""
        return self._between

    def _observe(self, index: int, sim: ScalarCollapse, state: SphericalState, metric) -> None:
        watching = self._watch_radius is not None
        if index != self.subcycler.depth - 1 or not (watching or self.record_centre):
            return
        a, alpha = metric()
        if self.record_centre:
            self.centre.append(
                (state.t, float(sim.r[0]), float(alpha[0]), float(a[0]), float(state.Pi[0]))
            )
        if watching:
            inside = sim.r < self._watch_radius
            if inside.any():
                ricci = np.abs(ricci_scalar(a[inside], state.Phi[inside], state.Pi[inside]))
                self._between = max(self._between, float(ricci.max()))

    def solve_metric(self, Phi: np.ndarray, Pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._latest is None:
            return self.base.solve_metric(Phi, Pi)
        state, a, alpha = self._latest
        if Phi is not state.Phi or Pi is not state.Pi:
            raise ValueError(
                "an adaptive run's metric is solved across its levels; ask for the "
                "state its last step returned"
            )
        return a, alpha


__all__ = ["AdaptiveCollapse"]
