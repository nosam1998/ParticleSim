"""Berger-Oliger subcycling for the spherical solver.

Issue #111. :mod:`particlesim.solvers.nr.hierarchy` builds the nested levels
and solves the constraints across them; this evolves them in time.

**Each level takes its own step, and finer levels take more of them.** The
Courant limit is set by the spacing, so a level refined by two can only take
half the step -- but it also covers only a fraction of the domain, so it needs
proportionally fewer cells to do it. That is the whole economy of the scheme:
a level resolving a scale ``2^-k`` costs ``2^k`` more work per coarse step and
lives ``2^-k`` as long, so each level costs about what the last one did and a
deep hierarchy is linear in its depth.

**The order of operations is not a convention.** A level steps first, so that
its child has a parent bracketing its own interval to take boundary data
from; then the child subcycles across that interval; then the child's
solution is restricted onto the cells it covers, because it is the better
answer there. Restricting before the child has caught up would mix two
different times.

**The boundary data in time is where the order is easiest to lose.** A
child's outer edge is an interface rather than a boundary, and it needs the
parent's solution at instants the parent never lands on. Interpolating
linearly between the parent's two endpoints is the textbook choice and is
second order -- 2.00 at every step size tried -- which would cap the whole
evolution through its own refinement boundary. The parent has already
evaluated its right-hand side at both ends, so a cubic Hermite through both
values and both slopes is available for nothing, and is fourth order.

**The lapse is where it is easiest to be wrong outright.** A level can solve
for its own ``a``, because the Misner-Sharp mass depends only on the matter
inside a radius. It cannot solve for its own lapse. The slicing condition
fixes ``d(ln alpha)/dr`` locally, but the constant is fixed at the outermost
boundary, and a child's own solve fixes it at the *child's* outer edge instead
-- as though that edge were the asymptotic region. It is only if every bit of
matter lies inside it; otherwise the child's lapse is off by
``exp(integral 4 pi r rho dr)`` over the matter it cannot see, the child runs at
the wrong rate of coordinate time, and the error is a constant factor that no
amount of refinement touches.

That was measured, not reasoned about. With the refinement boundary at
``r = 5``, outside an ingoing shell at ``r = 3``, a hand-wired two-level run
matched a uniform grid at the child's spacing to four figures and converged
at 17.3 per doubling. Moved to ``r = 2.5``, inside the shell, the same run
was off by ``6.7e-3`` at every resolution -- nine hundred times worse than not
refining at all. The first placement is vacuum at the interface; the second
is what a collapse run does, since the pulse disperses outward through every
refinement boundary on its way out. So the child's lapse is rescaled at every
stage to agree with its parent's at their shared edge, and the parent's is
itself taken from *its* parent, down to the coarsest level, which alone sees
the asymptotic boundary.

**Regridding asks only how deep.** The critical solution collapses onto the
origin, so every level is a ball centred there, a new one is created over the
inner part of the finest, and only the finest is ever retired. Levels do not
move and are never resized.

**Two triggers, as issue #111 asks.** The principled one is a Richardson
estimate: a parent that takes its own step over cells its child covers can
compare the result with the child's restriction, and at fourth order the
difference is fifteen times the child's local error. A level gets a child
where that error exceeds ``tolerance`` times the largest field the run has
held. It sees shape as well as amplitude, so it refines an infalling shell
before the shell reaches the origin.

The cheap one is the curvature radius ``1 / sqrt(8 pi rho)``, the length over
which the geometry varies: a level gets a child where it puts fewer than
``cells_per_radius`` cells across it. It needs no parent to compare with, so
it can grow a hierarchy from a single level, and near threshold it follows
the echoes by construction -- the critical solution ties amplitude to size,
``Phi ~ 1/L``, so the radius shrinks with each echo and the hierarchy deepens
by ``log2(e^3.44)``, five levels, per echo. What it cannot see is shape: an
infalling shell is weak until it is nearly at the origin, and crosses the
base grid unrefined.

Either or both may be set. A level is added when either asks and retired
only when every trigger set has been satisfied, with a margin, for
``patience`` consecutive checks.

For a while the Richardson estimate looked unusable here: after a strong
bounce it read grid-scale noise at the origin as error, created a level for
it that inherited the noise, and cascaded twelve levels deep. That noise was
the solver's, not the estimate's -- the ``Pi`` equation was written in a
form that made energy at the origin -- and with it fixed the same estimate
runs through the bounce five levels deep and retires them all afterwards.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from particlesim.core.interpolate import hermite
from particlesim.core.spherical import SphericalGrid
from particlesim.solvers.nr.hierarchy import (
    RATIO,
    Hierarchy,
    Level,
    _interpolate,
    prolong,
    restrict,
)
from particlesim.solvers.nr.spherical import ScalarCollapse, SphericalState

#: Cells at a level's outer edge driven by its parent rather than evolved.
#: Two is what the fourth-order centred stencil reaches for.
INTERFACE_CELLS = 2

#: Cells between a parent's mass anchor and its child's edge; see
#: :meth:`Subcycler._anchor`. One step reaches twelve cells -- four stages,
#: each through the three-cell radius of the dissipation -- and the child's
#: interface reads the two parent cells inside its edge.
ANCHOR_CELLS = 14

Metric = tuple[np.ndarray, np.ndarray]


@dataclass
class Interface:
    """A parent's solution across one of its steps, as its child needs it.

    Everything a child asks of its parent -- field values at the child's edge,
    and the lapse there -- is at some instant inside the parent's step, so
    this holds the parent's state and right-hand side at both ends of the
    step and interpolates between them with a cubic Hermite. ``upstream`` is
    whatever drives the parent in turn, and ``offset`` and ``span`` say where
    the parent's step sits inside the upstream one, so a question about the
    child's time can be passed all the way to the coarsest level.
    """

    sim: ScalarCollapse
    before: SphericalState
    after: SphericalState
    slope_before: tuple[np.ndarray, np.ndarray]
    slope_after: tuple[np.ndarray, np.ndarray]
    step: float
    upstream: Interface | None = None
    offset: float = 0.0
    span: float = 1.0
    #: The parent's anchor cell and the mass there at both ends of its step,
    #: with the flux that carries it; see :func:`step_anchored`.
    anchor: int | None = None
    mass_before: float = 0.0
    mass_after: float = 0.0
    flux_before: float = 0.0
    flux_after: float = 0.0
    _metrics: dict[float, Metric] = field(default_factory=dict, repr=False)

    def upstream_fraction(self, fraction: float) -> float:
        return self.offset + fraction * self.span

    def state_at(self, fraction: float) -> SphericalState:
        """The parent's fields ``fraction`` of the way through its step."""
        f = min(max(fraction, 0.0), 1.0)
        Phi, Pi = (
            hermite(
                getattr(self.before, name),
                getattr(self.after, name),
                self.slope_before[k],
                self.slope_after[k],
                self.step,
                f,
            )
            for k, name in enumerate(("Phi", "Pi"))
        )
        if self.upstream is not None:
            # The parent's own interface cells are not evolved, so their
            # slopes carry no information; take them from upstream directly.
            radii = self.sim.r[-INTERFACE_CELLS:]
            Phi[-INTERFACE_CELLS:], Pi[-INTERFACE_CELLS:] = self.upstream.fields_at(
                self.upstream_fraction(f), radii
            )
        return SphericalState(self.before.t + f * self.step, Phi, Pi)

    def fields_at(self, fraction: float, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(Phi, Pi)`` interpolated to ``radii``, with each field's parity."""
        state = self.state_at(fraction)
        level = Level(self.sim.grid, state.Phi, state.Pi)
        return (
            _interpolate(level, radii, state.Phi, odd=True),
            _interpolate(level, radii, state.Pi, odd=False),
        )

    def anchor_at(self, fraction: float) -> tuple[int, float] | None:
        """The parent's anchor mass at that instant, by the same cubic Hermite."""
        if self.anchor is None:
            return None
        f = min(max(fraction, 0.0), 1.0)
        mass = hermite(
            self.mass_before, self.mass_after, self.flux_before, self.flux_after, self.step, f
        )
        return self.anchor, float(mass)

    def metric_at(self, fraction: float) -> Metric:
        """The parent's metric at that instant, with its lapse properly normalised."""
        if fraction not in self._metrics:
            self._metrics[fraction] = normalised_metric(
                self.sim,
                self.state_at(fraction),
                self.upstream,
                self.upstream_fraction(fraction),
                self.anchor_at(fraction),
            )
        return self._metrics[fraction]

    def lapse_at(self, fraction: float, radius: float) -> float:
        """The parent's lapse at ``radius``, which is what fixes the child's."""
        _, alpha = self.metric_at(fraction)
        level = Level(self.sim.grid, alpha, alpha)
        return float(_interpolate(level, np.array([radius]), alpha, odd=False)[0])


def normalised_metric(
    sim: ScalarCollapse,
    state: SphericalState,
    interface: Interface | None,
    fraction: float,
    anchor: tuple[int, float] | None = None,
) -> Metric:
    """``(a, alpha)`` on one level, with the lapse taken from the parent at the edge.

    The level's own solve gets ``a`` right and normalises ``alpha a -> 1`` at
    its own outermost cell. The shape of the lapse inside the level is right
    too, since ``d(ln alpha)/dr`` is local; only the constant is wrong, so the
    whole profile is rescaled until its outermost value is the parent's there.
    The coarsest level has no parent and is normalised against the asymptotic
    boundary, which is the one place that normalisation is true.

    ``anchor`` is ``(cell, mass)`` for a level with a child: the mass there
    is given rather than integrated through the covered region; see
    :meth:`~particlesim.solvers.nr.spherical.ScalarCollapse.solve_anchored_metric`.
    """
    if anchor is None:
        a, alpha = sim.solve_metric(state.Phi, state.Pi)
    else:
        a, alpha = sim.solve_anchored_metric(state.Phi, state.Pi, *anchor)
    if interface is None:
        return a, alpha
    edge = interface.lapse_at(fraction, float(sim.r[-1]))
    return a, alpha * (edge / alpha[-1])


def level_slope(
    sim: ScalarCollapse,
    state: SphericalState,
    interface: Interface | None,
    fraction: float,
    anchor: tuple[int, float] | None = None,
) -> tuple[SphericalState, tuple[np.ndarray, np.ndarray], Metric | None]:
    """The level's state with its interface cells filled, its right-hand side,
    and the metric that went into it.

    Without an interface the level is the outermost one: its own metric
    solve is correct and its own outgoing condition applies.
    """
    if interface is None:
        if anchor is None:
            return state, sim.rhs(state), None
        metric = normalised_metric(sim, state, None, fraction, anchor)
        return state, sim.rhs(state, metric), metric
    phi, pi = interface.fields_at(fraction, sim.r[-INTERFACE_CELLS:])
    Phi, Pi = np.array(state.Phi), np.array(state.Pi)
    Phi[-INTERFACE_CELLS:], Pi[-INTERFACE_CELLS:] = phi, pi
    driven = state.with_fields(Phi, Pi)
    metric = normalised_metric(sim, driven, interface, fraction, anchor)
    dPhi, dPi = sim.rhs(driven, metric)
    # The interface cells are prescribed at every stage, so they must not
    # also integrate on their own.
    dPhi[-INTERFACE_CELLS:] = 0.0
    dPi[-INTERFACE_CELLS:] = 0.0
    return driven, (dPhi, dPi), metric


def mass_flux(sim: ScalarCollapse, state: SphericalState, metric: Metric, index: int) -> float:
    """``dm/dt`` at ``r[index]``: ``4 pi r^2 alpha Phi Pi / a^3``.

    The momentum constraint, ``da/dt = 4 pi r alpha Phi Pi``, written for the
    Misner-Sharp mass ``m = (r/2)(1 - 1/a^2)``. The mass inside a sphere
    changes only by what crosses it.
    """
    a, alpha = metric
    r = float(sim.r[index])
    flux = state.Phi[index] * state.Pi[index] * alpha[index] / a[index] ** 3
    return float(4.0 * np.pi * r**2 * flux)


def step_with_interface(
    sim: ScalarCollapse,
    state: SphericalState,
    dt: float,
    interface: Interface | None,
    offset: float = 0.0,
) -> SphericalState:
    """One fourth-order Runge-Kutta step of one level.

    ``offset`` is where this step begins as a fraction of the parent's step,
    so each stage asks the parent for the instant it actually needs rather
    than for the step's start. Getting that wrong is a first-order error in
    time at the interface, which is worse than the linear interpolation the
    Hermite is there to avoid.
    """
    if interface is None:
        return sim.step(state, dt)
    span = dt / interface.step

    def slope(fields: SphericalState, fraction: float):
        return level_slope(sim, fields, interface, fraction)[1]

    k1 = slope(state, offset)
    s2 = state.with_fields(state.Phi + 0.5 * dt * k1[0], state.Pi + 0.5 * dt * k1[1])
    k2 = slope(s2, offset + 0.5 * span)
    s3 = state.with_fields(state.Phi + 0.5 * dt * k2[0], state.Pi + 0.5 * dt * k2[1])
    k3 = slope(s3, offset + 0.5 * span)
    s4 = state.with_fields(state.Phi + dt * k3[0], state.Pi + dt * k3[1])
    k4 = slope(s4, offset + span)

    Phi = state.Phi + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    Pi = state.Pi + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    after = SphericalState(state.t + dt, Phi, Pi)
    return level_slope(sim, after, interface, offset + span)[0]


def step_anchored(
    sim: ScalarCollapse,
    state: SphericalState,
    dt: float,
    interface: Interface | None,
    offset: float,
    index: int,
    mass: float,
) -> tuple[SphericalState, float]:
    """:func:`step_with_interface` for a level with a child, carrying its anchor mass.

    The mass at the anchor cell is one more variable of the same Runge-Kutta
    step, with the flux through that sphere as its right-hand side, so it is
    fourth order in time like everything else. It starts every step from the
    finer levels' value (:meth:`Subcycler._anchor`) the way the covered
    fields start from their restriction.
    """
    span = dt / interface.step if interface is not None else 1.0

    def slope(fields: SphericalState, fraction: float, m: float):
        driven, k, metric = level_slope(sim, fields, interface, fraction, (index, m))
        return k, mass_flux(sim, driven, metric, index)

    k1, f1 = slope(state, offset, mass)
    s2 = state.with_fields(state.Phi + 0.5 * dt * k1[0], state.Pi + 0.5 * dt * k1[1])
    k2, f2 = slope(s2, offset + 0.5 * span, mass + 0.5 * dt * f1)
    s3 = state.with_fields(state.Phi + 0.5 * dt * k2[0], state.Pi + 0.5 * dt * k2[1])
    k3, f3 = slope(s3, offset + 0.5 * span, mass + 0.5 * dt * f2)
    s4 = state.with_fields(state.Phi + dt * k3[0], state.Pi + dt * k3[1])
    k4, f4 = slope(s4, offset + span, mass + dt * f3)

    Phi = state.Phi + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    Pi = state.Pi + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    after_mass = mass + dt / 6.0 * (f1 + 2 * f2 + 2 * f3 + f4)
    after = SphericalState(state.t + dt, Phi, Pi)
    if interface is not None:
        after = level_slope(sim, after, interface, offset + span, (index, after_mass))[0]
    return after, after_mass


class Subcycler:
    """A :class:`~particlesim.solvers.nr.hierarchy.Hierarchy`, evolved in time.

    One call to :meth:`step` advances the coarsest level by its own Courant
    step and every finer level by ``RATIO`` times as many steps of its own.

    With ``cells_per_radius`` set, levels below the ones given are created
    and retired as the solution asks; see :meth:`_regrid`. The given levels
    are kept whatever happens.
    """

    def __init__(
        self,
        hierarchy: Hierarchy,
        courant: float = 0.25,
        dissipation: float | None = None,
        tolerance: float | None = None,
        cells_per_radius: float | None = None,
        level_cells: int | None = None,
        max_depth: int = 24,
        patience: int = 4,
        on_step: Callable[[int, ScalarCollapse, SphericalState, Callable[[], Metric]], None]
        | None = None,
    ):
        self.courant, self.dissipation = courant, dissipation
        #: Called with ``(level, sim, state, metric)`` after every step of every
        #: level; ``metric()`` solves for that level's normalised metric then.
        self.on_step = on_step
        self.tolerance, self.cells_per_radius = tolerance, cells_per_radius
        self.max_depth, self.patience = max_depth, patience
        #: Cells in every level the regridding creates; the finest seed
        #: level's count unless given. A created level covers at most half its
        #: parent, so its interpolation stencil never reaches the parent's edge.
        self.level_cells = level_cells or hierarchy.levels[-1].grid.n
        regridding = tolerance is not None or cells_per_radius is not None
        if regridding and (
            self.level_cells % RATIO
            or not 2 * INTERFACE_CELLS < self.level_cells <= hierarchy.levels[-1].grid.n
        ):
            raise ValueError(
                f"level_cells must be even, above {2 * INTERFACE_CELLS} and at most the "
                f"finest seed level's {hierarchy.levels[-1].grid.n}, got {self.level_cells}"
            )
        #: The levels given at construction are kept; only deeper ones come
        #: and go.
        self.seed_depth = len(hierarchy.levels)
        if tolerance is not None and cells_per_radius is None and self.seed_depth < 2:
            raise ValueError(
                "a Richardson estimate compares a level with its parent, so from a "
                "single level nothing could ever be refined; seed two levels or set "
                "cells_per_radius as well"
            )
        #: Consecutive checks the finest level has been surplus.
        self._surplus = 0
        #: Richardson estimate of each level's local error per parent step,
        #: from its last restriction: ``level -> (parent radii, error)``.
        self.errors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        #: Levels placed since their last restriction; see :meth:`_restrict`.
        self._fresh: set[int] = set()
        self._largest_field = 0.0
        #: Every level created or removed, ``(t, level, action, extent)``.
        self.regrids: list[tuple[float, int, str, float]] = []
        self.sims = [
            ScalarCollapse(level.grid, courant=courant, dissipation=dissipation)
            for level in hierarchy.levels
        ]
        self.states = [
            SphericalState(0.0, np.array(level.Phi, dtype=float), np.array(level.Pi, dtype=float))
            for level in hierarchy.levels
        ]
        for fine, coarse in zip(self.sims[1:], self.sims[:-1], strict=True):
            if fine.grid.n < 2 * INTERFACE_CELLS:
                raise ValueError(
                    f"a refined level needs more than {2 * INTERFACE_CELLS} cells, since its "
                    f"outermost {INTERFACE_CELLS} are driven by its parent"
                )
            if abs(coarse.dt - RATIO * fine.dt) > 1e-12 * coarse.dt:
                raise ValueError("the levels must share a Courant number to subcycle 2:1")

    @property
    def depth(self) -> int:
        return len(self.sims)

    @property
    def t(self) -> float:
        return self.states[0].t

    @property
    def dt(self) -> float:
        """The coarsest level's step, which is what one :meth:`step` advances."""
        return self.sims[0].dt

    def step(self, dt: float | None = None) -> None:
        self._advance(0, self.dt if dt is None else dt, None, 0.0)

    def hierarchy(self) -> Hierarchy:
        """The current fields as a :class:`Hierarchy`, for the constraint solve."""
        return Hierarchy(
            [
                Level(sim.grid, state.Phi, state.Pi)
                for sim, state in zip(self.sims, self.states, strict=True)
            ]
        )

    def _advance(self, index: int, dt: float, interface: Interface | None, offset: float) -> None:
        sim, before = self.sims[index], self.states[index]
        anchor = self._anchor(index)
        if anchor is None:
            after = step_with_interface(sim, before, dt, interface, offset)
        else:
            after, after_mass = step_anchored(sim, before, dt, interface, offset, *anchor)

        if index + 1 < self.depth:
            span = dt / interface.step if interface is not None else 1.0
            pinned = None if anchor is None else (anchor[0], after_mass)
            start, slope_before, metric_before = level_slope(sim, before, interface, offset, anchor)
            end, slope_after, metric_after = level_slope(
                sim, after, interface, offset + span, pinned
            )
            anchored = {}
            if anchor is not None:
                anchored = {
                    "anchor": anchor[0],
                    "mass_before": anchor[1],
                    "mass_after": after_mass,
                    "flux_before": mass_flux(sim, start, metric_before, anchor[0]),
                    "flux_after": mass_flux(sim, end, metric_after, anchor[0]),
                }
            child = Interface(
                sim=sim,
                before=before,
                after=after,
                slope_before=slope_before,
                slope_after=slope_after,
                step=dt,
                upstream=interface,
                offset=offset,
                span=span,
                **anchored,
            )
            for k in range(RATIO):
                self._advance(index + 1, dt / RATIO, child, k / RATIO)
            after = self._restrict(index, after)

        self.states[index] = after
        if self.on_step is not None:
            end = offset + (dt / interface.step if interface is not None else 1.0)
            pinned = None if anchor is None else (anchor[0], after_mass)
            self.on_step(
                index, sim, after, lambda: normalised_metric(sim, after, interface, end, pinned)
            )
        regridding = self.tolerance is not None or self.cells_per_radius is not None
        if regridding and index == max(self.depth - 2, 0):
            self._regrid()

    def _anchor(self, index: int) -> tuple[int, float] | None:
        """``(cell, mass)`` where a level with a child takes its mass from the child.

        The cell sits ``ANCHOR_CELLS`` inside the child's edge: far enough in
        that nothing the level does to its covered interior during a step can
        reach the cells that feed the child's interface, and inside the region
        the child owns, where its mass is well resolved. ``None`` for the
        finest level, and for a child too small to leave room.
        """
        if index + 1 >= self.depth:
            return None
        sim, child = self.sims[index], self.sims[index + 1]
        cell = int(round(child.grid.r_max / sim.dr)) - ANCHOR_CELLS
        if cell < ANCHOR_CELLS:
            return None
        return cell, self._mass_at(index + 1, float(sim.r[cell]))

    def _mass_at(self, index: int, radius: float) -> float:
        """The Misner-Sharp mass inside ``radius`` from level ``index`` and those below.

        Called while every level from ``index`` down is at the same instant,
        so each takes its own anchor from the next, down to the finest.
        """
        sim, state = self.sims[index], self.states[index]
        anchor = self._anchor(index)
        if anchor is None:
            a, _ = sim.solve_metric(state.Phi, state.Pi)
        else:
            a, _ = sim.solve_anchored_metric(state.Phi, state.Pi, *anchor)
        mass = 0.5 * sim.r * (1.0 - 1.0 / a**2)
        level = Level(sim.grid, mass, mass)
        return float(_interpolate(level, np.array([radius]), mass, odd=True)[0])

    def _restrict(self, index: int, parent: SphericalState) -> SphericalState:
        """Carry the child's solution back onto the parent cells it evolved.

        The last two covered parent cells are left alone: their restriction
        stencil reaches into the child's interface cells, which hold the
        parent's own data interpolated, so restricting them would hand the
        parent back its own answer through two interpolations.
        """
        sim, child_sim = self.sims[index], self.sims[index + 1]
        child_state = self.states[index + 1]
        phi, pi = restrict(
            Level(child_sim.grid, child_state.Phi, child_state.Pi),
            Level(sim.grid, parent.Phi, parent.Pi),
        )
        covered = np.flatnonzero(sim.r < child_sim.grid.r_max)
        keep = covered[: max(len(covered) - INTERFACE_CELLS, 0)]
        if index + 1 in self._fresh:
            # The child was built by prolonging this parent, and prolonging
            # then restricting is not the identity: the two differ by the
            # interpolation's O(h^4) before either has taken a step, where the
            # estimate is after one step's O(h^5). So a level's first estimate
            # measures its own construction and is discarded.
            self._fresh.discard(index + 1)
            self.errors.pop(index + 1, None)
        else:
            # The parent evolved these cells on its own from the child's data;
            # the difference is the parent's local error less the child's,
            # which at fourth order is fifteen times the child's.
            difference = np.maximum(
                np.abs(parent.Phi[keep] - phi[keep]), np.abs(parent.Pi[keep] - pi[keep])
            )
            self.errors[index + 1] = (sim.r[keep], difference / (RATIO**4 - 1))
        Phi, Pi = np.array(parent.Phi), np.array(parent.Pi)
        Phi[keep], Pi[keep] = phi[keep], pi[keep]
        return parent.with_fields(Phi, Pi)

    def composite(self):
        """``(r, Phi, Pi, a, alpha)`` on every owned cell, finest first.

        The metric comes from :meth:`Hierarchy.solve_metric`, which chains
        the constraint outward through the levels and normalises the lapse
        once, at the outer boundary. A level's own ``solve_metric`` would
        normalise it at that level's edge instead -- the defect the evolution
        corrects at every stage -- so anything reporting a lapse from a
        refined run reads it from here.
        """
        hierarchy = self.hierarchy()
        a_levels, alpha_levels = hierarchy.solve_metric()
        order = range(self.depth - 1, -1, -1)
        owned = [hierarchy.owned(k) for k in range(self.depth)]
        pick = [
            (
                self.sims[k].r[owned[k]],
                self.states[k].Phi[owned[k]],
                self.states[k].Pi[owned[k]],
                a_levels[k],
                alpha_levels[k],
            )
            for k in order
        ]
        return tuple(np.concatenate([piece[i] for piece in pick]) for i in range(5))

    # --- regridding -----------------------------------------------------

    def cells_across(self, index: int, extent: float | None = None) -> float:
        """Cells of level ``index`` across the smallest curvature radius inside ``extent``.

        The curvature radius is ``1 / sqrt(8 pi rho)`` with ``rho`` the energy
        density, ``(Phi^2 + Pi^2) / (2 a^2)``. Measured in cells of proper
        length ``a dr`` the metric cancels, which is why none is solved for.
        """
        sim, state = self.sims[index], self.states[index]
        inside = sim.r < (sim.grid.r_max if extent is None else extent)
        density = float(np.max(state.Phi[inside] ** 2 + state.Pi[inside] ** 2))
        return 1.0 / (sim.dr * np.sqrt(4.0 * np.pi * max(density, np.finfo(float).tiny)))

    def field_scale(self) -> float:
        """The largest ``|Phi| + |Pi|`` the hierarchy has held, what the
        Richardson tolerance is relative to.

        The whole hierarchy's, not a level's: a level the pulse has not
        reached holds only Gaussian tails, and measured against those every
        error is large -- normalised per level, the seed region was refined to
        a spacing of 1e-4 while the pulse was four units away. And a running
        maximum, not the current one: an outgoing pulse loses amplitude as
        ``1/r``, and a tolerance that tightened as it did would refine ever
        harder the quieter the physics became.
        """
        peak = max(float(np.max(np.abs(st.Phi) + np.abs(st.Pi))) for st in self.states)
        self._largest_field = max(self._largest_field, peak)
        return max(self._largest_field, np.finfo(float).tiny)

    def flagged(self, index: int, extent: float, margin: float = 1.0) -> int:
        """Cells of level ``index``'s estimate inside ``extent`` over the tolerance.

        ``margin`` scales the tolerance, for the hysteresis in retiring.
        """
        if index not in self.errors:
            return 0
        radii, error = self.errors[index]
        limit = margin * self.tolerance * self.field_scale()
        return int(np.count_nonzero(error[radii < extent] > limit))

    def _regrid(self) -> None:
        """Add a level below the finest, or retire the finest. Nothing else.

        The critical solution collapses onto the origin, so the hierarchy is a
        set of nested shells centred there and the only adaptive question is
        how deep. Each new level has ``level_cells`` cells at half its
        parent's spacing.

        A level is added where either trigger asks for it over the region it
        would cover: its parent puts fewer than ``cells_per_radius`` cells
        across the curvature radius there, or the parent's estimated error
        exceeds the tolerance at two or more cells. It is retired once every
        trigger set is satisfied with room to spare -- the parent would put
        twice ``cells_per_radius`` cells across the radius, and its error is
        under a quarter of the tolerance -- for ``patience`` consecutive
        checks. The margins keep a level whose trigger sits at its threshold
        from flipping every step.

        Checked whenever the finest level and its parent are at the same
        instant, which is the only time a level can be added below the finest
        without interpolating it in time as well as space.
        """
        finest = self.depth - 1
        below = self.level_cells * self.sims[finest].dr / RATIO
        wanted = False
        if self.cells_per_radius is not None:
            wanted |= self.cells_across(finest, below) < self.cells_per_radius
        if self.tolerance is not None:
            wanted |= self.flagged(finest, below) >= 2
        if wanted:
            self._surplus = 0
            if self.depth < self.max_depth:
                self._add_level()
            return

        if finest < self.seed_depth:
            return
        extent = self.sims[finest].grid.r_max
        surplus = True
        if self.cells_per_radius is not None:
            surplus &= self.cells_across(finest) / RATIO >= 2.0 * self.cells_per_radius
        if self.tolerance is not None:
            surplus &= finest - 1 in self.errors and self.flagged(finest - 1, extent, 0.25) < 2
        if surplus:
            self._surplus += 1
            if self._surplus >= self.patience:
                self._truncate(finest)
        else:
            self._surplus = 0

    def _add_level(self) -> None:
        """A new finest level over the inner half of the current one."""
        parent_sim, parent_state = self.sims[-1], self.states[-1]
        spacing = parent_sim.dr / RATIO
        grid = SphericalGrid(r_max=self.level_cells * spacing, n=self.level_cells)
        source = Level(parent_sim.grid, parent_state.Phi, parent_state.Pi)
        Phi, Pi = prolong(source, grid), prolong(source, grid, field="Pi")
        self.sims.append(ScalarCollapse(grid, courant=self.courant, dissipation=self.dissipation))
        self.states.append(SphericalState(parent_state.t, Phi, Pi))
        self._fresh.add(self.depth - 1)
        self.regrids.append((parent_state.t, self.depth - 1, "created", grid.r_max))

    def _truncate(self, target: int) -> None:
        t = self.states[target].t
        del self.sims[target:], self.states[target:]
        for level in range(target, target + 2):
            self.errors.pop(level, None)
            self._fresh.discard(level)
        self._surplus = 0
        self.regrids.append((t, target, "removed", 0.0))


__all__ = [
    "INTERFACE_CELLS",
    "Interface",
    "Subcycler",
    "level_slope",
    "normalised_metric",
    "step_with_interface",
]
