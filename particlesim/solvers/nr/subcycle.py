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
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from particlesim.core.interpolate import hermite
from particlesim.solvers.nr.hierarchy import RATIO, Hierarchy, Level, _interpolate, restrict
from particlesim.solvers.nr.spherical import ScalarCollapse, SphericalState

#: Cells at a level's outer edge driven by its parent rather than evolved.
#: Two is what the fourth-order centred stencil reaches for.
INTERFACE_CELLS = 2

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

    def metric_at(self, fraction: float) -> Metric:
        """The parent's metric at that instant, with its lapse properly normalised."""
        if fraction not in self._metrics:
            self._metrics[fraction] = normalised_metric(
                self.sim, self.state_at(fraction), self.upstream, self.upstream_fraction(fraction)
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
) -> Metric:
    """``(a, alpha)`` on one level, with the lapse taken from the parent at the edge.

    The level's own solve gets ``a`` right and normalises ``alpha a -> 1`` at
    its own outermost cell. The shape of the lapse inside the level is right
    too, since ``d(ln alpha)/dr`` is local; only the constant is wrong, so the
    whole profile is rescaled until its outermost value is the parent's there.
    The coarsest level has no parent and is normalised against the asymptotic
    boundary, which is the one place that normalisation is true.
    """
    a, alpha = sim.solve_metric(state.Phi, state.Pi)
    if interface is None:
        return a, alpha
    edge = interface.lapse_at(fraction, float(sim.r[-1]))
    return a, alpha * (edge / alpha[-1])


def level_slope(
    sim: ScalarCollapse,
    state: SphericalState,
    interface: Interface | None,
    fraction: float,
) -> tuple[SphericalState, tuple[np.ndarray, np.ndarray]]:
    """The level's state with its interface cells filled, and its right-hand side.

    Without an interface the level is the outermost one: its own metric
    solve is correct and its own outgoing condition applies.
    """
    if interface is None:
        return state, sim.rhs(state)
    phi, pi = interface.fields_at(fraction, sim.r[-INTERFACE_CELLS:])
    Phi, Pi = np.array(state.Phi), np.array(state.Pi)
    Phi[-INTERFACE_CELLS:], Pi[-INTERFACE_CELLS:] = phi, pi
    driven = state.with_fields(Phi, Pi)
    dPhi, dPi = sim.rhs(driven, normalised_metric(sim, driven, interface, fraction))
    # The interface cells are prescribed at every stage, so they must not
    # also integrate on their own.
    dPhi[-INTERFACE_CELLS:] = 0.0
    dPi[-INTERFACE_CELLS:] = 0.0
    return driven, (dPhi, dPi)


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


class Subcycler:
    """A :class:`~particlesim.solvers.nr.hierarchy.Hierarchy`, evolved in time.

    One call to :meth:`step` advances the coarsest level by its own Courant
    step and every finer level by ``RATIO`` times as many steps of its own.
    """

    def __init__(
        self,
        hierarchy: Hierarchy,
        courant: float = 0.25,
        dissipation: float | None = None,
    ):
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
        after = step_with_interface(sim, before, dt, interface, offset)

        if index + 1 < self.depth:
            span = dt / interface.step if interface is not None else 1.0
            child = Interface(
                sim=sim,
                before=before,
                after=after,
                slope_before=level_slope(sim, before, interface, offset)[1],
                slope_after=level_slope(sim, after, interface, offset + span)[1],
                step=dt,
                upstream=interface,
                offset=offset,
                span=span,
            )
            for k in range(RATIO):
                self._advance(index + 1, dt / RATIO, child, k / RATIO)
            after = self._restrict(index, after)

        self.states[index] = after

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
        Phi, Pi = np.array(parent.Phi), np.array(parent.Pi)
        Phi[keep], Pi[keep] = phi[keep], pi[keep]
        return parent.with_fields(Phi, Pi)


__all__ = [
    "INTERFACE_CELLS",
    "Interface",
    "Subcycler",
    "level_slope",
    "normalised_metric",
    "step_with_interface",
]
