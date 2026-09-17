"""Geodesic completeness probes (design doc Section 3.5).

A spacetime is singular, in the sense that matters, when a geodesic runs out
of spacetime at finite affine parameter. That is the definition the report
card should be using, rather than a curvature invariant growing large, and
this module measures it directly: launch a bundle, integrate each member to
an affine budget, and count how many arrive.

The measurement has a failure mode worth naming. An integrator that stops
because its step size collapsed looks exactly like a geodesic that ran out of
spacetime. The two are distinguished here by refining the tolerance: a real
incompleteness keeps its terminal affine parameter as the tolerance tightens,
while a numerical stall moves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from particlesim.analysis.geodesics import GeodesicIntegrator


@dataclass
class ProbeOutcome:
    """One geodesic's fate."""

    index: int
    complete: bool
    affine_reached: float
    message: str = ""


@dataclass
class CompletenessReport:
    budget: float
    outcomes: list[ProbeOutcome] = field(default_factory=list)

    @property
    def complete_fraction(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(o.complete for o in self.outcomes) / len(self.outcomes)

    @property
    def geodesically_complete(self) -> bool:
        return all(o.complete for o in self.outcomes) and bool(self.outcomes)

    @property
    def shortest_affine(self) -> float:
        return min((o.affine_reached for o in self.outcomes), default=0.0)

    def summary(self) -> dict[str, float | bool | int]:
        return {
            "budget": self.budget,
            "probes": len(self.outcomes),
            "complete_fraction": self.complete_fraction,
            "geodesically_complete": self.geodesically_complete,
            "shortest_affine": self.shortest_affine,
        }


def probe_bundle(
    integrator: GeodesicIntegrator,
    starts: Sequence[np.ndarray],
    velocities: Sequence[np.ndarray],
    budget: float = 100.0,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    samples: int = 64,
) -> CompletenessReport:
    """Integrate a bundle of geodesics and record which ones survive the budget.

    ``samples`` sets how finely the affine parameter is recorded, and it is
    not cosmetic: a probe that terminates early reports the last sample it
    reached, so too few samples make every incomplete geodesic look as though
    it died immediately, which loses the one number that says how far it got.
    """
    if len(starts) != len(velocities):
        raise ValueError("need one initial velocity per starting point")
    report = CompletenessReport(budget=budget)
    for i, (x0, u0) in enumerate(zip(starts, velocities, strict=True)):
        try:
            res = integrator.integrate(x0, u0, budget, n_out=samples, rtol=rtol, atol=atol)
        except Exception as exc:  # noqa: BLE001 - a probe failing is data, not a crash
            report.outcomes.append(ProbeOutcome(i, False, 0.0, repr(exc)))
            continue
        reached = float(res.affine[-1]) if res.affine.size else 0.0
        complete = bool(res.success) and reached >= budget * (1.0 - 1e-9)
        report.outcomes.append(ProbeOutcome(i, complete, reached, "" if complete else res.message))
    return report


def radial_infall_bundle(
    integrator: GeodesicIntegrator,
    radii: Sequence[float],
    inward_speed: float = 0.3,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Timelike probes falling inward from several radii, on the equator."""
    starts, velocities = [], []
    for r0 in radii:
        x0 = np.array([0.0, float(r0), np.pi / 2, 0.0])
        u0 = integrator.from_eulerian_velocity(x0, [-inward_speed, 0.0, 0.0])
        starts.append(x0)
        velocities.append(u0)
    return starts, velocities


def is_numerically_stalled(
    integrator: GeodesicIntegrator,
    x0: np.ndarray,
    u0: np.ndarray,
    budget: float,
    tolerances: Sequence[float] = (1e-8, 1e-11),
) -> bool:
    """Whether a termination moves when the integrator is tightened.

    A genuine incompleteness terminates at the same affine parameter however
    accurately it is integrated. A stall does not, and calling one the other
    would turn a solver limitation into a claim about spacetime.
    """
    reached = []
    for rtol in tolerances:
        res = integrator.integrate(x0, u0, budget, n_out=256, rtol=rtol, atol=rtol * 1e-2)
        reached.append(float(res.affine[-1]) if res.affine.size else 0.0)
    scale = max(max(reached), 1e-12)
    # The comparison is coarser than machine precision on purpose: the
    # recorded value is quantised by the sample spacing, so only a move larger
    # than that counts as the termination shifting.
    return abs(reached[-1] - reached[0]) / scale > 2.0 / 256
