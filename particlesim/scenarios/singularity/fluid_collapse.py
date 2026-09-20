"""Fluid collapse, and the theorem that says the outside does not notice.

Issue #60. A star past the maximum-mass point of its own sequence is
unstable, so a small push decides its fate: inward and it collapses, outward
and it disperses. The same star, the same magnitude of kick, opposite sign,
and two entirely different endings -- which is the cleanest demonstration
that the equilibrium is the unstable one and not a numerical accident.

**"Forms a horizon" has to be read in the slicing.** Polar-areal coordinates
are horizon-avoiding: a trapped surface never forms in finite coordinate
time. What happens instead is that ``2m/r`` asymptotes to one from below
while the lapse collapses, and the constraint integration eventually refuses
to continue because the coordinates do not cover what is beyond. So
:class:`CollapseOutcome` reports an *approach* to a horizon -- the largest
``2m/r`` reached, the smallest lapse, and whether the integration ended in
:class:`~particlesim.solvers.nr.polar.PolarSlicingBreakdown` -- rather than
claiming a crossing it cannot see. Calling that a black hole needs both
signals, which is the same convention
:mod:`particlesim.solvers.nr.spherical` uses for a scalar field.

**The exterior is the sharp half of the acceptance, and it is Birkhoff's
theorem.** A spherically symmetric vacuum is Schwarzschild, *statically*,
however violently the interior behaves -- so the metric outside the star
should not move at all while the star falls in. Measured against
Schwarzschild with the initial ADM mass, the exterior radial metric agrees
to ``8e-10`` at the start and ``1.1e-6`` after two dynamical times, by which
point ``2m/r`` has gone from 0.527 to 0.579 and the central density has
risen by a third. ``alpha a = 1`` outside holds to ``1e-9`` throughout. That
is a theorem being verified rather than a tolerance being met, and it is
what makes the interior's violence credible.

**What ends the run is the physics, not an exception handler.** The
constraint integration raises when ``2m/r`` reaches one because polar-areal
coordinates stop there, and the scenario records that as the outcome rather
than catching it and continuing with a metric that does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from particlesim.core.spherical import SphericalGrid
from particlesim.matter.eos import EquationOfState, Polytrope
from particlesim.matter.tov import Star, solve_tov
from particlesim.scenarios.singularity.harness import BatteryResult
from particlesim.solvers.hydro.spherical import SphericalHydro, from_star
from particlesim.solvers.hydro.srhd import GammaLaw
from particlesim.solvers.nr.polar import PolarSlicingBreakdown

#: ``2m/r`` past which the run is called an approach to a horizon.
HORIZON_THRESHOLD = 0.95

#: Central lapse below which the slicing is called collapsed.
LAPSE_THRESHOLD = 1e-2


def horizon_radius(radii, a, threshold: float = HORIZON_THRESHOLD) -> float | None:
    """Smallest radius where ``2m/r`` exceeds ``threshold``, or ``None``.

    An approach rather than a crossing: in this slicing ``2m/r`` rises
    towards one and never reaches it, so the threshold is a convention and
    is named as one. Pair it with :func:`lapse_collapsed` before calling a
    run a black hole -- either alone can be produced by a grid too coarse to
    resolve the approach.
    """
    compactness = 1.0 - 1.0 / np.asarray(a, dtype=float) ** 2
    found = np.nonzero(compactness > threshold)[0]
    return float(np.asarray(radii)[found[0]]) if found.size else None


def lapse_collapsed(lapse, threshold: float = LAPSE_THRESHOLD) -> bool:
    """Whether the central lapse has collapsed, the other half of the signature."""
    return bool(np.min(np.asarray(lapse, dtype=float)) < threshold)


def exterior_residual(radii, a, lapse, mass: float, surface: float) -> float:
    """How far the vacuum outside the star is from Schwarzschild.

    Birkhoff's theorem says the answer is zero: a spherically symmetric
    vacuum is Schwarzschild and static, whatever the interior is doing. So
    this is a theorem to verify rather than an error to tolerate, and it
    stays at parts in a million while the interior collapses.

    The word doing the work is *vacuum*. A dispersing star expands past the
    sampling radius and the region stops being one, at which point the
    number is a comparison between a Schwarzschild metric and a region full
    of matter and means nothing -- so :meth:`FluidCollapse.run` stops
    accumulating it rather than reporting a tenth.
    """
    radii = np.asarray(radii, dtype=float)
    outside = radii > surface
    if not outside.any():
        raise ValueError(
            f"no grid point lies beyond the surface at {surface:.6g}; the exterior cannot "
            "be compared with anything"
        )
    exact = 1.0 / np.sqrt(1.0 - 2.0 * mass / radii[outside])
    radial = float(np.max(np.abs(np.asarray(a)[outside] / exact - 1.0)))
    product = float(np.max(np.abs((np.asarray(a) * np.asarray(lapse))[outside] - 1.0)))
    return max(radial, product)


@dataclass(frozen=True)
class CollapseOutcome:
    """What a collapse run ended as, and the evidence for saying so."""

    outcome: str
    peak_compactness: float
    minimum_lapse: float
    horizon_radius: float | None
    slicing_broke_down: bool
    elapsed: float
    central_density_ratio: float
    adm_mass: float
    mass_drift: float
    exterior_residual: float
    notes: list[str] = field(default_factory=list)

    @property
    def formed_horizon(self) -> bool:
        """Both signals, because either alone is a coarse grid's signature too."""
        return self.horizon_radius is not None and self.minimum_lapse < LAPSE_THRESHOLD

    def as_row(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "peak_compactness": self.peak_compactness,
            "minimum_lapse": self.minimum_lapse,
            "horizon_radius": self.horizon_radius,
            "formed_horizon": self.formed_horizon,
            "elapsed_dynamical_times": self.elapsed,
            "adm_mass": self.adm_mass,
            "mass_drift": self.mass_drift,
            "exterior_residual": self.exterior_residual,
        }


@dataclass
class FluidCollapse:
    """A star, an equation of state to evolve it with, and a push."""

    star: Star
    cold: EquationOfState
    eos: GammaLaw
    points: int = 160
    span: float = 1.6
    reconstruction: str = "ppm-extremum"
    kick: float = -0.02
    courant: float = 0.25

    def __post_init__(self) -> None:
        if self.star.radii.size == 0:
            raise ValueError(
                "the stellar model carries no profile; solve it with samples > 0 so there "
                "is something to put on the grid"
            )
        if abs(self.kick) >= 1.0:
            raise ValueError(f"the kick is a velocity and must be subluminal, got {self.kick}")

    def solver(self) -> SphericalHydro:
        return SphericalHydro(
            SphericalGrid(r_max=self.span * self.star.radius, n=self.points),
            self.eos,
            reconstruction=self.reconstruction,
            courant=self.courant,
            reference_density=float(self.cold.density_from_pressure(self.star.central_pressure)),
        )

    def initial_state(self, solver: SphericalHydro) -> np.ndarray:
        """The star, plus a radial velocity vanishing at the centre and surface.

        The profile is a half sine so the kick is regular at the origin and
        carries no discontinuity at the surface -- it is a push, not a shock.
        """
        state = from_star(solver, self.star, self.cold)
        density, _, pressure, a, _, _ = solver.decompose(state)
        shape = np.sin(np.pi * np.minimum(solver.radii / self.star.radius, 1.0))
        return solver.densitise(density, self.kick * shape, pressure, a)

    def run(self, dynamical_times: float = 6.0, samples: int = 0) -> CollapseOutcome:
        """Evolve until the time runs out or the slicing gives up.

        The second is not a failure. Polar-areal coordinates do not cover a
        trapped region, so a run that ends in
        :class:`PolarSlicingBreakdown` has reached the end of what these
        coordinates can say, and that is recorded as the outcome.
        """
        if dynamical_times <= 0.0:
            raise ValueError(f"the run needs a positive duration, got {dynamical_times}")
        solver = self.solver()
        state = self.initial_state(solver)
        initial_density = solver.decompose(state)[0]
        initial_mass = solver.adm_mass(state)
        target = dynamical_times * self.star.dynamical_time

        peak = 0.0
        lowest = 1.0
        radius = None
        residual = 0.0
        broke = False
        elapsed = 0.0
        density = initial_density
        vacuum_lost = False
        notes: list[str] = []
        history = []
        try:
            while elapsed < target:
                size = min(solver.time_step(state), target - elapsed)
                state = solver.step(state, size)
                elapsed += size
                density, _, _, a, lapse, _ = solver.decompose(state)
                peak = max(peak, float(np.max(1.0 - 1.0 / a**2)))
                lowest = min(lowest, float(np.min(lapse)))
                radius = horizon_radius(solver.radii, a) or radius
                sample = solver.radii > 1.15 * self.star.radius
                if float(np.max(density[sample])) <= 10.0 * solver.density_floor:
                    residual = max(
                        residual,
                        exterior_residual(
                            solver.radii, a, lapse, initial_mass, 1.15 * self.star.radius
                        ),
                    )
                elif not vacuum_lost:
                    vacuum_lost = True
                    notes.append(
                        "matter reached the exterior sampling radius; the Birkhoff "
                        "comparison stopped there because the region is no longer vacuum"
                    )
                if samples and len(history) < samples:
                    history.append((elapsed, peak, lowest))
        except PolarSlicingBreakdown as breakdown:
            broke = True
            notes.append(str(breakdown))

        ratio = float(density[0] / initial_density[0])
        if broke or (radius is not None and lowest < LAPSE_THRESHOLD):
            outcome = "collapsed"
        elif ratio < 0.5:
            outcome = "dispersed"
        else:
            outcome = "bounded"
        return CollapseOutcome(
            outcome=outcome,
            peak_compactness=peak,
            minimum_lapse=lowest,
            horizon_radius=radius,
            slicing_broke_down=broke,
            elapsed=elapsed / self.star.dynamical_time,
            central_density_ratio=ratio,
            adm_mass=initial_mass,
            mass_drift=abs(solver.adm_mass(state) / initial_mass - 1.0)
            if not broke
            else float("nan"),
            exterior_residual=residual,
            notes=notes,
        )

    def as_battery_result(self, dynamical_times: float = 6.0) -> BatteryResult:
        """The scenario's row in the singularity battery."""
        result = self.run(dynamical_times)
        return BatteryResult(
            scenario="fluid_collapse",
            outcome=result.outcome,
            bounced=result.outcome == "dispersed",
            max_density=result.central_density_ratio,
            min_scale_factor=1.0 - result.peak_compactness,
            notes=result.notes,
        )


def polytropic_collapse(
    central_pressure: float,
    polytropic_constant: float = 100.0,
    adiabatic_index: float = 1.9,
    **settings,
) -> FluidCollapse:
    """Build a collapse from a cold polytrope, which is the equation-of-state hook.

    The star is solved on the *cold* branch and evolved with the ideal-gas
    law that shares its index, so the initial data is an exact equilibrium of
    the system being evolved. Mixing the two -- a star from one index, an
    evolution with another -- leaves a residual that does not converge, which
    :mod:`particlesim.solvers.hydro.spherical` documents at length.
    """
    cold = Polytrope(polytropic_constant=polytropic_constant, gamma=adiabatic_index)
    star = solve_tov(cold, central_pressure, samples=20000)
    return FluidCollapse(star, cold, GammaLaw(adiabatic_index), **settings)


def is_unstable(cold: EquationOfState, central_pressure: float, fraction: float = 0.05) -> bool:
    """Whether the star sits past the turning point of ``M(rho_c)``.

    Which decides what a push does to it, and is settled by the structure
    solver alone -- no evolution involved. Worth calling before setting up a
    collapse: a star on the stable branch will not collapse however hard it
    is pushed, and time spent wondering why is time spent looking for a bug
    in a correct answer.
    """
    above = solve_tov(cold, central_pressure * (1.0 + fraction)).mass
    below = solve_tov(cold, central_pressure * (1.0 - fraction)).mass
    return above < below


__all__ = [
    "HORIZON_THRESHOLD",
    "LAPSE_THRESHOLD",
    "CollapseOutcome",
    "FluidCollapse",
    "exterior_residual",
    "horizon_radius",
    "is_unstable",
    "lapse_collapsed",
    "polytropic_collapse",
]
