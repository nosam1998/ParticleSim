"""Relativistic stellar structure, checked against the case that is exact.

Issue #58's acceptance -- a TOV star stable for ten dynamical times -- needs
a hydrodynamic evolution, which is issue #57's and #59's work, not this
one's. What this module does instead is *build* the star and hold the
integrator to two things that are exactly true, so that when an evolution
does arrive it starts from initial data known to be right rather than
plausible.

**The incompressible star is solvable in closed form.** For constant energy
density the Tolman-Oppenheimer-Volkoff equations integrate to

    p(r) = rho_0 [ sqrt(1 - 2 M r^2/R^3) - sqrt(1 - 2M/R) ]
                 / [ 3 sqrt(1 - 2M/R) - sqrt(1 - 2 M r^2/R^3) ]

so the integrator can be compared with arithmetic rather than with another
integrator. It agrees to one part in ``1e10`` in radius, mass *and* the whole
pressure profile, at compactness 0.2, 0.5 and 0.8 -- the last of which is
well beyond any real star.

**And that solution contains its own limit.** The central pressure diverges
when ``3 sqrt(1 - 2M/R) = 1``, which is ``2M/R = 8/9``: the Buchdahl bound.
No static star of any equation of state is more compact than that, so a
request for one is refused here rather than integrated into a singularity.

**The Newtonian limit is the second exact check, and it is a rate.** A
``Gamma = 2`` polytrope in Newtonian gravity is the ``n = 1`` Lane-Emden
case, whose solution is ``sin(xi)/xi`` and whose radius is

    R = sqrt(pi K / 2)

*independent of the central density*. The relativistic answer approaches it
as the star is made lighter, and the departure is **first order in the
compactness** with a coefficient near 0.7:

    rho_c      R          relative error   2M/R      ratio
    1e-3      10.047        1.98e-1        0.253     0.78
    1e-4      12.198        2.67e-2        0.0379    0.71
    1e-5      12.498        2.78e-3        0.00398   0.70
    1e-6      12.530        2.87e-4        0.00040   0.72

A tolerance would pass for any integrator that happened to land nearby. The
ratio holding at 0.7 across three decades is the statement that the
relativistic correction is the one general relativity predicts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from particlesim.matter.eos import EquationOfState, UniformDensity

#: ``2M/R`` above which no static star exists, for any equation of state.
BUCHDAHL_LIMIT = 8.0 / 9.0

#: Fraction of the central pressure at which the surface is declared.
SURFACE_FRACTION = 1e-12


@dataclass
class Star:
    """A solved stellar model, and the profile it came from."""

    radius: float
    mass: float
    central_pressure: float
    radii: np.ndarray = field(default_factory=lambda: np.empty(0))
    pressures: np.ndarray = field(default_factory=lambda: np.empty(0))
    masses: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def compactness(self) -> float:
        """``2M/R``, which the Buchdahl bound caps at ``8/9``."""
        return 2.0 * self.mass / self.radius

    @property
    def dynamical_time(self) -> float:
        """``sqrt(R^3/M)``: the timescale an evolution would be measured in."""
        return math.sqrt(self.radius**3 / self.mass)

    def as_row(self) -> dict:
        return {
            "radius": self.radius,
            "mass": self.mass,
            "compactness": self.compactness,
            "central_pressure": self.central_pressure,
            "dynamical_time": self.dynamical_time,
        }


def uniform_density_pressure(radii, radius: float, mass: float, density: float) -> np.ndarray:
    """The Schwarzschild interior solution: ``p(r)`` in closed form.

    Defined only for ``2M/R < 8/9``; past that the denominator changes sign
    and the "solution" describes negative pressure at the centre, which is
    the algebra reporting that no such star exists.
    """
    compactness = 2.0 * mass / radius
    if compactness >= BUCHDAHL_LIMIT:
        raise ValueError(
            f"the Schwarzschild interior solution needs 2M/R below {BUCHDAHL_LIMIT:.6f}, "
            f"got {compactness:.6f}; no static star is more compact than that"
        )
    radii = np.asarray(radii, dtype=float)
    surface = math.sqrt(1.0 - compactness)
    interior = np.sqrt(1.0 - 2.0 * mass * radii**2 / radius**3)
    return density * (interior - surface) / (3.0 * surface - interior)


def uniform_density_central_pressure(radius: float, mass: float, density: float) -> float:
    """``p_c`` of an incompressible star, which diverges at the Buchdahl bound."""
    return float(uniform_density_pressure(0.0, radius, mass, density))


def newtonian_polytrope_radius(polytropic_constant: float) -> float:
    """``sqrt(pi K / 2)``: the ``n = 1`` Lane-Emden radius, in geometric units.

    Independent of the central density, which is the peculiarity of
    ``Gamma = 2`` and the reason it makes such a clean check: an integrator
    that got the equations slightly wrong would not reproduce a radius that
    does not move.
    """
    if polytropic_constant <= 0.0:
        raise ValueError(f"the polytropic constant must be positive, got {polytropic_constant}")
    return math.sqrt(math.pi * polytropic_constant / 2.0)


def solve_tov(
    eos: EquationOfState,
    central_pressure: float,
    maximum_radius: float = 1e4,
    tolerance: float = 1e-10,
    samples: int = 0,
) -> Star:
    """Integrate the structure equations outward until the pressure vanishes.

    ``dm/dr = 4 pi r^2 e`` and
    ``dp/dr = -(e + p)(m + 4 pi r^3 p) / (r(r - 2m))``, in geometric units.
    The surface is found as an event on the pressure rather than by watching
    for a sign change, so the radius is interpolated by the integrator
    instead of being quantised to a step.
    """
    if central_pressure <= 0.0:
        raise ValueError(f"the central pressure must be positive, got {central_pressure}")

    def derivative(radius, state):
        mass, pressure = state
        if pressure <= 0.0:
            return [0.0, 0.0]
        energy = float(eos.energy_density_from_pressure(pressure))
        denominator = radius * (radius - 2.0 * mass)
        if denominator <= 0.0:
            raise ValueError(
                f"the metric function 1 - 2m/r vanished at r = {radius:.6g}; the model "
                "has collapsed inside its own horizon, which no static star does"
            )
        return [
            4.0 * math.pi * radius**2 * energy,
            -(energy + pressure) * (mass + 4.0 * math.pi * radius**3 * pressure) / denominator,
        ]

    def surface(_radius, state):
        return state[1] - central_pressure * SURFACE_FRACTION

    surface.terminal = True
    surface.direction = -1.0

    start = 1e-8
    central_energy = float(eos.energy_density_from_pressure(central_pressure))
    solution = solve_ivp(
        derivative,
        (start, maximum_radius),
        [4.0 / 3.0 * math.pi * start**3 * central_energy, central_pressure],
        events=surface,
        rtol=tolerance,
        atol=1e-16,
        dense_output=True,
    )
    if not solution.t_events[0].size:
        raise ValueError(
            f"the pressure had not fallen to the surface by r = {maximum_radius:.6g}; "
            "either the star is larger than the integration range or the equation of "
            "state has no surface"
        )
    radius = float(solution.t_events[0][0])
    mass = float(solution.y_events[0][0][0])

    if samples:
        grid = np.linspace(start, radius, int(samples))
        profile = solution.sol(grid)
        return Star(radius, mass, central_pressure, grid, profile[1], profile[0])
    return Star(radius, mass, central_pressure)


def solve_uniform_density(density: float, compactness: float, **kwargs) -> Star:
    """An incompressible star at a chosen ``2M/R``, integrated not assumed.

    The radius follows from the density and the compactness, the central
    pressure from the closed form, and the integration then has to reproduce
    both -- which is the check, rather than the construction.
    """
    if not 0.0 < compactness < BUCHDAHL_LIMIT:
        raise ValueError(
            f"compactness must lie in (0, {BUCHDAHL_LIMIT:.6f}), got {compactness}; "
            "the Buchdahl bound is where the central pressure diverges"
        )
    radius = math.sqrt(compactness / (8.0 / 3.0 * math.pi * density))
    mass = 4.0 / 3.0 * math.pi * density * radius**3
    central = uniform_density_central_pressure(radius, mass, density)
    return solve_tov(
        UniformDensity(density=density), central, maximum_radius=10.0 * radius, **kwargs
    )


def mass_radius_sequence(eos: EquationOfState, central_densities, **kwargs) -> list[Star]:
    """One star per central density, for an ``M(R)`` curve."""
    return [
        solve_tov(eos, float(eos.pressure(density)), **kwargs)
        for density in np.asarray(central_densities, dtype=float)
    ]


__all__ = [
    "BUCHDAHL_LIMIT",
    "SURFACE_FRACTION",
    "Star",
    "mass_radius_sequence",
    "newtonian_polytrope_radius",
    "solve_tov",
    "solve_uniform_density",
    "uniform_density_central_pressure",
    "uniform_density_pressure",
]
