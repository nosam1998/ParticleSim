"""Valencia hydrodynamics on a spherically symmetric curved background.

Issue #57's remaining task, and what issue #58's stability half was waiting
on. Polar-areal coordinates, the metric recovered from the constraints at
every stage rather than evolved, and the fluid evolved in conservative form
with the curvature terms as sources.

**The whole right-hand side vanishes on a solution of the
Tolman-Oppenheimer-Volkoff equation, and that is the check.** A static star
is an exact solution of this system, so every sign and factor in the flux
divergence and the three source terms has to cancel against the others. It
is not a tolerance: ``rhs(D)`` and ``rhs(tau)`` are *identically* zero
because the velocity is, and ``rhs(S_r)`` falls at the reconstruction's own
order. A single wrong sign anywhere leaves a residual that does not converge
at all, which is how the first version of this was caught disagreeing --
the star had been built with one adiabatic index and evolved with another,
and the residual sat flat at 1% while the grid was refined by eight.

**The static balance is the Tolman-Oppenheimer-Volkoff equation, written
out.** Setting the momentum right-hand side to zero at rest gives

    alpha p' + (e + p) alpha' = 0

and the polar slicing condition gives
``alpha'/alpha = (m + 4 pi r^3 p)/(r(r - 2m))``, whose product is exactly
the structure equation :func:`particlesim.matter.tov.solve_tov` integrates.
So the evolution and the stellar-structure solver are the same physics
reached two different ways, and the suite compares them rather than trusting
either.

**The metric derivatives are analytic, not differenced.** ``a'`` and
``alpha'`` appear in the sources, and both follow in closed form from the
constraints that produced ``a`` and ``alpha``:

    a'     = a^3 (4 pi r (tau + D) - m/r^2)
    alpha' = alpha [ m/(r^2 (1 - 2m/r)) + 4 pi r a^2 S^r_r ]

Differencing them instead would put a second, unrelated truncation error
into the source terms, and the static balance would then hold only to the
worse of the two.

**What is evolved is densitised.** ``sqrt(gamma) = a r^2`` multiplies the
conserved variables, which is what makes the update a plain difference of
fluxes and the total rest mass exact -- and it makes the mass constraint
self-contained, because ``tau + D = (E + Dh)/(a r^2)`` turns
``dm/dr = 4 pi r^2 (tau + D)`` into ``dm/dr = 4 pi (E + Dh) sqrt(1 - 2m/r)``,
which needs no recovery to integrate. The primitives are then recovered from
undensitised variables that are, component for component, the flat ones --
so :func:`particlesim.solvers.hydro.srhd.conserved_to_primitive` is reused
rather than rewritten, with the orthonormal velocity ``a v^r`` in place of
``v``.

**The origin needs no special flux.** The innermost face of a cell-centred
grid sits at ``r = 0``, where ``r^2`` multiplies every flux, so regularity
there is arithmetic rather than a boundary condition. The reconstruction
into the first cell uses parity -- density and pressure even, velocity odd
-- which is the only place the origin is mentioned at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from particlesim.core.spherical import SphericalGrid
from particlesim.solvers.hydro.reconstruct import GHOSTS, SCHEMES, reconstruct
from particlesim.solvers.hydro.riemann import SOLVERS, riemann_flux
from particlesim.solvers.hydro.srhd import (
    GammaLaw,
    characteristic_speeds,
    conserved_to_primitive,
    primitive_to_conserved,
)
from particlesim.solvers.nr.polar import Source, midpoints, solve_lapse, solve_mass

#: Rest-mass density of the atmosphere, as a fraction of the central density.
ATMOSPHERE = 1e-10

#: Largest orthonormal speed a reconstruction may produce.
VELOCITY_CEILING = 1.0 - 1e-12

#: Shu-Osher's third-order strong stability-preserving Runge-Kutta.
SSP_RK3 = ((0.0, 1.0), (0.75, 0.25), (1.0 / 3.0, 2.0 / 3.0))


@dataclass
class SphericalHydro:
    """A fluid in polar-areal spherical symmetry, with the metric constrained."""

    grid: SphericalGrid
    eos: GammaLaw = field(default_factory=GammaLaw)
    reconstruction: str = "minmod"
    solver: str = "hllc"
    courant: float = 0.25
    atmosphere: float = ATMOSPHERE
    reference_density: float = 1.0
    recovery_tolerance: float = 1e-13

    def __post_init__(self) -> None:
        if not self.grid.uniform:
            raise ValueError(
                "the metric constraints are integrated on a uniform radial grid; a graded "
                "grid would need the Runge-Kutta rewritten for a varying step"
            )
        if self.reconstruction not in SCHEMES:
            raise ValueError(
                f"unknown reconstruction {self.reconstruction!r}; expected one of {SCHEMES}"
            )
        if self.solver not in SOLVERS:
            raise ValueError(f"unknown Riemann solver {self.solver!r}; expected one of {SOLVERS}")
        if not 0.0 < self.courant <= 1.0:
            raise ValueError(f"the Courant number must lie in (0, 1], got {self.courant}")
        if self.reference_density <= 0.0:
            raise ValueError(
                f"the reference density must be positive, got {self.reference_density}; it is "
                "what the atmosphere is a fraction of"
            )

    # --- geometry -------------------------------------------------------

    @property
    def radii(self) -> np.ndarray:
        return self.grid.radii()

    @property
    def spacing(self) -> float:
        return float(self.grid.dr)

    @property
    def faces(self) -> np.ndarray:
        """``r`` at every face, from the origin outward. The first one is zero."""
        return np.arange(self.grid.n + 1) * self.spacing

    @property
    def density_floor(self) -> float:
        return self.atmosphere * self.reference_density

    def pressure_floor(self) -> float:
        """The atmosphere's pressure, kept far below its rest-mass energy."""
        return self.density_floor * self.atmosphere

    # --- the constraint solve -------------------------------------------

    def _mass(self, densitised_energy: np.ndarray) -> np.ndarray:
        """The Misner-Sharp mass. See :meth:`_mass_source` for the equation."""
        return solve_mass(self.radii, self.spacing, self._mass_source(densitised_energy))

    def _mass_source(self, densitised_energy: np.ndarray) -> Source:
        """``dm/dr = 4 pi (E + D_h) sqrt(1 - 2m/r)``, which needs no recovery.

        The factor is the whole reason to evolve densitised variables: the
        Eulerian energy density is the densitised one over ``a r^2``, and
        ``a`` is what the integration is solving for, so writing the source
        with ``sqrt(1 - 2m/r)`` in it removes the circularity rather than
        iterating around it.

        It is returned rather than consumed because the lapse solve wants it
        too. The slicing condition samples the mass at cell midpoints, and
        interpolating to those from values alone is wrong by 75% at the
        innermost one, where the mass is cubic in the radius. Handing the
        derivative over lets that interpolation be a Hermite fit, which is
        exact for a cubic.
        """
        total = 4.0 * np.pi * densitised_energy
        middle = midpoints(total)

        def slope(index, mid, radius, mass):
            value = middle[index] if mid else total[index]
            return value * math.sqrt(max(1.0 - 2.0 * mass / radius, 0.0))

        return slope

    def _lapse(
        self, mass: np.ndarray, radial_stress: np.ndarray, mass_slope: Source | None = None
    ) -> np.ndarray:
        """``d(ln alpha)/dr = (m + 4 pi r^3 S^r_r)/(r(r - 2m))``, the polar condition."""
        stress_mid = midpoints(radial_stress)

        def slope(index, mid, radius, value):
            stress = stress_mid[index] if mid else radial_stress[index]
            return (value + 4.0 * np.pi * radius**3 * stress) / (radius * (radius - 2.0 * value))

        return solve_lapse(self.radii, self.spacing, mass, slope, mass_slope)

    def decompose(self, state: np.ndarray):
        """``(rho, v, p, a, alpha, m)`` from the densitised conserved variables.

        The order is forced: the mass constraint first, because it alone can
        be integrated without knowing the primitives; then the undensitising,
        the recovery, and only then the lapse, whose source is the radial
        stress the recovery produced.
        """
        state = np.asarray(state, dtype=float)
        radii = self.radii
        mass_source = self._mass_source(state[0] + state[2])
        mass = solve_mass(radii, self.spacing, mass_source)
        a = 1.0 / np.sqrt(1.0 - 2.0 * mass / radii)
        volume = a * radii**2

        density, momentum, energy = self.regularise(
            state[0] / volume, state[1] / (volume * a), state[2] / volume
        )
        rest, velocity, pressure = conserved_to_primitive(
            density, momentum, energy, self.eos, self.recovery_tolerance
        )
        lapse = self._lapse(mass, momentum * velocity + pressure, mass_source)
        return rest, velocity, pressure, a, lapse, mass

    def regularise(self, density, momentum, energy):
        """Two floors, applied to the conserved variables before any recovery.

        Outside the star there is nothing to evolve and a great deal that can
        go wrong: a vacuum has no sound speed, and a scheme asked to advect a
        density of zero will produce a negative one. So the atmosphere is
        held at a fixed fraction of the central density -- a fraction, so a
        star and the same star rescaled behave identically -- and at rest.

        The second floor is the one that is easy to miss. The recovery has a
        root only if the energy can pay for the momentum: for a cold flow
        ``tau = sqrt(D^2 + S^2) - D`` exactly, and anything below that
        describes a fluid with negative internal energy. A single step that
        hands an atmosphere cell some momentum without the energy to carry it
        lands there, and without this floor the recovery does not misbehave
        -- it refuses, several thousand steps into a run, naming numbers that
        look perfectly ordinary. Both floors break conservation where they
        fire, which is why :meth:`floored_mass` reports how much.
        """
        at_floor = density <= self.density_floor
        density = np.where(at_floor, self.density_floor, density)
        momentum = np.where(at_floor, 0.0, momentum)
        # The margin above the cold limit has to exceed the round-off in the
        # limit itself, which is the mistake the first version made: an
        # absolute floor of 1e-23 sat far below the 5e-19 of cancellation in
        # ``sqrt(D^2 + S^2) - D``, so the floor fired and left the state
        # exactly at the limit, where the internal energy is zero and the
        # recovery has no root. Deep in a collapse, at a Lorentz factor of
        # nine, that is where the run stopped -- with a floor already in
        # place and doing nothing.
        cold = np.sqrt(density**2 + momentum**2) - density
        minimum = cold * (1.0 + 1e-12) + self.pressure_floor() / (self.eos.gamma - 1.0)
        return (
            density,
            momentum,
            np.where(
                at_floor,
                self.pressure_floor() / (self.eos.gamma - 1.0),
                np.maximum(energy, minimum),
            ),
        )

    def floored_mass(self, state: np.ndarray) -> float:
        """Rest mass the atmosphere floor adds on this state, over ``4 pi``."""
        state = np.asarray(state, dtype=float)
        mass = self._mass(state[0] + state[2])
        a = 1.0 / np.sqrt(1.0 - 2.0 * mass / self.radii)
        volume = a * self.radii**2
        raised = self.regularise(state[0] / volume, state[1] / (volume * a), state[2] / volume)[0]
        return float(np.sum(raised * volume - state[0]) * self.spacing)

    def regularised(self, state: np.ndarray) -> np.ndarray:
        """The same floors applied to the densitised state a stage produced."""
        state = np.asarray(state, dtype=float)
        mass = self._mass(state[0] + state[2])
        a = 1.0 / np.sqrt(1.0 - 2.0 * mass / self.radii)
        volume = a * self.radii**2
        density, momentum, energy = self.regularise(
            state[0] / volume, state[1] / (volume * a), state[2] / volume
        )
        return np.stack([volume * density, volume * a * momentum, volume * energy])

    # --- states ----------------------------------------------------------

    def densitise(self, density, velocity, pressure, a) -> np.ndarray:
        """``sqrt(gamma) (D, S_r, tau)`` from primitives and a known ``a``."""
        conserved = primitive_to_conserved(density, velocity, pressure, self.eos)
        volume = a * self.radii**2
        return np.stack([volume * conserved[0], volume * a * conserved[1], volume * conserved[2]])

    def totals(self, state: np.ndarray) -> np.ndarray:
        """The integrated conserved variables, over ``4 pi`` for the rest mass."""
        return np.sum(np.asarray(state), axis=1) * self.spacing

    def rest_mass(self, state: np.ndarray) -> float:
        return float(4.0 * np.pi * self.totals(state)[0])

    def adm_mass(self, state: np.ndarray) -> float:
        return float(self._mass(np.asarray(state)[0] + np.asarray(state)[2])[-1])

    # --- the right-hand side ---------------------------------------------

    def _pad(self, values: np.ndarray, odd: bool = False) -> np.ndarray:
        """Parity at the origin, zero gradient at the outer edge.

        The origin is not a boundary -- it is the centre of a sphere, and
        the field on the other side of it is the field on this side with the
        parity its rank demands. Copying instead would put a spurious
        gradient there and the star would breathe about it.
        """
        sign = -1.0 if odd else 1.0
        inner = sign * values[:GHOSTS][::-1]
        return np.concatenate([inner, values, np.full(GHOSTS, values[-1])])

    def _face_states(self, density, velocity, pressure):
        count = self.grid.n
        window = slice(GHOSTS - 1, count + GHOSTS)
        sides = ([], [])
        for values, odd in ((density, False), (velocity, True), (pressure, False)):
            low, high = reconstruct(self._pad(values, odd), self.reconstruction)
            sides[0].append(low[window])
            sides[1].append(high[window])
        built = []
        for side in sides:
            built.append(
                (
                    np.maximum(side[0], self.density_floor),
                    np.clip(side[1], -VELOCITY_CEILING, VELOCITY_CEILING),
                    np.maximum(side[2], self.pressure_floor()),
                )
            )
        return built[0], built[1]

    def rhs(self, state: np.ndarray) -> np.ndarray:
        density, velocity, pressure, a, lapse, mass = self.decompose(state)
        state = np.asarray(state, dtype=float)
        radii = self.radii
        volume = a * radii**2
        energy = (state[0] + state[2]) / volume
        momentum = state[1] / (volume * a)
        radial_stress = momentum * velocity + pressure

        left, right = self._face_states(density, velocity, pressure)
        flat = riemann_flux(left, right, self.eos, self.solver)
        face_radii = self.faces
        face_a = np.concatenate([[a[0]], midpoints(a), [a[-1]]])
        face_lapse = np.concatenate([[lapse[0]], midpoints(lapse), [lapse[-1]]])
        geometry = face_lapse * face_radii**2
        fluxes = np.stack([geometry * flat[0], geometry * face_a * flat[1], geometry * flat[2]])

        # Analytic metric derivatives: both follow from the constraints that
        # produced the metric, so the sources carry no second truncation error.
        slope_a = a**3 * (4.0 * np.pi * radii * energy - mass / radii**2)
        slope_lapse = lapse * (
            mass / (radii**2 * (1.0 - 2.0 * mass / radii))
            + 4.0 * np.pi * radii * a**2 * radial_stress
        )

        # ``K^r_r = 4 pi r S_r`` from the momentum constraint in polar
        # slicing, which vanishes for a star at rest and is what makes the
        # energy equation's source vanish there along with its flux.
        extrinsic = 4.0 * np.pi * radii * a * momentum
        source = np.stack(
            [
                np.zeros_like(radii),
                volume
                * (
                    lapse * (radial_stress * slope_a / a + 2.0 * pressure / radii)
                    - energy * slope_lapse
                ),
                volume * (lapse * radial_stress * extrinsic - momentum / a * slope_lapse),
            ]
        )
        return -(fluxes[:, 1:] - fluxes[:, :-1]) / self.spacing + source

    # --- stepping ---------------------------------------------------------

    def time_step(self, state: np.ndarray) -> float:
        density, velocity, pressure, a, lapse, _ = self.decompose(state)
        speeds = characteristic_speeds(density, velocity, pressure, self.eos)
        fastest = np.max(np.abs(np.stack(speeds)), axis=0)
        return float(self.courant * np.min(self.spacing * a / (lapse * fastest)))

    def step(self, state: np.ndarray, step_size: float) -> np.ndarray:
        """One SSP-RK3 step, with the floors applied to every stage.

        Applying them inside the recovery alone would leave the evolved
        state unphysical and merely hide it, so each stage's output is
        regularised before the next stage reads it.
        """
        updated = state
        for old, new in SSP_RK3:
            updated = self.regularised(
                old * state + new * (updated + step_size * self.rhs(updated))
            )
        return updated

    def run(self, state: np.ndarray, duration: float, steps: int | None = None) -> np.ndarray:
        if duration < 0.0:
            raise ValueError(f"the duration must not be negative, got {duration}")
        if steps is not None:
            if steps < 1:
                raise ValueError(f"a fixed-step run needs at least one step, got {steps}")
            size = duration / steps
            for _ in range(steps):
                state = self.step(state, size)
            return state
        elapsed = 0.0
        while elapsed < duration:
            size = min(self.time_step(state), duration - elapsed)
            state = self.step(state, size)
            elapsed += size
        return state


def from_star(solver: SphericalHydro, star, cold) -> np.ndarray:
    """Sample a solved stellar model onto the grid, densitised and at rest.

    ``a`` comes from the model's own enclosed mass rather than from a
    constraint solve, so the initial data is the structure solver's answer
    and not an approximation to it -- which is what makes the comparison
    afterwards a comparison rather than a round trip.
    """
    radii = solver.radii
    pressure = np.where(
        radii < star.radius, np.interp(radii, star.radii, star.pressures, right=0.0), 0.0
    )
    pressure = np.maximum(pressure, solver.pressure_floor())
    density = np.array([float(cold.density_from_pressure(value)) for value in pressure])
    density = np.maximum(density, solver.density_floor)
    enclosed = np.where(
        radii < star.radius,
        np.interp(radii, star.radii, star.masses, right=star.mass),
        star.mass,
    )
    a = 1.0 / np.sqrt(1.0 - 2.0 * enclosed / radii)
    return solver.densitise(density, np.zeros_like(radii), pressure, a)


__all__ = [
    "ATMOSPHERE",
    "SSP_RK3",
    "VELOCITY_CEILING",
    "SphericalHydro",
    "from_star",
]
