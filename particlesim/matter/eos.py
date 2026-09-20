"""Equations of state, and the two properties one has to be held to.

Issue #58. An equation of state is swappable, so the interface matters more
than any one member of it: the solver asks for pressure, energy density and
a sound speed, and everything else is the plugin's business. What makes the
interface testable is that two of those answers are not free.

**The first law.** Along a cold (isentropic) branch, ``dh = dp/rho`` exactly,
where ``h = 1 + eps + p/rho``. That is a statement relating three of the
quantities an equation of state returns, so a plugin whose ``specific_energy``
does not match its ``pressure`` fails it -- and nothing else would notice,
because each function on its own looks perfectly reasonable. The suite checks
it by finite differences on every plugin here.

**Causality.** ``c_s^2 = (dp/drho)/(de/drho)`` must not exceed one, and for a
polytrope it is not automatic: ``c_s^2 -> Gamma - 1`` as the density grows, so
any ``Gamma > 2`` goes superluminal above

    rho_max = [ (Gamma - 1) / (Gamma K (Gamma - 2)) ]^(1/(Gamma - 1))

which is exact, and where ``c_s^2`` equals one to eight decimals. A plugin
that returned 1.5 there without comment would be the failure worth guarding,
so :meth:`EquationOfState.causal_density_limit` reports the bound and
:meth:`check_causal` refuses past it.

**The tabulated reader builds its own energy.** Given ``(rho, p)`` pairs it
does not also read ``eps``: it integrates ``d eps = (p/rho^2) d rho``, which
is the first law again. So a table round-trips a polytrope to interpolation
accuracy rather than to whatever was in the third column, and a table whose
pressure is not monotonic is refused rather than interpolated through.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from particlesim.theories.base import Theory

#: Relative step used when a sound speed is taken by finite difference.
DERIVATIVE_STEP = 1e-6


class EquationOfState(ABC):
    """The cold branch: everything a stellar-structure solver needs.

    Written around the *cold* (isentropic) relation because that is what
    determines a star's structure. A thermal equation of state adds a second
    variable, and :class:`IdealGas` carries that as an extra rather than
    complicating this interface for the plugins that do not need it.
    """

    id: str = "abstract"
    provenance: str = ""
    validity_statement: str = "unrestricted"

    @abstractmethod
    def pressure(self, density):
        """``p(rho)`` on the cold branch."""

    @abstractmethod
    def specific_energy(self, density):
        """``eps(rho)``: internal energy per unit rest mass."""

    @abstractmethod
    def density_from_pressure(self, pressure):
        """``rho(p)``, the inverse the solver needs when integrating in ``p``."""

    def energy_density(self, density):
        """``e = rho (1 + eps)``, the source in Einstein's equations."""
        density = np.asarray(density, dtype=float)
        return density * (1.0 + self.specific_energy(density))

    def enthalpy(self, density):
        """``h = 1 + eps + p/rho``, the relativistic specific enthalpy."""
        density = np.asarray(density, dtype=float)
        return 1.0 + self.specific_energy(density) + self.pressure(density) / density

    def energy_density_from_pressure(self, pressure):
        """``e(p)``: what the stellar-structure integrator actually calls."""
        return self.energy_density(self.density_from_pressure(pressure))

    def sound_speed_squared(self, density):
        """``c_s^2 = (dp/drho)/(de/drho)``, by central difference by default.

        Overridden where a closed form exists. The numerical default is not
        laziness: it means a plugin that supplies only ``pressure`` and
        ``specific_energy`` still gets a sound speed consistent with them,
        rather than one that could silently disagree.
        """
        density = np.asarray(density, dtype=float)
        step = density * DERIVATIVE_STEP
        d_pressure = self.pressure(density + step) - self.pressure(density - step)
        d_energy = self.energy_density(density + step) - self.energy_density(density - step)
        return d_pressure / d_energy

    def causal_density_limit(self) -> float:
        """Density above which ``c_s^2`` exceeds one, or infinity if never."""
        return float("inf")

    def check_causal(self, density) -> None:
        """Refuse a density where the plugin is superluminal."""
        limit = self.causal_density_limit()
        if np.any(np.asarray(density, dtype=float) > limit):
            raise ValueError(
                f"{self.id} is superluminal above rho = {limit:.6g} and was asked for "
                f"{np.max(np.asarray(density, dtype=float)):.6g}; the sound speed there "
                "exceeds the speed of light, so the answer would not describe matter"
            )

    def describe(self) -> dict:
        return {
            "id": self.id,
            "provenance": self.provenance,
            "validity": self.validity_statement,
            "causal_density_limit": self.causal_density_limit(),
        }


@dataclass(frozen=True)
class Polytrope(EquationOfState):
    """``p = K rho^Gamma`` with ``eps = p/((Gamma-1) rho)``.

    The cold branch of an ideal gas, and the workhorse for stellar structure.
    Its specific energy is fixed by the first law rather than chosen, which
    is why the two cannot be varied independently.
    """

    polytropic_constant: float = 100.0
    gamma: float = 2.0
    id: str = "eos.polytrope"
    provenance: str = "cold polytrope, the isentrope of an ideal gas"

    def __post_init__(self) -> None:
        if self.polytropic_constant <= 0.0:
            raise ValueError(
                f"the polytropic constant must be positive, got {self.polytropic_constant}"
            )
        if self.gamma <= 1.0:
            raise ValueError(
                f"the adiabatic exponent must exceed one, got {self.gamma}; at one the "
                "specific energy is not defined and the gas has no restoring pressure"
            )

    @property
    def validity_statement(self) -> str:
        limit = self.causal_density_limit()
        if np.isfinite(limit):
            return f"causal only below rho = {limit:.6g}, where the sound speed reaches one"
        return "causal at every density"

    def pressure(self, density):
        return self.polytropic_constant * np.asarray(density, dtype=float) ** self.gamma

    def specific_energy(self, density):
        density = np.asarray(density, dtype=float)
        return self.pressure(density) / ((self.gamma - 1.0) * density)

    def density_from_pressure(self, pressure):
        pressure = np.asarray(pressure, dtype=float)
        return (pressure / self.polytropic_constant) ** (1.0 / self.gamma)

    def sound_speed_squared(self, density):
        """``Gamma p / (rho + Gamma p/(Gamma-1))`` in closed form."""
        density = np.asarray(density, dtype=float)
        pressure = self.pressure(density)
        return self.gamma * pressure / (density + self.gamma * pressure / (self.gamma - 1.0))

    def causal_density_limit(self) -> float:
        """Exact, and infinite for ``Gamma <= 2``.

        ``c_s^2`` rises to ``Gamma - 1`` as the density grows, so a polytrope
        stiffer than ``Gamma = 2`` is superluminal somewhere. Below that it
        approaches one from beneath and never crosses.
        """
        if self.gamma <= 2.0:
            return float("inf")
        numerator = self.gamma - 1.0
        denominator = self.gamma * self.polytropic_constant * (self.gamma - 2.0)
        return float((numerator / denominator) ** (1.0 / (self.gamma - 1.0)))


@dataclass(frozen=True)
class IdealGas(Polytrope):
    """A Gamma-law gas: the polytrope above, plus a thermal branch.

    ``p = (Gamma - 1) rho eps`` with ``eps`` free is what an evolution needs;
    the cold branch it inherits is the isentrope of the same gas, which is
    what a stellar-structure solver needs. Keeping both in one plugin is the
    point -- a star built on the cold branch and evolved on the thermal one
    has to be the same gas.
    """

    id: str = "eos.ideal_gas"
    provenance: str = "Gamma-law ideal gas with its own isentrope as the cold branch"

    def thermal_pressure(self, density, specific_energy):
        """``p = (Gamma - 1) rho eps``, with ``eps`` an independent variable."""
        density = np.asarray(density, dtype=float)
        return (self.gamma - 1.0) * density * np.asarray(specific_energy, dtype=float)

    def thermal_sound_speed_squared(self, density, specific_energy):
        """``Gamma (Gamma-1) eps / (1 + Gamma eps)``."""
        specific_energy = np.asarray(specific_energy, dtype=float)
        return (
            self.gamma * (self.gamma - 1.0) * specific_energy / (1.0 + self.gamma * specific_energy)
        )


@dataclass(frozen=True)
class UniformDensity(EquationOfState):
    """Incompressible matter: ``e = rho_0`` whatever the pressure.

    Not a realistic star -- the sound speed is infinite, which the validity
    statement says plainly -- but the one case where the relativistic
    stellar-structure equations have a closed-form solution, so it is how the
    integrator in :mod:`particlesim.matter.tov` is checked against arithmetic
    rather than against another integrator.
    """

    density: float = 1e-3
    id: str = "eos.uniform"
    provenance: str = "incompressible matter; the Schwarzschild interior solution"
    validity_statement = (
        "incompressible, so the sound speed is unbounded and no signal speed is "
        "respected; kept because it is the exactly solvable case"
    )

    def __post_init__(self) -> None:
        if self.density <= 0.0:
            raise ValueError(f"the density must be positive, got {self.density}")

    def pressure(self, density):
        raise NotImplementedError(
            "incompressible matter has no p(rho): the density is fixed and the pressure "
            "is whatever the structure equations require"
        )

    def specific_energy(self, density):
        return np.zeros_like(np.asarray(density, dtype=float))

    def density_from_pressure(self, pressure):
        return np.full_like(np.asarray(pressure, dtype=float), self.density)

    def energy_density_from_pressure(self, pressure):
        return np.full_like(np.asarray(pressure, dtype=float), self.density)

    def sound_speed_squared(self, density):
        return np.full_like(np.asarray(density, dtype=float), np.inf)


@dataclass(frozen=True)
class PiecewisePolytrope(EquationOfState):
    """Polytropic segments joined so that the pressure is continuous.

    Only the first ``polytropic_constant`` is free: continuity at each
    transition fixes the next one as ``K_(i+1) = K_i rho_i^(G_i - G_(i+1))``,
    and continuity of the specific energy fixes an additive constant per
    segment. Both are *derived* here rather than supplied, because a table
    of independently chosen constants is a discontinuous equation of state
    that still evaluates.
    """

    transitions: tuple[float, ...] = (1e-3,)
    exponents: tuple[float, ...] = (1.5, 2.5)
    polytropic_constant: float = 100.0
    id: str = "eos.piecewise_polytrope"
    provenance: str = "piecewise polytrope with pressure and energy continuity imposed"

    def __post_init__(self) -> None:
        if len(self.exponents) != len(self.transitions) + 1:
            raise ValueError(
                f"{len(self.transitions)} transitions need {len(self.transitions) + 1} "
                f"exponents, got {len(self.exponents)}"
            )
        if any(gamma <= 1.0 for gamma in self.exponents):
            raise ValueError(f"every exponent must exceed one, got {self.exponents}")
        if list(self.transitions) != sorted(self.transitions) or any(
            value <= 0.0 for value in self.transitions
        ):
            raise ValueError(
                f"transition densities must be positive and increasing, got {self.transitions}"
            )

    @property
    def constants(self) -> tuple[float, ...]:
        """``K_i`` per segment, with continuity of pressure imposed."""
        values = [float(self.polytropic_constant)]
        for index, boundary in enumerate(self.transitions):
            previous, following = self.exponents[index], self.exponents[index + 1]
            values.append(values[-1] * boundary ** (previous - following))
        return tuple(values)

    @property
    def offsets(self) -> tuple[float, ...]:
        """The additive constant in ``eps`` per segment, from continuity."""
        constants = self.constants
        values = [0.0]
        for index, boundary in enumerate(self.transitions):
            previous, following = self.exponents[index], self.exponents[index + 1]
            before = values[-1] + constants[index] * boundary ** (previous - 1.0) / (previous - 1.0)
            after = constants[index + 1] * boundary ** (following - 1.0) / (following - 1.0)
            values.append(before - after)
        return tuple(values)

    def _segment(self, density):
        return np.searchsorted(np.asarray(self.transitions, dtype=float), density, side="right")

    def pressure(self, density):
        density = np.asarray(density, dtype=float)
        index = self._segment(density)
        constants = np.asarray(self.constants)[index]
        exponents = np.asarray(self.exponents)[index]
        return constants * density**exponents

    def specific_energy(self, density):
        density = np.asarray(density, dtype=float)
        index = self._segment(density)
        constants = np.asarray(self.constants)[index]
        exponents = np.asarray(self.exponents)[index]
        offsets = np.asarray(self.offsets)[index]
        return offsets + constants * density ** (exponents - 1.0) / (exponents - 1.0)

    def density_from_pressure(self, pressure):
        pressure = np.asarray(pressure, dtype=float)
        boundaries = np.asarray([self.pressure(value) for value in self.transitions])
        index = np.searchsorted(boundaries, pressure, side="right")
        constants = np.asarray(self.constants)[index]
        exponents = np.asarray(self.exponents)[index]
        return (pressure / constants) ** (1.0 / exponents)


@dataclass
class Tabulated(EquationOfState):
    """A table of ``(rho, p)``, with the specific energy integrated from it.

    The third column is not read even when a table has one. ``eps`` follows
    from ``d eps = (p/rho^2) d rho``, so what comes back is thermodynamically
    consistent with the pressure by construction rather than by trust, and a
    table that round-trips a polytrope recovers ``p/((Gamma-1) rho)`` to
    interpolation accuracy.

    Interpolation is in log-log, because a tabulated equation of state spans
    decades and a linear interpolant between two points a decade apart is
    wrong by tens of percent in the middle.
    """

    densities: np.ndarray = field(default_factory=lambda: np.logspace(-6, -2, 64))
    pressures: np.ndarray = field(default_factory=lambda: 100.0 * np.logspace(-6, -2, 64) ** 2)
    id: str = "eos.tabulated"
    provenance: str = "tabulated equation of state with energy integrated from the first law"
    validity_statement: str = "valid only between the first and last tabulated density"

    def __post_init__(self) -> None:
        self.densities = np.asarray(self.densities, dtype=float)
        self.pressures = np.asarray(self.pressures, dtype=float)
        if self.densities.shape != self.pressures.shape or self.densities.ndim != 1:
            raise ValueError(
                f"densities and pressures must be one-dimensional and the same length, got "
                f"{self.densities.shape} and {self.pressures.shape}"
            )
        if self.densities.size < 2:
            raise ValueError("a table needs at least two points to interpolate between")
        if np.any(np.diff(self.densities) <= 0.0):
            raise ValueError("tabulated densities must be strictly increasing")
        if np.any(np.diff(self.pressures) <= 0.0):
            raise ValueError(
                "tabulated pressures must be strictly increasing: a table that falls "
                "somewhere has a negative sound speed there and is not matter"
            )
        if np.any(self.densities <= 0.0) or np.any(self.pressures <= 0.0):
            raise ValueError("log interpolation needs positive densities and pressures")
        self._energies = self._integrate_energy()

    def _integrate_energy(self) -> np.ndarray:
        """``eps`` from ``d eps = (p/rho^2) d rho``, cumulative from the first point."""
        integrand = self.pressures / self.densities**2
        steps = np.diff(self.densities)
        increments = 0.5 * (integrand[1:] + integrand[:-1]) * steps
        return np.concatenate([[0.0], np.cumsum(increments)])

    def _interpolate(self, values, table) -> np.ndarray:
        return np.exp(
            np.interp(
                np.log(np.asarray(values, dtype=float)), np.log(self.densities), np.log(table)
            )
        )

    def pressure(self, density):
        return self._interpolate(density, self.pressures)

    def specific_energy(self, density):
        density = np.asarray(density, dtype=float)
        return np.interp(np.log(density), np.log(self.densities), self._energies)

    def density_from_pressure(self, pressure):
        pressure = np.asarray(pressure, dtype=float)
        return np.exp(np.interp(np.log(pressure), np.log(self.pressures), np.log(self.densities)))

    @classmethod
    def from_polytrope(cls, polytrope: Polytrope, densities) -> Tabulated:
        """Sample a polytrope, for the round-trip the suite checks."""
        densities = np.asarray(densities, dtype=float)
        return cls(densities=densities, pressures=np.asarray(polytrope.pressure(densities)))


class EOSTheory(Theory):
    """An equation of state as a plugin, so it can sit in a :class:`TheoryStack`.

    Tier C: there is no action here and no general-relativistic limit to
    recover, because an equation of state is a constitutive relation rather
    than a theory of gravity. What the stack checks of it is that its frame
    and dimension match the gravity it is composed with, which is exactly the
    consistency an equation of state *can* violate -- a relation written
    against the Jordan metric and evaluated on the Einstein one is wrong by
    the conformal factor, and both halves look fine on their own.

    Build one with :func:`as_plugin` rather than subclassing, so the physics
    stays in the :class:`EquationOfState` and only the wiring is here.
    """

    tier = "C"
    matter: EquationOfState = None

    def gr_limit(self) -> dict[str, float]:
        """Empty: a constitutive relation has no coupling to switch off."""
        return {}

    def observable_predictions(self) -> dict:
        return dict(self.matter.describe())


def as_plugin(equation_of_state: EquationOfState, frame: str = "einstein") -> type[Theory]:
    """Wrap an equation of state so a :class:`TheoryStack` will accept it."""

    class Wrapped(EOSTheory):
        id = equation_of_state.id
        matter = equation_of_state
        provenance = equation_of_state.provenance
        validity_statement = equation_of_state.validity_statement

    Wrapped.frame = frame
    return Wrapped


__all__ = [
    "DERIVATIVE_STEP",
    "EOSTheory",
    "EquationOfState",
    "IdealGas",
    "PiecewisePolytrope",
    "Polytrope",
    "Tabulated",
    "UniformDensity",
    "as_plugin",
]
