"""Unit systems and conversions.

Internally the gravity code works in geometric units (G = c = 1) with an
explicit base length. Conversions to SI and Planck units go through this
module only (design doc ADR-003).

A dimension is a tuple of exponents ``(length, time, mass)``.
"""

from __future__ import annotations

from dataclasses import dataclass

# CODATA 2018 values.
C_SI = 299_792_458.0  # m / s
G_SI = 6.674_30e-11  # m^3 / (kg s^2)
HBAR_SI = 1.054_571_817e-34  # J s
M_SUN_SI = 1.988_47e30  # kg

PLANCK_LENGTH_SI = (HBAR_SI * G_SI / C_SI**3) ** 0.5
PLANCK_TIME_SI = PLANCK_LENGTH_SI / C_SI
PLANCK_MASS_SI = (HBAR_SI * C_SI / G_SI) ** 0.5

Dimension = tuple[float, float, float]

LENGTH: Dimension = (1, 0, 0)
TIME: Dimension = (0, 1, 0)
MASS: Dimension = (0, 0, 1)
VELOCITY: Dimension = (1, -1, 0)
ENERGY: Dimension = (2, -2, 1)
ENERGY_DENSITY: Dimension = (-1, -2, 1)
DIMENSIONLESS: Dimension = (0, 0, 0)


@dataclass(frozen=True)
class UnitSystem:
    """A unit system defined by its base length, time, and mass in SI."""

    name: str
    length_m: float
    time_s: float
    mass_kg: float

    def factor_to_si(self, dim: Dimension) -> float:
        """Multiply a value in this system by this to get SI."""
        ell, t, m = dim
        return self.length_m**ell * self.time_s**t * self.mass_kg**m

    def to_si(self, value, dim: Dimension):
        return value * self.factor_to_si(dim)

    def from_si(self, value, dim: Dimension):
        return value / self.factor_to_si(dim)

    def convert(self, value, dim: Dimension, other: UnitSystem):
        """Convert ``value`` with dimension ``dim`` from this system to ``other``."""
        return other.from_si(self.to_si(value, dim), dim)


SI = UnitSystem("SI", 1.0, 1.0, 1.0)
PLANCK = UnitSystem("Planck", PLANCK_LENGTH_SI, PLANCK_TIME_SI, PLANCK_MASS_SI)


def geometric(base_length_m: float, name: str | None = None) -> UnitSystem:
    """Geometric units (G = c = 1) with the given base length in metres.

    Time unit is ``L / c`` and mass unit is ``L c^2 / G`` so that G = c = 1.
    """
    return UnitSystem(
        name or f"geometric(L={base_length_m:g} m)",
        base_length_m,
        base_length_m / C_SI,
        base_length_m * C_SI**2 / G_SI,
    )


def geometric_solar_mass() -> UnitSystem:
    """Geometric units where one unit of mass is one solar mass."""
    return geometric(G_SI * M_SUN_SI / C_SI**2, "geometric(M_sun)")


def is_geometric(system: UnitSystem, rtol: float = 1e-9) -> bool:
    """True when c and G are both unity in ``system``."""
    c = system.length_m / system.time_s / C_SI
    g = system.length_m**3 / (system.mass_kg * system.time_s**2) / G_SI
    return abs(c - 1) < rtol and abs(g - 1) < rtol
