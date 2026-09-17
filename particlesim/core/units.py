"""Unit systems and conversions.

Internally the gravity code works in geometric units (G = c = 1) with an
explicit base length. Conversions to SI and Planck units go through this
module only (design doc ADR-003).

A dimension is a tuple of exponents ``(length, time, mass)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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


def dimension_mul(a: Dimension, b: Dimension) -> Dimension:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def dimension_div(a: Dimension, b: Dimension) -> Dimension:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dimension_pow(a: Dimension, n: float) -> Dimension:
    return (a[0] * n, a[1] * n, a[2] * n)


def format_dimension(d: Dimension) -> str:
    names = ("L", "T", "M")
    parts = [f"{n}^{e:g}" for n, e in zip(names, d, strict=True) if e != 0]
    return "dimensionless" if not parts else " ".join(parts)


class Quantity:
    """An array tagged with its physical dimension and unit system.

    The failure this prevents is specific and has cost real experiments:
    two arrays that look identical to NumPy, one in geometric units and one
    in SI, add without complaint and produce a number that is wrong by
    forty orders of magnitude while remaining finite, plottable and
    plausible. Here that addition raises.

    Operations follow physics rather than convenience. Addition requires
    the same dimension; if the systems differ the right operand is
    converted rather than rejected, because that conversion is
    unambiguous. Multiplication combines dimensions and requires a common
    system, since there is no single correct choice of output system.
    """

    __slots__ = ("value", "dim", "system")

    def __init__(self, value, dim: Dimension, system: UnitSystem):
        self.value = np.asarray(value, dtype=float)
        self.dim = tuple(float(e) for e in dim)  # type: ignore[assignment]
        self.system = system

    def __repr__(self) -> str:
        return (
            f"Quantity({np.array2string(self.value, threshold=6)}, "
            f"{format_dimension(self.dim)}, {self.system.name})"
        )

    def __array__(self, dtype=None, copy=None):
        """Bare NumPy access returns the raw magnitude in this system.

        This is deliberate and documented: once a Quantity is handed to
        NumPy the tagging is gone, so conversions must happen before that
        point, not after.
        """
        arr = self.value
        return arr.astype(dtype) if dtype is not None else arr

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    def to(self, system: UnitSystem) -> Quantity:
        return Quantity(self.system.convert(self.value, self.dim, system), self.dim, system)

    def _aligned(self, other: Quantity) -> np.ndarray:
        if self.dim != other.dim:
            raise ValueError(
                f"cannot combine {format_dimension(self.dim)} with "
                f"{format_dimension(other.dim)}: dimensions differ"
            )
        return other.to(self.system).value if other.system != self.system else other.value

    def __add__(self, other: Quantity) -> Quantity:
        if not isinstance(other, Quantity):
            raise TypeError("add Quantity to Quantity; a bare array has no units")
        return Quantity(self.value + self._aligned(other), self.dim, self.system)

    def __sub__(self, other: Quantity) -> Quantity:
        if not isinstance(other, Quantity):
            raise TypeError("subtract Quantity from Quantity; a bare array has no units")
        return Quantity(self.value - self._aligned(other), self.dim, self.system)

    def __mul__(self, other) -> Quantity:
        if isinstance(other, Quantity):
            self._require_same_system(other, "multiply")
            return Quantity(
                self.value * other.value, dimension_mul(self.dim, other.dim), self.system
            )
        return Quantity(self.value * np.asarray(other), self.dim, self.system)

    __rmul__ = __mul__

    def __truediv__(self, other) -> Quantity:
        if isinstance(other, Quantity):
            self._require_same_system(other, "divide")
            return Quantity(
                self.value / other.value, dimension_div(self.dim, other.dim), self.system
            )
        return Quantity(self.value / np.asarray(other), self.dim, self.system)

    def __pow__(self, n: float) -> Quantity:
        return Quantity(self.value**n, dimension_pow(self.dim, n), self.system)

    def __neg__(self) -> Quantity:
        return Quantity(-self.value, self.dim, self.system)

    def _require_same_system(self, other: Quantity, verb: str) -> None:
        if other.system != self.system:
            raise ValueError(
                f"cannot {verb} a quantity in {self.system.name} by one in "
                f"{other.system.name}: convert one with .to() first, since there is "
                "no single correct system for the result"
            )

    def is_dimensionless(self) -> bool:
        return self.dim == (0.0, 0.0, 0.0)

    def magnitude_in(self, system: UnitSystem) -> np.ndarray:
        return self.to(system).value
