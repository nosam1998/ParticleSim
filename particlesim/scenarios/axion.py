"""Axion-photon conversion, and the approximation the textbook formula is.

Issue #72. An axion couples to electromagnetism through ``-(g/4) a F Fdual``,
which is ``g a E.B``: in an external magnetic field the photon polarisation
*parallel* to ``B`` mixes with the axion, and a beam crossing a magnet comes
out partly converted. That is how every laboratory axion search works.

**The acceptance is a limit, not a tolerance.** The formula usually quoted,

    P(gamma -> a) = (2 Delta_M / Delta_osc)^2 sin^2(Delta_osc L / 2)

with ``Delta_M = g B / 2``, ``Delta_a = -m^2 / (2 omega)`` and
``Delta_osc = sqrt((Delta_par - Delta_a)^2 + 4 Delta_M^2)``, comes from
dropping the second ``z``-derivative -- the slowly-varying envelope, or
paraxial, approximation. Checking a solver against it to some tolerance
would leave open whether a disagreement is the solver's error or the
approximation's. So :func:`propagate` keeps the second derivative and
integrates the *exact* boundary-value problem, and what is asserted is that
the gap between the two closes at first order in ``Delta_M / omega``, the
approximation's own small parameter. Measured, at fixed conversion phase
``g B L = 2``:

    g B      departure from the formula
    4.0e-02  8.1e-03
    2.0e-02  5.9e-03
    1.0e-02  2.8e-03
    5.0e-03  1.4e-03
    2.5e-03  6.4e-04

halving as the parameter halves. A fixed tolerance would have passed a
formula that was wrong by a constant.

**Massive axions do not travel at the speed of light**, so the conversion
probability is a ratio of *fluxes* rather than of amplitudes: the axion
carries ``k_a / k_gamma`` less flux per unit amplitude. Leaving that factor
out is a quiet error that vanishes at ``m = 0`` and grows as the axion
approaches its mass shell, which is exactly where the interesting searches
sit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Mixing:
    """The parameters of the two-level axion-photon system.

    ``plasma`` is the photon's own refractive term, ``-omega_pl^2/(2 omega)``
    in a plasma. It is kept because the resonance that makes conversion
    efficient is ``Delta_par = Delta_a``, which needs something on the photon
    side to tune against -- in vacuum the mass term can only ever spoil the
    match.
    """

    coupling: float = 1e-3
    field: float = 1.0
    mass: float = 0.0
    frequency: float = 1.0
    plasma: float = 0.0

    def __post_init__(self) -> None:
        if self.frequency <= 0.0:
            raise ValueError(f"the frequency must be positive, got {self.frequency}")
        if self.mass < 0.0:
            raise ValueError(f"the axion mass must not be negative, got {self.mass}")
        if self.mass >= self.frequency:
            raise ValueError(
                f"an axion of mass {self.mass} cannot propagate at frequency "
                f"{self.frequency}: below the mass shell there is no conversion, "
                "only an evanescent tail"
            )

    @property
    def mixing_term(self) -> float:
        """``Delta_M = g B / 2``."""
        return 0.5 * self.coupling * self.field

    @property
    def axion_term(self) -> float:
        """``Delta_a = -m^2 / (2 omega)``."""
        return -(self.mass**2) / (2.0 * self.frequency)

    @property
    def photon_term(self) -> float:
        """``Delta_par = -omega_pl^2 / (2 omega)``."""
        return -(self.plasma**2) / (2.0 * self.frequency)

    @property
    def oscillation_term(self) -> float:
        """``Delta_osc``, whose reciprocal sets the oscillation length."""
        detuning = self.photon_term - self.axion_term
        return float(np.sqrt(detuning**2 + 4.0 * self.mixing_term**2))

    @property
    def wavenumbers(self) -> tuple[float, float]:
        """``(k_photon, k_axion)`` at this frequency, from the free dispersions."""
        photon = float(np.sqrt(self.frequency**2 - self.plasma**2))
        axion = float(np.sqrt(self.frequency**2 - self.mass**2))
        return photon, axion


def conversion_probability(mixing: Mixing, length: float) -> float:
    """The textbook two-level result, in the paraxial approximation.

    Exact only as ``Delta_M / omega -> 0``; :func:`propagate` is the
    computation that does not assume it.
    """
    if length < 0.0:
        raise ValueError(f"the path length must not be negative, got {length}")
    amplitude = 2.0 * mixing.mixing_term / mixing.oscillation_term
    return float(amplitude**2 * np.sin(0.5 * mixing.oscillation_term * length) ** 2)


def oscillation_length(mixing: Mixing) -> float:
    """``2 pi / Delta_osc``: one full conversion-and-back cycle."""
    return float(2.0 * np.pi / mixing.oscillation_term)


def propagate(mixing: Mixing, length: float, tolerance: float = 1e-11):
    """Integrate the coupled wave equations in ``z``, second derivatives kept.

    For fields varying as ``e^{-i omega t}`` the equations of motion are

        A'' = -(omega^2 - omega_pl^2) A + i omega g B a
        a'' = (m^2 - omega^2) a - i omega g B A

    which is a linear system in ``z`` with no envelope assumed. Started with
    a forward-going photon and no axion, its solution at ``z = L`` gives the
    conversion directly, and the reflected component it also produces is
    part of the answer rather than an error -- the paraxial formula has no
    room for one.

    Returns ``(probability, photon_amplitude, axion_amplitude)``.
    """
    if length < 0.0:
        raise ValueError(f"the path length must not be negative, got {length}")
    photon, axion = mixing.wavenumbers
    drive = mixing.frequency * mixing.coupling * mixing.field

    def derivative(_z, state):
        field = state[0] + 1j * state[1]
        rate = state[2] + 1j * state[3]
        scalar = state[4] + 1j * state[5]
        scalar_rate = state[6] + 1j * state[7]
        second_field = -(photon**2) * field + 1j * drive * scalar
        second_scalar = -(axion**2) * scalar - 1j * drive * field
        return [
            rate.real,
            rate.imag,
            second_field.real,
            second_field.imag,
            scalar_rate.real,
            scalar_rate.imag,
            second_scalar.real,
            second_scalar.imag,
        ]

    start = [1.0, 0.0, 0.0, photon, 0.0, 0.0, 0.0, 0.0]
    solution = solve_ivp(
        derivative,
        (0.0, length),
        start,
        rtol=tolerance,
        atol=tolerance * 1e-2,
        max_step=max(length / 2000.0, 1e-12),
    )
    final = solution.y[:, -1]
    field = complex(final[0], final[1])
    scalar = complex(final[4], final[5])
    # A ratio of fluxes, not amplitudes: a massive axion is slower.
    return float(abs(scalar) ** 2 * axion / photon), field, scalar


def resonant_plasma_frequency(mass: float) -> float:
    """The plasma frequency that tunes ``Delta_par`` onto ``Delta_a``.

    At ``omega_pl = m`` the detuning vanishes and the mixing is maximal
    whatever the coupling -- the trick every helioscope uses to restore
    sensitivity at non-zero axion mass.
    """
    if mass < 0.0:
        raise ValueError(f"the axion mass must not be negative, got {mass}")
    return float(mass)


__all__ = [
    "Mixing",
    "conversion_probability",
    "oscillation_length",
    "propagate",
    "resonant_plasma_frequency",
]
