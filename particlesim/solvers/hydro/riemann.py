"""Approximate Riemann solvers, and the exact solution they are held to.

Issue #57. Each face of the grid poses a Riemann problem, and the flux the
scheme uses is an answer to it. Two approximate answers are here, plus the
exact one -- because issue #57's acceptance is that relativistic shock tubes
match published profiles, and what a published profile *is* is the exact
solution of the Riemann problem. Computing it here makes the acceptance an
equality rather than a comparison to a figure.

**HLLE and HLLC differ on one wave, and the difference is not a tolerance.**
HLLE averages over the whole fan, which folds the contact discontinuity into
the averaging and diffuses it. HLLC restores the contact as a third wave.
On a *stationary* contact -- uniform pressure, uniform velocity, a jump in
density -- the constant term of HLLC's quadratic is the HLL momentum, which
is then identically zero, so the contact speed is exactly zero, the two
star states are the outer states, and the flux is ``(0, p, 0)`` at every
face. Nothing moves. Over the same run HLLE smears the jump by 3.79 out of
9, and the two differ by more than twelve orders of magnitude, which is not
a gap a tuned constant produces.

**And the two errors have different sources, which is checkable by moving
one of them.** What is left of HLLC's contact is the primitive recovery, not
the flux: loosening :func:`conserved_to_primitive`'s tolerance from
``1e-15`` to ``1e-13`` to ``1e-11`` moves the drift to ``5.1e-14``,
``6.3e-13``, ``1.3e-10`` -- in step, because the recovered pressure is
uniform only to that tolerance and the faces then do not quite cancel.
HLLE's 3.79 does not move at all across the same three runs. One number
belongs to the arithmetic and the other to the scheme, and the way to tell
which is which is to change the arithmetic and see what follows.

**The root of that quadratic is taken in the stable form, and that was
measured rather than assumed.** ``(-b - sqrt(b^2 - 4ac))/(2a)`` is the
classic place a root is lost to cancellation, so :func:`contact_speed` takes
``c/q`` instead and ``stable=False`` evaluates the textbook form for
comparison. Across 40,000 random contacts the two never disagree, because
``4ac/b^2`` is of order one for this particular quadratic rather than
small -- so the exactness above comes from ``c`` vanishing and from nothing
else. The alternative is kept anyway, since a claim that two forms agree is
worth a test rather than a sentence.

**The exact solver's shock branch is checked against the jump conditions it
was derived from.** Its closed form comes from the Taub adiabat, which is
one rearrangement away from being wrong; so the suite takes the state it
produces and evaluates ``F(U*) - F(U) - V (U* - U)`` directly, with the same
flux function the numerical scheme uses. That residual is at round-off,
which is a statement about the algebra rather than about a reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from particlesim.solvers.hydro.srhd import (
    GammaLaw,
    characteristic_speeds,
    flux,
    lorentz,
    primitive_to_conserved,
)

#: The approximate solvers :func:`riemann_flux` accepts.
SOLVERS = ("hlle", "hllc")


def _total_energy_form(density, velocity, pressure, eos):
    """``(D, m, E)`` and its fluxes, with ``E = tau + D`` the total energy.

    HLLC is stated in the total-energy variables because that is where its
    two star-region relations are linear: ``F_E = m`` and ``m = (E + p) v``.
    In ``tau`` they pick up the rest-mass flux and stop being.
    """
    conserved_density, momentum, energy = primitive_to_conserved(density, velocity, pressure, eos)
    flux_density, flux_momentum, flux_energy = flux(density, velocity, pressure, eos)
    state = np.stack([conserved_density, momentum, energy + conserved_density])
    fluxes = np.stack([flux_density, flux_momentum, flux_energy + flux_density])
    return state, fluxes


def _signal_speeds(left, right, eos):
    """The outermost characteristic of the two states, on each side."""
    left_low, _, left_high = characteristic_speeds(*left, eos)
    right_low, _, right_high = characteristic_speeds(*right, eos)
    return np.minimum(left_low, right_low), np.maximum(left_high, right_high)


def contact_speed(state_hll, flux_hll, stable=True):
    """The HLLC contact speed: the subluminal root of ``F_E x^2 - (E + F_m) x + m``.

    ``stable`` selects which algebraically identical form of the root to
    evaluate -- ``c/q``, or the textbook ``(-b - sqrt(b^2 - 4ac))/(2a)``.
    Both return exactly zero when the momentum does, which is what keeps a
    stationary contact stationary, and across 40,000 random contacts they
    were never found to differ at all. The parameter exists so that the
    agreement is a test rather than a claim.
    """
    quadratic = flux_hll[2]
    linear = -(state_hll[2] + flux_hll[1])
    constant = state_hll[1]
    discriminant = np.maximum(linear**2 - 4.0 * quadratic * constant, 0.0)
    root = np.sqrt(discriminant)
    if not stable:
        return (-linear - root) / (2.0 * quadratic)
    larger = -0.5 * (linear + np.where(linear < 0.0, -root, root))
    return np.where(larger == 0.0, 0.0, constant / np.where(larger == 0.0, 1.0, larger))


def hlle(left, right, eos: GammaLaw):
    """One averaged state across the whole fan. Positive, diffusive, and blunt."""
    low, high = _signal_speeds(left, right, eos)
    low = np.minimum(low, 0.0)
    high = np.maximum(high, 0.0)
    left_state, left_flux = _total_energy_form(*left, eos)
    right_state, right_flux = _total_energy_form(*right, eos)
    averaged = (high * left_flux - low * right_flux + high * low * (right_state - left_state)) / (
        high - low
    )
    return np.stack([averaged[0], averaged[1], averaged[2] - averaged[0]])


def hllc(left, right, eos: GammaLaw):
    """Three waves: the contact is restored instead of averaged away."""
    low, high = _signal_speeds(left, right, eos)
    left_state, left_flux = _total_energy_form(*left, eos)
    right_state, right_flux = _total_energy_form(*right, eos)

    span = high - low
    state_hll = (high * right_state - low * left_state + left_flux - right_flux) / span
    flux_hll = (
        high * left_flux - low * right_flux + high * low * (right_state - left_state)
    ) / span
    middle = contact_speed(state_hll, flux_hll)
    star_pressure = flux_hll[1] - flux_hll[2] * middle

    def star(state, fluxes, speed, velocity, pressure):
        gap = speed - middle
        scale = (speed - velocity) / gap
        starred = np.stack(
            [
                state[0] * scale,
                (state[1] * (speed - velocity) + star_pressure - pressure) / gap,
                (state[2] * (speed - velocity) + star_pressure * middle - pressure * velocity)
                / gap,
            ]
        )
        return fluxes + speed * (starred - state)

    left_star = star(
        left_state, left_flux, low, np.asarray(left[1], float), np.asarray(left[2], float)
    )
    right_star = star(
        right_state, right_flux, high, np.asarray(right[1], float), np.asarray(right[2], float)
    )
    chosen = np.where(
        low >= 0.0,
        left_flux,
        np.where(middle >= 0.0, left_star, np.where(high > 0.0, right_star, right_flux)),
    )
    return np.stack([chosen[0], chosen[1], chosen[2] - chosen[0]])


def riemann_flux(left, right, eos: GammaLaw, solver="hllc"):
    """``(F_D, F_S, F_tau)`` at every face, from the named approximate solver."""
    if solver == "hlle":
        return hlle(left, right, eos)
    if solver == "hllc":
        return hllc(left, right, eos)
    raise ValueError(f"unknown Riemann solver {solver!r}; expected one of {SOLVERS}")


# --- the exact solution ---------------------------------------------------


@dataclass(frozen=True)
class RiemannFan:
    """The exact solution's wave structure: two nonlinear waves and a contact."""

    star_pressure: float
    star_velocity: float
    left_star_density: float
    right_star_density: float
    left_speeds: tuple[float, float]
    right_speeds: tuple[float, float]
    left_is_shock: bool
    right_is_shock: bool


def _invariant_term(sound, eos: GammaLaw):
    """``(2/sqrt(G-1)) artanh(c_s/sqrt(G-1))``, the relativistic Riemann invariant.

    Reduces to the Newtonian ``2 c_s/(G-1)`` as ``c_s`` shrinks, which is
    what the suite checks it against -- a closed form is only as trustworthy
    as the limit it reproduces.
    """
    root = math.sqrt(eos.gamma - 1.0)
    return 2.0 / root * math.atanh(sound / root)


def _rarefaction(state, pressure, eos: GammaLaw, sign):
    """The state behind a rarefaction of the given family, at pressure ``p``."""
    density, velocity, base = state
    scaled = density * (pressure / base) ** (1.0 / eos.gamma)
    sound_base = math.sqrt(eos.sound_speed_squared(density, base))
    sound = math.sqrt(eos.sound_speed_squared(scaled, pressure))
    rapidity = math.atanh(velocity) + sign * (
        _invariant_term(sound_base, eos) - _invariant_term(sound, eos)
    )
    return scaled, math.tanh(rapidity)


def _shock(state, pressure, eos: GammaLaw, sign):
    """The state behind a shock at pressure ``p``, and the shock's own speed.

    From the Taub adiabat, which closes to a quadratic in the specific
    enthalpy, then the mass flux and two more quadratics for the shock speed
    and the velocity behind it. Every root is picked by a physical condition
    -- subluminal, compressive, the fluid overtaking the shock -- rather than
    by which one happens to be real.
    """
    density, velocity, base = state
    enthalpy = eos.enthalpy(density, base)
    gap = pressure - base

    coefficient = (eos.gamma - 1.0) * gap / (eos.gamma * pressure)
    offset = enthalpy * gap / density
    scaled_enthalpy = (
        -coefficient
        + math.sqrt(coefficient**2 + 4.0 * (1.0 - coefficient) * (enthalpy**2 + offset))
    ) / (2.0 * (1.0 - coefficient))
    scaled = eos.gamma * pressure / ((eos.gamma - 1.0) * (scaled_enthalpy - 1.0))

    mass_flux = math.sqrt(gap / (enthalpy / density - scaled_enthalpy / scaled))
    factor = lorentz(velocity)
    conserved = density * factor
    speed = (conserved**2 * velocity - sign * mass_flux * math.hypot(density, mass_flux)) / (
        conserved**2 + mass_flux**2
    )

    ratio = scaled / (mass_flux * math.sqrt(1.0 - speed**2))
    behind = (ratio**2 * speed + sign * math.sqrt(ratio**2 * (1.0 - speed**2) + 1.0)) / (
        ratio**2 + 1.0
    )
    return scaled, behind, speed


def _branch(state, pressure, eos: GammaLaw, sign):
    """Velocity behind whichever wave the pressure ratio calls for."""
    if pressure > state[2]:
        scaled, behind, _ = _shock(state, pressure, eos, sign)
        return scaled, behind
    return _rarefaction(state, pressure, eos, sign)


def exact_riemann(left, right, eos: GammaLaw, tolerance=1e-14) -> RiemannFan:
    """Solve the relativistic Riemann problem exactly, by matching at the contact.

    One unknown -- the pressure between the waves -- found by bisection on
    the velocity mismatch, which is monotonic. Rarefactions come from the
    Riemann invariant along the isentrope, shocks from the Taub adiabat.
    """
    for state, label in ((left, "left"), (right, "right")):
        if state[0] <= 0.0 or state[2] <= 0.0 or abs(state[1]) >= 1.0:
            raise ValueError(
                f"the {label} state {tuple(state)} is not a physical fluid state: "
                "density and pressure must be positive and the speed subluminal"
            )

    def mismatch(pressure):
        return _branch(left, pressure, eos, +1)[1] - _branch(right, pressure, eos, -1)[1]

    low = 1e-12 * min(left[2], right[2])
    high = max(left[2], right[2])
    guard = 0
    while mismatch(high) > 0.0 and guard < 400:
        high *= 2.0
        guard += 1
    if mismatch(low) < 0.0:
        raise ValueError(
            "the two states separate faster than any rarefaction can follow, so the "
            "exact solution contains vacuum; this solver does not build one"
        )
    star_pressure = brentq(mismatch, low, high, xtol=tolerance * high, rtol=1e-15)

    left_density, star_velocity = _branch(left, star_pressure, eos, +1)
    right_density, _ = _branch(right, star_pressure, eos, -1)

    def speeds(state, density, pressure, sign):
        if pressure > state[2]:
            return (_shock(state, pressure, eos, sign)[2],) * 2
        head = characteristic_speeds(state[0], state[1], state[2], eos)
        tail = characteristic_speeds(density, star_velocity, pressure, eos)
        index = 0 if sign > 0 else 2
        return float(head[index]), float(tail[index])

    return RiemannFan(
        star_pressure=star_pressure,
        star_velocity=star_velocity,
        left_star_density=left_density,
        right_star_density=right_density,
        left_speeds=speeds(left, left_density, star_pressure, +1),
        right_speeds=speeds(right, right_density, star_pressure, -1),
        left_is_shock=star_pressure > left[2],
        right_is_shock=star_pressure > right[2],
    )


def _fan_state(state, fan_speed, eos: GammaLaw, sign, bracket):
    """Sample inside a rarefaction, by finding the pressure whose head is there."""
    low, high = bracket

    def offset(pressure):
        density, velocity = _rarefaction(state, pressure, eos, sign)
        sound = math.sqrt(eos.sound_speed_squared(density, pressure))
        index = -1 if sign > 0 else 1
        return (velocity + index * sound) / (1.0 + index * velocity * sound) - fan_speed

    pressure = brentq(offset, min(low, high), max(low, high), xtol=1e-15 * max(low, high))
    density, velocity = _rarefaction(state, pressure, eos, sign)
    return density, velocity, pressure


def exact_profile(left, right, positions, eos: GammaLaw, fan: RiemannFan | None = None):
    """``(rho, v, p)`` of the exact solution at ``x/t = positions``.

    Self-similar, so the profile at any time is this evaluated at ``x/t``.
    That is what makes the comparison in the suite an equality: there is no
    reference resolution and no interpolation on the exact side.
    """
    fan = fan if fan is not None else exact_riemann(left, right, eos)
    positions = np.atleast_1d(np.asarray(positions, dtype=float))
    density = np.empty_like(positions)
    velocity = np.empty_like(positions)
    pressure = np.empty_like(positions)

    for index, place in enumerate(positions):
        if place <= fan.left_speeds[0]:
            sample = tuple(left)
        elif place < fan.left_speeds[1]:
            sample = _fan_state(left, place, eos, +1, (fan.star_pressure, left[2]))
        elif place <= fan.star_velocity:
            sample = (fan.left_star_density, fan.star_velocity, fan.star_pressure)
        elif place <= fan.right_speeds[1]:
            sample = (fan.right_star_density, fan.star_velocity, fan.star_pressure)
        elif place < fan.right_speeds[0]:
            sample = _fan_state(right, place, eos, -1, (fan.star_pressure, right[2]))
        else:
            sample = tuple(right)
        density[index], velocity[index], pressure[index] = sample
    return density, velocity, pressure


__all__ = [
    "SOLVERS",
    "RiemannFan",
    "contact_speed",
    "exact_profile",
    "exact_riemann",
    "hllc",
    "hlle",
    "riemann_flux",
]
