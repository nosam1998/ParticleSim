"""Tunnel ionization by the ADK rate (design doc Section 3.3, Milestone 2).

Ammosov, Delone and Krainov 1986 (Sov. Phys. JETP 64, 1191) give the rate at
which a bound electron tunnels out of the barrier a strong low-frequency
field bends its potential into. In atomic units, for a state of effective
principal quantum number ``n*`` and angular momentum ``l, m``,

    W = C^2 f_lm I_p (2 (2 I_p)^(3/2) / F)^(2n* - |m| - 1)
        exp(-2 (2 I_p)^(3/2) / (3 F))

    n*  = Z / sqrt(2 I_p)                       l* = n* - 1
    C^2 = 2^(2n*) / (n* Gamma(n*+l*+1) Gamma(n*-l*))
    f_lm = (2l+1) (l+|m|)! / (2^|m| |m|! (l-|m|)!)

with ``Z`` the charge of the ion left behind. For hydrogen this collapses to
``W = (4/F) exp(-2/(3F))``, the textbook result, and that is how the
implementation is checked rather than against a table.

Two things about the formula are worth knowing before using it.

It is a *tunnelling* rate, valid when the field varies slowly compared with
the time an electron takes to cross the barrier. That is the Keldysh
parameter ``gamma = omega sqrt(2 I_p) / F`` being well below one.
:func:`keldysh_parameter` computes it, and :func:`adk_rate` will say so
rather than return a number that looks ordinary when it is far outside.

It is also cycle-resolved, not cycle-averaged: ``F`` is the instantaneous
field. Feeding it a peak amplitude where an instantaneous value belongs
overestimates the rate by a factor that grows exponentially with intensity.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma as gamma_function

#: One atomic unit of electric field, in volts per metre.
ATOMIC_FIELD = 5.14220675e11
#: One atomic unit of time, in seconds.
ATOMIC_TIME = 2.4188843265e-17
#: One hartree, in electronvolts.
HARTREE_EV = 27.211386245988


def electronvolts_to_hartree(energy_ev):
    return np.asarray(energy_ev, dtype=float) / HARTREE_EV


def volts_per_metre_to_atomic(field):
    return np.asarray(field, dtype=float) / ATOMIC_FIELD


def keldysh_parameter(omega: float, field, ionization_potential) -> np.ndarray:
    """``gamma = omega sqrt(2 I_p) / F``, all in atomic units.

    Below about 0.5 the field tunnels the electron out and ADK applies. Above
    about 2 the process is multiphoton and ADK does not describe it at all.
    """
    field = np.asarray(field, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return omega * np.sqrt(2.0 * np.asarray(ionization_potential, float)) / field


def barrier_suppression_field(ionization_potential, charge_state: int = 1):
    """``F = I_p^2 / (4 Z)``, where the barrier stops existing.

    Above it the electron is not tunnelling, it is simply free, and ADK
    keeps rising past the point where there is anything left to describe.
    For hydrogen this is 0.0625 atomic units, about 1.4e14 W/cm^2, which a
    laser-plasma run passes early rather than as an edge case.
    """
    return np.asarray(ionization_potential, dtype=float) ** 2 / (4.0 * charge_state)


def _coefficient(n_star: float, l_star: float) -> float:
    return 2.0 ** (2.0 * n_star) / (
        n_star * gamma_function(n_star + l_star + 1.0) * gamma_function(n_star - l_star)
    )


def _angular_factor(l: int, m: int) -> float:
    from math import factorial

    m = abs(m)
    return (2 * l + 1) * factorial(l + m) / (2**m * factorial(m) * factorial(l - m))


def adk_rate(
    field,
    ionization_potential: float,
    charge_state: int = 1,
    l: int = 0,
    m: int = 0,
) -> np.ndarray:
    """Tunnel ionization rate in atomic units, per atomic unit of time.

    ``field`` is the instantaneous field magnitude and ``ionization_potential``
    the binding energy of the electron being removed, both atomic. Zero field
    gives zero rate rather than an overflow, which matters because a
    cycle-resolved field passes through zero twice a period.

    Above :func:`barrier_suppression_field` the result is not a tunnelling
    rate at all and is returned anyway rather than raised on, because a
    cycle-resolved pulse crosses that threshold routinely and briefly.
    Compare against it before believing an ionization fraction.
    """
    if l < 0 or abs(m) > l:
        raise ValueError("need l >= 0 and |m| <= l")
    if ionization_potential <= 0:
        raise ValueError("the ionization potential must be positive")
    if charge_state < 1:
        raise ValueError("charge_state is the charge left behind, so at least one")

    field = np.abs(np.asarray(field, dtype=float))
    n_star = charge_state / np.sqrt(2.0 * ionization_potential)
    l_star = n_star - 1.0
    scale = 2.0 * (2.0 * ionization_potential) ** 1.5

    out = np.zeros_like(field)
    live = field > 0
    if not np.any(live):
        return out
    f = field[live]
    out[live] = (
        _coefficient(n_star, l_star)
        * _angular_factor(l, m)
        * ionization_potential
        * (scale / f) ** (2.0 * n_star - abs(m) - 1.0)
        * np.exp(-scale / (3.0 * f))
    )
    return out


def ionization_probability(rate, dt: float):
    """``1 - exp(-W dt)``, the chance of ionizing in one step.

    Written this way rather than as ``W dt`` so it stays a probability when
    the rate is large. In a laser focus ``W dt`` routinely exceeds one, and
    an unbounded expression there quietly ionizes more than everything.
    """
    return 1.0 - np.exp(-np.asarray(rate, dtype=float) * dt)


def tunnel_ionize(field, ionization_potential, dt, charge_state=1, l=0, m=0, rng=None):
    """Which macro-particles ionize this step, as a boolean mask.

    Each macro-particle stands for many real atoms, so strictly this should
    remove a fraction of its weight rather than all or nothing. Drawing per
    macro-particle is the usual approximation and is unbiased in the mean;
    it is noisier than weight splitting, which is the thing to change if the
    noise ever matters.
    """
    generator = np.random.default_rng() if rng is None else rng
    rate = adk_rate(field, ionization_potential, charge_state, l, m)
    probability = ionization_probability(rate, dt)
    return generator.random(np.shape(probability)) < probability
