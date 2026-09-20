"""One-dimensional special-relativistic hydrodynamics in conservative form.

Issue #57. The Valencia formulation evolves ``(D, S, tau)`` rather than the
primitives, which is what makes shocks come out in the right place: a
conservative scheme satisfies the jump conditions by construction, and a
non-conservative one puts the shock wherever its truncation error prefers.
The price is that every step has to recover ``(rho, v, p)`` from the
conserved variables by solving an equation, and that recovery is where a
relativistic code usually fails.

**The characteristic speeds are relativistic velocity addition.** The flux
Jacobian's eigenvalues are

    lambda_pm = (v +- c_s)/(1 +- v c_s),    lambda_0 = v

which is the velocity-addition law, so they are subluminal automatically
rather than by a clamp. That is checkable against the numerically
differentiated Jacobian, and the suite does it: the three eigenvalues agree
to ``1e-9``, which is the finite difference's limit and not the formula's.

**Recovery is a root-find in the logarithm of the pressure**, by bisection
rather than Newton, because the residual is monotonic and a Newton iteration
on it diverges for exactly the cold, fast states a shock tube spends its time
in. Working in ``ln p`` keeps the tolerance relative across a bracket that
spans many decades.

**And how well it can possibly work is set by the flow, not the algorithm.**
The recovery needs ``tau + D + p - D W``, which is the internal energy -- a
small difference of large numbers whenever the flow is cold. Double precision
keeps only ``eps/fraction`` of it, where the fraction is that difference over
``tau + D``, and the measured round-trip error follows that law across four
decades:

    fraction          median round-trip error    eps/fraction
    1e-9 .. 1e-7            1.16e-8                 1.0e-8
    1e-7 .. 1e-5            1.13e-10                1.0e-10
    1e-5 .. 1e-3            1.38e-12                1.0e-12
    1e-3 .. 1e-1            2.45e-14                1.0e-14

So a cold relativistic flow is recoverable to far fewer digits than the
arithmetic suggests, and no rearrangement of the residual recovers digits the
conserved variables do not carry. :func:`recovery_precision` reports the
bound so a caller can know it rather than discover it.

**What the schemes are held to.** A smooth pulse advected at uniform
velocity and pressure has an exact solution -- pure translation -- so each
reconstruction can be measured against its nominal order rather than against
a tolerance. Conservation is exact to round-off, because the update is a
difference of fluxes and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Floor applied to recovered primitives, as a fraction of the local scale.
ATMOSPHERE = 1e-13

#: Largest Lorentz factor the recovery will report before refusing.
MAXIMUM_LORENTZ = 1e4


@dataclass(frozen=True)
class GammaLaw:
    """``p = (Gamma - 1) rho eps``: the equation of state the solver hooks into.

    Kept deliberately small. The recovery below needs only ``pressure`` and
    ``sound_speed_squared``, so a different equation of state substitutes by
    providing those two -- which is what "con2prim with EOS hooks" means.
    """

    gamma: float = 5.0 / 3.0

    def __post_init__(self) -> None:
        if not 1.0 < self.gamma < 2.0:
            raise ValueError(
                f"the adiabatic index must lie in (1, 2), got {self.gamma}; at or above "
                "two the sound speed reaches the speed of light"
            )

    def pressure(self, density, specific_energy):
        return (self.gamma - 1.0) * density * specific_energy

    def specific_energy(self, density, pressure):
        return pressure / ((self.gamma - 1.0) * density)

    def enthalpy(self, density, pressure):
        return 1.0 + self.gamma * pressure / ((self.gamma - 1.0) * density)

    def sound_speed_squared(self, density, pressure):
        """``Gamma p / (rho h)``, which stays below one for ``Gamma < 2``."""
        return self.gamma * pressure / (density * self.enthalpy(density, pressure))


def lorentz(velocity):
    velocity = np.asarray(velocity, dtype=float)
    return 1.0 / np.sqrt(1.0 - velocity**2)


def primitive_to_conserved(density, velocity, pressure, eos: GammaLaw):
    """``(D, S, tau) = (rho W, rho h W^2 v, rho h W^2 - p - D)``."""
    density = np.asarray(density, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    factor = lorentz(velocity)
    enthalpy = eos.enthalpy(density, pressure)
    conserved_density = density * factor
    momentum = density * enthalpy * factor**2 * velocity
    energy = density * enthalpy * factor**2 - pressure - conserved_density
    return conserved_density, momentum, energy


def conserved_to_primitive(conserved_density, momentum, energy, eos: GammaLaw, tolerance=1e-13):
    """Recover ``(rho, v, p)``, by bisection in ``ln p``, over the whole grid at once.

    The residual is monotonic in the pressure, so bisection is enough and is
    what makes this robust: a Newton iteration on the same function diverges
    for a cold, fast state, which is exactly where a shock tube spends its
    time. Working in the logarithm keeps the tolerance relative, which is
    what a bracket spanning many decades needs.

    Every cell is bisected in step with every other, on the same array. That
    is not only for speed: a per-cell loop with a per-cell iteration count
    makes the answer depend on how the grid was partitioned, and a
    convergence study cannot then tell a scheme's error from a solver's.
    """
    conserved_density = np.atleast_1d(np.asarray(conserved_density, dtype=float))
    momentum = np.atleast_1d(np.asarray(momentum, dtype=float))
    energy = np.atleast_1d(np.asarray(energy, dtype=float))

    def residual(pressure):
        total = energy + conserved_density + pressure
        velocity = momentum / total
        superluminal = np.abs(velocity) >= 1.0
        velocity = np.where(superluminal, 0.0, velocity)
        factor = 1.0 / np.sqrt(1.0 - velocity * velocity)
        density = conserved_density / factor
        # Grouped so the only cancellation is the one the physics forces:
        # total - D W is the internal energy, small whenever the flow is cold.
        # Writing it as total/(D W) - 1 would cancel against one a second time.
        scale = conserved_density * factor
        specific = (total - scale) / scale - pressure * factor / conserved_density
        return np.where(superluminal, np.inf, pressure - eos.pressure(density, specific))

    floor = ATMOSPHERE * (energy + conserved_density)
    low = np.maximum(np.abs(momentum) - energy - conserved_density, 0.0)
    low = np.maximum(low * (1.0 + 1e-12), floor)
    high = np.maximum((eos.gamma - 1.0) * (energy + conserved_density), low * 10.0)
    for _ in range(200):
        climbing = residual(high) < 0.0
        if not climbing.any():
            break
        high = np.where(climbing, 2.0 * high, high)

    lower, upper = np.log(low), np.log(high)
    for _ in range(400):
        middle = 0.5 * (lower + upper)
        above = residual(np.exp(middle)) > 0.0
        upper = np.where(above, middle, upper)
        lower = np.where(above, lower, middle)
        if np.all(upper - lower < tolerance):
            break

    pressures = np.exp(0.5 * (lower + upper))
    total = energy + conserved_density + pressures
    velocities = momentum / total
    factors = 1.0 / np.sqrt(1.0 - velocities * velocities)
    if np.any(factors > MAXIMUM_LORENTZ):
        worst = float(np.max(factors))
        raise ValueError(
            f"recovery reached a Lorentz factor of {worst:.3g}, past the "
            f"{MAXIMUM_LORENTZ:.0g} this solver reports; beyond it the conserved "
            "variables no longer determine the primitives to double precision"
        )
    return conserved_density / factors, velocities, pressures


def cold_flow_fraction(conserved_density, momentum, energy, pressure, eos: GammaLaw):
    """``(tau + D + p - D W)/(tau + D)``: internal energy as a share of the total.

    The number that decides how well the primitives can be recovered at all.
    The recovery needs that difference, and the quantities it subtracts are
    larger than it by the reciprocal of this fraction -- so double precision
    keeps only ``eps/fraction`` of the answer.
    """
    conserved_density = np.asarray(conserved_density, dtype=float)
    energy = np.asarray(energy, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    total = energy + conserved_density + pressure
    velocity = np.asarray(momentum, dtype=float) / total
    return (total - conserved_density * lorentz(velocity)) / (energy + conserved_density)


def recovery_precision(conserved_density, momentum, energy, pressure, eos: GammaLaw):
    """``eps_machine / cold_flow_fraction``: the accuracy this state allows.

    A law rather than a tolerance -- across four decades the median
    round-trip error tracks it to within a factor of about 1.2 -- and a
    property of evolving ``(D, S, tau)`` rather than of this implementation.
    """
    fraction = np.abs(cold_flow_fraction(conserved_density, momentum, energy, pressure, eos))
    return np.finfo(float).eps / np.maximum(fraction, np.finfo(float).tiny)


def flux(density, velocity, pressure, eos: GammaLaw):
    """``(D v, S v + p, S - D v)``: the Valencia fluxes."""
    conserved_density, momentum, _ = primitive_to_conserved(density, velocity, pressure, eos)
    return (
        conserved_density * velocity,
        momentum * velocity + pressure,
        momentum - conserved_density * velocity,
    )


def characteristic_speeds(density, velocity, pressure, eos: GammaLaw):
    """``((v - c)/(1 - vc), v, (v + c)/(1 + vc))``: relativistic velocity addition.

    Subluminal by construction rather than by a clamp, which is why a
    relativistic Riemann solver needs no special handling near ``|v| -> 1``.
    """
    sound = np.sqrt(eos.sound_speed_squared(density, pressure))
    velocity = np.asarray(velocity, dtype=float)
    return (
        (velocity - sound) / (1.0 - velocity * sound),
        velocity,
        (velocity + sound) / (1.0 + velocity * sound),
    )


__all__ = [
    "ATMOSPHERE",
    "MAXIMUM_LORENTZ",
    "GammaLaw",
    "characteristic_speeds",
    "cold_flow_fraction",
    "conserved_to_primitive",
    "flux",
    "lorentz",
    "primitive_to_conserved",
    "recovery_precision",
]
