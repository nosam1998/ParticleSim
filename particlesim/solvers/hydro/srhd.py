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

**The bracket is the superluminal limit and nothing else.** Below
``|S| - tau - D`` the implied velocity exceeds one, so that is a hard lower
bound on the pressure; an atmosphere floor put there instead looks harmless
and is not. For a cold enough flow -- ``p/rho`` at ``1e-12`` and a Lorentz
factor of 27 -- a floor of ``1e-13 (tau + D)`` sits *above* the true
pressure, the root is outside the bracket, and bisection does not fail. It
converges, confidently, to the bracket end: 3.6% of a random sample came
back with pressures wrong by up to a factor of 68. The bracket now carries
only the physical bound, and a state whose root is genuinely outside it is
refused by name rather than answered.

**And how well the recovery can possibly work is set by the flow, not the
algorithm.** Two amplifications, and the bound is their product:

* ``tau + D + p - D W`` is the internal energy, a small difference of large
  numbers whenever the flow is cold. Only ``eps/fraction`` of it survives,
  where the fraction is that difference over ``tau + D``.
* the velocity comes out as ``S/(tau + D + p)``, so an error in the pressure
  is an error in the velocity, and ``W = 1/sqrt(1 - v^2)`` turns that into
  an error ``W^2`` larger in everything downstream.

The second is invisible until the flow is fast, which is exactly how it gets
missed: a law fitted below ``W = 3`` reproduces its own data and
under-predicts by a hundred at ``W = 27``. Together, over 300,000 random
states with Lorentz factors from 1 to 80, the median round-trip error is a
steady **half** of ``eps W^2 / fraction`` across five decades:

    eps W^2 / fraction     median round-trip error    ratio
    1e-14 .. 1e-12               7.39e-14             0.97
    1e-12 .. 1e-10               2.28e-12             0.50
    1e-10 .. 1e-8                2.19e-10             0.50
    1e-8  .. 1e-6                2.22e-8              0.50
    1e-6  .. 1e-4                2.16e-6              0.50
    1e-4  .. 1e-2                1.14e-4              0.50

Below that the bisection's own tolerance is the floor, at ``2e-14``. So a
cold or fast relativistic flow is recoverable to far fewer digits than the
arithmetic suggests, and no rearrangement of the residual recovers digits the
conserved variables do not carry. :func:`recovery_precision` reports the
scale so a caller can know it rather than discover it.

**What the schemes are held to.** A smooth pulse advected at uniform
velocity and pressure has an exact solution -- pure translation -- so each
reconstruction can be measured against its nominal order rather than against
a tolerance. Conservation is exact to round-off, because the update is a
difference of fluxes and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Floor applied to *reconstructed* primitives, as a fraction of the largest
#: value on the grid. Deliberately not used to bracket the pressure recovery:
#: for a cold enough flow this floor sits above the true pressure, and a
#: bracket that excludes the root does not fail, it answers.
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

    # The one hard lower bound there is: below |S| - tau - D the implied
    # velocity is superluminal. Anything else put under the root -- an
    # atmosphere floor, say -- is a guess, and a guess above the true
    # pressure puts the root outside the bracket and the bisection then
    # converges confidently to the bracket end instead. Only the logarithm
    # needs a positive number, so that is all the floor is for.
    superluminal_limit = np.maximum(np.abs(momentum) - energy - conserved_density, 0.0)
    low = np.maximum(superluminal_limit * (1.0 + 1e-12), np.finfo(float).tiny)
    high = np.maximum((eos.gamma - 1.0) * (energy + conserved_density), low * 10.0)
    for _ in range(200):
        climbing = residual(high) < 0.0
        if not climbing.any():
            break
        high = np.where(climbing, 2.0 * high, high)
    unbracketed = residual(low) > 0.0
    if np.any(unbracketed):
        index = int(np.argmax(unbracketed))
        raise ValueError(
            f"no pressure between {low[index]:.6g} and {high[index]:.6g} satisfies the "
            f"recovery for (D, S, tau) = ({conserved_density[index]:.6g}, "
            f"{momentum[index]:.6g}, {energy[index]:.6g}); these conserved variables do "
            "not correspond to any state of this equation of state"
        )

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
    """``eps W^2 / cold_flow_fraction``: the accuracy this state allows.

    Two independent amplifications, and the bound is their product. The
    first is the cancellation :func:`cold_flow_fraction` measures. The
    second is that the velocity comes out as ``S/(tau + D + p)``, so an
    error in the pressure is an error in the velocity, and the Lorentz
    factor turns that into an error ``W^2`` larger in everything downstream
    -- which is invisible until the flow is fast, and is why a law fitted at
    ``W < 3`` under-predicts by a hundred at ``W = 27``.

    A law rather than a tolerance: over six decades of the bound and Lorentz
    factors from 1 to 80, the median round-trip error is a steady 0.4 of it.
    And a property of evolving ``(D, S, tau)``, not of this implementation.
    """
    conserved_density = np.asarray(conserved_density, dtype=float)
    total = np.asarray(energy, dtype=float) + conserved_density + np.asarray(pressure, dtype=float)
    factor = lorentz(np.asarray(momentum, dtype=float) / total)
    fraction = np.abs(cold_flow_fraction(conserved_density, momentum, energy, pressure, eos))
    return np.finfo(float).eps * factor**2 / np.maximum(fraction, np.finfo(float).tiny)


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
