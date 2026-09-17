"""Macro-particles and relativistic pushers (design doc Section 5.4, Section 11).

Momentum is carried as ``u = gamma v`` rather than ``v``, so nothing has to
be clamped as a particle approaches the speed of light and the Lorentz
factor is recovered from ``gamma = sqrt(1 + |u|^2)``. Units are normalized,
``c = 1``.

Two pushers are provided and they are not interchangeable.

:func:`boris` is the standard. It splits the electric impulse in half around
a pure rotation by the magnetic field, so the rotation preserves ``|u|``
exactly and the energy error does not accumulate: a particle in a static
magnetic field stays on its circle indefinitely, whatever the step. That
property is why it survives runs no higher-order method survives.

:func:`vay` exists because Boris gets one important case wrong. In crossed
fields the Boris rotation uses a Lorentz factor evaluated between the two
electric half-kicks, which does not reduce to the correct drift for a
relativistic particle: the simulated ``E x B`` drift picks up a spurious
dependence on the step size and can exceed the speed of light's worth of
drift in the wrong direction. Vay's ordering is Lorentz-invariant and gets
the drift right. It does not conserve ``|u|`` in a pure magnetic field quite
as cleanly, so it is a choice rather than an upgrade.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


def lorentz_factor(u: np.ndarray) -> np.ndarray:
    """``sqrt(1 + |u|^2)`` for momentum-per-mass ``u = gamma v``."""
    return np.sqrt(1.0 + np.sum(np.asarray(u, dtype=float) ** 2, axis=-1))


@dataclass(frozen=True)
class Species:
    """A population of macro-particles sharing a charge-to-mass ratio.

    ``position`` is ``(N, ndim)`` and ``momentum`` is always ``(N, 3)``: a
    one-dimensional run still needs all three momentum components, because
    a laser drives transverse motion that a 1D position cannot express but a
    3D momentum must.

    ``weight`` is the number of real particles each macro-particle stands
    for. It multiplies charge and mass together, so it never changes the
    trajectory, only the current deposited.

    ``dtype`` controls particle storage only. Accumulators are always double
    precision, whatever this says: single-precision storage halves the memory
    a large run needs and costs nothing visible in a trajectory, but summing
    millions of contributions into a single-precision grid loses the
    charge-conservation identity that the deposition exists to provide.
    """

    charge: float
    mass: float
    position: np.ndarray
    momentum: np.ndarray
    weight: np.ndarray
    name: str = "species"

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError("mass must be positive")
        if self.position.ndim != 2:
            raise ValueError("position must be (N, ndim)")
        if self.momentum.shape != (len(self.position), 3):
            raise ValueError(
                f"momentum must be (N, 3); got {self.momentum.shape} for "
                f"{len(self.position)} particles. A one-dimensional run still "
                "carries three momentum components"
            )
        if self.weight.shape != (len(self.position),):
            raise ValueError("weight must be (N,)")

    @classmethod
    def create(
        cls,
        charge: float,
        mass: float,
        position,
        momentum=None,
        weight=1.0,
        name: str = "species",
        dtype=np.float64,
    ) -> Species:
        position = np.atleast_2d(np.asarray(position, dtype=dtype))
        n = len(position)
        if momentum is None:
            momentum = np.zeros((n, 3), dtype=dtype)
        else:
            momentum = np.asarray(momentum, dtype=dtype).reshape(n, 3)
        weight = np.broadcast_to(np.asarray(weight, dtype=dtype), (n,)).copy()
        return cls(float(charge), float(mass), position, momentum, weight, name)

    @property
    def count(self) -> int:
        return len(self.position)

    @property
    def ndim(self) -> int:
        return self.position.shape[1]

    @property
    def charge_over_mass(self) -> float:
        return self.charge / self.mass

    @property
    def gamma(self) -> np.ndarray:
        return lorentz_factor(self.momentum)

    @property
    def velocity(self) -> np.ndarray:
        return self.momentum / self.gamma[:, None]

    @property
    def kinetic_energy(self) -> np.ndarray:
        """``(gamma - 1) m`` per macro-particle, weighted."""
        return self.weight * self.mass * (self.gamma - 1.0)

    def with_momentum(self, momentum: np.ndarray) -> Species:
        return replace(self, momentum=momentum.astype(self.momentum.dtype, copy=False))

    def with_position(self, position: np.ndarray) -> Species:
        return replace(self, position=position.astype(self.position.dtype, copy=False))


def boris(u: np.ndarray, E: np.ndarray, B: np.ndarray, qm: float, dt: float) -> np.ndarray:
    """One Boris step: half electric kick, magnetic rotation, half kick.

    The rotation is exactly orthogonal, so ``|u|`` is unchanged by the
    magnetic part to round-off rather than to the order of the scheme. That
    is the whole reason this method is still the default after fifty years.
    """
    u = np.asarray(u, dtype=float)
    half = 0.5 * qm * dt
    u_minus = u + half * E
    t = half * B / lorentz_factor(u_minus)[:, None]
    s = 2.0 * t / (1.0 + np.sum(t**2, axis=-1))[:, None]
    u_prime = u_minus + np.cross(u_minus, t)
    u_plus = u_minus + np.cross(u_prime, s)
    return u_plus + half * E


def vay(u: np.ndarray, E: np.ndarray, B: np.ndarray, qm: float, dt: float) -> np.ndarray:
    """One Vay step, which gets the relativistic ``E x B`` drift right.

    Vay 2008, Phys. Plasmas 15, 056701. The Lorentz factor used for the
    rotation is solved for rather than taken from an intermediate state,
    which is what makes the result independent of the step size in crossed
    fields where Boris is not.
    """
    u = np.asarray(u, dtype=float)
    half = 0.5 * qm * dt
    v = u / lorentz_factor(u)[:, None]
    u_prime = u + half * (2.0 * E + np.cross(v, B))

    tau = half * B
    tau2 = np.sum(tau**2, axis=-1)
    u_star = np.sum(u_prime * tau, axis=-1)
    gamma_prime2 = 1.0 + np.sum(u_prime**2, axis=-1)
    sigma = gamma_prime2 - tau2
    gamma = np.sqrt(0.5 * (sigma + np.sqrt(sigma**2 + 4.0 * (tau2 + u_star**2))))

    t = tau / gamma[:, None]
    s = 1.0 / (1.0 + np.sum(t**2, axis=-1))
    dotted = np.sum(u_prime * t, axis=-1)
    return s[:, None] * (u_prime + dotted[:, None] * t + np.cross(u_prime, t))


PUSHERS = {"boris": boris, "vay": vay}


def push_momentum(
    species: Species, E: np.ndarray, B: np.ndarray, dt: float, scheme: str = "boris"
) -> Species:
    try:
        pusher = PUSHERS[scheme]
    except KeyError:
        raise ValueError(f"unknown pusher {scheme!r}; choose from {sorted(PUSHERS)}") from None
    return species.with_momentum(pusher(species.momentum, E, B, species.charge_over_mass, dt))


def push_position(species: Species, dt: float, extent=None) -> Species:
    """Advance positions by ``v dt``, wrapping into ``extent`` if given."""
    moved = species.position + dt * species.velocity[:, : species.ndim]
    if extent is not None:
        moved = np.mod(moved, np.asarray(extent, dtype=float))
    return species.with_position(moved)
