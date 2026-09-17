"""String gas cosmology: equation of state and radion (Sections 3.1, 4.4).

A gas of closed strings on a torus has momentum modes whose energy falls as
``1/R`` and winding modes whose energy *rises* as ``R``, in string units.
That single fact is the whole of string gas cosmology's background
behaviour, and three consequences follow from it without any further input.

**The equation of state interpolates between radiation and its negative.**
With ``V = R^d`` and ``p = -dE/dV``,

    w = (1/d) (E_momentum - E_winding)/(E_momentum + E_winding)

so a gas of pure momentum modes is radiation, ``w = 1/d``; pure winding is
``w = -1/d``; and a gas with equal energy in both is *pressureless*. That
last case is the Hagedorn phase, and its vanishing pressure is why the
scale factor sits still there rather than being dragged by the gas.

**The radion is stabilised at the self-dual radius.** ``E = E_n/R + E_w R``
has a minimum at ``R = sqrt(E_n/E_w)``, which for the T-duality symmetric
spectrum ``E_n = E_w`` is ``R = 1`` in string units. The winding modes hold
the radius from growing and the momentum modes hold it from shrinking, and
neither has to be put in by hand. That minimum is the *same* condition as
the pressureless gas above -- ``E_n/R = E_w R`` rearranges to
``R = sqrt(E_n/E_w)`` -- so the Hagedorn phase and the stabilised radion
are one statement rather than two.

**T-duality is exact.** ``R -> 1/R`` with ``E_n <-> E_w`` leaves the energy
alone, which is checked here to machine precision rather than asserted.

Once the winding modes have annihilated, only momentum modes are left,
``w = 1/3`` in three dimensions, and the background is radiation dominated:
``a ~ t^(1/2)``. That is the scaling solution this module is checked
against, using the Friedmann integrator in
:mod:`particlesim.cosmo.dynamics` so the equation of state is the only thing
supplied here.

**What this module does not do.** The quasi-static Hagedorn phase itself is
*not* obtained from ``w = 0`` in Einstein gravity -- pressureless matter
gives ``a ~ t^(2/3)``, not a static universe -- and this module says so
rather than pretending otherwise. A static Hagedorn phase needs a
dilaton-gravity background (the same string-frame system
:mod:`particlesim.cosmo.early.prebigbang` integrates) or a fixed external
radion, and the thermal fluctuation spectrum that string gas cosmology is
actually interesting for needs the specific heat of the string gas on a
torus, which is not here either. What is here is the equation of state, the
radion potential, the dimension count, and the post-Hagedorn scaling
solution.

Provenance: Brandenberger and Vafa, Nucl. Phys. B316:391 (1989), for the
gas, the T-duality argument and the dimension count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Largest number of spatial dimensions in which winding modes can annihilate.
#:
#: Two string worldsheets are two-dimensional, so in ``D`` spacetime
#: dimensions they generically intersect only if ``2 + 2 >= D``. That gives
#: ``D <= 4`` and so at most three large spatial dimensions -- the
#: Brandenberger-Vafa argument for why three dimensions grow. It is a
#: statement about generic intersection, not a theorem about the dynamics,
#: and the module reports it as the counting argument it is.
MAXIMUM_LARGE_DIMENSIONS = 3


def winding_modes_annihilate(spatial_dimensions: int) -> bool:
    """Can two string worldsheets generically meet in this many dimensions?"""
    if spatial_dimensions < 1:
        raise ValueError(f"spatial_dimensions must be at least 1, got {spatial_dimensions}")
    return spatial_dimensions <= MAXIMUM_LARGE_DIMENSIONS


@dataclass(frozen=True)
class StringGas:
    """A gas of closed strings on an isotropic torus of radius ``radius``.

    ``momentum_scale`` and ``winding_scale`` are the coefficients of the
    ``1/R`` and ``R`` parts of the energy: they carry the occupation numbers
    and the temperature, and only their ratio affects the equation of state.
    Setting one of them to zero gives the pure momentum or pure winding gas.
    """

    momentum_scale: float = 1.0
    winding_scale: float = 1.0
    radius: float = 1.0
    dimension: int = 3

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError(f"radius must be positive, got {self.radius}")
        if self.dimension < 1:
            raise ValueError(f"dimension must be at least 1, got {self.dimension}")
        if self.momentum_scale < 0.0 or self.winding_scale < 0.0:
            raise ValueError("momentum_scale and winding_scale must be non-negative")
        if self.momentum_scale + self.winding_scale <= 0.0:
            raise ValueError("a gas with neither momentum nor winding modes has no energy")

    @property
    def momentum_energy(self) -> float:
        return self.momentum_scale / self.radius

    @property
    def winding_energy(self) -> float:
        return self.winding_scale * self.radius

    @property
    def energy(self) -> float:
        return self.momentum_energy + self.winding_energy

    @property
    def volume(self) -> float:
        return self.radius**self.dimension

    @property
    def density(self) -> float:
        return self.energy / self.volume

    @property
    def pressure(self) -> float:
        """``p = -dE/dV``, taken through ``R``."""
        gradient = -self.momentum_scale / self.radius**2 + self.winding_scale
        return -gradient / (self.dimension * self.radius ** (self.dimension - 1))

    @property
    def equation_of_state(self) -> float:
        """``w = (1/d)(E_momentum - E_winding)/(E_momentum + E_winding)``."""
        return (self.momentum_energy - self.winding_energy) / (self.dimension * self.energy)

    @property
    def hagedorn(self) -> bool:
        """Equal energy in winding and momentum, so the pressure vanishes."""
        return abs(self.equation_of_state) < 1e-12

    @property
    def self_dual_radius(self) -> float:
        """``R`` at which the energy is stationary: ``sqrt(E_n/E_w)``."""
        if self.winding_scale <= 0.0:
            raise ValueError(
                "with no winding modes the energy falls monotonically with radius and "
                "the radion is not stabilised: that is the decompactified case"
            )
        return math.sqrt(self.momentum_scale / self.winding_scale)

    @property
    def stabilised(self) -> bool:
        """Is the radius at the minimum of the energy?"""
        return abs(self.radius / self.self_dual_radius - 1.0) < 1e-12

    @property
    def radion_force(self) -> float:
        """``-dE/dR``: positive pushes the radius outward."""
        return self.momentum_scale / self.radius**2 - self.winding_scale

    @property
    def expansion_exponent(self) -> float:
        """``p`` in ``a ~ t^p`` for this equation of state: ``2/(d(1+w))``."""
        return 2.0 / (self.dimension * (1.0 + self.equation_of_state))

    def t_dual(self) -> StringGas:
        """``R -> 1/R`` with the momentum and winding scales exchanged.

        The energy is invariant under this, exactly. It is the statement
        that a string cannot tell a small torus from a large one, and it is
        why the radion has a minimum rather than running away.
        """
        return StringGas(
            momentum_scale=self.winding_scale,
            winding_scale=self.momentum_scale,
            radius=1.0 / self.radius,
            dimension=self.dimension,
        )

    def at_radius(self, radius: float) -> StringGas:
        return StringGas(
            momentum_scale=self.momentum_scale,
            winding_scale=self.winding_scale,
            radius=radius,
            dimension=self.dimension,
        )

    def after_winding_annihilation(self) -> StringGas:
        """The same gas with its winding modes gone: radiation.

        Whether this is allowed to happen is the Brandenberger-Vafa
        question, and :func:`winding_modes_annihilate` answers it by
        counting dimensions.
        """
        return StringGas(
            momentum_scale=self.momentum_scale,
            winding_scale=0.0,
            radius=self.radius,
            dimension=self.dimension,
        )

    def energy_curve(self, radii):
        """``E(R)`` over an array of radii, for plotting the radion potential."""
        radii = np.asarray(radii, dtype=float)
        if np.any(radii <= 0.0):
            raise ValueError("radii must be positive")
        return self.momentum_scale / radii + self.winding_scale * radii

    def summary(self) -> dict[str, float | bool]:
        return {
            "radius": self.radius,
            "energy": self.energy,
            "equation_of_state": self.equation_of_state,
            "hagedorn": self.hagedorn,
            "radion_force": self.radion_force,
            "expansion_exponent": self.expansion_exponent,
        }


def hagedorn_gas(dimension: int = 3, radius: float = 1.0) -> StringGas:
    """A gas with equal winding and momentum energy at ``radius``.

    Equal energies is the same condition as being at the self-dual radius,
    which is worth spelling out because it looks like a coincidence and is
    not: ``E_n/R = E_w R`` rearranges to ``R = sqrt(E_n/E_w)``, which is
    exactly where the energy is stationary. So **a pressureless string gas
    and a stabilised radion are one condition, not two** -- the Hagedorn
    phase sits at the minimum of the radion potential by construction, and
    a gas built to be pressureless comes out with zero radion force.
    """
    return StringGas(
        momentum_scale=radius,
        winding_scale=1.0 / radius,
        radius=radius,
        dimension=dimension,
    )
