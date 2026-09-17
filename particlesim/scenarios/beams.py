"""Beam instabilities: filamentation, Weibel and hosing (Section 3.3, M2).

Two counter-streaming beams are unstable in more than one way, and which way
depends on where the perturbation points. Along the beams it is the
electrostatic two-stream instability, which
:mod:`particlesim.scenarios.plasma` covers. *Across* them it is the
current-filamentation instability -- the Weibel instability driven by a
beam's own anisotropy -- which is what this module is about: the beams break
into current filaments that attract one another, the magnetic field between
them grows, and the filaments merge.

The growth rate is derived here rather than quoted, because the published
forms differ by factors of the beam Lorentz factor depending on whether the
perturbing field lies along the drift or across it, and picking the wrong
one makes a benchmark agree with the wrong number.

Linearizing cold symmetric beams at ``+-beta0`` along ``y``, perturbed along
``x``, with the transverse pair ``E_y`` and ``B_z``:

    delta u_x:  -i w du_x = -beta_s dB_z          (across the drift, mass gamma)
    delta u_y:  -i w du_y = -dE_y                 (along it, mass gamma^3)

The two beams' ``x`` responses are equal and opposite, so the density
perturbations cancel, ``E_x`` stays zero, and the mode is purely
electromagnetic. Feeding the current back through Maxwell gives

    w^2 = k^2 + wp^2/gamma^3 + wp^2 k^2 beta0^2 / (w^2 gamma)

and setting ``w = i G`` for a purely growing mode leaves a quadratic in
``G^2``. As ``k`` grows the rate saturates at ``beta0 wp / sqrt(gamma)``,
which is the textbook maximum and is what says the derivation landed in the
right place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from particlesim.scenarios.plasma import solve_poisson
from particlesim.solvers.pic.deposition import deposit_charge
from particlesim.solvers.pic.particles import Species
from particlesim.solvers.pic.yee import Fields, YeeGrid


def filamentation_growth_rate(k: float, drift: float, plasma_frequency: float = 1.0) -> float:
    """Growth rate of the cold current-filamentation mode.

    ``G^4 + G^2 (k^2 + wp^2/gamma^3) - wp^2 k^2 beta0^2 / gamma = 0``, taking
    the positive root. Zero at ``k = 0`` and rising monotonically to
    :func:`filamentation_maximum_rate`.
    """
    if not 0.0 < drift < 1.0:
        raise ValueError("the drift speed must lie strictly between zero and one")
    gamma = 1.0 / np.sqrt(1.0 - drift**2)
    wp2 = plasma_frequency**2
    a = k**2 + wp2 / gamma**3
    b = wp2 * k**2 * drift**2 / gamma
    squared = 0.5 * (-a + np.sqrt(a**2 + 4.0 * b))
    return float(np.sqrt(squared)) if squared > 0 else 0.0


def filamentation_maximum_rate(drift: float, plasma_frequency: float = 1.0) -> float:
    """``beta0 wp / sqrt(gamma)``, the short-wavelength limit.

    Approached rather than attained: the rate rises with ``k`` and never
    turns over, so in practice the fastest mode is set by whatever cuts the
    spectrum off -- a finite beam temperature, or the grid.
    """
    gamma = 1.0 / np.sqrt(1.0 - drift**2)
    return float(drift * plasma_frequency / np.sqrt(gamma))


@dataclass(frozen=True)
class BeamSetup:
    """Loaded beams and the field that balances them."""

    fields: Fields
    species: Species
    density: float
    plasma_frequency: float
    wavenumber: float
    drift: float

    @property
    def growth_rate(self) -> float:
        return filamentation_growth_rate(self.wavenumber, self.drift, self.plasma_frequency)


def counter_streaming_beams(
    grid: YeeGrid,
    density: float = 1.0,
    drift: float = 0.5,
    per_cell: int = 64,
    amplitude: float = 1e-4,
    mode: int = 1,
    order: int = 1,
) -> BeamSetup:
    """Two cold beams along ``y``, perturbed across themselves in ``x``.

    The seed displaces the two beams in opposite directions, which is what
    makes a current filament rather than a density ripple. Displacing them
    the same way seeds the electrostatic mode instead, and since that one
    grows faster at long wavelength it would be what the run measured.

    The initial field is zero: the beams are neutral cell by cell, and their
    currents cancel exactly before the seed tilts them.
    """
    length = grid.extent[0]
    count = grid.shape[0] * per_cell
    half = count // 2
    base = ((np.arange(half) + 0.5) / half * length).reshape(half, 1)
    k = 2.0 * np.pi * mode / length
    shift = amplitude * 2.0 * np.pi / k

    forward = base + shift * np.sin(k * base)
    backward = base - shift * np.sin(k * base)
    positions = np.mod(np.concatenate([forward, backward], axis=0), length)

    gamma = 1.0 / np.sqrt(1.0 - drift**2)
    u = np.zeros((count, 3))
    u[:half, 1] = gamma * drift
    u[half:, 1] = -gamma * drift

    species = Species.create(-1.0, 1.0, positions, u, weight=density * length / count, name="beams")
    rho = deposit_charge(grid, species, order)
    D = solve_poisson(grid, rho - rho.mean(), reference=density)
    fields = Fields(D[0], D[1], D[2], *(grid.zeros() for _ in range(3)))
    return BeamSetup(
        fields=fields,
        species=species,
        density=density,
        plasma_frequency=float(np.sqrt(density)),
        wavenumber=float(k),
        drift=float(drift),
    )


def magnetic_mode_amplitude(fields: Fields, mode: int = 1) -> float:
    """Amplitude of one spatial mode of ``B_z``, the filamentation signature.

    ``B_z`` rather than ``E_x``: the electrostatic mode lives in ``E_x`` and
    the filamentation mode in ``B_z``, and measuring the wrong one on a run
    that carries both gives whichever grew faster, not the one intended.
    """
    return float(abs(np.fft.fft(np.asarray(fields.Bz, dtype=float))[mode]))
