"""Cold-plasma scenarios and their linear theory (design doc Section 10, M2).

Two classical tests of a particle-in-cell core, set up so that what they
measure is the code rather than the initial conditions.

**Cold plasma oscillation.** Displace a uniform electron population
sinusoidally against an immobile neutralizing background and it oscillates
at the plasma frequency, which in normalized units (``c = eps_0 = 1``, and
electrons of charge ``-1`` and mass ``1``) is ``omega_p = sqrt(n)``. The
answer does not depend on the wavenumber, the amplitude, or anything else
the setup chooses, which is what makes it a good gate.

**Two-stream instability.** Two counter-streaming cold beams are unstable.
Linear theory gives a growth rate that peaks at ``k v0 = sqrt(3/8) omega_p``
with ``Im(omega) = omega_p / (2 sqrt(2))``, and :func:`two_stream_growth_rate`
solves the dispersion relation for any ``k`` rather than only at the peak.

Both start from a *quiet* particle loading: particles on a regular lattice
rather than sampled at random. Random loading carries shot noise of order
``1/sqrt(N)`` in every mode, which for the frequencies wanted here would
swamp the measurement, and adding particles buys precision only as the
square root. A quiet start has no noise to begin with.

The initial field is solved for. Starting from zero fields with charge
already present would leave ``div D - rho`` at the wrong value forever,
since the cycle conserves it exactly; the plasma would then oscillate about
a displaced equilibrium.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from particlesim.solvers.pic.deposition import deposit_charge
from particlesim.solvers.pic.particles import Species
from particlesim.solvers.pic.yee import Fields, YeeGrid


def solve_poisson(
    grid: YeeGrid, rho: np.ndarray, reference: float | None = None
) -> tuple[np.ndarray, ...]:
    """``D`` with ``div D = rho``, for the solver's own discrete divergence.

    Solved in Fourier space against the exact eigenvalues of the composite
    operator the solver uses -- a backward difference of a forward difference,
    whose symbol is ``-4 sin(k dx / 2)^2 / dx^2`` -- rather than against the
    continuum ``-k^2``. Using the continuum symbol would leave a residual of
    order ``(k dx)^2``, which is small, permanent, and indistinguishable from
    a physical field.

    The mean of ``rho`` must vanish: a periodic box cannot hold net charge,
    and a solver that quietly subtracted it would be changing the problem.
    ``reference`` is the density scale to judge that against, which matters
    when the perturbation itself is tiny: an unperturbed plasma leaves a
    ``rho`` that is pure round-off, and asking whether round-off is neutral
    relative to itself has no answer. It defaults to the largest value in
    ``rho``.
    """
    rho = np.asarray(rho, dtype=float)
    total = float(rho.mean())
    scale = float(np.abs(rho).max()) if reference is None else abs(float(reference))
    if scale > 0 and abs(total) > 1e-10 * scale:
        raise ValueError(
            f"net charge density {total:.3e} against a scale of {scale:.3e}. A "
            "periodic box cannot hold net charge, so the background is not "
            "neutralizing the particles"
        )

    symbol = np.zeros(grid.shape)
    for axis, (n, dx) in enumerate(zip(grid.shape, grid.spacing, strict=True)):
        m = np.fft.fftfreq(n) * n
        piece = -4.0 * np.sin(np.pi * m / n) ** 2 / dx**2
        symbol = symbol + piece.reshape([-1 if a == axis else 1 for a in range(grid.ndim)])

    # div D = rho with D = -grad phi means laplacian phi = -rho, so the
    # potential carries the minus sign. Dropping it returns a field whose
    # divergence is -rho: an anti-restoring initial condition that still
    # oscillates at the right frequency, and so is invisible in the one
    # measurement most likely to be checked.
    spectrum = np.fft.fftn(rho)
    with np.errstate(divide="ignore", invalid="ignore"):
        potential = np.where(symbol != 0, -spectrum / symbol, 0.0)
    phi = np.real(np.fft.ifftn(potential))

    # D = -grad phi, with the forward difference that pairs with the
    # backward difference the divergence uses.
    out = []
    for axis, dx in enumerate(grid.spacing):
        out.append(-(np.roll(phi, -1, axis=axis) - phi) / dx)
    while len(out) < 3:
        out.append(np.zeros(grid.shape))
    return tuple(out)


@dataclass(frozen=True)
class PlasmaSetup:
    """A loaded plasma and the field that balances it."""

    fields: Fields
    species: Species
    density: float
    plasma_frequency: float
    wavenumber: float


def _quiet_positions_1d(grid: YeeGrid, per_cell: int) -> np.ndarray:
    """Particles on a regular lattice, offset half a sub-cell from the edges."""
    n = grid.shape[0] * per_cell
    return ((np.arange(n) + 0.5) / n * grid.extent[0]).reshape(n, 1)


def cold_plasma_oscillation(
    grid: YeeGrid,
    density: float = 1.0,
    per_cell: int = 8,
    amplitude: float = 1e-4,
    mode: int = 1,
    order: int = 1,
) -> PlasmaSetup:
    """Uniform electrons displaced sinusoidally against fixed ions.

    ``amplitude`` is the displacement as a fraction of the wavelength. It has
    to stay small: the oscillation is harmonic only in the linear regime, and
    once neighbouring particle sheets cross, the frequency shifts and then
    the cold-plasma answer is not the right one to compare against.
    """
    length = grid.extent[0]
    positions = _quiet_positions_1d(grid, per_cell)
    k = 2.0 * np.pi * mode / length
    displaced = positions + (amplitude * 2.0 * np.pi / k) * np.sin(k * positions)

    weight = density * length / len(positions)
    species = Species.create(-1.0, 1.0, np.mod(displaced, length), weight=weight, name="electrons")

    rho = deposit_charge(grid, species, order)
    # Immobile ions, exactly neutralizing. They carry no current, so they
    # never appear again after this line.
    rho = rho - rho.mean()
    D = solve_poisson(grid, rho, reference=density)
    fields = Fields(D[0], D[1], D[2], *(grid.zeros() for _ in range(3)))
    return PlasmaSetup(fields, species, density, float(np.sqrt(density)), float(k))


def two_stream(
    grid: YeeGrid,
    density: float = 1.0,
    drift: float = 0.2,
    per_cell: int = 16,
    amplitude: float = 1e-5,
    mode: int = 1,
    order: int = 1,
) -> PlasmaSetup:
    """Two counter-streaming cold beams, each of half the total density.

    Both beams are displaced identically, which seeds a clean single mode but
    not the growing eigenvector: a symmetric density perturbation projects
    onto the stable branches too, and they beat against the growing one for
    the first few e-foldings. That is why the fit window starts well above
    the seed rather than at the beginning, and why the measured rate is
    quoted against a window whose bounds are stated.
    """
    length = grid.extent[0]
    half = _quiet_positions_1d(grid, per_cell // 2)
    k = 2.0 * np.pi * mode / length
    gamma = 1.0 / np.sqrt(1.0 - drift**2)
    displacement = amplitude * 2.0 * np.pi / k

    positions = np.concatenate([half, half], axis=0)
    u = np.zeros((len(positions), 3))
    u[: len(half), 0] = gamma * drift
    u[len(half) :, 0] = -gamma * drift

    positions = positions + displacement * np.sin(k * positions)

    weight = density * length / len(positions)
    species = Species.create(-1.0, 1.0, np.mod(positions, length), u, weight=weight, name="beams")
    rho = deposit_charge(grid, species, order)
    rho = rho - rho.mean()
    D = solve_poisson(grid, rho, reference=density)
    fields = Fields(D[0], D[1], D[2], *(grid.zeros() for _ in range(3)))
    return PlasmaSetup(fields, species, density, float(np.sqrt(density)), float(k))


def two_stream_growth_rate(
    k: float,
    drift: float,
    plasma_frequency: float = 1.0,
    relativistic: bool = True,
) -> float:
    """Growth rate from the cold two-stream dispersion relation.

    For two beams of equal density drifting at ``+-v0``,

        1 = (W^2 / 2) [ 1/(omega - k v0)^2 + 1/(omega + k v0)^2 ]

    which is a quadratic in ``omega^2``. The unstable branch is the root with
    ``omega^2 < 0`` and its growth rate is ``sqrt(-omega^2)``. Returns zero
    where the mode is stable.

    ``W^2 = omega_p^2 / gamma^3`` when ``relativistic``, because a
    longitudinal perturbation of a drifting beam sees the longitudinal mass
    ``gamma^3 m`` rather than ``m``. The growth rate therefore carries
    ``gamma^-3/2``: a 0.8% shift at ``v0 = 0.1`` and 6.8% at ``v0 = 0.3``, so
    for anything but the slowest beams it is the difference between agreeing
    with a simulation and not.
    Setting it false gives the textbook non-relativistic result.

    The classical maximum, ``omega_p / (2 sqrt(2))`` at ``k v0 = sqrt(3/8)
    omega_p``, falls out of this rather than being hard-coded.
    """
    a = (k * drift) ** 2
    wp2 = plasma_frequency**2
    if relativistic:
        gamma = 1.0 / np.sqrt(1.0 - drift**2)
        wp2 = wp2 / gamma**3
    # omega^4 - (2a + W^2) omega^2 + a^2 - a W^2 = 0
    b = -(2.0 * a + wp2)
    c = a**2 - a * wp2
    disc = b**2 - 4.0 * c
    if disc < 0:
        return 0.0
    root = 0.5 * (-b - np.sqrt(disc))
    return float(np.sqrt(-root)) if root < 0 else 0.0


def two_stream_fastest_mode(drift: float, plasma_frequency: float = 1.0) -> float:
    """The wavenumber whose growth rate is largest, ``sqrt(3/8) W / v0``."""
    if drift <= 0:
        raise ValueError(
            "beams at rest are stable, so there is no fastest-growing mode; give a positive drift"
        )
    gamma = 1.0 / np.sqrt(1.0 - drift**2)
    return float(np.sqrt(3.0 / 8.0) * plasma_frequency / gamma**1.5 / drift)


def mode_amplitude(field: np.ndarray, mode: int) -> complex:
    """The complex Fourier coefficient of one spatial mode."""
    return complex(np.fft.fft(np.asarray(field, dtype=float))[mode])


def oscillation_frequency(samples, dt: float) -> float:
    """Frequency of a sampled sinusoid, by least squares on its recurrence.

    A pure sinusoid sampled uniformly obeys ``x[n+1] + x[n-1] = 2 cos(w dt)
    x[n]`` exactly, whatever its amplitude and phase. Solving that in the
    least-squares sense over the whole series uses every sample and needs no
    window, no interpolation of a spectral peak, and no assumption about how
    many periods were run. It recovers the frequency of a clean signal to
    round-off.
    """
    x = np.asarray(samples, dtype=float)
    if len(x) < 3:
        raise ValueError("need at least three samples")
    middle = x[1:-1]
    ratio = float(np.sum(middle * (x[2:] + x[:-2])) / (2.0 * np.sum(middle**2)))
    if not -1.0 <= ratio <= 1.0:
        raise ValueError(
            f"the recurrence gives cos(w dt) = {ratio:.6f}, outside [-1, 1]. The "
            "series is not a single sinusoid: it is growing, decaying, or "
            "carrying more than one mode"
        )
    return float(np.arccos(ratio) / dt)


def exponential_window(times, amplitudes, low: float = 1e-2, high: float = 0.3):
    """Pick the clean exponential band out of a growth curve.

    An instability run has three phases and only the middle one is a growth
    rate. First a transient while the seeded perturbation settles onto the
    unstable branch, then exponential growth, then saturation as the beams
    trap and the mode stops growing. Fitting across the saturated tail is the
    easy mistake: it is where most of the samples are, and including it drags
    the fitted rate down by a large factor while still looking like a fit.

    The band is taken between ``low`` and ``high`` times the largest
    amplitude reached, and cut off at the first time ``high`` is crossed so
    that post-saturation excursions cannot come back in.

    The defaults are measured rather than assumed. Against the cold
    two-stream theory, at drifts from 0.05 to 0.3, the fitted rate comes out

        low = 1e-3, high = 0.3   +1.3% to +1.6%   (transient still in)
        low = 1e-2, high = 0.3   -0.4% to -0.5%   (the flat band)
        low = 2e-2, high = 0.5   -2.2% to -2.4%   (saturation creeping in)
        low = 5e-2, high = 0.7   about -49%       (mostly saturated)

    so the window is worth about two per cent either way and the residual
    half a per cent is a property of the method, not a precision claim. The
    chosen pair is the one where the local growth rate is flat across the
    whole band.
    """
    t = np.asarray(times, dtype=float)
    a = np.abs(np.asarray(amplitudes, dtype=float))
    peak = float(a.max())
    if peak <= 0:
        raise ValueError("the mode never grew")
    crossed = np.nonzero(a >= high * peak)[0]
    end = int(crossed[0]) if crossed.size else len(a)
    keep = np.zeros(len(a), dtype=bool)
    keep[:end] = a[:end] >= low * peak
    if keep.sum() < 3:
        raise ValueError(
            f"only {int(keep.sum())} samples between {low:g} and {high:g} of the "
            "peak. The run is too short to show exponential growth, or it "
            "saturated before it could"
        )
    return t[keep], a[keep]


def growth_rate(times, amplitudes, window=None) -> float:
    """Exponential growth rate from a straight-line fit to ``ln |amplitude|``.

    ``window`` is an explicit ``(start, end)`` in time. Leaving it out uses
    :func:`exponential_window` to find the band automatically.
    """
    t = np.asarray(times, dtype=float)
    a = np.abs(np.asarray(amplitudes, dtype=float))
    if window is None:
        t, a = exponential_window(t, a)
    else:
        lo, hi = window
        keep = (t >= lo) & (t <= hi)
        t, a = t[keep], a[keep]
    if len(t) < 3:
        raise ValueError("need at least three points inside the window")
    return float(np.polyfit(t, np.log(a), 1)[0])
