"""One-dimensional laser wakefield (design doc Section 10, Milestone 2).

A laser pulse pushes electrons out of its way by the ponderomotive force,
and they oscillate back through the ions behind it. The resulting plasma
wave trails the pulse at nearly the speed of light and carries a
longitudinal field that can be orders of magnitude above what a metal
cavity survives, which is the point of the whole field.

In one dimension and the quasi-static limit -- the pulse's own shape frozen
over the time an electron takes to cross it -- the wake potential obeys one
ordinary differential equation,

    d^2 phi / ds^2 = (k_p^2 / 2) [ (1 + a^2(s)) / (1 + phi)^2 - 1 ]

with ``s = ct - z`` measuring distance behind the pulse front, ``phi`` the
potential in units of ``mc^2/e``, and ``a^2`` the cycle-averaged normalized
vector potential. The longitudinal field follows as

    E_z / E_wb = (1 / k_p) d phi / ds,      E_wb = m c omega_p / e

This is not a linearization. It is the full cold one-dimensional result, and
it is what a particle-in-cell run is held to here, integrated numerically
rather than approximated by a formula. :func:`linear_wake_amplitude` gives
the small-amplitude closed form, and the nonlinear solver is checked against
it in the limit where the two must agree rather than being trusted on its
own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from particlesim.core.interpolate import midpoints
from particlesim.solvers.pic.boundaries import Conducting, PerfectlyMatchedLayer, Periodic
from particlesim.solvers.pic.cycle import advance
from particlesim.solvers.pic.deposition import deposit_charge
from particlesim.solvers.pic.laser import LaserPulse, PlaneWaveSource
from particlesim.solvers.pic.particles import PUSHERS, Species
from particlesim.solvers.pic.shapes import SUPPORTED_ORDERS
from particlesim.solvers.pic.yee import Fields, YeeGrid, YeeSolver


def linear_wake_amplitude(a0: float, sigma: float, plasma_wavenumber: float) -> float:
    """``(sqrt(pi)/4) a0^2 k_p sigma exp(-k_p^2 sigma^2 / 4)``.

    The wake left behind a Gaussian pulse ``a = a0 exp(-s^2 / 2 sigma^2)``
    when ``a0`` is small enough that the plasma responds linearly. It peaks
    at ``k_p sigma = sqrt(2)``, where a pulse is called resonant, giving
    ``0.3801 a0^2``: making the pulse longer than that does not help, because
    the exponential in the pulse length takes back what the length gives.

    The cycle average is already in this: the drive is ``a^2 / 2`` because a
    linearly polarized carrier spends half its time near zero.
    """
    k = plasma_wavenumber
    return float(np.sqrt(np.pi) / 4.0 * a0**2 * k * sigma * np.exp(-(k**2) * sigma**2 / 4.0))


def resonant_length(plasma_wavenumber: float) -> float:
    """``sqrt(2) / k_p``, the pulse length that drives the largest wake."""
    return float(np.sqrt(2.0) / plasma_wavenumber)


@dataclass(frozen=True)
class WakeSolution:
    """The wake behind a pulse, on the co-moving coordinate ``s``."""

    s: np.ndarray
    potential: np.ndarray
    field: np.ndarray
    plasma_wavenumber: float

    def amplitude(self, behind: float) -> float:
        """Peak ``|E_z| / E_wb`` at ``s`` beyond ``behind``.

        Taken behind the pulse rather than over the whole solution, because
        inside the pulse the field is the ponderomotive push itself and not
        the wake it leaves.
        """
        keep = self.s > behind
        if not np.any(keep):
            raise ValueError(f"the solution does not extend past s = {behind:g}")
        return float(np.abs(self.field[keep]).max())


def nonlinear_wake(s: np.ndarray, a_squared: np.ndarray, plasma_wavenumber: float) -> WakeSolution:
    """Integrate the cold one-dimensional wake equation behind a pulse.

    Fourth-order Runge-Kutta on a uniform ``s``. The source carries
    ``1 / (1 + phi)^2``, which stiffens as the wake steepens, and a step too
    coarse for it walks straight past ``1 + phi = 0`` into a region where
    the solution is finite, smooth, and meaningless. The integration refuses
    there instead.

    Refusing is not the same as saying the drive is too strong. At
    ``a0 = 30`` and a resonant pulse the refusal appears at two thousand
    points and disappears at twenty thousand, so the first thing to try is a
    finer ``s``. If it survives refinement, the wake really has reached
    ``1 + phi = 0``, which is where a cold fluid stops having a solution at
    all.

    **The drive at the half steps goes through** :func:`midpoints`. The
    Runge-Kutta stages want ``a^2`` halfway between the points it was sampled
    at, and this averaged the two neighbours for it, which is second-order
    accurate. Four stages do not repair a second-order integrand: the error
    enters each step with an ``O(ds)`` weight and there are ``1/ds`` steps,
    so the solve measured 2.00 where the scheme claims 4. Interpolating to
    fourth order takes the error at the default twenty thousand points from
    ``1.3e-8`` to ``7.1e-13``.
    """
    s = np.asarray(s, dtype=float)
    a_squared = np.asarray(a_squared, dtype=float)
    if s.shape != a_squared.shape:
        raise ValueError("s and a_squared must have the same shape")
    if len(s) < 3:
        raise ValueError("need at least three points")
    ds = float(s[1] - s[0])
    half = 0.5 * (k := plasma_wavenumber) ** 2

    def source(a2: float, phi: float) -> float:
        if 1.0 + phi <= 1e-9:
            raise ValueError(
                "the integration reached 1 + phi = 0, past which the cold "
                "fluid solution is finite, smooth and meaningless. Refine s "
                "first: this is usually a step too coarse for how the source "
                "stiffens. If it survives refinement, the wake has genuinely "
                "reached wave breaking"
            )
        return half * ((1.0 + a2) / (1.0 + phi) ** 2 - 1.0)

    phi = np.zeros_like(s)
    dphi = np.zeros_like(s)
    midpoint = midpoints(a_squared)
    for i in range(len(s) - 1):
        p, d = phi[i], dphi[i]
        k1p, k1d = d, source(a_squared[i], p)
        k2p, k2d = d + 0.5 * ds * k1d, source(midpoint[i], p + 0.5 * ds * k1p)
        k3p, k3d = d + 0.5 * ds * k2d, source(midpoint[i], p + 0.5 * ds * k2p)
        k4p, k4d = d + ds * k3d, source(a_squared[i + 1], p + ds * k3p)
        phi[i + 1] = p + ds / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
        dphi[i + 1] = d + ds / 6.0 * (k1d + 2 * k2d + 2 * k3d + k4d)
    return WakeSolution(s, phi, dphi / k, float(k))


def gaussian_wake(
    a0: float,
    sigma: float,
    plasma_wavenumber: float,
    front: float = 4.0,
    length: float = 6.0,
    points: int = 20001,
) -> WakeSolution:
    """The wake behind a Gaussian pulse, in units of the pulse and plasma scales.

    ``front`` is how many ``sigma`` ahead of the peak the integration starts,
    and ``length`` how many plasma wavelengths of wake to follow. The
    cycle-averaged drive is ``a^2 / 2``: a linearly polarized carrier spends
    half its time near zero, and using the envelope's square instead
    overestimates the wake by a factor of two.
    """
    s = np.linspace(0.0, front * sigma + length * 2 * np.pi / plasma_wavenumber, points)
    peak = front * sigma
    a_squared = 0.5 * a0**2 * np.exp(-((s - peak) ** 2) / sigma**2)
    return nonlinear_wake(s, a_squared, plasma_wavenumber)


def uniform_plasma(
    grid: YeeGrid,
    density: float,
    per_cell: int = 20,
    start: float = 0.0,
    ramp: float = 0.0,
    order: int = 1,
) -> tuple[Fields, Species, np.ndarray]:
    """Electrons on a neutralizing ion background, filling ``x >= start``.

    ``ramp`` softens the leading edge over that distance. A step in density
    launches a wake of its own, at the same wavelength as the one being
    measured and with no way to tell them apart afterwards, so the ramp is
    not cosmetic.

    The ions sit exactly where the electrons start, so the initial field is
    zero everywhere rather than solved for. That is not a shortcut, it is
    what neutral means, and spreading the ions uniformly over the box
    instead -- which is right when the plasma fills it and wrong here --
    leaves net negative charge in the plasma and net positive in the vacuum.
    Measured, that mistake puts six times the wave-breaking field on the
    initial slice, and since the cycle conserves ``div D - rho`` exactly, it
    never goes away.

    The ions are immobile and carry no current, so they never appear again.
    The cycle's invariant then holds their density: ``div D - rho_e`` starts
    at ``rho_ion`` and stays there.
    """
    length = grid.extent[0]
    count = grid.shape[0] * per_cell
    positions = ((np.arange(count) + 0.5) / count * length).reshape(count, 1)
    positions = positions[positions[:, 0] >= start]

    profile = np.ones(len(positions))
    if ramp > 0:
        edge = np.clip((positions[:, 0] - start) / ramp, 0.0, 1.0)
        profile = np.sin(0.5 * np.pi * edge) ** 2
    weight = density * length / count * profile

    species = Species.create(-1.0, 1.0, positions, weight=weight, name="electrons")
    ion_density = -deposit_charge(grid, species, order)
    fields = Fields(*(grid.zeros() for _ in range(6)))
    return fields, species, ion_density


def measure_wake(
    grid: YeeGrid,
    fields: Fields,
    plasma_frequency: float,
    pulse_width: float,
    plasma_start: float = 0.0,
) -> tuple[float, float]:
    """Peak ``|E_z| / E_wb`` behind the pulse, and where the pulse is.

    The pulse is located by its transverse field, and the search starts
    ``3 * pulse_width`` behind that so the ponderomotive push inside the
    pulse is not counted as wake. It stops at ``plasma_start`` because the
    vacuum ahead of the plasma holds no wake and the density ramp holds a
    transient.
    """
    x = grid.coordinates((0.0,))[0]
    transverse = np.abs(fields.Dz) + np.abs(fields.Dy)
    head = float(x[int(transverse.argmax())])
    region = (x < head - 3.0 * pulse_width) & (x > plasma_start + 3.0 * pulse_width)
    if not np.any(region):
        raise ValueError(
            "no wake region between the density ramp and the pulse; the pulse "
            "has not travelled far enough into the plasma"
        )
    return float(np.abs(fields.Dx[region]).max() / plasma_frequency), head


#: The field boundaries a :class:`LaserWakefield` run can close its box with.
BOUNDARIES = ("pml", "conducting", "periodic")


@dataclass(frozen=True)
class LaserWakefield:
    """A one-dimensional laser-wakefield run, as data.

    Everything :meth:`run` does is fixed by these fields, so a run can be
    stored, compared and handed to another code (issue #82's WarpX adapter)
    rather than living in the body of a test.

    Units are :mod:`~particlesim.solvers.pic.yee`'s: ``c = 1``, and an
    electron has charge ``-1`` and mass ``1``, so ``density`` is
    ``omega_p^2``. The length unit is the caller's; the tests use the laser
    wavelength.

    The box is ``cells`` cells of ``spacing``. With ``boundary="pml"`` the
    outer ``pml_cells`` at each end are an absorbing layer inside that box.
    The pulse enters by total-field/scattered-field at ``source_index``,
    where its peak passes at ``pulse.start``. Electrons fill everything from
    ``plasma_start`` on, rising over ``ramp`` as :func:`uniform_plasma`
    describes, on an immobile neutralizing background. ``steps`` cycles of
    :func:`~particlesim.solvers.pic.cycle.advance` follow, at
    ``courant`` times the Courant limit.
    """

    pulse: LaserPulse
    density: float
    cells: int
    spacing: float
    steps: int
    plasma_start: float = 0.0
    ramp: float = 0.0
    per_cell: int = 16
    source_index: int = 8
    courant: float = 0.5
    boundary: str = "pml"
    pml_cells: int = 12
    order: int = 1
    pusher: str = "boris"

    def __post_init__(self) -> None:
        if self.cells < 2 or self.spacing <= 0:
            raise ValueError("need at least two cells of positive spacing")
        if self.density < 0 or self.steps < 0 or self.per_cell < 1:
            raise ValueError("density and steps must be non-negative, per_cell positive")
        if self.plasma_start < 0 or self.ramp < 0:
            raise ValueError("plasma_start and ramp must be non-negative")
        if not 1 <= self.source_index < self.cells - 1:
            raise ValueError(f"the source must lie inside the box, got {self.source_index}")
        if self.boundary not in BOUNDARIES:
            raise ValueError(f"boundary must be one of {BOUNDARIES}")
        if self.order not in SUPPORTED_ORDERS:
            raise ValueError(f"order must be one of {SUPPORTED_ORDERS}")
        if self.pusher not in PUSHERS:
            raise ValueError(f"pusher must be one of {sorted(PUSHERS)}")

    @property
    def grid(self) -> YeeGrid:
        return YeeGrid((self.cells,), (self.spacing,))

    @property
    def plasma_frequency(self) -> float:
        return float(np.sqrt(self.density))

    def solver(self) -> YeeSolver:
        if self.boundary == "pml":
            boundary = PerfectlyMatchedLayer(thickness=self.pml_cells)
        elif self.boundary == "conducting":
            boundary = Conducting()
        else:
            boundary = Periodic()
        return YeeSolver(self.grid, courant=self.courant, boundary=boundary)

    def run(self) -> tuple[Fields, Species]:
        """The fields and electrons after ``steps`` cycles."""
        solver = self.solver()
        source = PlaneWaveSource(self.pulse, index=self.source_index)
        source.attach(solver)
        fields, species, _ = uniform_plasma(
            solver.grid, self.density, self.per_cell, self.plasma_start, self.ramp, self.order
        )
        for _ in range(self.steps):
            fields, species = advance(
                solver, fields, species, order=self.order, scheme=self.pusher, source=source
            )
        return fields, species

    def wake(self, fields: Fields) -> float:
        """Peak ``|E_x| / E_wb`` behind the pulse, by :func:`measure_wake`."""
        measured, _ = measure_wake(
            self.grid, fields, self.plasma_frequency, self.pulse.tau, self.plasma_start
        )
        return measured


def resonant_benchmark(
    a0: float,
    cells_per_wavelength: int = 16,
    per_cell: int = 16,
    density_ratio: float = 0.1,
) -> LaserWakefield:
    """Issue #34's benchmark: a resonant Gaussian pulse into an underdense ramp.

    Lengths are in laser wavelengths. ``density_ratio`` is ``omega_p /
    omega_0``, and the pulse is resonant: ``k_p sigma = sqrt(2)`` for
    ``a = a0 exp(-s^2 / 2 sigma^2)``. The box is 110 wavelengths with
    24-cell absorbing layers, and the source sits at cell 120. The plasma
    starts 22 wavelengths beyond the source, rising over 3. The run lasts
    until the peak has gone 56 wavelengths past the source, at half the
    Courant limit.
    """
    omega0 = 2.0 * np.pi
    k_p = density_ratio * omega0
    sigma = resonant_length(k_p)
    pulse = LaserPulse(a0=a0, wavelength=1.0, duration=sigma * np.sqrt(2) * np.sqrt(2 * np.log(2)))
    dx = 1.0 / cells_per_wavelength
    courant = 0.5
    return LaserWakefield(
        pulse=pulse,
        density=k_p**2,
        cells=int(round(110.0 / dx)),
        spacing=dx,
        steps=int(round((pulse.start + 56.0) / (courant * dx))),
        plasma_start=120 * dx + 22.0,
        ramp=3.0,
        per_cell=per_cell,
        source_index=120,
        courant=courant,
        boundary="pml",
        pml_cells=24,
    )
