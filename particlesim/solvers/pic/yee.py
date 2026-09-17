"""Yee finite-difference time-domain Maxwell solver (design doc Section 5.4).

Units are normalized: ``c = epsilon_0 = mu_0 = 1``. Lengths and times are
therefore in the same unit, and a plane wave in vacuum advances one length
unit per time unit. Picking a physical scale is the caller's business and
belongs in :mod:`particlesim.core.units`, not here.

The lattice is Yee's: each field component sits where the curl that drives
it is naturally centred, so both curls are exact centred differences and the
scheme is second order in space and time without any component ever needing
to be interpolated. Writing ``(i, j, k)`` for a cell corner,

    Ex, Dx  (i+1/2, j,     k    )      Bx, Hx  (i,     j+1/2, k+1/2)
    Ey, Dy  (i,     j+1/2, k    )      By, Hy  (i+1/2, j,     k+1/2)
    Ez, Dz  (i,     j,     k+1/2)      Bz, Hz  (i+1/2, j+1/2, k    )

A one-dimensional run drops the ``j`` and ``k`` indices and a two-dimensional
run drops ``k``; the component layout is unchanged, which is why the same
update serves both. Derivatives along a dropped axis are zero, so in 1D the
scheme reduces on its own to a static ``Bx``, a longitudinal ``Dx`` driven
only by the current, and two decoupled transverse polarizations.

The evolved pair is ``(D, B)`` rather than ``(E, B)``:

    dD/dt = curl H(D, B) - J
    dB/dt = -curl E(D, B)

with the medium supplying ``E`` and ``H``. In vacuum this is an identity and
costs nothing, but it is what lets a nonlinear electrodynamics -- Born-Infeld
among them -- enter as a constitutive relation rather than as a fork of the
solver.

Time is staggered by half a step: ``D`` lives on integer steps and ``B`` on
half-integer ones, which makes both updates centred and the scheme
time-reversible up to the current.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

COMPONENTS = ("x", "y", "z")


@dataclass(frozen=True)
class YeeGrid:
    """A uniform Cartesian grid in one or two dimensions.

    ``shape`` and ``spacing`` must have the same length, which is the
    dimensionality. Three dimensions belong to Milestone 4 and are refused
    here rather than half-supported.
    """

    shape: tuple[int, ...]
    spacing: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.shape) != len(self.spacing):
            raise ValueError("shape and spacing must have the same length")
        if not 1 <= len(self.shape) <= 2:
            raise ValueError("only one- and two-dimensional grids are supported")
        if any(n < 2 for n in self.shape):
            raise ValueError("each axis needs at least two cells")
        if any(d <= 0 for d in self.spacing):
            raise ValueError("spacings must be positive")

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def extent(self) -> tuple[float, ...]:
        return tuple(n * d for n, d in zip(self.shape, self.spacing, strict=True))

    @property
    def courant_limit(self) -> float:
        """Largest stable step, ``1 / sqrt(sum 1/dx_i^2)`` with ``c = 1``.

        This is the Courant-Friedrichs-Lewy limit for the Yee scheme. At
        exactly this value in one dimension the scheme becomes exact -- the
        "magic time step" -- which is a poor default precisely because it
        hides every dispersion error the scheme would otherwise show.
        """
        return 1.0 / np.sqrt(sum(1.0 / d**2 for d in self.spacing))

    def zeros(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=float)

    def coordinates(self, offsets: Sequence[float]) -> tuple[np.ndarray, ...]:
        """Meshed coordinates of a component offset by ``offsets`` cells.

        ``offsets`` is in units of cells, so ``0.5`` means the half-cell
        stagger. Only the first ``ndim`` entries are used.
        """
        axes = [
            (np.arange(n) + off) * d
            for n, d, off in zip(self.shape, self.spacing, offsets, strict=False)
        ]
        return tuple(np.meshgrid(*axes, indexing="ij"))


#: Half-cell offsets of each component, in the order (x, y, z).
E_OFFSETS = {"x": (0.5, 0.0, 0.0), "y": (0.0, 0.5, 0.0), "z": (0.0, 0.0, 0.5)}
B_OFFSETS = {"x": (0.0, 0.5, 0.5), "y": (0.5, 0.0, 0.5), "z": (0.5, 0.5, 0.0)}


@dataclass(frozen=True)
class Fields:
    """Electric displacement and magnetic induction on a Yee lattice.

    ``D`` is held at integer time steps and ``B`` at half-integer ones, so a
    :class:`Fields` is a snapshot of a leapfrog rather than of a single
    instant. ``time`` is the time of ``D``.
    """

    Dx: np.ndarray
    Dy: np.ndarray
    Dz: np.ndarray
    Bx: np.ndarray
    By: np.ndarray
    Bz: np.ndarray
    time: float = 0.0

    @classmethod
    def zeros(cls, grid: YeeGrid) -> Fields:
        return cls(*(grid.zeros() for _ in range(6)))

    @property
    def D(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.Dx, self.Dy, self.Dz)

    @property
    def B(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.Bx, self.By, self.Bz)

    def with_D(self, D, time: float | None = None) -> Fields:
        return replace(self, Dx=D[0], Dy=D[1], Dz=D[2], time=self.time if time is None else time)

    def with_B(self, B) -> Fields:
        return replace(self, Bx=B[0], By=B[1], Bz=B[2])


class Constitutive(Protocol):
    """Maps the evolved pair ``(D, B)`` to the pair the curls act on.

    A medium implementing this must deal with the staggering itself. ``Dx``
    and ``Bx`` do not live at the same lattice point, so any relation that
    genuinely mixes them has to say how it averages one onto the other, and
    there is no single right answer to hide in the solver.
    """

    def electric(self, D, B) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def magnetic(self, D, B) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


class Vacuum:
    """``E = D`` and ``H = B``, the normalized vacuum relation."""

    def electric(self, D, B):
        return D

    def magnetic(self, D, B):
        return B

    def energy_density(self, D, B):
        """``(E.D + H.B) / 2``, which for a linear medium is the field energy."""
        return 0.5 * sum(d**2 + b**2 for d, b in zip(D, B, strict=True))


def _forward(f: np.ndarray, axis: int, delta: float) -> np.ndarray:
    """Difference from integer to half-integer positions along ``axis``."""
    return (np.roll(f, -1, axis=axis) - f) / delta


def _backward(f: np.ndarray, axis: int, delta: float) -> np.ndarray:
    """Difference from half-integer to integer positions along ``axis``."""
    return (f - np.roll(f, 1, axis=axis)) / delta


def yee_frequency(k: Sequence[float], spacing: Sequence[float], dt: float) -> float:
    """Angular frequency of a plane wave on the Yee lattice.

    Solves the scheme's own dispersion relation

        sin(omega dt / 2)^2 / dt^2 = sum_i sin(k_i dx_i / 2)^2 / dx_i^2

    which differs from the continuum ``omega = |k|`` by an amount that
    vanishes as the second power of the spacing. Reproducing this rather than
    the continuum relation is what tells a Yee implementation apart from one
    that merely looks like a wave solver.

    Raises if the requested mode cannot propagate, which happens when the
    right-hand side exceeds ``1/dt^2``. Below the Courant limit no mode on
    the grid can do that, so reaching this means the step is already past
    the limit; past it the scheme is evanescent rather than merely
    inaccurate, and an ordinary-looking frequency would be a fiction.
    """
    rhs = sum((np.sin(ki * di / 2.0) / di) ** 2 for ki, di in zip(k, spacing, strict=True))
    s = np.sqrt(rhs) * dt
    if s > 1.0:
        raise ValueError(
            f"this mode does not propagate on the lattice: sin(omega dt/2) would "
            f"have to be {s:.4f}. Refine the grid or lower the time step"
        )
    return float(2.0 * np.arcsin(s) / dt)


def plane_wave(
    grid: YeeGrid,
    k: Sequence[float],
    dt: float,
    amplitude: float = 1.0,
    polarization: str = "TM",
    phase: float = 0.0,
) -> Fields:
    """The exact discrete plane wave of the Yee scheme.

    Not a sampled continuum solution. The amplitudes below are the ones that
    make the discrete update reproduce the mode step for step, so a run
    started from this state stays on it to round-off, and any drift is the
    implementation's rather than the discretization's.

    ``polarization`` is ``"TM"`` for the mode carrying ``Ez``, ``Bx`` and
    ``By``, or ``"TE"`` for the one carrying ``Ex``, ``Ey`` and ``Bz``. In one
    dimension the two are the independent transverse polarizations.

    ``B`` is placed half a step behind ``D``, as the leapfrog expects.
    """
    if polarization not in ("TE", "TM"):
        raise ValueError("polarization must be 'TE' or 'TM'")
    k = tuple(k)
    if len(k) != grid.ndim:
        raise ValueError(f"k needs {grid.ndim} components for this grid")

    omega = yee_frequency(k, grid.spacing, dt)
    sin_half = np.sin(omega * dt / 2.0)
    kk = k + (0.0,) * (3 - len(k))
    dd = tuple(grid.spacing) + (1.0,) * (3 - grid.ndim)

    def phase_at(offsets, t):
        coords = grid.coordinates(offsets)
        total = sum(ki * c for ki, c in zip(kk, coords, strict=False))
        return total - omega * t + phase

    fields = Fields.zeros(grid)
    # The B amplitudes come from matching the discrete update term by term;
    # see the module docstring's reference to the dispersion relation.
    if polarization == "TM":
        Ez = amplitude * np.cos(phase_at(E_OFFSETS["z"], 0.0))
        Bx0 = amplitude * (dt / dd[1]) * np.sin(kk[1] * dd[1] / 2.0) / sin_half
        By0 = -amplitude * (dt / dd[0]) * np.sin(kk[0] * dd[0] / 2.0) / sin_half
        fields = replace(
            fields,
            Dz=Ez,
            Bx=Bx0 * np.cos(phase_at(B_OFFSETS["x"], -dt / 2.0)),
            By=By0 * np.cos(phase_at(B_OFFSETS["y"], -dt / 2.0)),
        )
    else:
        Bz0 = amplitude
        Ex0 = -amplitude * (dt / dd[1]) * np.sin(kk[1] * dd[1] / 2.0) / sin_half
        Ey0 = amplitude * (dt / dd[0]) * np.sin(kk[0] * dd[0] / 2.0) / sin_half
        fields = replace(
            fields,
            Dx=Ex0 * np.cos(phase_at(E_OFFSETS["x"], 0.0)),
            Dy=Ey0 * np.cos(phase_at(E_OFFSETS["y"], 0.0)),
            Bz=Bz0 * np.cos(phase_at(B_OFFSETS["z"], -dt / 2.0)),
        )
    return fields


class YeeSolver:
    """Leapfrog Maxwell on the Yee lattice."""

    def __init__(
        self,
        grid: YeeGrid,
        dt: float | None = None,
        courant: float = 0.5,
        medium: Constitutive | None = None,
        boundary=None,
    ):
        from particlesim.solvers.pic.boundaries import Periodic

        self.grid = grid
        if dt is None:
            dt = courant * grid.courant_limit
        limit = grid.courant_limit
        if dt > limit * (1 + 1e-12):
            raise ValueError(
                f"dt = {dt:g} exceeds the Courant limit {limit:g} for this grid. "
                "The Yee scheme is unconditionally unstable above it, so this "
                "would not merely be inaccurate"
            )
        self.dt = float(dt)
        self.medium = medium if medium is not None else Vacuum()
        self.boundary = boundary if boundary is not None else Periodic()
        self.boundary.attach(self)

    # --- curls ----------------------------------------------------------

    def curl_E(self, E) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``curl E`` at the ``B`` positions, using forward differences.

        Terms along an axis the grid does not have are zero, which is what
        reduces the update to the right thing in one and two dimensions
        without a separate code path.
        """
        d = self.boundary.forward
        nd = self.grid.ndim
        h = self.grid.spacing
        zero = 0.0
        dy_Ez = d(E[2], 1, h[1], "Ez_y") if nd > 1 else zero
        dz_Ey = zero
        dz_Ex = zero
        dx_Ez = d(E[2], 0, h[0], "Ez_x")
        dx_Ey = d(E[1], 0, h[0], "Ey_x")
        dy_Ex = d(E[0], 1, h[1], "Ex_y") if nd > 1 else zero
        return (dy_Ez - dz_Ey, dz_Ex - dx_Ez, dx_Ey - dy_Ex)

    def curl_H(self, H) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``curl H`` at the ``D`` positions, using backward differences."""
        d = self.boundary.backward
        nd = self.grid.ndim
        h = self.grid.spacing
        zero = 0.0
        dy_Hz = d(H[2], 1, h[1], "Hz_y") if nd > 1 else zero
        dz_Hy = zero
        dz_Hx = zero
        dx_Hz = d(H[2], 0, h[0], "Hz_x")
        dx_Hy = d(H[1], 0, h[0], "Hy_x")
        dy_Hx = d(H[0], 1, h[1], "Hx_y") if nd > 1 else zero
        return (dy_Hz - dz_Hy, dz_Hx - dx_Hz, dx_Hy - dy_Hx)

    # --- evolution ------------------------------------------------------

    def advance_magnetic(self, fields: Fields) -> Fields:
        """Carry ``B`` from one half step to the next."""
        cE = self.curl_E(self.medium.electric(fields.D, fields.B))
        B = tuple(b - self.dt * c for b, c in zip(fields.B, cE, strict=True))
        return fields.with_B(self.boundary.after_magnetic(B))

    def advance_electric(self, fields: Fields, current=None) -> Fields:
        """Carry ``D`` forward a full step against ``B`` at the half step.

        ``current`` is ``J`` at that half step, laid out like ``D``.
        """
        dt = self.dt
        cH = self.curl_H(self.medium.magnetic(fields.D, fields.B))
        if current is None:
            D = tuple(d + dt * c for d, c in zip(fields.D, cH, strict=True))
        else:
            D = tuple(d + dt * (c - j) for d, c, j in zip(fields.D, cH, current, strict=True))
        return fields.with_D(self.boundary.after_electric(D), time=fields.time + dt)

    def step(self, fields: Fields, current=None) -> Fields:
        """Advance ``B`` by a half step, then ``D`` by a full one.

        ``current`` is ``J`` at the half step, in the same component layout
        as ``D``. Passing ``None`` is vacuum propagation.
        """
        return self.advance_electric(self.advance_magnetic(fields), current)

    def run(self, fields: Fields, steps: int, current=None) -> Fields:
        for _ in range(steps):
            fields = self.step(fields, current)
        return fields

    # --- diagnostics ----------------------------------------------------

    def energy(self, fields: Fields) -> float:
        """``(1/2) integral (E.D + H.B)``, in normalized units.

        The electric and magnetic halves are half a time step apart, so this
        does not sit still even on an exact discrete mode: it oscillates at
        twice the wave frequency with a relative amplitude of about
        ``omega dt / 4``, measured and confirmed to halve when ``dt`` does.
        That is an artefact of reading a leapfrog at one instant, not a leak,
        and the useful statement is that the oscillation is bounded. A run
        going unstable shows up as growth through that band, long before the
        fields overflow.

        For a travelling wave the two halves are in phase and the value is
        conserved to round-off instead.

        A medium supplying ``energy_density`` is asked for it. Nonlinear
        electrodynamics does not store ``(E.D + H.B)/2``: Born-Infeld's
        energy is its Hamiltonian, and using the linear expression there
        would report a number that is not conserved and not the energy.
        """
        cell = float(np.prod(self.grid.spacing))
        density = getattr(self.medium, "energy_density", None)
        if density is not None:
            return float(np.sum(density(fields.D, fields.B))) * cell
        E = self.medium.electric(fields.D, fields.B)
        H = self.medium.magnetic(fields.D, fields.B)
        total = sum(float(np.sum(e * d)) for e, d in zip(E, fields.D, strict=True))
        total += sum(float(np.sum(h * b)) for h, b in zip(H, fields.B, strict=True))
        return 0.5 * total * cell

    def divergence_D(self, fields: Fields) -> np.ndarray:
        """``div D`` at cell corners, the quantity Gauss's law constrains.

        Backward differences, matching the positions ``curl H`` is evaluated
        at, so that the discrete identity ``div curl = 0`` holds exactly and
        ``div D`` can only change through the current.
        """
        h = self.grid.spacing
        out = _backward(fields.Dx, 0, h[0])
        if self.grid.ndim > 1:
            out = out + _backward(fields.Dy, 1, h[1])
        return out
