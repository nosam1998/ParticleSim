"""Laser injection and the moving window (design doc Section 3.3, Milestone 2).

A laser has to enter the box without also appearing behind the point it
enters at. The naive way, adding a current at one cell, radiates
symmetrically: half the energy goes the wrong way, and unless it is absorbed
it comes back. The total-field/scattered-field construction avoids that
instead of cleaning up after it.

Split the grid at a plane. To the right of it every array holds the total
field, to the left only the scattered field. Every Yee update that reaches
across the plane then mixes the two conventions, and there are exactly two
such updates in one dimension. Correcting them by the incident field turns
the plane into a one-way source: the incident wave appears travelling right
and nothing at all appears travelling left, to the accuracy with which the
incident field is known on the lattice.

That accuracy has a floor, and it is the lattice's own dispersion. A pulse
carries a spread of wavenumbers, each travelling at its own numerical phase
velocity, so no single analytic incident field is exact for all of them. The
residue leaks backward through the plane. Evaluating the incident field at
the phase velocity of the carrier rather than at ``c`` shrinks it by about
an order of magnitude, which is what :class:`PlaneWaveSource` does.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from particlesim.solvers.pic.particles import Species
from particlesim.solvers.pic.yee import Fields, YeeGrid, YeeSolver, yee_frequency

ENVELOPES = ("gaussian", "sin2")


@dataclass(frozen=True)
class LaserPulse:
    """A linearly polarized pulse, specified the way an experiment would be.

    ``a0`` is the normalized vector potential ``eA/mc``, which for these
    units makes the peak electric field ``a0 * omega0``. ``duration`` is the
    intensity full width at half maximum, the quantity a pulse is actually
    quoted by; the field envelope is therefore wider than it by ``sqrt(2)``.

    ``energy`` is the analytic energy per unit area the pulse carries, which
    is what an injection is checked against. For a Gaussian field envelope
    ``exp(-(t/tau)^2)`` with a carrier, the cycle average of ``cos^2`` gives

        integral (E^2 + B^2) / 2 dt = E0^2 integral g(t)^2 cos^2 dt
                                    = E0^2 tau sqrt(pi/2) / 2

    with ``tau`` the field envelope's ``1/e`` half-width. The carrier
    contributes the factor of a half exactly once the pulse is many cycles
    long, which is the regime the formula is used in and the docstring of
    :meth:`energy` says so.
    """

    a0: float = 1.0
    wavelength: float = 1.0
    duration: float = 10.0
    polarization: str = "z"
    delay: float | None = None
    envelope: str = "gaussian"
    phase: float = 0.0

    def __post_init__(self) -> None:
        if self.wavelength <= 0 or self.duration <= 0:
            raise ValueError("wavelength and duration must be positive")
        if self.polarization not in ("y", "z"):
            raise ValueError("polarization must be 'y' or 'z'")
        if self.envelope not in ENVELOPES:
            raise ValueError(f"envelope must be one of {ENVELOPES}")

    @property
    def omega(self) -> float:
        return 2.0 * np.pi / self.wavelength

    @property
    def peak_field(self) -> float:
        """``E0 = a0 omega0`` in normalized units."""
        return self.a0 * self.omega

    @property
    def tau(self) -> float:
        """Field-envelope ``1/e`` half-width, from the intensity FWHM."""
        return self.duration / np.sqrt(2.0 * np.log(2.0))

    @property
    def start(self) -> float:
        """When the peak arrives. Far enough out that the pulse starts near zero."""
        return 3.0 * self.tau if self.delay is None else self.delay

    def envelope_at(self, t):
        t = np.asarray(t, dtype=float)
        s = (t - self.start) / self.tau
        if self.envelope == "gaussian":
            return np.exp(-(s**2))
        half = 0.5 * np.pi * np.sqrt(2.0)  # sin^2 support matched to the FWHM
        return np.where(np.abs(s) < half, np.cos(0.5 * np.pi * s / half) ** 2, 0.0)

    def field_at(self, t):
        """The incident transverse electric field at the source plane."""
        t = np.asarray(t, dtype=float)
        return (
            self.peak_field
            * self.envelope_at(t)
            * np.cos(self.omega * (t - self.start) + self.phase)
        )

    @property
    def energy(self) -> float:
        """Energy per unit area, ``E0^2 tau sqrt(pi/2) / 2`` for a Gaussian.

        Exact only in the many-cycle limit, where the carrier's ``cos^2``
        averages to a half independently of the envelope. At two cycles the
        cross term is percent-level and this stops being the right thing to
        compare against, so :func:`numeric_pulse_energy` is what the tests
        use and this is the closed form it is checked against.
        """
        if self.envelope != "gaussian":
            raise ValueError("the closed form is for the Gaussian envelope")
        return 0.5 * self.peak_field**2 * self.tau * np.sqrt(np.pi / 2.0)

    def cycles(self) -> float:
        """Field-envelope half-widths per carrier period, the many-cycle test."""
        return self.tau / self.wavelength


def numeric_pulse_energy(pulse: LaserPulse, dt: float, span: float = 8.0) -> float:
    """Energy per unit area by quadrature on the specified waveform.

    Carries no approximation about the number of cycles, so it is the right
    thing to hold an injection to. The window is ``span`` envelope widths
    either side of the peak, beyond which a Gaussian contributes nothing a
    double can represent.
    """
    t = np.arange(pulse.start - span * pulse.tau, pulse.start + span * pulse.tau, dt)
    return float(np.trapezoid(pulse.field_at(t) ** 2, t))


class PlaneWaveSource:
    """One-way injection at a plane, by total-field/scattered-field.

    ``index`` is the cell the plane sits at: cells at or above it hold the
    total field, cells below hold only what scatters back. Both corrections
    below follow from that single convention rather than from tuning.
    """

    def __init__(self, pulse: LaserPulse, index: int = 8):
        self.pulse = pulse
        self.index = int(index)

    def attach(self, solver: YeeSolver) -> None:
        grid = solver.grid
        if grid.ndim != 1:
            raise ValueError(
                "this source is one-dimensional; a two-dimensional box needs a "
                "correction on every face, which is issue #37's work"
            )
        if not 1 <= self.index < grid.shape[0] - 1:
            raise ValueError(f"the source plane must lie inside the grid, got {self.index}")
        self.solver = solver
        self.dx = grid.spacing[0]
        self.dt = solver.dt
        # Phase velocity of the carrier on this lattice, not c. A pulse
        # spans wavenumbers and no single velocity is right for all of them,
        # but the carrier's is right for most of the energy.
        k = 2.0 * np.pi / self.pulse.wavelength
        self.velocity = yee_frequency((k,), grid.spacing, solver.dt) / k

    def _electric_component(self) -> str:
        return "Dz" if self.pulse.polarization == "z" else "Dy"

    def _magnetic_component(self) -> str:
        # Ez pairs with By, Ey with Bz.
        return "By" if self.pulse.polarization == "z" else "Bz"

    def after_magnetic(self, fields: Fields, time: float) -> Fields:
        """Remove the incident electric field from the update that reached across.

        The sign follows the polarization because the two curls do:
        ``dBy/dt = +dEz/dx`` but ``dBz/dt = -dEy/dx``.
        """
        name = self._magnetic_component()
        array = np.array(getattr(fields, name), copy=True)
        sign = 1.0 if self.pulse.polarization == "z" else -1.0
        array[self.index - 1] -= sign * (self.dt / self.dx) * float(self.pulse.field_at(time))
        return replace(fields, **{name: array})

    def after_electric(self, fields: Fields, time: float) -> Fields:
        """Add back the incident magnetic field the update was missing.

        This one does not depend on the polarization. The curl's sign flips
        between ``Dz`` and ``Dy``, and so does the incident field's, since a
        wave travelling in ``+x`` has ``By = -Ez`` but ``Bz = +Ey``. The two
        flips cancel.
        """
        name = self._electric_component()
        array = np.array(getattr(fields, name), copy=True)
        retarded = float(self.pulse.field_at(time + 0.5 * self.dx / self.velocity))
        array[self.index] += (self.dt / self.dx) * retarded
        return replace(fields, **{name: array})

    def step(self, fields: Fields) -> Fields:
        """One field step with the source applied at both halves."""
        time = fields.time
        fields = self.solver.advance_magnetic(fields)
        fields = self.after_magnetic(fields, time)
        fields = self.solver.advance_electric(fields)
        return self.after_electric(fields, time + 0.5 * self.dt)


@dataclass
class MovingWindow:
    """Slide the box along at the speed of light, keeping the pulse in view.

    A wakefield run follows a pulse over distances thousands of times the
    box, so the box moves instead. Advancing by whole cells rather than by a
    continuous shift is what keeps it free: shifting the fields by an
    integer number of cells is a relabelling, so it introduces no
    interpolation error at all, and a continuous shift would filter the
    pulse a little on every step.
    """

    velocity: float = 1.0
    start: float = 0.0
    shifted: int = 0

    def due(self, grid: YeeGrid, time: float) -> int:
        if time < self.start:
            return 0
        wanted = int((time - self.start) * self.velocity / grid.spacing[0])
        return max(0, wanted - self.shifted)

    def advance(
        self,
        grid: YeeGrid,
        fields: Fields,
        species: Species | None = None,
        boundary=None,
    ):
        """Shift by however many whole cells are owed, discarding what leaves.

        ``boundary`` is the solver's boundary condition, whose own state has
        to be relabelled along with the fields. A perfectly matched layer
        keeps a convolution history per cell and a window that leaves it
        behind stops absorbing.
        """
        cells = self.due(grid, fields.time)
        if cells == 0:
            return fields, species
        if cells >= grid.shape[0]:
            raise ValueError(
                f"the window owes {cells} cells but the box is only "
                f"{grid.shape[0]} wide. Advance it more often"
            )
        self.shifted += cells
        if boundary is not None and hasattr(boundary, "shift"):
            boundary.shift(cells)

        moved = {}
        for name in ("Dx", "Dy", "Dz", "Bx", "By", "Bz"):
            array = np.roll(getattr(fields, name), -cells, axis=0)
            array[-cells:] = 0.0
            moved[name] = array
        fields = replace(fields, **moved)

        if species is not None:
            shift = cells * grid.spacing[0]
            position = species.position - np.array([shift] + [0.0] * (species.ndim - 1))
            keep = position[:, 0] >= 0.0
            species = Species(
                species.charge,
                species.mass,
                position[keep],
                species.momentum[keep],
                species.weight[keep],
                species.name,
            )
        return fields, species
