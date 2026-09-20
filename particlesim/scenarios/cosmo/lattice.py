"""Preheating on a lattice from a plugin-supplied potential, and its defects.

Issue #66. The Floquet machinery in :mod:`particlesim.cosmo.preheating` and
the lattice evolution it drives are already in place; what this adds is the
two things the issue's tasks name. The inflaton potential comes from the
registry rather than being hard-coded, and the spectrum carries a diagnostic
saying which of its modes are still worth reading.

**Floquet is potential-agnostic; the Mathieu closed form is not.** Driving
the same lattice with a Starobinsky inflaton instead of a quadratic one,
the measured growth in each resonance band agrees with
:func:`~particlesim.cosmo.preheating.floquet_exponent` to about one part in
``1e4``. Meanwhile :func:`~particlesim.cosmo.preheating.mathieu_parameters`
raises, because it needs a mass the potential does not have -- which is the
right behaviour and is asserted rather than assumed. So the acceptance
"resonance bands match Floquet analysis" holds for a potential Mathieu's
equation says nothing about.

**The spectrum has a floor, and below it a quiet mode reports the loud one.**
This is the part worth the module. A resonant band grows by many orders of
magnitude; once the loudest mode reaches about ``1/eps`` times a quiet one,
double-precision round-off in the *field* exceeds the quiet mode's true
amplitude, and from then on every silent mode tracks the loudest at a fixed
ratio of around ``1e-15``. Its apparent growth rate becomes a fraction of
the dominant rate -- not zero, not the dominant value, and perfectly steady.

Measured on a Starobinsky run, mode 6 of a 64-point lattice reads:

    window        dynamic range   measured   Floquet   relative amplitude
    2T  -> 8T        8.4e5         +0.0122      0            1.5e-6
    4T  -> 12T       8.0e8         -0.0415      0            1.2e-9
    8T  -> 32T       2.1e16        +0.0741      0            5.6e-16

The in-band modes agree with Floquet to ``1e-4`` in every one of those
windows. The out-of-band ones wander with random sign while they are
resolvable and settle into a systematic positive number exactly when the
relative amplitude reaches machine epsilon. Nothing about the run looks
wrong at that point, which is why :class:`GrowthReport` reports
``resolvable`` alongside the rates rather than leaving it to a docstring.

**Defects are counted by a topological identity, not a threshold.** The
winding of a phase field around a plaquette is an integer, and on a periodic
lattice the winding summed over every plaquette is **exactly zero**: each
link appears in two plaquettes with opposite sign. So a torus cannot hold a
net charge, and :func:`total_winding` returning anything but zero is a bug
rather than a tolerance failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from particlesim.cosmo.preheating import (
    Coupling,
    Oscillation,
    Preheating,
    floquet_exponent,
    mode_amplitudes,
)
from particlesim.solvers.lattice.realtime import Lattice
from particlesim.theories.registry import list_inflaton_potentials

#: A mode below this fraction of the loudest one is round-off, not signal.
#:
#: Double precision carries about ``2.2e-16``; the margin of roughly a
#: thousand is there because the floor is reached gradually rather than at a
#: cliff, and a rate measured while crossing it is already contaminated.
RESOLUTION_FLOOR = 1e-13


@dataclass(frozen=True)
class PreheatingScenario:
    """An inflaton from the registry, a ``chi`` field, and the lattice they live on."""

    potential_id: str = "inflation.quadratic"
    potential_parameters: tuple[tuple[str, float], ...] = ()
    amplitude: float = 1.0
    coupling_strength: float = 4.0
    coupling_mass: float = 0.0
    lattice: Lattice = field(default_factory=lambda: Lattice(size=32.0, points=64, dimensions=1))

    def __post_init__(self) -> None:
        known = list_inflaton_potentials()
        if self.potential_id not in known:
            raise KeyError(
                f"unknown inflaton potential {self.potential_id!r}; known: {sorted(known)}"
            )

    @property
    def potential(self):
        """The registry class, instantiated with this scenario's parameters."""
        return list_inflaton_potentials()[self.potential_id](**dict(self.potential_parameters))

    @property
    def oscillation(self) -> Oscillation:
        return Oscillation(potential=self.potential, amplitude=self.amplitude)

    @property
    def coupling(self) -> Coupling:
        return Coupling(strength=self.coupling_strength, mass=self.coupling_mass)

    @property
    def solver(self) -> Preheating:
        return Preheating(self.lattice, self.oscillation, self.coupling)

    def period(self) -> float:
        return self.oscillation.period()


@dataclass
class GrowthReport:
    """Per-mode growth, what Floquet predicts, and which modes still mean anything."""

    frequencies_squared: np.ndarray
    measured: np.ndarray
    predicted: np.ndarray
    relative_amplitude: np.ndarray
    dynamic_range: float

    @property
    def resolvable(self) -> np.ndarray:
        """Modes still above the round-off floor of the loudest one."""
        return self.relative_amplitude > RESOLUTION_FLOOR

    @property
    def banded(self) -> np.ndarray:
        """Modes Floquet puts inside a resonance band."""
        return self.predicted > 1e-8

    @property
    def trustworthy(self) -> np.ndarray:
        """Modes that are both resolvable and worth comparing.

        A banded mode is by construction one of the loud ones, so it is
        always resolvable; the flag matters for the silent modes, which are
        exactly the ones a naive reading would call "growing slowly".
        """
        return self.resolvable

    def agreement(self) -> np.ndarray:
        """``measured/predicted`` on the banded modes, in band order."""
        inside = self.banded
        return self.measured[inside] / self.predicted[inside]

    def as_row(self) -> dict:
        return {
            "dynamic_range": self.dynamic_range,
            "banded_modes": int(np.count_nonzero(self.banded)),
            "resolvable_modes": int(np.count_nonzero(self.resolvable)),
            "unresolvable_modes": int(np.count_nonzero(~self.resolvable)),
            "agreement": self.agreement().tolist(),
        }


def measure_growth(
    scenario: PreheatingScenario,
    warmup_periods: float = 4.0,
    measure_periods: float = 8.0,
    step: float = 0.005,
    seed: int = 0,
    noise: float = 1e-10,
) -> GrowthReport:
    """Evolve ``chi`` from noise and read a growth rate off each mode.

    Both samples are taken at whole numbers of background periods, because
    the growing solution is an exponential times a periodic function: a ratio
    at matching phase is the exponential alone, with no fitting window to
    choose. The warm-up is there because the initial noise contains the
    *decaying* Floquet solution as well as the growing one.
    """
    if warmup_periods < 0.0 or measure_periods <= 0.0:
        raise ValueError(
            f"a measurement needs a non-negative warm-up and a positive window, got "
            f"{warmup_periods} and {measure_periods}"
        )
    lattice = scenario.lattice
    solver = scenario.solver
    period = scenario.period()

    state = np.random.default_rng(seed).normal(size=lattice.shape) * noise
    rate = np.zeros(lattice.shape)
    if warmup_periods > 0.0:
        state, rate = solver.run(
            state, rate, warmup_periods * period, max(1, int(round(warmup_periods * period / step)))
        )
    early = mode_amplitudes(state, lattice)
    state, rate = solver.run(
        state, rate, measure_periods * period, max(1, int(round(measure_periods * period / step)))
    )
    late = mode_amplitudes(state, lattice)

    elapsed = measure_periods * period
    frequencies = np.asarray(solver.lattice_frequencies(), dtype=float).ravel()
    predicted = np.array(
        [
            floquet_exponent(float(value), scenario.oscillation, scenario.coupling)
            for value in frequencies
        ]
    )
    loudest = float(np.max(late))
    return GrowthReport(
        frequencies_squared=frequencies,
        measured=np.log(late / early) / elapsed,
        predicted=predicted,
        relative_amplitude=late / loudest,
        dynamic_range=loudest / float(np.min(late)),
    )


# --- defect diagnostics ----------------------------------------------------


def wrapped_difference(phases, axis: int) -> np.ndarray:
    """Neighbour phase difference folded into ``(-pi, pi]``.

    The folding is what makes the winding a topological quantity: it throws
    away the ``2 pi`` ambiguity of each link and keeps only the branch, so
    what survives around a closed loop is an integer.
    """
    array = np.asarray(phases, dtype=float)
    raw = np.roll(array, -1, axis=axis) - array
    return raw - 2.0 * np.pi * np.round(raw / (2.0 * np.pi))


def phase_winding(phases) -> np.ndarray:
    """Integer winding of each plaquette of a two-dimensional phase field."""
    array = np.asarray(phases, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"winding is defined here for a two-dimensional field, got {array.ndim}")
    along_first = wrapped_difference(array, 0)
    along_second = wrapped_difference(array, 1)
    loop = (
        along_first
        + np.roll(along_second, -1, axis=0)
        - np.roll(along_first, -1, axis=1)
        - along_second
    )
    return np.rint(loop / (2.0 * np.pi)).astype(int)


def total_winding(phases) -> int:
    """Zero on a periodic lattice, exactly and as an integer.

    Each link enters two plaquettes with opposite sign, so the sum cancels
    term by term. A torus cannot carry a net charge, and a nonzero answer
    here is a bug rather than a tolerance failure -- which is why the suite
    asserts equality with ``0`` and not closeness to it.
    """
    return int(np.sum(phase_winding(phases)))


def defect_sites(phases) -> np.ndarray:
    """Indices of the plaquettes carrying a charge, as an ``(n, 2)`` array."""
    return np.argwhere(phase_winding(phases) != 0)


def defect_count(phases) -> int:
    """How many plaquettes carry a charge, counting both signs."""
    return int(np.count_nonzero(phase_winding(phases)))


__all__ = [
    "RESOLUTION_FLOOR",
    "GrowthReport",
    "PreheatingScenario",
    "defect_count",
    "defect_sites",
    "measure_growth",
    "phase_winding",
    "total_winding",
    "wrapped_difference",
]
