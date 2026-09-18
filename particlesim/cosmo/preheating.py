"""Parametric resonance, and the Floquet chart it has to reproduce.

Issue #66, Level C4. An inflaton oscillating in its own potential drives a
coupled field through a time-dependent mass, and modes in certain bands of
``k`` grow exponentially. The bands are not a simulation output to be
eyeballed -- they are the Floquet spectrum of a linear ODE with periodic
coefficients, computable to machine precision and independent of any
lattice. That is what makes this acceptance sharp.

**The interaction.** A coupling ``g^2 phi^2 chi^2 / 2`` gives ``chi`` an
effective mass that follows the inflaton:

    chi_k'' + (k^2 + m_chi^2 + g^2 phi(t)^2) chi_k = 0

For the quadratic inflaton ``phi = Phi cos(m t)`` this is Mathieu's
equation, with

    A_k = (k^2 + m_chi^2)/m^2 + 2q,    q = g^2 Phi^2 / (4 m^2)

and ``q`` is the whole story: ``q << 1`` is narrow resonance, a thin band
near ``A = 1`` whose exponent has a closed form; ``q >> 1`` is broad
resonance, wide bands and no closed form at all.

**Floquet, not a growth-rate fit.** The monodromy matrix is the ODE's
solution operator over one period of the background. Its eigenvalues are
the Floquet multipliers, and ``mu = ln|lambda| / T`` is exact -- no window
to choose, no transient to wait out, and *zero* outside a band rather than
small. Measured against the narrow-resonance closed form
``mu = (m/2) sqrt(q^2 - (A-1)^2)`` the agreement at band centre is 1.0000,
0.9997 and 0.9988 for ``q`` of 0.02, 0.05 and 0.1 -- converging as the
closed form's own expansion parameter shrinks, which is the right way for
a leading-order formula to be checked. Outside the band the exponent comes
back at 1e-13, so the edges are sharp to round-off.

**The background is integrated, not assumed.** ``phi(t) = Phi cos(m t)``
only holds for a quadratic potential. The period is found from the motion
itself, by detecting the turning point, so any potential from
:mod:`particlesim.cosmo.potentials` can drive the resonance and the
Mathieu parameters are a special case rather than the definition.

**The lattice comparison uses the lattice's own dispersion.** A mode on a
lattice does not have ``omega^2 = k^2 + ...`` but the nearest-neighbour
value ``(4/h^2) sum sin^2(k_i h/2) + ...`` (see
:mod:`particlesim.solvers.lattice.realtime`). Feeding the continuum ``k^2``
into the Floquet reference instead moves the exponent by 19% and 17% at
the two modes that actually grow in the test configuration -- against the
0.1% the lattice dispersion gives -- and that discrepancy would read as a
physics disagreement rather than as a bookkeeping error.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp

from particlesim.cosmo.potentials import Potential, Quadratic
from particlesim.solvers.lattice.realtime import Lattice, laplacian


@dataclass(frozen=True)
class Oscillation:
    """A homogeneous field released from rest at ``amplitude``.

    The period is measured from the trajectory rather than taken from a
    formula, so a potential that is not quadratic -- where the period
    depends on the amplitude -- works without a special case.
    """

    potential: Potential = Quadratic(mass=1.0)
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        if self.amplitude <= 0.0:
            raise ValueError(f"the amplitude must be positive, got {self.amplitude}")

    def _derivative(self, _time, state):
        return [state[1], -float(self.potential.gradient(state[0]))]

    def period(self) -> float:
        """Time to return to rest at ``+amplitude``, by turning-point detection.

        Released from rest, the field reaches the opposite turning point in
        half a period; doubling that is exact for a symmetric potential and
        is what is checked against ``2 pi / m`` for the quadratic case.

        Cached, because scanning a band calls this once per wavenumber and
        the answer does not depend on the wavenumber. Without the cache a
        121-point scan spends most of its time re-deriving one number.
        """
        return _period(self)

    def _measure_period(self) -> float:
        def turning(_time, state):
            return state[1]

        turning.direction = 0.0
        turning.terminal = False

        guess = 100.0 / max(float(np.sqrt(abs(self.potential.curvature(self.amplitude)))), 1e-12)
        solution = solve_ivp(
            self._derivative,
            (0.0, guess),
            [self.amplitude, 0.0],
            events=turning,
            rtol=1e-12,
            atol=1e-14,
            dense_output=True,
        )
        events = solution.t_events[0]
        crossings = events[events > 1e-9]
        if crossings.size == 0:
            raise ValueError(
                "the field did not reach a turning point; the potential may be unbounded "
                f"or the amplitude {self.amplitude} outside its well"
            )
        return float(2.0 * crossings[0])


@lru_cache(maxsize=256)
def _period(oscillation: Oscillation) -> float:
    """The period of one oscillation, memoised on the frozen dataclass."""
    return oscillation._measure_period()

    def trajectory(self, times) -> np.ndarray:
        """``phi(t)`` at the requested times, from the same integration."""
        requested = np.atleast_1d(np.asarray(times, dtype=float))
        solution = solve_ivp(
            self._derivative,
            (0.0, float(requested.max()) + 1e-12),
            [self.amplitude, 0.0],
            t_eval=requested,
            rtol=1e-12,
            atol=1e-14,
        )
        return solution.y[0]


@dataclass(frozen=True)
class Coupling:
    """``g^2 phi^2 chi^2 / 2``: the inflaton drives ``chi``'s mass."""

    strength: float = 0.1
    mass: float = 0.0

    def __post_init__(self) -> None:
        if self.strength < 0.0:
            raise ValueError(f"the coupling must not be negative, got {self.strength}")

    def effective_mass_squared(self, field):
        return self.mass**2 + self.strength**2 * np.asarray(field, dtype=float) ** 2


def mathieu_parameters(
    frequency_squared: float, oscillation: Oscillation, coupling: Coupling
) -> tuple[float, float]:
    """``(A, q)`` for a quadratic inflaton, from the physical parameters.

    ``frequency_squared`` is the mode's own ``omega^2`` -- ``k^2`` in the
    continuum, or the lattice's dispersion on a lattice. Only meaningful
    for :class:`~particlesim.cosmo.potentials.Quadratic`, since Mathieu's
    equation is what a quadratic potential gives; the Floquet machinery
    below needs none of this.
    """
    mass = float(oscillation.potential.mass)
    resonance = coupling.strength**2 * oscillation.amplitude**2 / (4.0 * mass**2)
    return (frequency_squared + coupling.mass**2) / mass**2 + 2.0 * resonance, resonance


def narrow_resonance_exponent(centre: float, resonance: float, mass: float) -> float:
    """``(m/2) sqrt(q^2 - (A-1)^2)``: the first Mathieu band, to leading order.

    Valid for ``q << 1``. Kept as a function rather than inlined in a test
    because its job is to be the thing :func:`floquet_exponent` converges to
    as ``q`` shrinks -- a statement that needs both sides callable.
    """
    inside = resonance**2 - (centre - 1.0) ** 2
    return float(0.5 * mass * np.sqrt(inside)) if inside > 0.0 else 0.0


def monodromy(frequency_squared: float, oscillation: Oscillation, coupling: Coupling) -> np.ndarray:
    """The mode equation's solution operator over one background period.

    Both unit initial conditions are carried in one integration alongside
    the background, so the two columns share a step sequence and the
    background is never re-solved. The period used is the *field's*, not
    the coefficient's: for a symmetric potential ``phi^2`` repeats twice as
    often, so this monodromy is the square of the shorter one -- which
    leaves ``ln|lambda| / T`` unchanged and avoids having to know the
    symmetry in advance.
    """
    period = oscillation.period()

    def derivative(_time, state):
        """``[phi, phi', chi_a, chi_a', chi_b, chi_b']`` in one system.

        The background travels *with* the modes rather than being looked up
        from a separate integration. Looking it up costs a full re-solve per
        right-hand side evaluation, which turns an O(N) integration into an
        O(N^2) one -- and it is also less accurate, since the two
        integrations would not share a step sequence.
        """
        field, rate = state[0], state[1]
        mass_squared = frequency_squared + float(coupling.effective_mass_squared(field))
        return [
            rate,
            -float(oscillation.potential.gradient(field)),
            state[3],
            -mass_squared * state[2],
            state[5],
            -mass_squared * state[4],
        ]

    solution = solve_ivp(
        derivative,
        (0.0, period),
        [oscillation.amplitude, 0.0, 1.0, 0.0, 0.0, 1.0],
        rtol=1e-11,
        atol=1e-13,
        max_step=period / 200.0,
    )
    final = solution.y[:, -1]
    return np.array([[final[2], final[4]], [final[3], final[5]]])


def floquet_exponent(
    frequency_squared: float, oscillation: Oscillation, coupling: Coupling
) -> float:
    """``mu = max ln|lambda| / T``: the growth rate, exactly and with no fit.

    Zero -- to round-off, not to a tolerance -- outside a resonance band,
    because the multipliers then sit on the unit circle. That sharpness is
    what makes "the bands match" a testable claim rather than an
    impression.
    """
    multipliers = np.linalg.eigvals(monodromy(frequency_squared, oscillation, coupling))
    return float(np.max(np.log(np.abs(multipliers))) / oscillation.period())


def resonance_bands(
    frequencies_squared, oscillation: Oscillation, coupling: Coupling, threshold: float = 1e-8
) -> list[tuple[float, float]]:
    """Contiguous runs of ``omega^2`` whose Floquet exponent exceeds ``threshold``.

    Returned as intervals in ``omega^2`` rather than a mask, so the band
    edges can be compared with the closed form's ``|A - 1| < q`` directly.
    """
    values = np.asarray(frequencies_squared, dtype=float)
    unstable = np.array(
        [floquet_exponent(float(value), oscillation, coupling) > threshold for value in values]
    )
    bands: list[tuple[float, float]] = []
    start = None
    for index, flag in enumerate(unstable):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            bands.append((float(values[start]), float(values[index - 1])))
            start = None
    if start is not None:
        bands.append((float(values[start]), float(values[-1])))
    return bands


# --- the lattice realisation ---------------------------------------------


@dataclass(frozen=True)
class Preheating:
    """``chi`` on a lattice, driven by a homogeneous inflaton.

    The inflaton is a background here: no back-reaction, which is exactly
    the regime Floquet theory describes and therefore the regime in which
    the two can be compared at all. Once ``chi`` grows enough to drain the
    inflaton the linear analysis stops applying, and the comparison this
    module exists to make would no longer be meaningful.
    """

    lattice: Lattice
    oscillation: Oscillation
    coupling: Coupling

    def acceleration(self, field, inflaton: float) -> np.ndarray:
        mass_squared = float(self.coupling.effective_mass_squared(inflaton))
        return laplacian(field, self.lattice) - mass_squared * np.asarray(field, dtype=float)

    def run(self, field, momentum, final: float, steps: int):
        """Velocity Verlet to ``final``, returning the final field and momentum.

        The inflaton is stepped by the *same* leapfrog, in the same loop,
        rather than sampled from a separate solution. Two integrations of
        the same background do not agree to round-off, and the mismatch
        would show up as a spurious drift in the resonance -- which is the
        one thing this module is trying to measure.
        """
        if steps < 1:
            raise ValueError(f"at least one step is needed, got {steps}")
        step = final / int(steps)
        current = np.asarray(field, dtype=float)
        rate = np.asarray(momentum, dtype=float)
        inflaton, velocity = self.oscillation.amplitude, 0.0

        force = lambda value: -float(self.oscillation.potential.gradient(value))  # noqa: E731
        for _ in range(int(steps)):
            half = rate + 0.5 * step * self.acceleration(current, inflaton)
            half_velocity = velocity + 0.5 * step * force(inflaton)
            current = current + step * half
            inflaton = inflaton + step * half_velocity
            rate = half + 0.5 * step * self.acceleration(current, inflaton)
            velocity = half_velocity + 0.5 * step * force(inflaton)
        return current, rate

    def lattice_frequencies(self) -> np.ndarray:
        """``omega^2`` of each lattice mode, with ``chi``'s bare mass included."""
        return self.lattice.dispersion(self.coupling.mass**2)


def mode_amplitudes(field, lattice: Lattice) -> np.ndarray:
    """``|chi_k|`` per mode, for reading a growth rate off the lattice."""
    axes = tuple(range(lattice.dimensions))
    return np.abs(np.fft.rfftn(np.asarray(field, dtype=float), axes=axes))


__all__ = [
    "Coupling",
    "Oscillation",
    "Preheating",
    "floquet_exponent",
    "mathieu_parameters",
    "mode_amplitudes",
    "monodromy",
    "narrow_resonance_exponent",
    "resonance_bands",
]
