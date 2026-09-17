"""One-dimensional free-electron laser (design doc Section 3.3, Milestone 2).

An undulator wiggles a relativistic beam, the beam radiates, and the
radiation bunches the beam at its own wavelength, which makes it radiate
coherently. The instability is exponential and its e-folding length is the
gain length.

Following the electrons through every undulator period would mean resolving
the wiggle motion over hundreds of periods to see a gain length. The
period-averaged model does not: averaging over one period leaves the phase
of each electron relative to the radiation, its energy, and the radiation
amplitude, which is the whole of the physics at this level.

In the universal scaling of Bonifacio, Pellegrini and Narducci (1984), with
``zbar = 4 pi rho z / lambda_u``,

    d theta_j / d zbar = p_j
    d p_j    / d zbar = -(A e^{i theta_j} + conj)
    d A      / d zbar = <e^{-i theta_j}> + i delta A

and every parameter of the machine has disappeared into ``rho`` and the
detuning ``delta``. That is what makes the result universal: linearizing
gives ``A''' = i A``, whose growing root is ``exp(i pi / 6)``, so the field
grows at ``sqrt(3)/2`` per unit ``zbar`` and the power at ``sqrt(3)``.
Undoing the scaling turns that into

    L_g = lambda_u / (4 pi sqrt(3) rho)

which is the gain length this module is checked against. The check is of the
``sqrt(3)/2``, because that is the part a simulation can be wrong about; the
conversion is the definition of the scaling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Growing root of ``A''' = i A``: the field's e-folding rate in ``zbar``.
FIELD_GROWTH_RATE = float(np.sqrt(3.0) / 2.0)
#: Power grows twice as fast as the field.
POWER_GROWTH_RATE = float(np.sqrt(3.0))


def undulator_parameter(peak_field_tesla: float, period_metres: float) -> float:
    """``K = e B0 lambda_u / (2 pi m_e c)``, the undulator's strength.

    Dimensionless, and of order one for the undulators that are actually
    built. It is what sets both the resonance and, through the coupling, the
    gain.
    """
    if peak_field_tesla < 0 or period_metres <= 0:
        raise ValueError("the peak field must be non-negative and the period positive")
    # e / (2 pi m_e c) in SI, so K = 93.36 * B[T] * lambda_u[m].
    return float(93.3729 * peak_field_tesla * period_metres)


def resonant_wavelength(
    period: float, gamma: float, undulator_strength: float, helical: bool = False
) -> float:
    """``lambda_r = lambda_u (1 + K^2/2) / (2 gamma^2)`` for a planar undulator.

    A helical undulator has ``1 + K^2`` instead, because the electron's
    transverse velocity is constant rather than oscillating: the factor of a
    half is the mean square of a sinusoid, and using the planar form for a
    helical machine misses the resonance by a wavelength's worth of
    bandwidth many times over.
    """
    if gamma <= 1 or period <= 0:
        raise ValueError("need gamma > 1 and a positive period")
    coupling = undulator_strength**2 if helical else 0.5 * undulator_strength**2
    return float(period * (1.0 + coupling) / (2.0 * gamma**2))


def gain_length(rho: float, period: float) -> float:
    """``L_g = lambda_u / (4 pi sqrt(3) rho)``, the power e-folding length.

    This is the scaling's definition rather than a result: ``rho`` is defined
    so that the universal equations grow at ``sqrt(3)`` per unit ``zbar``,
    and ``zbar = 4 pi rho z / lambda_u``. What a simulation can get wrong is
    the ``sqrt(3)``, and that is what :func:`run_fel` measures.
    """
    if rho <= 0 or period <= 0:
        raise ValueError("rho and the period must be positive")
    return float(period / (4.0 * np.pi * np.sqrt(3.0) * rho))


def linear_growth_rate(detuning: float = 0.0) -> float:
    """Field growth rate from the linearized equations, at any detuning.

    Linearizing leaves a third-order equation whose characteristic
    polynomial is

        lambda^3 - i delta lambda^2 - i = 0

    and the growth rate is the largest real part among its roots. At
    resonance that is lambda cubed equals i, whose growing root is
    ``exp(i pi / 6)`` and whose real part is ``sqrt(3)/2``.

    Away from resonance the curve is asymmetric: growth survives far below
    the resonance and cuts off sharply above it. That is a property of the
    cubic rather than of any particular machine, which is why it is worth
    checking a simulation against rather than only checking the peak.
    """
    roots = np.roots([1.0, -1j * float(detuning), 0.0, -1j])
    largest = float(np.max(roots.real))
    # A stable mode's growth rate is zero, not the 1e-17 a numerical root
    # finder leaves behind. Returning the residue would make "is it stable"
    # a question about the solver rather than about the physics.
    return largest if largest > 1e-12 else 0.0


def saturation_power_fraction(rho: float) -> float:
    """``rho``: the fraction of the beam power the radiation reaches.

    The other half of what ``rho`` means, and the reason a small ``rho`` is
    bad twice over -- a long gain length and a low saturation.
    """
    return float(rho)


@dataclass(frozen=True)
class FELRun:
    """Amplitude, bunching and beam energy along the undulator."""

    zbar: np.ndarray
    amplitude: np.ndarray
    bunching: np.ndarray
    mean_energy: np.ndarray
    detuning: float

    @property
    def power(self) -> np.ndarray:
        return np.abs(self.amplitude) ** 2

    def growth_rate(self, low: float = 1e-2, high: float = 0.3) -> float:
        """Field e-folding rate over the exponential band.

        The band is bounded the same way an instability's always is: above
        the start-up, where the bunching has not yet settled onto the growing
        root, and below saturation, where the electrons have given up as much
        energy as they are going to.
        """
        magnitude = np.abs(self.amplitude)
        peak = float(magnitude.max())
        if peak <= 0:
            raise ValueError("the field never grew")
        crossed = np.nonzero(magnitude >= high * peak)[0]
        end = int(crossed[0]) if crossed.size else len(magnitude)
        keep = np.zeros(len(magnitude), dtype=bool)
        keep[:end] = magnitude[:end] >= low * peak
        if keep.sum() < 4:
            raise ValueError(
                "too few points in the exponential band; the run either "
                "saturated immediately or never left start-up"
            )
        return float(np.polyfit(self.zbar[keep], np.log(magnitude[keep]), 1)[0])


def run_fel(
    particles: int = 512,
    length: float = 16.0,
    step: float = 0.01,
    seed_field: complex = 1e-4 + 0j,
    detuning: float = 0.0,
    energy_spread: float = 0.0,
    rng_seed: int = 0,
) -> FELRun:
    """Integrate the period-averaged equations along the undulator.

    ``particles`` are loaded at evenly spaced phases -- a quiet start. Random
    phases carry shot noise of order ``1/sqrt(N)`` in the bunching, which for
    a seeded run is a second seed of unknown size competing with the intended
    one, and the measured start-up would then depend on the draw.

    Fourth-order Runge-Kutta. The phase equation is not stiff and the
    amplitude equation is an average, so nothing here needs more.
    """
    if particles < 8:
        raise ValueError("need at least eight particles for a meaningful average")
    if step <= 0 or length <= 0:
        raise ValueError("step and length must be positive")

    theta = 2.0 * np.pi * np.arange(particles) / particles
    momentum = np.zeros(particles)
    if energy_spread > 0:
        momentum = np.random.default_rng(rng_seed).normal(0.0, energy_spread, particles)
    field = complex(seed_field)

    def derivatives(theta, momentum, field):
        drive = field * np.exp(1j * theta)
        return (
            momentum,
            -2.0 * np.real(drive),
            np.mean(np.exp(-1j * theta)) + 1j * detuning * field,
        )

    steps = int(round(length / step))
    zbar = np.empty(steps + 1)
    amplitude = np.empty(steps + 1, dtype=complex)
    bunching = np.empty(steps + 1)
    mean_energy = np.empty(steps + 1)

    def record(index, z, theta, momentum, field):
        zbar[index] = z
        amplitude[index] = field
        bunching[index] = abs(np.mean(np.exp(-1j * theta)))
        mean_energy[index] = float(np.mean(momentum))

    record(0, 0.0, theta, momentum, field)
    z = 0.0
    for index in range(steps):
        k1 = derivatives(theta, momentum, field)
        k2 = derivatives(
            theta + 0.5 * step * k1[0], momentum + 0.5 * step * k1[1], field + 0.5 * step * k1[2]
        )
        k3 = derivatives(
            theta + 0.5 * step * k2[0], momentum + 0.5 * step * k2[1], field + 0.5 * step * k2[2]
        )
        k4 = derivatives(theta + step * k3[0], momentum + step * k3[1], field + step * k3[2])
        theta = theta + step / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        momentum = momentum + step / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        field = field + step / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        z += step
        record(index + 1, z, theta, momentum, field)

    return FELRun(zbar, amplitude, bunching, mean_energy, float(detuning))


def gain_curve(detunings, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Growth rate against detuning: the FEL's bandwidth, measured.

    Peaks at resonance and falls away either side. The width in ``delta`` is
    of order one, which in unscaled terms is a relative bandwidth of order
    ``rho`` -- the third thing ``rho`` sets.
    """
    detunings = np.asarray(detunings, dtype=float)
    rates = []
    for value in detunings:
        try:
            rates.append(run_fel(detuning=float(value), **kwargs).growth_rate())
        except ValueError:
            rates.append(0.0)
    return detunings, np.array(rates)
