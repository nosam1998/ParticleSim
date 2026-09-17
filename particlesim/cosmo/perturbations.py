"""Primordial perturbations: Mukhanov-Sasaki modes and delta N (Section 3.1).

This is the part of Level C1 that produces something observable. A
background history from :mod:`particlesim.cosmo.inflation` fixes ``H(N)``
and ``epsilon(N)``; here each comoving wavenumber is evolved through horizon
crossing and the primordial power spectra, ``n_s``, ``r`` and the running
are read off.

**The curvature perturbation is evolved, not the Mukhanov variable.** The
two are equivalent, ``v = z R`` with ``z = a sqrt(2 epsilon)``, and the
choice is numerical rather than physical. In terms of ``R`` the equation is

    R'' + (3 - epsilon + 2 phi''/phi') R' + (k/aH)^2 R = 0

whose coefficients need only ``phi`` and ``phi'``, both of which the
background already carries; the ``v`` form needs ``z''/z``, which is a
second derivative of ``epsilon`` and so a third derivative of the field.
More importantly ``R`` tends to a constant outside the horizon, so the
number the spectrum wants is the plateau of the solution and its
convergence is visible as a drift. ``v`` grows like ``a`` there, and reading
a constant off a growing solution means dividing two large numbers.

The price is that ``phi'`` sits in a denominator, so a trajectory that
brings the field momentarily to rest cannot be handled this way. Monotonic
single-field inflation never does, and :func:`mode_power` says so rather
than returning a number if it happens.

**Amplitudes are carried in logarithms.** Bunch-Davies normalisation puts
``|R|`` at ``1/(2 a sqrt(epsilon k))``, and with ``a = exp(N)`` a run of a
few hundred e-folds would overflow a double before any physics happened.
Since the mode equation is linear and homogeneous, the initial amplitude is
set to one and the normalisation is added afterwards in logs, which makes
the dynamic range of the run irrelevant.

**Real and imaginary parts are evolved as two real solutions.** The
coefficients are real and the initial data is complex only through ``R'``,
so the two parts satisfy the same equation with different initial
conditions. That is four real states for the scalar and four for the
tensor, integrated together, and no complex arithmetic anywhere.

The tensor mode is the same equation without the ``epsilon`` terms, for a
canonically normalised massless field ``chi``, with

    P_t = 8 (k^3/2 pi^2) |chi|^2

the ``8`` being two polarisations times the ``(2/M_p)^2`` from the
normalisation of ``h_ij``. In the slow-roll limit that reproduces
``P_t = 2 H^2/pi^2`` and ``r = 16 epsilon``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from particlesim.cosmo.inflation import InflationRun, MultiFieldRun, evolve_fields
from particlesim.cosmo.potentials import MultiFieldPotential

#: E-folds before horizon crossing at which a mode is started.
#:
#: The Bunch-Davies initial condition is the ``k >> aH`` limit, wrong by
#: ``O((aH/k)^2) = exp(-2 * subhorizon)``. Six e-folds puts that at 6e-6,
#: three orders below the accuracy anything here is quoted to, at the cost
#: of about sixty oscillations to integrate through.
SUBHORIZON_EFOLDS = 6.0
#: E-folds after crossing at which the frozen amplitude is read.
#:
#: The residual gradient correction is also ``O((aH/k)^2)``, so the same
#: six e-folds buys the same 6e-6 -- and because it is counted from each
#: mode's own crossing, what remains is common to every ``k`` and cancels
#: out of the spectral index rather than tilting it.
FREEZE_EFOLDS = 6.0


@dataclass(frozen=True)
class ModeResult:
    """One wavenumber's frozen power, and the evidence that it is frozen."""

    wavenumber: float
    crossing_efolds: float
    scalar_power: float
    tensor_power: float
    drift: float

    @property
    def tensor_to_scalar(self) -> float:
        return self.tensor_power / self.scalar_power


def crossing_efolds(run: InflationRun, wavenumber: float) -> float:
    """E-fold at which ``k = aH``.

    ``ln(aH)`` grows monotonically while ``epsilon < 1``, so the root is
    unique and bracketed by the run itself. A wavenumber that crosses
    outside the run is an error, not an extrapolation: it would be a mode
    whose initial condition was never imposed.
    """
    low, high = float(run.efolds[0]), run.total_efolds
    target = math.log(float(wavenumber))

    def excess(number: float) -> float:
        return float(run.log_comoving_hubble(number)) - target

    if excess(low) > 0.0:
        raise ValueError(
            f"k = {wavenumber:g} is already outside the horizon at the start of the "
            "run: begin the background earlier, or use a larger k"
        )
    if excess(high) < 0.0:
        raise ValueError(
            f"k = {wavenumber:g} never crosses the horizon during the run, which "
            f"covers {high - low:.3g} e-folds"
        )
    return float(brentq(excess, low, high, xtol=1e-12, rtol=1e-14))


def pivot_wavenumber(run: InflationRun, efolds_remaining: float) -> float:
    """Wavenumber that crosses the horizon ``efolds_remaining`` before the end.

    The e-fold count is the run's own, measured to where ``epsilon = 1``,
    not the slow-roll estimate used to pick the initial field. For a
    plateau potential the two differ by more than an e-fold, which is worth
    a part in a thousand of ``n_s`` -- the size of the criterion this
    module is checked against.
    """
    number = run.at_efolds_remaining(efolds_remaining)
    return float(np.exp(run.log_comoving_hubble(number)))


def mode_power(
    run: InflationRun,
    wavenumber: float,
    subhorizon_efolds: float = SUBHORIZON_EFOLDS,
    freeze_efolds: float = FREEZE_EFOLDS,
    rtol: float = 1e-10,
    atol: float = 1e-14,
) -> ModeResult:
    """Scalar and tensor power at ``wavenumber``, from Bunch-Davies data.

    The mode starts ``subhorizon_efolds`` before its own horizon crossing
    and is read ``freeze_efolds`` after it. Both are counted from the
    crossing rather than fixed in absolute e-folds, so every wavenumber in a
    spectrum gets the same treatment and whatever error the truncation
    leaves is common to all of them.
    """
    wavenumber = float(wavenumber)
    crossing = crossing_efolds(run, wavenumber)
    start = crossing - float(subhorizon_efolds)
    stop = crossing + float(freeze_efolds)
    if start < float(run.efolds[0]) - 1e-12:
        raise ValueError(
            f"k = {wavenumber:g} crosses the horizon {crossing - float(run.efolds[0]):.3g} "
            f"e-folds into the run but needs {subhorizon_efolds:g} e-folds of "
            "sub-horizon evolution first: start the background earlier"
        )
    if stop > run.total_efolds + 1e-12:
        raise ValueError(
            f"k = {wavenumber:g} crosses the horizon {run.total_efolds - crossing:.3g} "
            f"e-folds before the end of the run but needs {freeze_efolds:g} e-folds "
            "afterwards to freeze out"
        )

    log_k = math.log(wavenumber)

    def coefficients(number):
        phi, velocity = run.state(number)
        epsilon = 0.5 * velocity**2
        value = run.potential.value(phi)
        acceleration = -(3.0 - epsilon) * (velocity + run.potential.gradient(phi) / value)
        log_ratio = log_k - number - 0.5 * np.log(value / (3.0 - epsilon))
        return epsilon, velocity, acceleration, np.exp(2.0 * log_ratio)

    epsilon0, velocity0, acceleration0, ratio0 = coefficients(start)
    if abs(float(velocity0)) < 1e-30:
        raise ValueError(
            "phi' vanishes at the start of the mode integration, so the curvature "
            "equation's friction term is singular there. This solver assumes a "
            "monotonic trajectory"
        )
    slope = 1.0 + float(acceleration0) / float(velocity0)
    frequency = math.sqrt(float(ratio0))

    def right_hand_side(number, state):
        epsilon, velocity, acceleration, ratio = coefficients(number)
        scalar_damping = 3.0 - epsilon + 2.0 * acceleration / velocity
        tensor_damping = 3.0 - epsilon
        out = np.empty(8)
        for offset, damping in ((0, scalar_damping), (4, tensor_damping)):
            for part in (0, 2):
                amplitude = state[offset + part]
                derivative = state[offset + part + 1]
                out[offset + part] = derivative
                out[offset + part + 1] = -damping * derivative - ratio * amplitude
        return out

    initial = np.array(
        [1.0, -slope, 0.0, -frequency, 1.0, -1.0, 0.0, -frequency],
        dtype=float,
    )
    solution = solve_ivp(
        right_hand_side,
        (start, stop),
        initial,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(f"mode integration failed at k = {wavenumber:g}: {solution.message}")

    def powers(number):
        state = solution.sol(number)
        scalar = state[0] ** 2 + state[2] ** 2
        tensor = state[4] ** 2 + state[6] ** 2
        # ln P_R = 2 ln k - ln(8 pi^2) - 2 N_start - ln epsilon_start + ln|R/R_start|^2
        log_scalar = (
            2.0 * log_k
            - math.log(8.0 * math.pi**2)
            - 2.0 * start
            - math.log(float(epsilon0))
            + math.log(scalar)
        )
        # ln P_t = 2 ln k - ln(pi^2/2) - 2 N_start + ln|chi/chi_start|^2
        log_tensor = 2.0 * log_k - math.log(0.5 * math.pi**2) - 2.0 * start + math.log(tensor)
        return math.exp(log_scalar), math.exp(log_tensor)

    scalar_power, tensor_power = powers(stop)
    earlier, _ = powers(stop - 1.0)
    return ModeResult(
        wavenumber=wavenumber,
        crossing_efolds=crossing,
        scalar_power=scalar_power,
        tensor_power=tensor_power,
        drift=abs(scalar_power - earlier) / scalar_power,
    )


@dataclass(frozen=True)
class Spectrum:
    """A primordial power spectrum and the observables fitted to it."""

    wavenumbers: np.ndarray
    scalar_power: np.ndarray
    tensor_power: np.ndarray
    pivot: float
    efolds_remaining: float
    spectral_index: float
    running: float
    tensor_index: float
    drift: float

    @property
    def pivot_scalar_power(self) -> float:
        return float(np.interp(math.log(self.pivot), np.log(self.wavenumbers), self.scalar_power))

    @property
    def tensor_to_scalar(self) -> float:
        logs = np.log(self.wavenumbers)
        pivot = math.log(self.pivot)
        return float(
            np.interp(pivot, logs, self.tensor_power) / np.interp(pivot, logs, self.scalar_power)
        )

    def summary(self) -> dict[str, float]:
        return {
            "efolds_remaining": self.efolds_remaining,
            "spectral_index": self.spectral_index,
            "tensor_to_scalar": self.tensor_to_scalar,
            "running": self.running,
            "tensor_index": self.tensor_index,
            "scalar_amplitude": self.pivot_scalar_power,
            "drift": self.drift,
        }


def primordial_spectra(run: InflationRun, wavenumbers, **kwargs):
    """Scalar and tensor power at each of ``wavenumbers``.

    This is the primordial ``P(k)`` itself, over whatever range of
    wavenumbers the background run supports -- every mode that crosses the
    horizon with room to start sub-horizon and room to freeze out, which for
    a sixty-e-fold run is around twenty decades. :func:`power_spectrum` is
    the narrow, pivot-centred use of it; a plot of the spectrum or an input
    to a transfer function wants this.
    """
    results = [mode_power(run, float(k), **kwargs) for k in np.atleast_1d(wavenumbers)]
    scalar = np.array([r.scalar_power for r in results])
    tensor = np.array([r.tensor_power for r in results])
    return scalar, tensor


def power_spectrum(
    run: InflationRun,
    efolds_remaining: float = 55.0,
    spread: float = 0.5,
    points: int = 5,
    **kwargs,
) -> Spectrum:
    """Spectra and fitted observables around the pivot.

    ``points`` wavenumbers are spaced evenly in ``ln k`` over
    ``+-spread``, and ``ln P`` is fitted with a quadratic: the linear
    coefficient is ``n_s - 1`` and twice the quadratic one is the running.
    A fit rather than a two-point difference because the running is wanted
    as well, and a quadratic rather than anything higher because over half a
    decade the next term is below the noise of the fit.
    """
    if points < 3:
        raise ValueError(f"a quadratic fit needs at least three points, got {points}")
    pivot = pivot_wavenumber(run, efolds_remaining)
    offsets = np.linspace(-float(spread), float(spread), int(points))
    wavenumbers = pivot * np.exp(offsets)
    results = [mode_power(run, float(k), **kwargs) for k in wavenumbers]
    scalar = np.array([r.scalar_power for r in results])
    tensor = np.array([r.tensor_power for r in results])
    scalar_fit = np.polynomial.polynomial.polyfit(offsets, np.log(scalar), 2)
    tensor_fit = np.polynomial.polynomial.polyfit(offsets, np.log(tensor), 2)
    return Spectrum(
        wavenumbers=wavenumbers,
        scalar_power=scalar,
        tensor_power=tensor,
        pivot=pivot,
        efolds_remaining=float(efolds_remaining),
        spectral_index=1.0 + float(scalar_fit[1]),
        running=2.0 * float(scalar_fit[2]),
        tensor_index=float(tensor_fit[1]),
        drift=max(r.drift for r in results),
    )


def end_of_inflation_surface(_phi, velocity) -> float:
    """Stopping surface ``epsilon_H = 1``: the end of inflation."""
    return 0.5 * float(np.sum(np.asarray(velocity, dtype=float) ** 2)) - 1.0


def efolds_to_surface(
    potential: MultiFieldPotential,
    phi,
    surface: Callable[[np.ndarray, np.ndarray], float] | None = None,
    **kwargs,
) -> float:
    """E-folds from ``phi`` on the slow-roll attractor to ``surface``.

    This is the separate-universe number ``N`` that ``delta N`` differentiates:
    each neighbouring universe starts from its own field value, on the
    attractor, and runs to the *same* final surface. Which surface that is
    changes the answer whenever the perturbation has not yet become
    adiabatic, so it is an argument rather than a convention.
    """
    surface = end_of_inflation_surface if surface is None else surface
    run = evolve_fields(potential, phi, stop=surface, **kwargs)
    if not run.ended:
        raise ValueError(
            f"the trajectory from {np.asarray(phi)} never reached the stopping "
            "surface within the maximum number of e-folds"
        )
    return run.total_efolds


def delta_n_gradient(
    potential: MultiFieldPotential,
    phi,
    surface: Callable[[np.ndarray, np.ndarray], float] | None = None,
    step: float = 1e-3,
    **kwargs,
) -> np.ndarray:
    """``dN/dphi_i``, by central differences on the exact background.

    Differentiating the evolution rather than evaluating a closed-form
    expression keeps this valid for potentials that are not sum-separable,
    where no closed form exists. The step is a field displacement and the
    error is ``O(step^2)`` from the difference against ``O(1/step)`` from
    the integrator's tolerance; at ``step = 1e-3`` and ``rtol = 1e-11``
    both sit near ``1e-7`` relative.
    """
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    gradient = np.empty_like(phi)
    for index in range(phi.size):
        forward, backward = phi.copy(), phi.copy()
        forward[index] += step
        backward[index] -= step
        gradient[index] = (
            efolds_to_surface(potential, forward, surface, **kwargs)
            - efolds_to_surface(potential, backward, surface, **kwargs)
        ) / (2.0 * step)
    return gradient


@dataclass(frozen=True)
class DeltaNSpectrum:
    """A ``delta N`` power spectrum for a multi-field trajectory."""

    efolds_remaining: float
    field: np.ndarray
    gradient: np.ndarray
    hubble: float
    epsilon: float
    scalar_power: float
    spectral_index: float

    @property
    def tensor_power(self) -> float:
        return 2.0 * self.hubble**2 / math.pi**2

    @property
    def tensor_to_scalar(self) -> float:
        return self.tensor_power / self.scalar_power

    def summary(self) -> dict[str, float]:
        return {
            "efolds_remaining": self.efolds_remaining,
            "scalar_amplitude": self.scalar_power,
            "spectral_index": self.spectral_index,
            "tensor_to_scalar": self.tensor_to_scalar,
            "epsilon": self.epsilon,
        }


def delta_n_spectrum(
    run: MultiFieldRun,
    efolds_remaining: float = 55.0,
    surface: Callable[[np.ndarray, np.ndarray], float] | None = None,
    spread: float = 0.25,
    step: float = 1e-3,
) -> DeltaNSpectrum:
    """Curvature power and tilt from ``delta N`` at a pivot on ``run``.

        P_R = (H/2 pi)^2 sum_i (dN/dphi_i)^2

    with each field given the same ``H/2 pi`` of fluctuation at crossing.
    The tilt follows from differentiating that along the trajectory, using
    ``d ln k = dN`` at crossing:

        n_s - 1 = -2 epsilon + d ln(sum_i N_i^2) / dN

    which is why the gradient is evaluated at three points rather than one.
    This is the slow-roll super-horizon answer, not a mode-by-mode
    solution: it captures the multi-field physics -- an isocurvature mode
    feeding the curvature perturbation, which is what makes ``sum_i N_i^2``
    evolve at all -- but not the sub-horizon evolution the single-field
    solver in this module integrates. For one field the two agree, and a
    test checks they do.
    """
    pivot = run.total_efolds - float(efolds_remaining)
    if pivot - spread < float(run.efolds[0]):
        raise ValueError(
            f"the run covers {run.total_efolds - float(run.efolds[0]):.3g} e-folds and "
            f"the pivot at {efolds_remaining:g} before its end needs {spread:g} more"
        )

    def total(number: float) -> tuple[np.ndarray, np.ndarray]:
        fields, _ = run.state(number)
        gradient = delta_n_gradient(run.potential, fields, surface, step=step)
        return fields, gradient

    fields, gradient = total(pivot)
    _, ahead = total(pivot + spread)
    _, behind = total(pivot - spread)
    slope = (math.log(float(np.sum(ahead**2))) - math.log(float(np.sum(behind**2)))) / (
        2.0 * spread
    )

    _, velocity = run.state(pivot)
    epsilon = 0.5 * float(np.sum(velocity**2))
    hubble = float(run.hubble_at(pivot))
    power = float(np.sum(gradient**2)) * (hubble / (2.0 * math.pi)) ** 2
    return DeltaNSpectrum(
        efolds_remaining=float(efolds_remaining),
        field=fields,
        gradient=gradient,
        hubble=hubble,
        epsilon=epsilon,
        scalar_power=power,
        spectral_index=1.0 - 2.0 * epsilon + slope,
    )
