"""Coupled field-space perturbations: the mode matrix through horizon crossing.

Issue #124. :mod:`particlesim.cosmo.perturbations` solves Mukhanov-Sasaki
for a *single* field and covers the multi-field case with ``delta N``,

    P_R = (H/2pi)^2 sum_i (dN/dphi_i)^2

which is a super-horizon, slow-roll statement. It captures an isocurvature
mode feeding the curvature perturbation -- that is what makes the sum evolve
at all -- but it never integrates the perturbations *through* horizon
crossing, so it cannot give an isocurvature spectrum, a correlated
isocurvature-adiabatic fraction, or anything at all for a trajectory that is
not on the slow-roll attractor when its modes cross.

This module evolves the coupled system in the flat gauge instead.

**The equation, derived rather than quoted.** In cosmic time the flat-gauge
field perturbations obey

    ddot(dphi_i) + 3 H dot(dphi_i) + (k/a)^2 dphi_i + M_ij dphi_j = 0

with the effective mass matrix

    M_ij = V_ij - (1/a^3) d/dt ( a^3 dot(phi_i) dot(phi_j) / H )

in units ``M_p = 1``. The second term is the gravitational back-reaction:
the metric perturbation has been solved for and substituted, and dropping it
leaves the Hessian alone, which is a different and wrong system. Expanding
it in e-folds, with ``dot(phi_i) = H phi_i'`` and ``H'/H = -epsilon``:

    (1/a^3) d/dt(a^3 H phi_i' phi_j')
        = H d/dN(H phi_i' phi_j') + 3 H^2 phi_i' phi_j'
        = H^2 [ -epsilon phi_i' phi_j' + phi_i'' phi_j' + phi_i' phi_j''
                + 3 phi_i' phi_j' ]

so that

    M_ij / H^2 = V_ij/H^2 - [ phi_i'' phi_j' + phi_i' phi_j''
                              + (3 - epsilon) phi_i' phi_j' ]

and, converting the wave equation itself with
``ddot(f) = H^2 (f'' - epsilon f')``,

    dphi_i'' + (3 - epsilon) dphi_i' + [ (k/aH)^2 delta_ij + M_ij/H^2 ] dphi_j = 0

**The single-field limit is an identity, not a tolerance.** Substituting
``dphi = phi' R`` into the above and using ``epsilon' = phi' phi''`` together
with the background equation ``phi'' = -(3-epsilon) phi' - V_phi/H^2``
returns

    R'' + (3 - epsilon + 2 phi''/phi') R' + (k/aH)^2 R = 0

which is exactly what :func:`particlesim.cosmo.perturbations.mode_power`
integrates. So with one field this module and that one solve the same
equation in different variables, and the test suite holds them to the
*exact* power-law result rather than to agreement with each other.

**What is evolved is a matrix.** Each field carries an independent
Bunch-Davies vacuum, so there are ``F`` independent solutions of an
``F``-dimensional system and the object is an ``F x F`` matrix of mode
functions ``dphi_{i a}`` -- field ``i``, solution ``a``. Started
deep inside the horizon it is diagonal; the off-diagonal entries are the
coupling doing its work. Every spectrum is then a sum over ``a`` of a
projection over ``i``, which is what keeps the independent vacua from being
added as if they were correlated.

**Projections.** With ``sigma' = sqrt(2 epsilon)`` and the unit vector along
the trajectory ``e_i = phi_i'/sigma'``, the adiabatic and entropic
perturbations are

    R_a = e_i dphi_{i a} / sigma' = phi_i' dphi_{i a} / (2 epsilon)
    S_a^alpha = s^alpha_i dphi_{i a} / sigma'

for any orthonormal basis ``s^alpha`` of the complement of ``e``. Both are
divided by ``sigma'``, so ``P_S`` is on the same footing as ``P_R`` and the
cross-correlation is dimensionless.

**Amplitudes are carried in logarithms**, as in the single-field module: the
system is linear and homogeneous, so the matrix starts at the identity and
the Bunch-Davies normalisation ``|dphi|^2 = 1/(2 k a^2)`` is restored
afterwards, which makes the run's dynamic range irrelevant.

**What is assumed.** Canonical kinetic terms. A curved field-space metric
adds Christoffel terms to the covariant derivative along the trajectory and
a Riemann term to the mass matrix; neither is here, and neither is silently
set to zero somewhere it would be wrong -- there is simply no field-space
metric in :class:`~particlesim.cosmo.potentials.MultiFieldPotential` to
carry one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from particlesim.cosmo.inflation import MultiFieldRun
from particlesim.cosmo.perturbations import FREEZE_EFOLDS, SUBHORIZON_EFOLDS


def log_comoving_hubble(run: MultiFieldRun, number):
    """``ln(aH)`` for a multi-field run, with ``a = exp(N)``."""
    number = np.asarray(number, dtype=float)
    return number + np.log(run.hubble_at(number))


def crossing_efolds(run: MultiFieldRun, wavenumber: float) -> float:
    """E-fold at which ``k = aH``, bracketed by the run itself."""
    low, high = float(run.efolds[0]), run.total_efolds
    target = math.log(float(wavenumber))

    def excess(number: float) -> float:
        return float(log_comoving_hubble(run, number)) - target

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


def pivot_wavenumber(run: MultiFieldRun, efolds_remaining: float) -> float:
    """The wavenumber crossing ``efolds_remaining`` before the end of the run."""
    number = run.total_efolds - float(efolds_remaining)
    if number < float(run.efolds[0]):
        raise ValueError(
            f"the run covers {run.total_efolds - float(run.efolds[0]):.3g} e-folds, so "
            f"it does not reach back {efolds_remaining:g} e-folds before its end"
        )
    return float(np.exp(log_comoving_hubble(run, number)))


def background_at(run: MultiFieldRun, number: float):
    """``(phi, phi', phi'', epsilon, H^2)`` at one e-fold, fields on the last axis."""
    phi, velocity = run.state(float(number))
    phi = np.asarray(phi, dtype=float).reshape(-1)
    velocity = np.asarray(velocity, dtype=float).reshape(-1)
    epsilon = 0.5 * float(np.sum(velocity**2))
    value = float(run.potential.value(phi))
    hubble_squared = value / (3.0 - epsilon)
    gradient = np.asarray(run.potential.gradient(phi), dtype=float)
    acceleration = -(3.0 - epsilon) * velocity - gradient / hubble_squared
    return phi, velocity, acceleration, epsilon, hubble_squared


def mass_matrix(run: MultiFieldRun, number: float) -> np.ndarray:
    """``M_ij / H^2`` at one e-fold: the Hessian *and* the back-reaction.

    The two pieces are of the same order in slow roll, so the second is not
    a correction that can be dropped for a first version. For a single field
    it is what turns the Hessian into the combination that makes the
    curvature perturbation constant outside the horizon; drop it and ``R``
    drifts on super-horizon scales instead of freezing, which a spectrum
    read at a fixed number of e-folds after crossing would report as a
    spurious tilt.
    """
    phi, velocity, acceleration, epsilon, hubble_squared = background_at(run, number)
    hessian = np.asarray(run.potential.hessian(phi), dtype=float)
    back_reaction = (
        np.outer(acceleration, velocity)
        + np.outer(velocity, acceleration)
        + (3.0 - epsilon) * np.outer(velocity, velocity)
    )
    return hessian / hubble_squared - back_reaction


def trajectory_basis(velocity) -> tuple[np.ndarray, np.ndarray]:
    """``(e, s)``: the unit vector along the trajectory and a basis of its complement.

    The entropic basis is obtained from a QR factorisation of a matrix whose
    first column is ``e``, which gives an orthonormal set with the right span
    and is stable when the trajectory nearly aligns with a coordinate axis.
    The sign convention is fixed by forcing the diagonal of ``R`` positive,
    so the basis does not flip discontinuously along a run -- and the
    spectra below are sums of squares over the entropic directions, so no
    result depends on that choice anyway.
    """
    velocity = np.asarray(velocity, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(velocity))
    if norm <= 0.0:
        raise ValueError(
            "the trajectory has come to rest, so the adiabatic direction is "
            "undefined: the adiabatic-entropic split has no meaning there"
        )
    along = velocity / norm
    fields = along.size
    seed = np.eye(fields)
    seed[:, 0] = along
    factor, upper = np.linalg.qr(seed)
    factor = factor * np.sign(np.where(np.diag(upper) == 0.0, 1.0, np.diag(upper)))
    return along, factor[:, 1:]


@dataclass(frozen=True)
class ModeMatrix:
    """The evolved mode matrix at one wavenumber, and the spectra read off it."""

    wavenumber: float
    crossing_efolds: float
    curvature_power: float
    entropy_power: float
    cross_power: float
    drift: float
    fields: int

    @property
    def correlation(self) -> float:
        """``C_RS / sqrt(P_R P_S)``, in ``[-1, 1]``."""
        denominator = math.sqrt(self.curvature_power * self.entropy_power)
        return self.cross_power / denominator if denominator > 0.0 else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "wavenumber": self.wavenumber,
            "crossing_efolds": self.crossing_efolds,
            "curvature_power": self.curvature_power,
            "entropy_power": self.entropy_power,
            "cross_power": self.cross_power,
            "correlation": self.correlation,
            "drift": self.drift,
        }


def mode_matrix(
    run: MultiFieldRun,
    wavenumber: float,
    subhorizon_efolds: float = SUBHORIZON_EFOLDS,
    freeze_efolds: float = FREEZE_EFOLDS,
    rtol: float = 1e-11,
    atol: float = 1e-14,
) -> ModeMatrix:
    """Evolve the ``F x F`` mode matrix through horizon crossing.

    Started ``subhorizon_efolds`` before the mode's own crossing with each
    field in its own Bunch-Davies vacuum -- so the matrix begins at the
    identity -- and read ``freeze_efolds`` afterwards.

    The state is ``(2, F, F)`` values and the same again for derivatives:
    real and imaginary parts, solution index, field index. Real arithmetic
    throughout, because the coefficients are real and only the initial data
    is complex.
    """
    wavenumber = float(wavenumber)
    crossing = crossing_efolds(run, wavenumber)
    start = crossing - float(subhorizon_efolds)
    stop = crossing + float(freeze_efolds)
    if start < float(run.efolds[0]) - 1e-12:
        raise ValueError(
            f"k = {wavenumber:g} crosses the horizon "
            f"{crossing - float(run.efolds[0]):.3g} e-folds into the run but needs "
            f"{subhorizon_efolds:g} e-folds of sub-horizon evolution first"
        )
    if stop > run.total_efolds + 1e-12:
        raise ValueError(
            f"k = {wavenumber:g} crosses the horizon "
            f"{run.total_efolds - crossing:.3g} e-folds before the end of the run "
            f"but needs {freeze_efolds:g} e-folds afterwards to freeze out"
        )

    fields = int(run.field.shape[-1])
    log_k = math.log(wavenumber)
    size = 2 * fields * fields

    def ratio_at(number: float) -> float:
        return float(np.exp(2.0 * (log_k - log_comoving_hubble(run, number))))

    def right_hand_side(number, state):
        value = state[:size].reshape(2, fields, fields)
        derivative = state[size:].reshape(2, fields, fields)
        _, _, _, epsilon, _ = background_at(run, number)
        operator = mass_matrix(run, number) + ratio_at(number) * np.eye(fields)
        acceleration = -(3.0 - epsilon) * derivative - value @ operator.T
        return np.concatenate([derivative.ravel(), acceleration.ravel()])

    frequency = math.sqrt(ratio_at(start))
    identity = np.eye(fields)
    value0 = np.stack([identity, np.zeros_like(identity)])
    derivative0 = np.stack([-identity, -frequency * identity])
    initial = np.concatenate([value0.ravel(), derivative0.ravel()])

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
        raise RuntimeError(
            f"mode matrix integration failed at k = {wavenumber:g}: {solution.message}"
        )

    # ln P = 2 ln k - ln(4 pi^2) - 2 N_start, from |dphi|^2 = 1/(2 k a^2).
    log_prefactor = 2.0 * log_k - math.log(4.0 * math.pi**2) - 2.0 * start

    def spectra(number: float):
        state = solution.sol(number)
        value = state[:size].reshape(2, fields, fields)
        _, velocity, _, epsilon, _ = background_at(run, number)
        along, entropic = trajectory_basis(velocity)
        sigma = math.sqrt(2.0 * epsilon)

        curvature = value @ along / sigma  # (2, F) over solutions
        entropy = value @ entropic / sigma  # (2, F, F-1)
        scale = math.exp(log_prefactor)
        return (
            scale * float(np.sum(curvature**2)),
            scale * float(np.sum(entropy**2)),
            scale * float(np.sum(curvature[..., None] * entropy)),
        )

    curvature_power, entropy_power, cross_power = spectra(stop)
    earlier = spectra(stop - 1.0)[0]
    return ModeMatrix(
        wavenumber=wavenumber,
        crossing_efolds=crossing,
        curvature_power=curvature_power,
        entropy_power=entropy_power,
        cross_power=cross_power,
        drift=abs(curvature_power - earlier) / curvature_power,
        fields=fields,
    )


@dataclass(frozen=True)
class MultiFieldSpectrum:
    """Spectra and the fitted index, from a pivot-centred set of wavenumbers."""

    wavenumbers: np.ndarray
    curvature_power: np.ndarray
    entropy_power: np.ndarray
    cross_power: np.ndarray
    pivot: float
    efolds_remaining: float
    spectral_index: float
    running: float
    drift: float

    @property
    def pivot_curvature_power(self) -> float:
        logs = np.log(self.wavenumbers)
        return float(np.interp(math.log(self.pivot), logs, self.curvature_power))

    @property
    def pivot_entropy_fraction(self) -> float:
        """``P_S / P_R`` at the pivot: zero once the trajectory is adiabatic."""
        logs = np.log(self.wavenumbers)
        pivot = math.log(self.pivot)
        curvature = float(np.interp(pivot, logs, self.curvature_power))
        return float(np.interp(pivot, logs, self.entropy_power)) / curvature

    def summary(self) -> dict[str, float]:
        return {
            "efolds_remaining": self.efolds_remaining,
            "spectral_index": self.spectral_index,
            "running": self.running,
            "curvature_amplitude": self.pivot_curvature_power,
            "entropy_fraction": self.pivot_entropy_fraction,
            "drift": self.drift,
        }


def multifield_spectrum(
    run: MultiFieldRun,
    efolds_remaining: float = 55.0,
    spread: float = 0.5,
    points: int = 5,
    **kwargs,
) -> MultiFieldSpectrum:
    """Spectra and observables around the pivot, fitted as in the single-field case."""
    if points < 3:
        raise ValueError(f"a quadratic fit needs at least three points, got {points}")
    pivot = pivot_wavenumber(run, efolds_remaining)
    offsets = np.linspace(-float(spread), float(spread), int(points))
    wavenumbers = pivot * np.exp(offsets)
    results = [mode_matrix(run, float(k), **kwargs) for k in wavenumbers]
    curvature = np.array([r.curvature_power for r in results])
    fit = np.polynomial.polynomial.polyfit(offsets, np.log(curvature), 2)
    return MultiFieldSpectrum(
        wavenumbers=wavenumbers,
        curvature_power=curvature,
        entropy_power=np.array([r.entropy_power for r in results]),
        cross_power=np.array([r.cross_power for r in results]),
        pivot=pivot,
        efolds_remaining=float(efolds_remaining),
        spectral_index=1.0 + float(fit[1]),
        running=2.0 * float(fit[2]),
        drift=max(r.drift for r in results),
    )


__all__ = [
    "ModeMatrix",
    "MultiFieldSpectrum",
    "background_at",
    "crossing_efolds",
    "log_comoving_hubble",
    "mass_matrix",
    "mode_matrix",
    "multifield_spectrum",
    "pivot_wavenumber",
    "trajectory_basis",
]
