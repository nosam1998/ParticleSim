"""Euclidean hybrid Monte Carlo for scalar and compact gauge fields.

Issue #63. Hybrid Monte Carlo proposes a configuration by integrating a
fictitious Hamiltonian and accepts it with a Metropolis test. The accept
step is what makes the algorithm exact: the integrator can be as crude as
you like and the distribution sampled is still the right one, provided the
proposal is *reversible* and *volume preserving*. Leapfrog is both, which is
why it is the one used.

**That gives three checks that hold to machine precision or to statistics,
independently of any physics.**

* **The force is the gradient of the action.** Compared here against a
  central difference of :meth:`action` itself, to about ``1e-9``. Almost
  every wrong lattice result traces back to a sign or a shifted index in the
  force, and nothing downstream would notice: a wrong force still integrates,
  still accepts, and samples a different theory.
* **The trajectory is reversible.** Integrate forward, flip the momenta,
  integrate back, and the configuration returns to ``1e-15``.
* **``<exp(-dH)> = 1`` exactly.** Not approximately, and not only in the
  small-step limit: it is a consequence of detailed balance and holds at any
  step size, so it tests the accept step and the integrator together. A
  broken force or a non-symplectic update breaks it while leaving the
  acceptance rate looking healthy.

**And one that is exact in the physics.** Two-dimensional compact ``U(1)``
factorises: the character expansion gives

    Z = sum_n I_n(beta)^V,   <cos theta_p> = sum_n I_n' I_n^(V-1) / sum_n I_n^V

for ``V`` plaquettes, tending to ``I_1(beta)/I_0(beta)`` as the volume grows.
The finite-volume form is the one to compare against, and the distinction is
not academic: at ``beta = 1`` on a ``2 x 2`` lattice the exact answer is
``0.5052`` against an infinite-volume ``0.4464``, so a correct simulation
sits **13% away** from the textbook number. An acceptance of "within 1% of
``I_1/I_0``" is unreachable on any lattice small enough to test quickly, and
reaching for it would mean either a wrong verdict or a run that is not a
test. The exact finite-volume value is both sharper and affordable.

**Autocorrelation is not a refinement here, it is the difference between
agreeing and disagreeing.** Successive trajectories are correlated, so the
naive error on a mean is too small by ``sqrt(2 tau_int)``. On the runs below
that factor is about two, which is the difference between a two-sigma
discrepancy and a one-sigma one. :func:`integrated_autocorrelation` uses the
self-consistent window of Madras and Sokal, which stops summing once the
window reaches ``6 tau`` rather than running into the noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.special import iv, ivp

#: Terms kept either side of zero in the character expansion.
CHARACTER_TERMS = 60

#: Zero crossings below which a chain has not sampled both wells of a broken phase.
ERGODICITY_FLOOR = 30

#: Bins a binning analysis keeps, below which the spread of bin means is itself noise.
MINIMUM_BINS = 16


class Model(Protocol):
    """What hybrid Monte Carlo needs from a theory."""

    def action(self, state: np.ndarray) -> float: ...

    def force(self, state: np.ndarray) -> np.ndarray: ...

    def cold_start(self) -> np.ndarray: ...


@dataclass(frozen=True)
class Phi4:
    """``S = sum_x [ (1/2) sum_mu (dphi)^2 + (1/2) m^2 phi^2 + lambda phi^4 / 4! ]``.

    At ``coupling = 0`` the theory is Gaussian and its propagator is known in
    closed form, which is what :meth:`propagator` returns and what makes the
    free limit a complete test of the sampler rather than a smoke test.
    """

    shape: tuple[int, ...] = (8, 8)
    mass_squared: float = 0.5
    coupling: float = 0.0

    def __post_init__(self) -> None:
        if any(n < 2 for n in self.shape):
            raise ValueError(f"every lattice direction needs at least two sites, got {self.shape}")
        if self.coupling < 0.0:
            raise ValueError(
                f"the quartic coupling must not be negative, got {self.coupling}; "
                "the action is then unbounded below and there is nothing to sample"
            )

    @property
    def sites(self) -> int:
        return int(np.prod(self.shape))

    def cold_start(self) -> np.ndarray:
        return np.zeros(self.shape)

    def action(self, state: np.ndarray) -> float:
        kinetic = sum(np.sum((np.roll(state, -1, axis) - state) ** 2) for axis in range(state.ndim))
        quartic = self.coupling * np.sum(state**4) / 24.0
        return float(kinetic / 2.0 + self.mass_squared * np.sum(state**2) / 2.0 + quartic)

    def force(self, state: np.ndarray) -> np.ndarray:
        """``-dS/dphi``: the lattice Laplacian minus the potential's slope."""
        laplacian = (
            sum(np.roll(state, -1, axis) + np.roll(state, 1, axis) for axis in range(state.ndim))
            - 2 * state.ndim * state
        )
        return laplacian - self.mass_squared * state - self.coupling * state**3 / 6.0

    def magnetisation(self, state: np.ndarray) -> float:
        return float(np.mean(state))

    def propagator(self) -> np.ndarray:
        """``1/(khat^2 + m^2)`` with ``khat^2 = sum_mu 4 sin^2(k_mu/2)``.

        The *lattice* dispersion, not ``k^2``: a free-field check against the
        continuum propagator would fail at large momentum for a correct code
        and be indistinguishable from a bug.
        """
        if self.coupling != 0.0:
            raise ValueError(
                f"the closed-form propagator is the free one and the coupling is "
                f"{self.coupling}; there is no exact expression to compare against"
            )
        squared = np.zeros(self.shape)
        for axis, count in enumerate(self.shape):
            momenta = 2.0 * np.pi * np.fft.fftfreq(count)
            shape = [1] * len(self.shape)
            shape[axis] = count
            squared = squared + (4.0 * np.sin(momenta / 2.0) ** 2).reshape(shape)
        return 1.0 / (squared + self.mass_squared)

    @staticmethod
    def mode_power(state: np.ndarray) -> np.ndarray:
        """``|phi(k)|^2`` with the convention the propagator above is written in."""
        return np.abs(np.fft.fftn(state) / np.sqrt(state.size)) ** 2


@dataclass(frozen=True)
class CompactU1:
    """Two-dimensional compact ``U(1)``: ``S = beta sum_p (1 - cos theta_p)``.

    Two dimensions specifically, because that is where the theory factorises
    and the plaquette is known exactly at finite volume.
    """

    size: int = 4
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError(f"the lattice needs at least two sites a side, got {self.size}")
        if self.beta <= 0.0:
            raise ValueError(f"the inverse coupling must be positive, got {self.beta}")

    @property
    def plaquettes(self) -> int:
        return self.size**2

    def cold_start(self) -> np.ndarray:
        return np.zeros((2, self.size, self.size))

    def plaquette_angles(self, links: np.ndarray) -> np.ndarray:
        """``theta_1(x) + theta_2(x+1) - theta_1(x+2) - theta_2(x)``."""
        first, second = links
        return first + np.roll(second, -1, axis=0) - np.roll(first, -1, axis=1) - second

    def action(self, links: np.ndarray) -> float:
        return float(self.beta * np.sum(1.0 - np.cos(self.plaquette_angles(links))))

    def force(self, links: np.ndarray) -> np.ndarray:
        """``-dS/dtheta``. Each link sits in exactly two plaquettes, with opposite signs."""
        sines = np.sin(self.plaquette_angles(links))
        return -self.beta * np.stack(
            [sines - np.roll(sines, 1, axis=1), -sines + np.roll(sines, 1, axis=0)]
        )

    def mean_plaquette(self, links: np.ndarray) -> float:
        return float(np.mean(np.cos(self.plaquette_angles(links))))

    def exact_plaquette(self) -> float:
        """The character sum at *this* volume, which is what a run reproduces.

        Ratios to ``I_0`` are carried rather than the Bessel functions
        themselves: ``I_n(beta)^V`` overflows for a few hundred plaquettes
        while the ratios stay inside one.
        """
        orders = np.arange(-CHARACTER_TERMS, CHARACTER_TERMS + 1)
        ratio = iv(orders, self.beta) / iv(0, self.beta)
        derivative = ivp(orders, self.beta, 1) / iv(0, self.beta)
        volume = self.plaquettes
        return float(np.sum(derivative * ratio ** (volume - 1)) / np.sum(ratio**volume))

    @staticmethod
    def infinite_volume_plaquette(beta: float) -> float:
        """``I_1(beta)/I_0(beta)``: the textbook value, and the wrong one to test against."""
        return float(iv(1, beta) / iv(0, beta))


@dataclass
class Chain:
    """One run's measurements, with the error that accounts for correlation."""

    values: np.ndarray
    energy_changes: np.ndarray
    acceptance: float

    @property
    def mean(self) -> float:
        return float(np.mean(self.values))

    @property
    def naive_error(self) -> float:
        """What the standard deviation says, which is always too small."""
        return float(np.std(self.values) / np.sqrt(len(self.values)))

    @property
    def tau(self) -> float:
        return integrated_autocorrelation(self.values)

    @property
    def error(self) -> float:
        """``naive x sqrt(2 tau_int)``, the one to quote."""
        return self.naive_error * np.sqrt(2.0 * self.tau)

    @property
    def exchange_average(self) -> float:
        """``<exp(-dH)>``, which detailed balance fixes at exactly 1."""
        return float(np.mean(np.exp(-self.energy_changes)))

    @property
    def sign_changes(self) -> int:
        """How often the measured quantity crossed zero.

        For an order parameter this counts tunnelling events between the two
        wells, and it is the diagnostic that catches the worst failure in
        this module. Local hybrid Monte Carlo cannot tunnel once the barrier
        is high: the chain settles in one well, every block of a jackknife
        agrees about it, and the run reports a confident number for a
        quantity it never sampled. Measured on ``phi^4`` below the
        transition, the flips fall 808, 278, 67, 0 as the mass is lowered --
        and the error bar *shrinks* along the way, which is why the run has
        to be asked this question rather than trusted.
        """
        if self.values.size < 2:
            return 0
        signs = np.sign(self.values)
        return int(np.count_nonzero(signs[1:] != signs[:-1]))

    @property
    def ergodic(self) -> bool:
        """Whether the chain crossed zero often enough to have sampled both wells.

        Thirty crossings is a low bar deliberately: it is not a statement
        that the run is converged, only that it is not the degenerate case
        where the answer and its error describe a single well.
        """
        return self.sign_changes >= ERGODICITY_FLOOR

    def pull(self, expected: float) -> float:
        """How many of its own error bars the mean sits from ``expected``."""
        return (self.mean - expected) / self.error


@dataclass
class HybridMonteCarlo:
    """Leapfrog proposal, Metropolis accept. Exact at any step size."""

    model: Model
    step: float = 0.15
    steps: int = 10
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self) -> None:
        if self.step <= 0.0 or self.steps < 1:
            raise ValueError(
                f"a trajectory needs a positive step and at least one of them, "
                f"got {self.step} and {self.steps}"
            )

    def hamiltonian(self, state: np.ndarray, momentum: np.ndarray) -> float:
        return self.model.action(state) + float(np.sum(momentum**2)) / 2.0

    def leapfrog(self, state: np.ndarray, momentum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Half kick, then alternate drift and kick, then a half kick.

        Written as a single half-step at each end rather than a half-step
        inside the loop, so the map is manifestly a composition of shears --
        which is where volume preservation and reversibility come from.
        """
        state = np.array(state, copy=True)
        momentum = momentum + 0.5 * self.step * self.model.force(state)
        for index in range(self.steps):
            state = state + self.step * momentum
            scale = self.step if index < self.steps - 1 else 0.5 * self.step
            momentum = momentum + scale * self.model.force(state)
        return state, momentum

    def trajectory(self, state: np.ndarray) -> tuple[np.ndarray, bool, float]:
        momentum = self.rng.normal(size=np.shape(state))
        before = self.hamiltonian(state, momentum)
        proposal, final = self.leapfrog(state, momentum)
        change = self.hamiltonian(proposal, final) - before
        if self.rng.random() < np.exp(-change):
            return proposal, True, change
        return state, False, change

    def run(self, sweeps: int, thermalise: int = 0, observable=None, state=None) -> Chain:
        if thermalise >= sweeps:
            raise ValueError(
                f"thermalising for {thermalise} of {sweeps} sweeps leaves nothing to measure"
            )
        state = self.model.cold_start() if state is None else np.array(state, copy=True)
        values: list[float] = []
        changes: list[float] = []
        accepted = 0
        for sweep in range(sweeps):
            state, was_accepted, change = self.trajectory(state)
            accepted += int(was_accepted)
            changes.append(change)
            if sweep >= thermalise and observable is not None:
                values.append(float(observable(state)))
        return Chain(
            values=np.array(values),
            energy_changes=np.array(changes),
            acceptance=accepted / sweeps,
        )

    def reversibility_residual(self, state: np.ndarray) -> float:
        """Forward, flip, back: how far the configuration fails to return.

        Round-off only for a correct leapfrog, so this is a bug detector
        rather than a tolerance to tune.
        """
        momentum = self.rng.normal(size=np.shape(state))
        forward, carried = self.leapfrog(state, momentum)
        back, returned = self.leapfrog(forward, -carried)
        return float(max(np.max(np.abs(back - state)), np.max(np.abs(-returned - momentum))))

    def force_residual(self, state: np.ndarray, delta: float = 1e-6) -> float:
        """The force against a central difference of the action.

        The check that catches the failure mode nothing else would: a wrong
        force integrates, accepts, and samples a different theory.
        """
        state = np.asarray(state, dtype=float)
        analytic = self.model.force(state)
        worst = 0.0
        for index in np.ndindex(state.shape):
            plus = state.copy()
            minus = state.copy()
            plus[index] += delta
            minus[index] -= delta
            numeric = -(self.model.action(plus) - self.model.action(minus)) / (2.0 * delta)
            worst = max(worst, abs(numeric - analytic[index]))
        return float(worst)


def integrated_autocorrelation(series, window: int | None = None) -> float:
    """``tau_int`` with the self-consistent window of Madras and Sokal.

    Summing the autocorrelation function to the end of the series adds noise
    without signal and typically *underestimates* the error, which is the
    failure that makes a correct simulation look wrong. Stopping once the
    window reaches ``6 tau`` is the standard remedy.

    ``window`` caps the lag and defaults to a quarter of the series, so that
    the ``6 tau`` criterion is what ends the sum rather than an arbitrary
    constant. A fixed cap is a trap near a critical point: ``tau`` there runs
    to a hundred or more, ``6 tau`` exceeds any modest constant, and the
    function silently returns the cap-truncated value -- an *underestimate*,
    in exactly the regime where the error matters most. Passing a small
    ``window`` therefore biases the answer low rather than merely saving
    time.
    """
    values = np.asarray(series, dtype=float)
    if values.size < 4:
        return 0.5
    if window is None:
        window = max(4, values.size // 4)
    centred = values - values.mean()
    variance = float(np.dot(centred, centred) / centred.size)
    if variance == 0.0:
        return 0.5
    running = 0.5
    tau = 0.5
    for lag in range(1, min(window, centred.size // 4)):
        running += float(np.dot(centred[:-lag], centred[lag:]) / (centred.size - lag) / variance)
        if lag >= 6.0 * running:
            break
        tau = running
    return max(tau, 0.5)


def binned_errors(series, sizes=None):
    """Error on the mean against bin size: ``(sizes, errors)``.

    The picture a single ``tau_int`` cannot give. Averaging over bins of
    increasing length decorrelates the data, so the error rises from the
    naive value and flattens once the bin is long compared with the
    correlation time. The **plateau** is the honest error, and whether the
    curve has actually reached one is the thing worth looking at: a reported
    ``tau`` of 9.5 says nothing about whether the run was long enough to
    measure 9.5.

    The plateau sits at ``sqrt(2 tau_int)`` times the naive error, which the
    suite checks against an autoregressive process of known ``tau``.

    Bin sizes default to powers of two that still leave at least sixteen
    bins, because the spread of a handful of bin means is itself too noisy
    to read a plateau from.
    """
    values = np.asarray(series, dtype=float)
    if values.size < MINIMUM_BINS:
        raise ValueError(
            f"binning needs at least {MINIMUM_BINS} samples to leave a readable curve, "
            f"got {values.size}"
        )
    if sizes is None:
        largest = max(1, values.size // MINIMUM_BINS)
        sizes = [size for size in (2**power for power in range(64)) if size <= largest]
    chosen = []
    errors = []
    for size in sizes:
        count = values.size // int(size)
        if count < 2:
            continue
        means = values[: count * int(size)].reshape(count, int(size)).mean(axis=1)
        chosen.append(int(size))
        errors.append(float(np.std(means, ddof=1) / np.sqrt(count)))
    return np.array(chosen), np.array(errors)


def plateau_error(series) -> float:
    """The binning estimate of the error on the mean.

    Read as the median of the upper half of the binning curve, where the
    bins are long compared with the correlation time and the curve has
    flattened. A median rather than the last point, because the bin count
    falls as the bin grows and the tail of the curve is noisy -- which is
    itself visible in :func:`binned_errors` and is why the picture is worth
    drawing.

    This is an *independent* estimate of the same quantity
    :attr:`Chain.error` gets by summing the autocorrelation function. The
    two use different information, so their agreeing is a check on both;
    the suite holds them to ten percent of each other on a process whose
    ``tau`` is known exactly.
    """
    _, errors = binned_errors(series)
    return float(np.median(errors[len(errors) // 2 :]))


def binder_cumulant(magnetisations) -> float:
    """``1 - <m^4>/(3 <m^2>^2)``: zero for a Gaussian, 2/3 for a two-state signal.

    The quantity whose crossing in lattice size locates a critical coupling,
    and dimensionless, which is why the crossing is the estimator rather than
    any single lattice's value.
    """
    values = np.asarray(magnetisations, dtype=float)
    second = float(np.mean(values**2))
    if second == 0.0:
        return 0.0
    return 1.0 - float(np.mean(values**4)) / (3.0 * second**2)


__all__ = [
    "CHARACTER_TERMS",
    "ERGODICITY_FLOOR",
    "MINIMUM_BINS",
    "Chain",
    "CompactU1",
    "HybridMonteCarlo",
    "Model",
    "Phi4",
    "binder_cumulant",
    "binned_errors",
    "integrated_autocorrelation",
    "plateau_error",
]
