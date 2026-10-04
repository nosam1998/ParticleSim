"""Bosonic Lorentzian IKKT by complex Langevin (issue #86).

The mass-deformed, gauge-fixed SU(N) action follows Nishimura (2022),
https://arxiv.org/abs/2205.04726, equations (17), (30)--(33). The fermion
Pfaffian is omitted. This truncation can exhibit one expanding direction;
it does not establish three-dimensional space or a model of our universe.

Spatial matrices are complexified Hermitian matrices. Ordered temporal
eigenvalues use exponential gaps, with their trace removed. The action
includes the squared Vandermonde from (17) and the gap Jacobian. Drift
derivatives use the holomorphic action, never a conjugated norm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

PROVENANCE = "Nishimura (2022), arXiv:2205.04726; bosonic, mass-deformed Lorentzian IKKT"
VALIDITY = (
    "Experimental bosonic truncation: the fermion Pfaffian is omitted. Emergent-spacetime "
    "claims are suggestive and debated. Hermitian-part eigenvalues are an approximate, "
    "nonholomorphic diagnostic. Inspect drift tails, thermalization, step-size and "
    "stabilization dependence; finite trajectories do not certify Langevin correctness."
)
DRIFT_EDGES = np.r_[np.linspace(0, 1000, 501), np.geomspace(1050, 1e6, 64)]


def _traceless(a):
    n = a.shape[-1]
    return a - jnp.trace(a, axis1=-1, axis2=-2)[..., None, None] * jnp.eye(n) / n


def temporal_eigenvalues(tau):
    """Trace-free ordered eigenvalues, holomorphically extended to complex gaps."""
    alpha = jnp.concatenate((jnp.zeros(1, dtype=tau.dtype), jnp.cumsum(jnp.exp(tau))))
    return alpha - jnp.mean(alpha)


def _drift(tau, spatial, gamma):
    n = spatial.shape[-1]
    alpha = temporal_eigenvalues(tau)
    difference = alpha[:, None] - alpha[None, :]
    squares = jnp.sum(spatial @ spatial, axis=0)
    sandwich = jnp.sum(spatial[:, None] @ spatial[None, :] @ spatial[:, None], axis=0)
    double_commutator = squares @ spatial + spatial @ squares - 2 * sandwich
    spatial_drift = -1j * n * (difference**2 * spatial - double_commutator + gamma * spatial)
    temporal_gradient = (
        1j * n * (2 * jnp.einsum("ab,iab,iba->a", difference, spatial, spatial) - gamma * alpha)
    )
    # The diagonal is excluded before inversion, rather than evaluating 1/0.
    inverse = 1 / (difference + jnp.eye(n)) - jnp.eye(n)
    temporal_gradient -= 2 * jnp.sum(inverse, axis=1)
    temporal_gradient -= jnp.mean(temporal_gradient)
    tau_drift = 1 - jnp.exp(tau) * jnp.cumsum(temporal_gradient[::-1])[::-1][1:]
    return tau_drift, spatial_drift


class LangevinState(NamedTuple):
    tau: Any
    spatial: Any
    key: Any
    time: Any


@dataclass(frozen=True)
class BosonicIKKT:
    """Nine spatial matrices and one diagonal temporal matrix; g^2 = 1/N."""

    size: int = 32
    gamma: float = 0.0
    provenance: str = PROVENANCE
    validity_statement: str = VALIDITY

    def __post_init__(self):
        if type(self.size) is not int or self.size < 2:
            raise ValueError("size must be an integer at least two")
        if not np.isfinite(self.gamma) or self.gamma < 0:
            raise ValueError("gamma must be finite and non-negative")

    def initialize(self, seed: int = 0, *, noise_seed: int | None = None) -> LangevinState:
        """Start near the Euclidean contour at gamma=0, or near Hermitian at gamma>0.

        A positive-mass run still needs thermalization and, when reproducing
        the expanding branch, adiabatic continuation from large gamma.
        """
        n = self.size
        rng = np.random.default_rng(seed)
        angle = np.pi / 8 if self.gamma == 0 else 0.0
        tau = jnp.full(n - 1, np.log(4 / (n - 1)) - 3j * angle)
        z = rng.normal(size=(9, n, n)) + 1j * rng.normal(size=(9, n, n))
        spatial = (z + z.conj().swapaxes(-1, -2)) * (0.15 / np.sqrt(n)) * np.exp(1j * angle)
        key = jax.random.key(seed + 1 if noise_seed is None else noise_seed)
        return LangevinState(tau, _traceless(jnp.asarray(spatial)), key, jnp.array(0.0))

    def action(self, tau, spatial):
        """Effective action, including the temporal gauge-fixing measure."""
        self._check_shapes(tau, spatial)
        n = self.size
        alpha = temporal_eigenvalues(tau)
        difference = alpha[:, None] - alpha[None, :]
        temporal = difference * spatial
        commutator = spatial[:, None] @ spatial[None, :] - spatial[None, :] @ spatial[:, None]
        quartic = -0.5j * n * jnp.einsum("iab,iba->", temporal, temporal)
        quartic += 0.25j * n * jnp.einsum("ijab,ijba->", commutator, commutator)
        mass = (
            -0.5j * n * self.gamma * (jnp.sum(alpha**2) - jnp.einsum("iab,iba->", spatial, spatial))
        )
        vandermonde = 2 * jnp.sum(jnp.log(difference[jnp.tril_indices(n, -1)]))
        return quartic + mass - vandermonde - jnp.sum(tau)

    def drift(self, tau, spatial):
        """Minus the action gradient; spatial indices follow dS/d(A_i)_{ba}."""
        self._check_shapes(tau, spatial)
        return _drift(tau, spatial, self.gamma)

    def _check_shapes(self, tau, spatial):
        if tau.shape != (self.size - 1,) or spatial.shape != (9, self.size, self.size):
            raise ValueError("state shapes do not match the model size")


@dataclass(frozen=True)
class DriftRecord:
    steps: int
    minimum_adaptive_step: float
    maximum_drift: float
    time_in_bins: np.ndarray


@jax.jit
def _advance(state, target, gamma, step_size, drift_limit, stabilization, max_steps):
    bins = jnp.asarray(DRIFT_EDGES)
    initial = (state, 0, step_size, jnp.array(0.0), jnp.zeros(len(DRIFT_EDGES) + 1), True)

    def running(carry):
        current, count, _, _, _, finite = carry
        return (current.time < target) & (count < max_steps) & finite

    def step(carry):
        current, count, minimum, maximum, histogram, _ = carry
        kt, ka = _drift(current.tau, current.spatial, gamma)
        magnitude = jnp.maximum(jnp.max(jnp.abs(kt)), jnp.max(jnp.abs(ka)))
        adaptive_step = step_size * jnp.minimum(1.0, drift_limit / magnitude)
        dt = jnp.minimum(target - current.time, adaptive_step)
        key, temporal_key, real_key, imag_key = jax.random.split(current.key, 4)
        z = jax.random.normal(real_key, current.spatial.shape, dtype=jnp.float64)
        z = z + 1j * jax.random.normal(imag_key, current.spatial.shape, dtype=jnp.float64)
        noise = _traceless((z + z.conj().swapaxes(-1, -2)) / 2)
        tau = (
            current.tau
            + dt * kt
            + jnp.sqrt(2 * dt)
            * jax.random.normal(temporal_key, current.tau.shape, dtype=jnp.float64)
        )
        spatial = current.spatial + dt * ka + jnp.sqrt(2 * dt) * noise
        spatial = (spatial + stabilization * spatial.conj().swapaxes(-1, -2)) / (1 + stabilization)
        finite = (
            jnp.isfinite(magnitude) & jnp.all(jnp.isfinite(tau)) & jnp.all(jnp.isfinite(spatial))
        )
        finite &= (dt > 0) & (current.time + dt > current.time)
        # Weight by Langevin time: counting adaptive steps overweights high drift.
        histogram = histogram.at[jnp.searchsorted(bins, magnitude)].add(dt)
        out = LangevinState(tau, _traceless(spatial), key, current.time + dt)
        return (
            out,
            count + 1,
            jnp.minimum(minimum, adaptive_step),
            jnp.maximum(maximum, magnitude),
            histogram,
            finite,
        )

    return jax.lax.while_loop(running, step, initial)


def _measure(state, block_size):
    a = np.asarray(state.spatial)
    block_count = a.shape[-1] - block_size + 1
    extents, eigenvalues, departures = [], [], []
    for offset in range(block_count):
        block = a[:, offset : offset + block_size, offset : offset + block_size]
        extents.append(np.einsum("iab,iba->", block, block) / block_size)
        hermitian = (block + block.conj().swapaxes(-1, -2)) / 2
        tensor = np.einsum("iab,jba->ij", hermitian, hermitian).real / block_size
        eigenvalues.append(np.linalg.eigvalsh(tensor)[::-1])
        departures.append(np.linalg.norm(block - hermitian) / max(np.linalg.norm(block), 1e-300))
    return np.asarray(temporal_eigenvalues(state.tau)), extents, eigenvalues, departures


@dataclass(frozen=True)
class IKKTObservations:
    alpha: np.ndarray
    extent_squared: np.ndarray
    hermitian_eigenvalues: np.ndarray
    nonhermiticity: np.ndarray
    langevin_times: np.ndarray
    drifts: tuple[DriftRecord, ...]
    size: int
    gamma: float
    block_size: int
    step_size: float
    stabilization: float
    drift_limit: float = 100.0
    provenance: str = PROVENANCE
    validity_statement: str = VALIDITY

    def physical_times(self):
        """Time from block averages of ensemble-mean alpha, centered on the middle block."""
        alpha = self.alpha.mean(axis=0)
        blocks = np.convolve(alpha, np.ones(self.block_size) / self.block_size, mode="valid")
        times = np.r_[0.0, np.cumsum(np.abs(np.diff(blocks)))]
        return times - (times[0] + times[-1]) / 2

    def summary(self):
        """Serializable observables with truncation and validity visible in every report."""
        return {
            "model": "bosonic mass-deformed Lorentzian IKKT",
            "fermions": "omitted",
            "status": "experimental",
            "provenance": self.provenance,
            "validity_statement": self.validity_statement,
            "size": self.size,
            "gamma": self.gamma,
            "block_size": self.block_size,
            "samples": len(self.alpha),
            "step_size": self.step_size,
            "stabilization": self.stabilization,
            "drift_limit": self.drift_limit,
            "langevin_times": self.langevin_times.tolist(),
            "physical_times": self.physical_times().tolist(),
            "hermitian_eigenvalues": self.hermitian_eigenvalues.mean(axis=0).tolist(),
            "extent_phase": (np.angle(self.extent_squared.mean(axis=0)) / 2).tolist(),
            "nonhermiticity": self.nonhermiticity.mean(axis=0).tolist(),
            "drift_bin_edges": DRIFT_EDGES.tolist(),
            "drift_time_in_bins": np.sum([d.time_in_bins for d in self.drifts], axis=0).tolist(),
            "maximum_drift": max(d.maximum_drift for d in self.drifts),
            "minimum_adaptive_step": min(d.minimum_adaptive_step for d in self.drifts),
        }


@dataclass(frozen=True)
class ComplexLangevin:
    model: BosonicIKKT
    step_size: float = 2e-5
    drift_limit: float = 100.0
    stabilization: float = 0.0
    max_steps: int = 1_000_000

    def __post_init__(self):
        for name in ("step_size", "drift_limit"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.stabilization) or not 0 <= self.stabilization <= 1:
            raise ValueError("stabilization must lie between zero and one")
        if self.model.gamma == 0 and self.stabilization != 0:
            raise ValueError("stabilization is unjustified on the undeformed Euclidean branch")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")

    def advance(self, state: LangevinState, duration: float) -> tuple[LangevinState, DriftRecord]:
        """Advance exactly duration in Langevin time, with adaptive internal steps."""
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive")
        self.model._check_shapes(state.tau, state.spatial)
        target = float(state.time) + duration
        if not np.isfinite(target) or target <= float(state.time):
            raise ValueError("duration must advance a finite Langevin clock")
        result = _advance(
            state,
            target,
            self.model.gamma,
            self.step_size,
            self.drift_limit,
            self.stabilization,
            self.max_steps,
        )
        current, count, minimum, maximum, histogram, finite = result
        if not bool(finite):
            raise RuntimeError("nonfinite Langevin state or time step; the trajectory is invalid")
        if float(current.time) < target:
            raise RuntimeError("maximum Langevin steps reached before the requested duration")
        record = DriftRecord(int(count), float(minimum), float(maximum), np.asarray(histogram))
        return current, record

    def sample(self, state, count: int, *, interval: float = 0.02, block_size: int = 4):
        """Equal-time samples. Thermalization and continuation are explicit advance calls."""
        if type(count) is not int or count < 1:
            raise ValueError("count must be a positive integer")
        if type(block_size) is not int or not 1 <= block_size <= self.model.size:
            raise ValueError("block_size must lie between one and the matrix size")
        measurements, times, records = [], [], []
        for _ in range(count):
            state, record = self.advance(state, interval)
            measurements.append(_measure(state, block_size))
            times.append(float(state.time))
            records.append(record)
        alpha, extents, eigenvalues, departures = map(np.asarray, zip(*measurements, strict=True))
        observations = IKKTObservations(
            alpha,
            extents,
            eigenvalues,
            departures,
            np.asarray(times),
            tuple(records),
            self.model.size,
            self.model.gamma,
            block_size,
            self.step_size,
            self.stabilization,
            self.drift_limit,
        )
        return state, observations
