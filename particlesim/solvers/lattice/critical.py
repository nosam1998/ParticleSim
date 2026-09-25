"""The two-dimensional φ⁴ critical coupling, with cluster updates (issue #63).

:mod:`~particlesim.solvers.lattice.euclidean` located the transition and
stopped short of a number, for three measured reasons: local updates stop
tunnelling below it, their autocorrelation grows with the lattice, and the
published number is a renormalised continuum ratio a lattice does not
measure directly. This module removes the first two and does the
conversion.

**Clusters for the sign, hybrid Monte Carlo for the size.** Write
``phi_x = s_x |phi_x|``. At fixed ``|phi|`` the action is an Ising model in
the signs, with bond ``J_xy = |phi_x| |phi_y|``, and Swendsen and Wang's
algorithm updates it exactly: bond aligned neighbours with probability
``1 - exp(-2 J_xy)`` and flip each cluster with probability one half (Brower
and Tamayo 1989). A cluster flip carries a whole domain across the barrier
in one step, which is what local updates cannot do. It never changes
``|phi|``, so a hybrid Monte Carlo trajectory between flips does that. At
``lambda = 1`` and ``L = 16`` the pair gives a Binder cumulant of
0.6036 ± 0.0008 against pure hybrid Monte Carlo's 0.6008 ± 0.0026, with
``tau_int`` 4 against 16 and a hundred times as many sign changes. At the
transition ``tau_int`` stays near 10 from ``L = 32`` to ``64``.

**The crossing.** The Binder cumulant ``U = 1 - <M^4>/(3 <M^2>^2)`` at the
critical point tends to its value for the two-dimensional Ising class on a
periodic square, ``U* = 0.6106901`` (Salas and Sokal 2000), since φ⁴ is in
that class. Each lattice gives the bare mass where ``U_L = U*``, and those
converge to the critical bare mass as ``L`` grows. ``sum phi^2`` is
conjugate to ``m^2``, so one run reweights to nearby masses and the crossing
is found to a fraction of the spacing between runs.

**The conversion.** The literature quotes ``f = lambda/mu^2`` for
``(lambda/4) phi^4``, with ``mu^2`` the mass after normal ordering. In two
dimensions that is the only renormalisation needed, and on the lattice

    m_0^2 = mu^2 - 3 lambda A(mu^2),   A(mu^2) = int d^2k/(2 pi)^2 1/(khat^2 + mu^2)

with the lattice dispersion ``khat^2 = sum 4 sin^2(k/2)``. On the square
lattice ``A`` has a closed form, ``K(m = (2/z)^2)/(pi z)`` with
``z = 2 + mu^2/2``. This module's :class:`~.euclidean.Phi4` writes the
quartic as ``g phi^4/4!``, so ``lambda = g/6``.

**What sets the error.** ``mu^2`` is a small difference of two numbers near
``3 lambda A``, so ``df/dm_0^2`` is large: about 130 at ``lambda = 0.25``.
Half a percent on ``f`` needs the critical bare mass to ``4e-4``, and the
reweighted crossing gives it to ``3e-4`` at ``L = 64``.

**The continuum limit** is ``lambda -> 0`` in lattice units, and it is not
linear. Schaich and Loinaz (2009) showed that a ``lambda ln lambda`` term is
needed, which is why fits through ``lambda ~ 1`` found 10.26 and theirs
10.8. :func:`continuum_fit` fits ``f_0 + a lambda + b lambda ln lambda``.
The measurements are in ``docs/benchmarks.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import ellipk

from particlesim.solvers.lattice.euclidean import HybridMonteCarlo, Phi4

#: The Binder cumulant at the critical point of the 2D Ising class, periodic square.
ISING_BINDER = 0.6106901


def embedded_ising_flip(state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One Swendsen-Wang update of the signs of ``state``, its magnitudes fixed.

    The nearest-neighbour term of the action is ``-sum phi_x phi_y``, an Ising
    coupling ``|phi_x| |phi_y|`` between the signs. Aligned neighbours are
    bonded with probability ``1 - exp(-2 phi_x phi_y)``, and each connected
    cluster flips with probability one half.
    """
    state = np.asarray(state, dtype=float)
    flat = state.ravel()
    index = np.arange(flat.size).reshape(state.shape)
    rows, cols = [], []
    for axis in range(state.ndim):
        neighbour = np.roll(index, -1, axis).ravel()
        product = flat * flat[neighbour]
        bonded = (product > 0) & (rng.random(flat.size) < -np.expm1(-2.0 * product))
        rows.append(index.ravel()[bonded])
        cols.append(neighbour[bonded])
    rows_all, cols_all = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix(
        (np.ones(rows_all.size, dtype=np.int8), (rows_all, cols_all)),
        shape=(flat.size, flat.size),
    )
    count, labels = connected_components(graph, directed=False)
    flips = rng.random(count) < 0.5
    return np.where(flips[labels], -flat, flat).reshape(state.shape)


def lattice_tadpole(mass_squared: float) -> float:
    """``int d^2k/(2 pi)^2 1/(khat^2 + mu^2)`` over the Brillouin zone of the square lattice.

    ``K(m)/(pi z)`` with ``z = 2 + mu^2/2`` and ``m = (2/z)^2``. It diverges
    as ``ln(32/mu^2)/(4 pi)`` when ``mu^2 -> 0``, the logarithm that normal
    ordering removes.
    """
    if mass_squared <= 0.0:
        raise ValueError(f"the tadpole needs a positive mass squared, got {mass_squared}")
    z = 2.0 + mass_squared / 2.0
    return float(ellipk((2.0 / z) ** 2) / (np.pi * z))


def renormalised_mass(bare: float, quartic: float) -> float:
    """``mu^2`` from ``m_0^2 = mu^2 - 3 lambda A(mu^2)``, for ``(lambda/4) phi^4``."""
    return float(
        brentq(lambda mu2: mu2 - 3.0 * quartic * lattice_tadpole(mu2) - bare, 1e-12, 100.0)
    )


def critical_ratio(bare: float, quartic: float) -> float:
    """``lambda/mu^2`` at a critical bare mass, the number the literature quotes."""
    return quartic / renormalised_mass(bare, quartic)


def _weights(square_sum, run_mass: float, mass: float) -> np.ndarray:
    """Reweighting factors from ``m^2 = run_mass`` to ``mass``, normalised."""
    square_sum = np.asarray(square_sum, dtype=float)
    exponent = -(mass - run_mass) / 2.0 * (square_sum - square_sum.mean())
    weights = np.exp(exponent - exponent.max())
    return weights / weights.sum()


def reweighted_binder(magnetisation, square_sum, run_mass: float, mass: float) -> float:
    """``U`` at ``m^2 = mass`` from a run at ``run_mass``.

    ``square_sum`` is ``sum_x phi_x^2`` on each configuration, the quantity
    ``m^2`` multiplies in the action. Reliable while ``mass`` is within a
    few ``1/sqrt(var(square_sum))`` of ``run_mass``.
    """
    m = np.asarray(magnetisation, dtype=float)
    w = _weights(square_sum, run_mass, mass)
    return float(1.0 - np.sum(w * m**4) / (3.0 * np.sum(w * m**2) ** 2))


def binder_crossing(magnetisation, square_sum, run_mass: float, width: float) -> float:
    """The ``m^2`` within ``width`` of ``run_mass`` where the reweighted ``U`` is ``U*``."""
    return float(
        brentq(
            lambda mass: (
                reweighted_binder(magnetisation, square_sum, run_mass, mass) - ISING_BINDER
            ),
            run_mass - width,
            run_mass + width,
        )
    )


def jackknife_crossing(
    magnetisation, square_sum, run_mass: float, width: float, blocks: int = 20
) -> tuple[float, float]:
    """``(crossing, error)``, the error from ``blocks`` jackknife blocks.

    Blocks rather than single samples, so that each is long compared with
    the autocorrelation time.
    """
    m = np.asarray(magnetisation, dtype=float)
    s = np.asarray(square_sum, dtype=float)
    size = m.size // blocks
    m, s = m[: blocks * size], s[: blocks * size]
    best = binder_crossing(m, s, run_mass, width)
    left_out = [
        binder_crossing(
            np.delete(m, np.s_[k * size : (k + 1) * size]),
            np.delete(s, np.s_[k * size : (k + 1) * size]),
            run_mass,
            width,
        )
        for k in range(blocks)
    ]
    return best, float(np.sqrt((blocks - 1) * np.var(left_out)))


def continuum_fit(couplings, ratios, errors) -> tuple[np.ndarray, np.ndarray]:
    """Weighted fit of ``f_0 + a lambda + b lambda ln lambda``: ``(coefficients, errors)``."""
    lam = np.asarray(couplings, dtype=float)
    f = np.asarray(ratios, dtype=float)
    sigma = np.asarray(errors, dtype=float)
    design = np.stack([np.ones_like(lam), lam, lam * np.log(lam)], axis=1) / sigma[:, None]
    covariance = np.linalg.inv(design.T @ design)
    coefficients = covariance @ design.T @ (f / sigma)
    return coefficients, np.sqrt(np.diag(covariance))


@dataclass
class ClusterHybrid:
    """A hybrid Monte Carlo trajectory then a cluster flip of the signs, per sweep.

    ``quartic`` is ``lambda`` in ``(lambda/4) phi^4``, the literature's
    convention; the model underneath is :class:`~.euclidean.Phi4` with
    ``g = 6 lambda``. The trajectory's step shrinks as ``L^(-1/2)`` to hold
    the acceptance near 0.8 as the volume grows, with its length held at
    about 1.5.
    """

    size: int
    mass_squared: float
    quartic: float
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self) -> None:
        self.model = Phi4((self.size, self.size), self.mass_squared, 6.0 * self.quartic)
        step = min(0.15, 0.15 * np.sqrt(32.0 / self.size))
        self.hybrid = HybridMonteCarlo(
            self.model, step=step, steps=max(6, int(round(1.5 / step))), rng=self.rng
        )

    def sweep(self, state: np.ndarray) -> tuple[np.ndarray, bool]:
        state, accepted, _ = self.hybrid.trajectory(state)
        return embedded_ising_flip(state, self.rng), accepted

    def run(self, sweeps: int, thermalise: int = 200, state=None):
        """``(magnetisation, square_sum, acceptance, final state)`` over ``sweeps`` sweeps.

        The start is a small random field thermalised by flips and short
        trajectories. A cold start at a large lattice is far enough from
        equilibrium that the first trajectories are all rejected.
        """
        if state is None:
            state = 0.1 * self.rng.normal(size=(self.size, self.size))
            warm = HybridMonteCarlo(self.model, step=0.05, steps=10, rng=self.rng)
            for _ in range(thermalise):
                state, _, _ = warm.trajectory(state)
                state = embedded_ising_flip(state, self.rng)
        for _ in range(thermalise):
            state, _ = self.sweep(state)
        magnetisation = np.empty(sweeps)
        square_sum = np.empty(sweeps)
        accepted = 0
        for index in range(sweeps):
            state, was = self.sweep(state)
            accepted += int(was)
            magnetisation[index] = state.mean()
            square_sum[index] = float(np.sum(state**2))
        return magnetisation, square_sum, accepted / sweeps, state


__all__ = [
    "ISING_BINDER",
    "ClusterHybrid",
    "binder_crossing",
    "continuum_fit",
    "critical_ratio",
    "embedded_ising_flip",
    "jackknife_crossing",
    "lattice_tadpole",
    "renormalised_mass",
    "reweighted_binder",
]
