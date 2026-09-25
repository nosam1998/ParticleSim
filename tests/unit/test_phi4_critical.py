"""The two-dimensional φ⁴ critical coupling: clusters, the crossing, the conversion (issue #63)."""

from __future__ import annotations

import itertools

import numpy as np

from particlesim.solvers.lattice.critical import (
    ISING_BINDER,
    ClusterHybrid,
    continuum_fit,
    critical_ratio,
    embedded_ising_flip,
    jackknife_crossing,
    lattice_tadpole,
    renormalised_mass,
    reweighted_binder,
)
from particlesim.solvers.lattice.euclidean import HybridMonteCarlo, Phi4, binder_cumulant


def test_the_cluster_flip_samples_the_ising_weights_of_the_signs():
    """With the magnitudes fixed, the signs are an Ising model, and the flip samples it.

    On a ``2 x 3`` torus there are 64 sign patterns, each with weight
    ``exp(sum_x sum_mu phi_x phi_(x+mu))``. The chain's histogram is held to
    those weights by a chi-square over 63 degrees of freedom, measured at
    60.6, 64.2 and 59.7 for three seeds. Every third flip is kept: successive
    flips are correlated, and unthinned the same test reads up to 118. A
    wrong bond probability, ``1 - exp(-J)`` in place of ``1 - exp(-2J)``,
    gives 85 000.
    """
    rng = np.random.default_rng(7)
    magnitudes = rng.uniform(0.3, 0.9, size=(2, 3))
    patterns = np.array(list(itertools.product((-1.0, 1.0), repeat=6)))

    def coupling(signs):
        field = magnitudes * signs.reshape(2, 3)
        return sum(np.sum(field * np.roll(field, -1, axis)) for axis in (0, 1))

    weights = np.exp([coupling(p) for p in patterns])
    expected = weights / weights.sum()
    code = {tuple(p): i for i, p in enumerate(patterns)}
    counts = np.zeros(len(patterns))
    state = magnitudes.copy()
    samples, thinning = 13000, 3
    for step in range(samples * thinning):
        state = embedded_ising_flip(state, rng)
        if step % thinning == 0:
            counts[code[tuple(np.sign(state).ravel())]] += 1
    assert np.allclose(np.abs(state), magnitudes)
    chi_square = np.sum((counts - samples * expected) ** 2 / (samples * expected))
    # The 99.9th percentile of chi-square with 63 degrees of freedom is 103.
    assert chi_square < 103, chi_square


def test_with_the_clusters_the_chain_samples_the_same_theory_and_tunnels():
    """Against plain hybrid Monte Carlo near the transition, at ``lambda = 1`` and ``L = 8``."""
    mass, quartic = -1.30, 1.0
    rng = np.random.default_rng(11)
    clustered = ClusterHybrid(8, mass, quartic, rng=rng)
    magnetisation, square_sum, acceptance, _ = clustered.run(6000, 200)
    plain = HybridMonteCarlo(
        Phi4((8, 8), mass, 6.0 * quartic), step=0.15, steps=10, rng=np.random.default_rng(12)
    )
    chain = plain.run(8000, 500, observable=lambda s: float(np.mean(s**2)))
    assert acceptance > 0.7
    assert abs(square_sum.mean() / 64 - chain.mean) < 4 * max(chain.error, 0.005)
    flips = np.count_nonzero(np.sign(magnetisation[1:]) != np.sign(magnetisation[:-1]))
    assert flips > 1000
    assert 0.4 < binder_cumulant(magnetisation) < 0.66


def test_the_tadpole_is_the_lattice_sum_and_its_logarithm():
    """The closed form against the sum over a ``1024 x 1024`` lattice, and the ``mu -> 0`` limit."""
    momenta = 2 * np.pi * np.arange(1024) / 1024
    squared = 4 * np.sin(momenta / 2) ** 2
    for mass in (1.0, 0.1, 0.01):
        summed = np.mean(1.0 / (squared[:, None] + squared[None, :] + mass))
        assert abs(lattice_tadpole(mass) - summed) < 1e-12
    assert abs(lattice_tadpole(1e-6) - np.log(32e6) / (4 * np.pi)) < 1e-5


def test_the_renormalised_mass_undoes_the_counterterm():
    for quartic, mass in ((1.0, 0.09), (0.25, 0.023), (0.0625, 0.0057)):
        bare = mass - 3 * quartic * lattice_tadpole(mass)
        assert abs(renormalised_mass(bare, quartic) - mass) < 1e-10
        assert abs(critical_ratio(bare, quartic) - quartic / mass) < 1e-6
    # The amplification: at lambda = 0.25 a shift of 1e-3 in the bare mass moves f by 1.3%.
    near = critical_ratio(-0.405, 0.25)
    assert 0.011 < abs(critical_ratio(-0.406, 0.25) - near) / near < 0.015


def test_reweighting_is_exact_at_the_run_mass_and_finds_the_crossing():
    rng = np.random.default_rng(3)
    magnetisation = rng.normal(size=4000)
    square_sum = rng.normal(100.0, 5.0, size=4000)
    assert (
        abs(
            reweighted_binder(magnetisation, square_sum, -0.4, -0.4)
            - binder_cumulant(magnetisation)
        )
        < 1e-12
    )
    # A two-state signal reads 2/3, a Gaussian 0; mixing them by m^2 gives a crossing.
    sign = np.where(rng.random(4000) < 0.5, -1.0, 1.0)
    ordered = rng.random(4000) < 0.5
    signal = np.where(ordered, sign, rng.normal(size=4000))
    energy = np.where(ordered, 110.0, 90.0) + rng.normal(0, 1.0, size=4000)
    crossing, error = jackknife_crossing(signal, energy, 0.0, 0.5)
    assert abs(reweighted_binder(signal, energy, 0.0, crossing) - ISING_BINDER) < 1e-9
    assert 0.0 < error < 0.05


def test_the_continuum_fit_recovers_a_lambda_log_lambda_term():
    couplings = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    truth = np.array([11.06, -1.2, 0.9])
    ratios = truth[0] + truth[1] * couplings + truth[2] * couplings * np.log(couplings)
    coefficients, errors = continuum_fit(couplings, ratios, np.full(5, 0.01))
    assert np.allclose(coefficients, truth, atol=1e-10)
    assert errors[0] < 0.05
