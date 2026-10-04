"""Published checks for the explicitly bosonic IKKT truncation."""

import numpy as np
import pytest

from particlesim.solvers.matrix.ikkt import BosonicIKKT, ComplexLangevin


@pytest.mark.slow
@pytest.mark.benchmark
def test_undeformed_contour_phases_match_the_analytic_identity():
    """Equations (9)--(10): half-phases are -3 pi/8 and pi/8, at any N."""
    model = BosonicIKKT(size=8, gamma=0)
    sampler = ComplexLangevin(model)
    state, _ = sampler.advance(model.initialize(seed=17), 1.0)
    _, samples = sampler.sample(state, 100, interval=0.02, block_size=8)
    temporal_extent = np.mean(np.sum(samples.alpha**2, axis=1) / 8)
    spatial_extent = np.mean(samples.extent_squared)
    phases = np.angle([temporal_extent, spatial_extent]) / 2
    assert phases == pytest.approx([-3 * np.pi / 8, np.pi / 8], abs=0.03), phases


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.research
def test_n32_dimension_profile_matches_the_published_bosonic_curve():
    """One expanding direction, compared with the late-time fit in figure 3.

    The paper's coefficients have errors a=3.55(9), b=0.38(5), c=-5(1).
    A 15% profile tolerance is narrower than their propagated uncertainty
    over this late-time window. This compares the curve, not independently
    fitted coefficients, which are strongly correlated over a short window.
    """
    from examples.ikkt_benchmark import comparison, run_benchmark

    _, samples = run_benchmark()
    result = comparison(samples)
    assert len(result["late_times"]) >= 4, result
    assert max(result["relative_errors"]) < 0.15, result
    largest = np.asarray(result["largest_eigenvalue"])
    second = np.asarray(result["second_eigenvalue"])
    assert np.min(largest / second) > 8, result
    assert np.all(np.diff(largest) > 0), result
    # The Hermitian-part approximation is measured, never silently assumed.
    t = samples.physical_times()
    assert np.max(samples.nonhermiticity.mean(axis=0)[np.abs(t) >= 2]) < 0.15
