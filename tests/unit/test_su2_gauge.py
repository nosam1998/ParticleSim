"""``SU(2)`` lattice gauge theory, held to what cannot pass by accident.

Issue #64 asks that known ``SU(2)`` plaquette values be reproduced. In two
dimensions those values are not tabulated numbers but an exact character sum,
so the comparison is against a closed form at the run's own volume. The rest
of the file is structural: gauge invariance, unitarity, and a force checked
by differentiating *along the group* rather than along a coordinate. Those
are the checks a wrong staple fails immediately, while a plaquette comparison
might not.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.solvers.lattice.euclidean import CompactU1, integrated_autocorrelation
from particlesim.solvers.lattice.gauge import (
    GaugeHybridMonteCarlo,
    SU2Gauge,
    dagger,
    su2_exponential,
    unitarity_residual,
)


def links_and_engine(size=4, beta=2.0, seed=3, step=0.1, steps=12):
    model = SU2Gauge(size=size, beta=beta)
    links = model.random_links(np.random.default_rng(seed))
    engine = GaugeHybridMonteCarlo(model, step=step, steps=steps, rng=np.random.default_rng(seed))
    return model, links, engine


# --- the group -------------------------------------------------------------


def test_the_exponential_lands_in_the_group_exactly():
    """``U^dagger U = 1`` and ``det U = 1`` to round-off, including at zero.

    Closed form rather than a series, so a trajectory never needs
    reprojecting. That matters beyond tidiness: a reprojection inside the
    leapfrog would break reversibility, and the Metropolis test would then be
    sampling the wrong distribution while still looking healthy.
    """
    rng = np.random.default_rng(1)
    assert unitarity_residual(su2_exponential(rng.normal(size=(6, 6, 3)))) < 1e-14
    assert unitarity_residual(su2_exponential(np.zeros((4, 3)))) < 1e-15
    identity = su2_exponential(np.zeros((1, 3)))[0]
    assert np.allclose(identity, np.eye(2))


def test_the_exponential_composes_along_one_direction():
    """``exp(a p) exp(b p) = exp((a+b) p)``, which a wrong half-angle would fail."""
    rng = np.random.default_rng(2)
    coefficients = rng.normal(size=(5, 3))
    composed = su2_exponential(coefficients, 0.3) @ su2_exponential(coefficients, 0.4)
    assert np.max(np.abs(composed - su2_exponential(coefficients, 0.7))) < 1e-13


def test_unitarity_residual_notices_a_matrix_that_is_not_in_the_group():
    assert unitarity_residual(np.eye(2, dtype=complex) * 1.5) > 1.0


# --- gauge invariance ------------------------------------------------------


def test_the_action_does_not_move_under_a_gauge_transformation():
    """``U_mu(x) -> g(x) U_mu(x) g(x+mu)^dagger`` leaves the action alone.

    This is a statement about the indices, so a staple assembled with a
    factor daggered the wrong way fails it outright -- while still producing
    a real action, a plausible acceptance rate and a wrong answer. It is the
    first check for that reason.
    """
    model, links, _ = links_and_engine()
    rng = np.random.default_rng(7)
    for _ in range(3):
        transformation = su2_exponential(rng.normal(size=(model.size, model.size, 3)))
        transformed = model.gauge_transform(links, transformation)
        assert unitarity_residual(transformed) < 1e-13
        assert abs(model.action(transformed) - model.action(links)) < 1e-12


def test_every_plaquette_trace_is_invariant_site_by_site():
    """Stronger than the action: ``U_p -> g(x) U_p g(x)^dagger``, so its trace is fixed.

    A staple error that happened to cancel in the total would still move the
    individual traces, so this closes that gap.
    """
    model, links, _ = links_and_engine()
    transformation = su2_exponential(np.random.default_rng(8).normal(size=(4, 4, 3)))
    before = np.trace(model.plaquettes(links), axis1=-2, axis2=-1)
    after = np.trace(
        model.plaquettes(model.gauge_transform(links, transformation)), axis1=-2, axis2=-1
    )
    assert np.max(np.abs(after - before)) < 1e-12


def test_the_staples_account_for_exactly_two_plaquettes_each():
    """``sum_x Re Tr[U_mu A_mu] = 2 sum_x Re Tr U_p`` in both directions.

    Each plaquette contains two links of each direction, so the identity is a
    counting statement -- and it pins the staples independently of the force
    that uses them.
    """
    model, links, _ = links_and_engine()
    staples = model.staples(links)
    total = np.trace(model.plaquettes(links), axis1=-2, axis2=-1).real.sum()
    for direction in (0, 1):
        contracted = np.trace(links[direction] @ staples[direction], axis1=-2, axis2=-1).real.sum()
        assert contracted == pytest.approx(2.0 * total, rel=1e-12)


# --- the integrator --------------------------------------------------------


def test_the_force_is_the_derivative_along_the_group():
    """Differentiated along ``U -> exp(i omega T_a) U``, not along a coordinate.

    That is what makes this a test of the staple assembly *and* the generator
    convention together. A wrong factor of a half in ``T_a = sigma_a/2`` would
    pass every unitarity check and fail here.
    """
    _, links, engine = links_and_engine(size=3)
    assert engine.force_residual(links) < 1e-8


def test_the_trajectory_is_reversible_and_stays_in_the_group():
    """Reversible to ``1e-13``, and unitary afterwards without reprojection."""
    _, links, engine = links_and_engine()
    assert engine.reversibility_residual(links) < 1e-13
    evolved, _ = engine.leapfrog(links, engine.draw_momenta())
    assert unitarity_residual(evolved) < 1e-13


def test_the_exchange_average_is_one():
    """``<exp(-dH)> = 1`` by detailed balance, on the group as on a flat space."""
    model = SU2Gauge(size=3, beta=2.0)
    engine = GaugeHybridMonteCarlo(model, step=0.15, steps=8, rng=np.random.default_rng(31))
    chain = engine.run(sweeps=2000, thermalise=200)
    assert chain.exchange_average == pytest.approx(1.0, abs=0.02)
    assert chain.acceptance > 0.8


# --- the plaquette ---------------------------------------------------------


def test_the_cold_start_has_unit_plaquettes_and_no_action():
    model = SU2Gauge(size=4, beta=2.0)
    links = model.cold_start()
    assert model.action(links) == pytest.approx(0.0, abs=1e-12)
    assert model.mean_plaquette(links) == pytest.approx(1.0)
    assert unitarity_residual(links) < 1e-15


def test_a_run_reproduces_the_exact_plaquette():
    """Against the character sum at the run's own volume, at two couplings."""
    for size, beta, seed in ((2, 1.0, 41), (3, 2.0, 42)):
        model = SU2Gauge(size=size, beta=beta)
        engine = GaugeHybridMonteCarlo(model, step=0.15, steps=8, rng=np.random.default_rng(seed))
        chain = engine.run(sweeps=4000, thermalise=400)
        tau = integrated_autocorrelation(chain.values)
        error = float(np.std(chain.values) / np.sqrt(chain.values.size) * np.sqrt(2.0 * tau))
        assert abs(chain.mean - model.exact_plaquette()) < 3.0 * error


def test_the_character_sum_converges_to_the_bessel_ratio():
    infinite = SU2Gauge.infinite_volume_plaquette(2.0)
    deviations = [
        abs(SU2Gauge(size=size, beta=2.0).exact_plaquette() / infinite - 1.0) for size in (2, 3, 4)
    ]
    for coarse, fine in zip(deviations, deviations[1:], strict=False):
        assert fine < coarse
    assert deviations[-1] < 1e-8


def test_strong_and_weak_coupling_limits_are_the_expected_ones():
    """``I_2/I_1`` runs from zero at weak ``beta`` to one at strong.

    A plaquette above one or below zero would be unphysical, and the two ends
    fix the normalisation of ``(1/2) Tr U_p`` that everything else rests on.
    """
    assert SU2Gauge.infinite_volume_plaquette(0.01) < 0.01
    assert SU2Gauge.infinite_volume_plaquette(100.0) == pytest.approx(1.0, abs=0.02)
    values = [SU2Gauge.infinite_volume_plaquette(b) for b in (0.5, 1.0, 2.0, 4.0, 8.0)]
    for low, high in zip(values, values[1:], strict=False):
        assert high > low


def test_the_non_abelian_finite_volume_correction_is_far_smaller():
    """``SU(2)`` at ``V = 4`` is 0.08% from its limit where ``U(1)`` is 13%.

    Same character expansion, same factorisation, and a correction differing
    by more than two orders of magnitude -- because the first subleading
    representation is suppressed by 0.12 here against 0.45 there, and that
    ratio enters as its ``V``-th power. It is why
    :mod:`particlesim.solvers.lattice.euclidean` has to insist on the
    finite-volume form and this module does not.
    """
    beta = 1.0
    non_abelian = SU2Gauge(size=2, beta=beta)
    abelian = CompactU1(size=2, beta=beta)
    assert non_abelian.subleading_suppression() == pytest.approx(0.1201, abs=1e-3)

    non_abelian_shift = abs(
        non_abelian.exact_plaquette() / SU2Gauge.infinite_volume_plaquette(beta) - 1.0
    )
    abelian_shift = abs(abelian.exact_plaquette() / CompactU1.infinite_volume_plaquette(beta) - 1.0)
    assert non_abelian_shift == pytest.approx(8.0e-4, rel=0.05)
    assert abelian_shift == pytest.approx(1.32e-1, rel=0.05)
    assert abelian_shift / non_abelian_shift > 100.0


# --- what the model refuses ------------------------------------------------


def test_degenerate_lattices_and_couplings_are_refused():
    with pytest.raises(ValueError, match="at least two sites"):
        SU2Gauge(size=1)
    with pytest.raises(ValueError, match="must be positive"):
        SU2Gauge(beta=0.0)


def test_a_trajectory_needs_a_step_and_a_count():
    with pytest.raises(ValueError, match="positive step"):
        GaugeHybridMonteCarlo(SU2Gauge(), step=0.0)
    with pytest.raises(ValueError, match="positive step"):
        GaugeHybridMonteCarlo(SU2Gauge(), steps=0)


def test_a_run_that_measures_nothing_is_refused():
    with pytest.raises(ValueError, match="leaves nothing to measure"):
        GaugeHybridMonteCarlo(SU2Gauge()).run(sweeps=50, thermalise=50)


def test_the_dagger_helper_is_the_conjugate_transpose():
    rng = np.random.default_rng(99)
    matrices = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))
    assert np.allclose(dagger(matrices), np.conj(np.transpose(matrices, (0, 2, 1))))
