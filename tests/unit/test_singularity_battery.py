"""The singularity battery: dust collapse, FLRW, and mixmaster (design doc 3.5)."""

import numpy as np
import pytest

from particlesim.scenarios.singularity import (
    BianchiIX,
    FLRWBackground,
    OppenheimerSnyder,
    epoch_parameters,
    is_kasner_epoch,
    kasner_exponents,
    kasner_map,
    kasner_parameter,
    kasner_sequence,
    lqc_correction,
    normalized_exponents,
)
from particlesim.scenarios.singularity.flrw import (
    W_LAMBDA,
    W_MATTER,
    W_RADIATION,
    matter_dominated_scale_factor,
    radiation_dominated_scale_factor,
)

# --- Oppenheimer-Snyder ---------------------------------------------------


def test_collapse_time_matches_the_closed_form():
    os_ = OppenheimerSnyder(mass=1.0, r0=10.0)
    expected = 0.5 * np.pi * np.sqrt(10.0**3 / 2.0)
    assert os_.collapse_proper_time == pytest.approx(expected, rel=1e-14)
    # The surface reaches zero radius exactly at that proper time.
    assert os_.radius(np.pi) == pytest.approx(0.0, abs=1e-14)
    assert os_.proper_time(np.pi) == pytest.approx(expected, rel=1e-14)


def test_trajectory_is_the_cycloid():
    os_ = OppenheimerSnyder(mass=2.0, r0=12.0)
    tau, r = os_.trajectory(n=50)
    assert r[0] == pytest.approx(12.0) and r[-1] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(r) < 0)  # monotonic collapse
    assert np.all(np.diff(tau) > 0)
    # Parametric inversion round-trips.
    mid = os_.eta_at_proper_time(tau[25])
    assert os_.radius(mid) == pytest.approx(r[25], rel=1e-10)


def test_horizon_is_crossed_strictly_before_the_singularity():
    os_ = OppenheimerSnyder(mass=1.0, r0=10.0)
    assert os_.horizon_radius == 2.0
    t_h = os_.horizon_crossing_proper_time
    assert 0 < t_h < os_.collapse_proper_time
    assert os_.proper_time_to_radius(2.0) == pytest.approx(t_h)


def test_density_and_curvature_diverge_at_the_crunch():
    os_ = OppenheimerSnyder(mass=1.0, r0=10.0)
    eta = np.array([0.0, 0.5 * np.pi, 0.99 * np.pi])
    rho = os_.density(eta)
    assert np.all(np.diff(rho) > 0)
    # Uniform density at release is 3M / (4 pi r0^3).
    assert rho[0] == pytest.approx(3.0 / (4 * np.pi * 1000.0))
    k = os_.kretschmann_at_surface(eta)
    assert np.all(np.diff(k) > 0)
    # At the surface the exterior is vacuum, so 48 M^2 / R^6 applies exactly.
    assert k[0] == pytest.approx(48.0 / 10.0**6)


def test_dust_must_start_outside_its_own_horizon():
    with pytest.raises(ValueError, match="inside the horizon"):
        OppenheimerSnyder(mass=1.0, r0=1.5)
    with pytest.raises(ValueError, match="mass must be positive"):
        OppenheimerSnyder(mass=0.0, r0=10.0)


# --- FLRW -----------------------------------------------------------------


def test_flat_matter_universe_follows_the_two_thirds_power_law():
    bg = FLRWBackground(components={W_MATTER: 1.0})
    sol = bg.evolve(a0=0.05, t_max=8.0, expanding=True, n_out=200)
    # Fit a ~ (t - t_bb)^(2/3) by comparing to the closed form shifted to match.
    a, t = sol.a, sol.t
    # Since a ∝ (t - t_bb)^(2/3), the quantity a^(3/2) is linear in t.
    slope, intercept = np.polyfit(t, a**1.5, 1)
    residual = np.abs(a**1.5 - (slope * t + intercept)).max()
    assert residual < 1e-6 * (a**1.5).max()
    assert matter_dominated_scale_factor(np.array([8.0]))[0] == pytest.approx(8.0 ** (2 / 3))


def test_flat_radiation_universe_follows_the_half_power_law():
    bg = FLRWBackground(components={W_RADIATION: 1.0})
    sol = bg.evolve(a0=0.05, t_max=4.0, expanding=True, n_out=200)
    # a ∝ (t - t_bb)^(1/2), so a^2 is linear in t.
    slope, intercept = np.polyfit(sol.t, sol.a**2, 1)
    residual = np.abs(sol.a**2 - (slope * sol.t + intercept)).max()
    assert residual < 1e-6 * (sol.a**2).max()
    assert radiation_dominated_scale_factor(np.array([4.0]))[0] == pytest.approx(2.0)


def test_collapsing_matter_universe_reaches_a_singularity():
    bg = FLRWBackground(components={W_MATTER: 1.0})
    sol = bg.evolve(a0=1.0, t_max=50.0, expanding=False)
    assert sol.outcome == "singularity"
    assert not sol.bounced
    assert sol.min_scale_factor < 1e-7
    assert sol.density.max() > 1e10


def test_closed_universe_recollapses():
    """Positive curvature turns expansion around; that is the turning point."""
    closed = FLRWBackground(components={W_MATTER: 1.0}, curvature=1.0)
    sol = closed.evolve(a0=0.5, t_max=50.0, expanding=True)
    assert sol.outcome == "turning_point"
    # The turning point is where the density term balances curvature.
    a_max = sol.a.max()
    assert closed.hubble_squared(a_max) == pytest.approx(0.0, abs=1e-6)


def test_lambda_dominated_universe_expands_without_end():
    bg = FLRWBackground(components={W_LAMBDA: 1.0})
    sol = bg.evolve(a0=1.0, t_max=5.0, expanding=True)
    assert sol.outcome == "ran_to_t_max"
    # de Sitter: H is constant, so ln a is linear in t.
    slope, intercept = np.polyfit(sol.t, np.log(sol.a), 1)
    assert np.abs(np.log(sol.a) - (slope * sol.t + intercept)).max() < 1e-8
    assert slope == pytest.approx(np.sqrt(8 * np.pi / 3), rel=1e-8)


def test_lqc_correction_replaces_the_singularity_with_a_bounce():
    """The defining Tier B result: collapse halts exactly at the critical
    density rather than running to zero scale factor."""
    rho_c = 10.0
    lqc = FLRWBackground(components={W_MATTER: 1.0}, density_correction=lqc_correction(rho_c))
    sol = lqc.evolve(a0=1.0, t_max=50.0, expanding=False)
    assert sol.outcome == "turning_point"
    assert sol.bounced and sol.bounce_time > 0
    assert sol.min_scale_factor > 0.1
    assert sol.density.max() == pytest.approx(rho_c, rel=1e-9)

    # Without the correction the same data hits a singularity.
    plain = FLRWBackground(components={W_MATTER: 1.0})
    assert plain.evolve(a0=1.0, t_max=50.0, expanding=False).outcome == "singularity"


def test_lqc_correction_validates_its_input():
    with pytest.raises(ValueError, match="positive"):
        lqc_correction(0.0)


# --- Bianchi IX -----------------------------------------------------------


@pytest.mark.parametrize("u", [1.0, 1.3, 2.0, 5.0, 50.0])
def test_kasner_exponents_satisfy_their_defining_relations(u):
    p = kasner_exponents(u)
    assert sum(p) == pytest.approx(1.0, abs=1e-14)
    assert sum(x * x for x in p) == pytest.approx(1.0, abs=1e-14)
    assert sum(1 for x in p if x < 0) <= 1  # at most one negative exponent


def test_kasner_parameter_inverts_the_exponents():
    for u in (1.2, 2.5, 7.9):
        assert kasner_parameter(np.array(kasner_exponents(u))) == pytest.approx(u, rel=1e-9)


def test_kasner_map_branches():
    assert kasner_map(3.5) == pytest.approx(2.5)  # era continues
    assert kasner_map(1.5) == pytest.approx(2.0)  # era ends
    assert kasner_sequence(3.5, 4) == pytest.approx([3.5, 2.5, 1.5, 2.0])
    with pytest.raises(ValueError):
        kasner_map(1.0)
    with pytest.raises(ValueError):
        kasner_exponents(0.5)


def test_normalization_and_surface_test():
    p = np.array(kasner_exponents(3.0))
    np.testing.assert_allclose(normalized_exponents(p), p, rtol=1e-12)
    # Rescaling the exponents leaves the surface test and the parameter alone.
    np.testing.assert_allclose(normalized_exponents(0.37 * p), p, rtol=1e-12)
    assert is_kasner_epoch(0.37 * p)
    assert not is_kasner_epoch(np.array([0.5, 0.3, 0.2]))
    with pytest.raises(ValueError, match="sum to zero"):
        normalized_exponents(np.array([1.0, -1.0, 0.0]))


def test_kasner_initial_data_satisfies_the_constraint():
    """Setting the exponents alone is not admissible data: the curvature
    potential does not vanish unless the scale factors are small."""
    b = BianchiIX.from_kasner(2.5)
    assert abs(b.constraint(b.state)) < 1e-14
    naive = BianchiIX(0.0, 0.0, 0.0, *kasner_exponents(2.5))
    assert abs(naive.constraint(naive.state)) > 0.1


def test_constraint_is_conserved_while_its_terms_are_not():
    """The point of a first integral: the pieces move, the combination does
    not. A wrong relative factor would still vanish on a Kasner epoch."""
    b = BianchiIX.from_kasner(2.5)
    sol = b.evolve(-20.0, n_out=2000)
    residuals = np.array([abs(b.constraint(sol.y[:, i])) for i in range(sol.y.shape[1])])
    kinetic = np.array(
        [
            sol.y[3, i] * sol.y[4, i] + sol.y[4, i] * sol.y[5, i] + sol.y[5, i] * sol.y[3, i]
            for i in range(sol.y.shape[1])
        ]
    )
    swing = kinetic.max() - kinetic.min()
    assert residuals.max() < 1e-7
    # The individual terms move by orders of magnitude more than the
    # combination drifts. An absolute threshold on the swing would be the
    # wrong test: on a Kasner epoch the kinetic term sits near zero by
    # construction, and it is the ratio that says the integral is conserved
    # rather than merely small.
    assert swing > 1000 * residuals.max()


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.parametrize("u0", [2.5, 3.5])
def test_mixmaster_reproduces_the_bkl_map(u0):
    """The acceptance criterion: the epochs an integration actually visits
    must be the ones the discrete map predicts, including both of its
    branches."""
    b = BianchiIX.from_kasner(u0, scale=-2.0)
    sol = b.evolve(-80.0, n_out=16000)
    observed = epoch_parameters(sol.y[3:, :])
    assert len(observed) >= 3

    predicted = [u0]
    while len(predicted) < len(observed):
        predicted.append(kasner_map(predicted[-1]))

    for got, want in zip(observed[:3], predicted[:3], strict=True):
        assert got == pytest.approx(want, abs=0.06), f"{observed} vs {predicted}"
