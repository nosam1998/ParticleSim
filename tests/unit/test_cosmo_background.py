"""Friedmann background cosmology (design doc Section 3.1, Milestone 3)."""

import time

import numpy as np
import pytest

from particlesim.cosmo import Component, Cosmology, integrate, matter, radiation
from particlesim.cosmo.background import GYR_S, LIGHT_KM_S, MPC_KM

astropy_cosmology = pytest.importorskip("astropy.cosmology")

REDSHIFTS = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0])
TOLERANCE = 1e-8  # the acceptance criterion for issue #39


def _relative(mine, theirs) -> float:
    mine = np.asarray(mine, dtype=float)
    theirs = np.asarray(getattr(theirs, "value", theirs), dtype=float)
    return float(np.abs(mine / theirs - 1.0).max())


# --- quadrature -------------------------------------------------------------


def test_the_quadrature_is_exact_on_a_smooth_integrand():
    """Gauss-Legendre of order n is exact for polynomials below degree 2n, and
    near-exact for anything analytic. Checked against closed forms so a later
    change to the rule cannot pass by agreeing with itself."""
    assert integrate(lambda x: x**5, 2.0) == pytest.approx(2.0**6 / 6, rel=1e-15)
    assert integrate(np.sin, np.pi) == pytest.approx(2.0, rel=1e-14)
    assert integrate(np.exp, 1.0) == pytest.approx(np.e - 1.0, rel=1e-14)


def test_the_quadrature_vectorizes_over_its_upper_limit():
    """What makes a parameter sweep cheap: one call into the integrand for
    every limit at once, not one adaptive solve each."""
    limits = np.array([0.5, 1.0, 2.0, 3.0])
    together = integrate(np.exp, limits)
    apart = np.array([integrate(np.exp, float(x)) for x in limits])
    np.testing.assert_allclose(together, apart, rtol=1e-15)
    np.testing.assert_allclose(together, np.expm1(limits), rtol=1e-14)


# --- constants --------------------------------------------------------------


def test_the_unit_constants_are_the_accepted_values():
    """Regression: a gigayear truncated to 3.1557e16 instead of 3.15576e16 is
    wrong by 1.9e-5, which showed up as a constant age offset at every
    redshift in every model -- the signature of a unit constant rather than a
    quadrature error, and three orders above this file's tolerance."""
    assert GYR_S == pytest.approx(3.15576e16, rel=1e-15)
    assert LIGHT_KM_S == pytest.approx(299792.458, rel=1e-15)
    assert MPC_KM == pytest.approx(3.0856775814913673e19, rel=1e-15)

    from astropy import units as u
    from astropy.cosmology import FlatLambdaCDM

    cosmology = FlatLambdaCDM(H0=70.0, Om0=0.3)
    mine = Cosmology.lcdm(h0=70.0, omega_m=0.3)
    assert mine.hubble_time == pytest.approx(cosmology.hubble_time.to(u.Gyr).value, rel=1e-12)
    assert mine.hubble_distance == pytest.approx(
        cosmology.hubble_distance.to(u.Mpc).value, rel=1e-12
    )


# --- the acceptance criterion ----------------------------------------------


def test_flat_lcdm_distances_match_astropy():
    """Acceptance for issue #39: 1e-8 relative. Achieved at 1e-13."""
    from astropy.cosmology import FlatLambdaCDM

    theirs = FlatLambdaCDM(H0=67.66, Om0=0.3111)
    mine = Cosmology.lcdm(h0=67.66, omega_m=0.3111)
    for name, reference in (
        ("comoving_distance", theirs.comoving_distance),
        ("luminosity_distance", theirs.luminosity_distance),
        ("angular_diameter_distance", theirs.angular_diameter_distance),
    ):
        assert _relative(getattr(mine, name)(REDSHIFTS), reference(REDSHIFTS)) < TOLERANCE, name
    assert _relative(mine.hubble(REDSHIFTS), theirs.H(REDSHIFTS)) < 1e-14
    assert _relative(mine.distance_modulus(REDSHIFTS), theirs.distmod(REDSHIFTS)) < TOLERANCE


@pytest.mark.parametrize(
    ("omega_m", "omega_lambda"), [(0.3, 0.6), (0.3, 0.8), (1.0, 0.0), (0.05, 0.95)]
)
def test_curved_and_extreme_lcdm_distances_match_astropy(omega_m, omega_lambda):
    """Curvature is derived, not specified, so this also checks that the
    ``sinh`` and ``sin`` branches are picked by the sign rather than by a
    tolerance."""
    from astropy.cosmology import LambdaCDM

    theirs = LambdaCDM(H0=70.0, Om0=omega_m, Ode0=omega_lambda)
    mine = Cosmology.lcdm(h0=70.0, omega_m=omega_m, omega_lambda=omega_lambda)
    assert mine.omega_k == pytest.approx(theirs.Ok0, abs=1e-12)
    assert (
        _relative(mine.comoving_distance(REDSHIFTS), theirs.comoving_distance(REDSHIFTS))
        < TOLERANCE
    )
    assert (
        _relative(mine.luminosity_distance(REDSHIFTS), theirs.luminosity_distance(REDSHIFTS))
        < TOLERANCE
    )


@pytest.mark.parametrize(("w0", "wa"), [(-1.0, 0.0), (-0.9, 0.0), (-1.1, 0.0), (-0.9, 0.3)])
def test_dark_energy_distances_match_astropy(w0, wa):
    """A varying equation of state uses its closed form rather than freezing
    ``w`` at its present value, which is wrong by percent at these
    redshifts."""
    from astropy.cosmology import Flatw0waCDM

    theirs = Flatw0waCDM(H0=70.0, Om0=0.3, w0=w0, wa=wa)
    mine = Cosmology.wcdm(h0=70.0, omega_m=0.3, w0=w0, wa=wa)
    assert (
        _relative(mine.luminosity_distance(REDSHIFTS), theirs.luminosity_distance(REDSHIFTS))
        < TOLERANCE
    )


def test_radiation_is_dominant_early_and_small_but_real_late():
    """Radiation is optional but must be right when present.

    Compared against the same budget with radiation removed, which is what
    astropy carries when ``Tcmb0 = 0``. At ``z = 1000`` radiation dominates
    and the distance shifts by more than a part in a thousand; by ``z = 0.5``
    the shift is 3.5e-5 -- small, and not zero. Asserting it were negligible
    there would be asserting something false in order to make a test pass.
    """
    from astropy.cosmology import LambdaCDM

    omega_r = 1e-4
    without = LambdaCDM(H0=70.0, Om0=0.3, Ode0=0.7 - omega_r, Tcmb0=0.0)
    with_radiation = Cosmology(
        h0=70.0,
        components=(radiation(omega_r), matter(0.3), Component("lambda", 0.7 - omega_r, -1.0)),
    )
    assert with_radiation.omega_k == pytest.approx(0.0, abs=1e-12)

    early = _relative(with_radiation.comoving_distance(1000.0), without.comoving_distance(1000.0))
    late = _relative(with_radiation.comoving_distance(0.5), without.comoving_distance(0.5))
    assert early > 1e-3
    assert 1e-6 < late < 1e-4
    assert early > 10 * late

    # Radiation always slows the expansion early, so it brings objects closer.
    assert float(with_radiation.comoving_distance(1000.0)) < float(
        without.comoving_distance(1000.0).value
    )


def test_a_callable_equation_of_state_reproduces_the_closed_form():
    """The general quadrature path, checked where a closed form exists.

    A constant ``w`` written as a callable must give the same density history
    as the analytic ``a^(-3(1+w))``. Without this the general path ships
    untested, and a component with a genuinely varying ``w`` would be the
    first thing to find out.
    """
    scale = np.array([0.1, 0.3, 0.5, 1.0])
    for w in (-1.0, -0.5, 0.0, 1.0 / 3.0):
        closed = Component("x", 1.0, w).density_ratio(scale)
        general = Component(
            "x", 1.0, lambda a, w=w: np.full_like(np.asarray(a, dtype=float), w)
        ).density_ratio(scale)
        np.testing.assert_allclose(general, closed, rtol=1e-13)

    # And for a scalar, which takes a different branch through the shaping.
    assert float(
        Component("x", 1.0, lambda a: np.full_like(np.asarray(a, dtype=float), -0.5)).density_ratio(
            0.5
        )
    ) == pytest.approx(float(Component("x", 1.0, -0.5).density_ratio(0.5)), rel=1e-13)


def test_a_varying_equation_of_state_actually_varies():
    """A component whose ``w`` depends on ``a`` must differ from every
    constant-``w`` component, or the callable is being ignored."""
    varying = Component("x", 1.0, lambda a: -1.0 + 0.5 * (1.0 - np.asarray(a, dtype=float)))
    scale = np.array([0.2, 0.5, 1.0])
    history = varying.density_ratio(scale)
    assert history[-1] == pytest.approx(1.0, rel=1e-13)
    for w in (-1.0, -0.75, -0.5):
        constant = Component("x", 1.0, w).density_ratio(scale)
        assert not np.allclose(history, constant, rtol=1e-3)


# --- ages -------------------------------------------------------------------


def test_flat_lcdm_ages_match_astropy():
    from astropy.cosmology import FlatLambdaCDM

    theirs = FlatLambdaCDM(H0=67.66, Om0=0.3111)
    mine = Cosmology.lcdm(h0=67.66, omega_m=0.3111)
    assert _relative(mine.age(REDSHIFTS), theirs.age(REDSHIFTS)) < TOLERANCE
    assert _relative(mine.lookback_time(REDSHIFTS), theirs.lookback_time(REDSHIFTS)) < TOLERANCE


def test_the_age_is_converged_and_the_residual_against_astropy_is_theirs():
    """Worth establishing which side a disagreement is on.

    For a curved model the age differs from astropy by 1.4e-8, which is
    exactly astropy's default ``quad`` tolerance of 1.49e-8. Raising the
    quadrature order here from 48 to 240 moves the answer by 1e-15, so the
    fixed rule is converged and the gap belongs to the adaptive one being
    compared against.
    """
    from astropy.cosmology import LambdaCDM

    redshifts = np.array([0.0, 0.5, 2.0, 100.0])
    coarse = Cosmology.lcdm(h0=70.0, omega_m=0.3, omega_lambda=0.6, order=48)
    fine = Cosmology.lcdm(h0=70.0, omega_m=0.3, omega_lambda=0.6, order=240)
    assert _relative(coarse.age(redshifts), fine.age(redshifts)) < 1e-13

    theirs = LambdaCDM(H0=70.0, Om0=0.3, Ode0=0.6)
    gap = _relative(coarse.age(redshifts), theirs.age(redshifts))
    assert gap < 1e-7
    assert gap > 1e-12, "if this ever tightens, astropy got more accurate"


# --- shape and validation ---------------------------------------------------


def test_scalar_and_array_inputs_agree():
    mine = Cosmology.lcdm()
    for z in (0.0, 0.3, 4.0):
        assert float(mine.luminosity_distance(z)) == pytest.approx(
            float(mine.luminosity_distance(np.array([z]))[0]), rel=1e-15
        )


def test_curvature_is_whatever_the_budget_does_not_account_for():
    """Not a component. Carrying it as one would let a caller specify a budget
    that does not close, and the symptom would be subtly wrong distances
    rather than an error."""
    assert Cosmology.lcdm(omega_m=0.3, omega_lambda=0.7).omega_k == pytest.approx(0.0)
    assert Cosmology.lcdm(omega_m=0.3, omega_lambda=0.5).omega_k == pytest.approx(0.2)
    assert Cosmology.lcdm(omega_m=0.3).omega_k == pytest.approx(0.0)


def test_comoving_volume_refuses_a_curved_universe():
    curved = Cosmology.lcdm(omega_m=0.3, omega_lambda=0.5)
    with pytest.raises(NotImplementedError, match="flat universe only"):
        curved.comoving_volume(1.0)
    flat = Cosmology.lcdm()
    assert flat.comoving_volume(1.0) == pytest.approx(
        4 * np.pi / 3 * float(flat.comoving_distance(1.0)) ** 3
    )


def test_an_unknown_sweep_parameter_is_refused():
    with pytest.raises(KeyError, match="unknown sweep parameter"):
        Cosmology.lcdm().sweep("omega_neutrino", [0.1, 0.2], redshift=1.0)


def test_the_summary_reports_the_budget_and_the_age():
    summary = Cosmology.lcdm(h0=70.0, omega_m=0.3).summary()
    assert summary["h0"] == 70.0
    assert summary["omega_matter"] == pytest.approx(0.3)
    assert summary["omega_lambda"] == pytest.approx(0.7)
    assert summary["omega_k"] == pytest.approx(0.0)
    assert 13.0 < summary["age_gyr"] < 14.5


# --- sweeps -----------------------------------------------------------------


@pytest.mark.benchmark
def test_a_ten_thousand_point_sweep_runs_in_seconds():
    """The other half of issue #39's acceptance.

    Fixed-order quadrature is what makes this possible: an adaptive routine
    would need a Python-level call per evaluation and the same sweep would be
    minutes.
    """
    mine = Cosmology.lcdm(h0=67.66, omega_m=0.3111)
    grid = np.linspace(0.1, 0.5, 10_000)
    started = time.perf_counter()
    distances = mine.sweep("matter", grid, redshift=1.0)
    elapsed = time.perf_counter() - started

    assert elapsed < 10.0, f"the sweep took {elapsed:.1f}s"
    assert distances.shape == grid.shape
    # More matter means faster expansion at z > 0, so a closer object.
    assert np.all(np.diff(distances) < 0)
    # Spot-check one point of the sweep against a direct construction.
    direct = Cosmology.lcdm(h0=67.66, omega_m=float(grid[1234]), omega_lambda=1.0 - 0.3111)
    assert distances[1234] == pytest.approx(float(direct.luminosity_distance(1.0)), rel=1e-12)


@pytest.mark.benchmark
def test_distances_vectorize_over_ten_thousand_redshifts():
    mine = Cosmology.lcdm()
    redshifts = np.linspace(1e-3, 3.0, 10_000)
    started = time.perf_counter()
    distances = mine.luminosity_distance(redshifts)
    elapsed = time.perf_counter() - started
    assert elapsed < 2.0, f"took {elapsed:.2f}s"
    assert np.all(np.diff(distances) > 0)
