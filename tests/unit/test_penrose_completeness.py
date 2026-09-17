"""Causal structure: Penrose compactification and geodesic completeness."""

import numpy as np
import pytest
import sympy as sp

from particlesim.analysis import penrose
from particlesim.analysis.completeness import (
    is_numerically_stalled,
    probe_bundle,
    radial_infall_bundle,
)
from particlesim.analysis.geodesics import GeodesicIntegrator

T_, R_, TH_, PH_ = sp.symbols("t r theta phi", positive=True)
COORDS = [T_, R_, TH_, PH_]


def minkowski_integrator() -> GeodesicIntegrator:
    metric = sp.diag(-1, 1, R_**2, R_**2 * sp.sin(TH_) ** 2)
    return GeodesicIntegrator(metric, COORDS)


def schwarzschild_integrator(mass: float = 1.0) -> GeodesicIntegrator:
    f = 1 - 2 * sp.Symbol("M", positive=True) / R_
    metric = sp.diag(-f, 1 / f, R_**2, R_**2 * sp.sin(TH_) ** 2)
    return GeodesicIntegrator(metric, COORDS, {sp.Symbol("M", positive=True): mass})


# --- Penrose compactification ---------------------------------------------


def test_minkowski_null_rays_stay_at_forty_five_degrees():
    """The property the whole construction exists to preserve."""
    lam = np.linspace(0.1, 50.0, 40)
    # An outgoing radial null ray: t = r + const.
    t, r = lam + 3.0, lam
    big_t, x = penrose.minkowski(t, r)
    slope = np.diff(big_t) / np.diff(x)
    np.testing.assert_allclose(slope, 1.0, rtol=1e-9)


def test_minkowski_compactification_is_bounded():
    r = np.logspace(-3, 8, 60)
    big_t, x = penrose.minkowski(np.zeros_like(r), r)
    assert np.all(np.abs(big_t) < np.pi) and np.all(np.abs(x) < np.pi)
    # Spatial infinity is approached but never reached.
    assert x.max() < np.pi and x.max() > 3.0


def test_schwarzschild_horizon_maps_to_forty_five_degree_lines():
    """The horizon is U V = 0, so T = +/- X exactly, not approximately."""
    t = np.linspace(-30.0, 30.0, 25)
    big_t, x = penrose.schwarzschild(t, np.full_like(t, 2.0), 1.0)
    np.testing.assert_allclose(np.abs(big_t), np.abs(x), atol=1e-14)


def test_schwarzschild_singularity_is_a_horizontal_line():
    """On r = 0 the Kruskal product is one, so arctan(U) + arctan(1/U) = pi/2
    identically. A horizontal line means a spacelike singularity: unavoidable
    once inside, which is the physical content of the picture."""
    t = np.linspace(-20.0, 20.0, 17)
    big_t, _ = penrose.schwarzschild(t, np.full_like(t, 1e-12), 1.0)
    np.testing.assert_allclose(big_t, penrose.schwarzschild_singularity_conformal_time(), atol=1e-9)
    assert penrose.singularity_character(1.0, 0.0) == "spacelike"


def test_tortoise_pushes_the_horizon_to_minus_infinity():
    mass = 1.0
    near = penrose.tortoise(np.array([2.0 + 1e-9]), mass)[0]
    assert near < -30.0
    # Far away it approaches r.
    far = penrose.tortoise(np.array([1e6]), mass)[0]
    assert far == pytest.approx(1e6, rel=1e-4)


def test_kruskal_product_identifies_horizon_and_singularity():
    mass = 1.0
    r = np.array([0.5, 1.0, 2.0, 4.0, 10.0])
    u, v = penrose.kruskal(np.zeros_like(r), r, mass)
    product = u * v
    expected = (1.0 - r / (2 * mass)) * np.exp(r / (2 * mass))
    np.testing.assert_allclose(product, expected, rtol=1e-10)
    # Horizon is exactly zero, interior positive, exterior negative.
    assert product[2] == pytest.approx(0.0, abs=1e-12)
    assert product[0] > 0 and product[-1] < 0


@pytest.mark.parametrize(
    ("charge", "expected"),
    [(0.0, 1), (0.6, 2), (1.0, 1), (1.5, 0)],
)
def test_reissner_nordstrom_horizon_count(charge, expected):
    """Two horizons below extremality, one at it, none above, and one for
    uncharged Schwarzschild: the inner root sits at r = 0 there, which is the
    singularity rather than a surface."""
    roots = penrose.reissner_nordstrom_horizons(1.0, charge)
    assert len(roots) == expected
    for r in roots:
        assert penrose.reissner_nordstrom_lapse(np.array([r]), 1.0, charge)[0] == pytest.approx(
            0.0, abs=1e-12
        )


def test_charge_makes_the_singularity_timelike():
    """This is what lets the Reissner-Nordström diagram continue past r = 0,
    and it is read off the sign of the metric function rather than asserted."""
    assert penrose.singularity_character(1.0, 0.0) == "spacelike"
    assert penrose.singularity_character(1.0, 0.6) == "timelike"
    assert penrose.reissner_nordstrom_lapse(np.array([1e-6]), 1.0, 0.6)[0] > 0
    assert penrose.reissner_nordstrom_lapse(np.array([1e-6]), 1.0, 0.0)[0] < 0


def test_reissner_nordstrom_rejects_unphysical_mass():
    with pytest.raises(ValueError, match="mass must be positive"):
        penrose.reissner_nordstrom_horizons(0.0, 0.5)


# --- geodesic completeness ------------------------------------------------


def test_minkowski_is_geodesically_complete():
    geo = minkowski_integrator()
    starts, vels = radial_infall_bundle(geo, [5.0, 10.0, 20.0])
    report = probe_bundle(geo, starts, vels, budget=4.0)
    assert report.geodesically_complete
    assert report.complete_fraction == 1.0
    assert report.summary()["probes"] == 3


def test_probe_bundle_validates_its_input():
    geo = minkowski_integrator()
    with pytest.raises(ValueError, match="one initial velocity"):
        probe_bundle(geo, [np.zeros(4)], [])


@pytest.mark.slow
def test_schwarzschild_infall_is_incomplete():
    """A probe falling inward runs out of spacetime at finite affine
    parameter. That termination, not a large curvature invariant, is what
    'singular' means."""
    geo = schwarzschild_integrator()
    starts, vels = radial_infall_bundle(geo, [6.0], inward_speed=0.5)
    report = probe_bundle(geo, starts, vels, budget=200.0)
    assert not report.geodesically_complete
    assert report.complete_fraction == 0.0
    assert 0.0 < report.shortest_affine < 200.0


@pytest.mark.slow
def test_a_real_incompleteness_does_not_move_with_tolerance():
    """The check that separates a singularity from a solver giving up."""
    geo = schwarzschild_integrator()
    starts, vels = radial_infall_bundle(geo, [6.0], inward_speed=0.5)
    assert not is_numerically_stalled(geo, starts[0], vels[0], budget=200.0)


@pytest.mark.slow
def test_circular_orbit_probe_is_complete():
    """Not everything in Schwarzschild is incomplete, or the test above would
    be measuring the integrator rather than the spacetime."""
    geo = schwarzschild_integrator()
    r0 = 10.0
    ut = 1 / np.sqrt(1 - 3 / r0)
    x0 = np.array([0.0, r0, np.pi / 2, 0.0])
    u0 = np.array([ut, 0.0, 0.0, np.sqrt(1.0 / r0**3) * ut])
    report = probe_bundle(geo, [x0], [u0], budget=150.0)
    assert report.geodesically_complete
