import numpy as np
import pytest
import sympy as sp

from particlesim.analysis.geodesics import GeodesicIntegrator
from particlesim.scenarios.warp.metrics import COORDS, make_metric

t, r, th, ph = sp.symbols("t r theta phi", positive=True)
M = sp.Symbol("M", positive=True)


def schwarzschild_integrator():
    f = 1 - 2 * M / r
    g = sp.diag(-f, 1 / f, r**2, r**2 * sp.sin(th) ** 2)
    return GeodesicIntegrator(g, [t, r, th, ph], {M: 1.0})


def test_circular_orbit_stays_circular_and_has_keplerian_period():
    geo = schwarzschild_integrator()
    r0 = 10.0
    omega = np.sqrt(1.0 / r0**3)  # dφ/dt = sqrt(M/r³)
    ut = 1 / np.sqrt(1 - 3 / r0)
    x0 = [0.0, r0, np.pi / 2, 0.0]
    u0 = [ut, 0.0, 0.0, omega * ut]
    assert geo.norm(np.array(x0), np.array(u0)) == pytest.approx(-1.0, abs=1e-12)
    period_t = 2 * np.pi / omega
    res = geo.integrate(x0, u0, affine_max=2 * period_t / ut, n_out=400)
    assert res.success
    assert np.abs(res.x[:, 1] - r0).max() < 1e-7
    assert np.abs(res.norm + 1).max() < 1e-8
    # After one coordinate period, φ has advanced by 2π.
    phi_at_period = np.interp(period_t, res.x[:, 0], res.x[:, 3])
    assert phi_at_period == pytest.approx(2 * np.pi, abs=1e-6)


def test_photon_sphere_null_orbit():
    geo = schwarzschild_integrator()
    r0 = 3.0
    x0 = np.array([0.0, r0, np.pi / 2, 0.0])
    # Null: g_tt (u^t)² + r² (u^φ)² = 0 -> u^φ = u^t sqrt(f)/r
    ut = 1.0
    uph = ut * np.sqrt(1 - 2 / r0) / r0
    u0 = geo.normalize(x0, [0.0, 0.0, uph], kind="null")
    assert u0[0] == pytest.approx(ut, rel=1e-12)
    res = geo.integrate(x0, u0, affine_max=20.0, n_out=100)
    assert res.success
    # Unstable orbit: stays close over a short affine interval.
    assert np.abs(res.x[:, 1] - r0).max() < 1e-5
    assert np.abs(res.norm).max() < 1e-8


def test_eulerian_observer_inside_alcubierre_bubble_rides_it():
    """Unit lapse makes Eulerian observers geodesic; inside the bubble f = 1 so
    x - v t stays constant. The exact centre r_s = 0 is a coordinate
    singularity of the shape function, so start slightly off-centre."""
    m = make_metric("alcubierre", {"v_s": 2.0, "R": 5.0, "sigma": 2.0})
    geo = GeodesicIntegrator(m.metric(), COORDS, m.params)
    x0 = np.array([0.0, 0.5, 0.3, 0.2])
    # Coordinate time is spacelike inside a superluminal bubble, so spatial
    # components alone are ambiguous there.
    with pytest.raises(ValueError, match="not timelike"):
        geo.normalize(x0, [2.0, 0.0, 0.0])
    u0 = geo.from_eulerian_velocity(x0, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(u0, [1.0, 2.0, 0.0, 0.0], atol=1e-7)
    assert geo.norm(x0, u0) == pytest.approx(-1.0, abs=1e-12)
    res = geo.integrate(x0, u0, affine_max=5.0, n_out=50)
    assert res.success
    np.testing.assert_allclose(res.x[:, 1] - 2.0 * res.x[:, 0], 0.5, atol=1e-6)
    np.testing.assert_allclose(res.x[:, 2], 0.3, atol=1e-6)
    np.testing.assert_allclose(res.x[:, 3], 0.2, atol=1e-6)
    # And the singular centre fails fast instead of hanging.
    with pytest.raises(ValueError, match="non-finite"):
        geo.integrate(np.zeros(4), [1.0, 2.0, 0.0, 0.0], affine_max=1.0, n_out=3)


def test_eulerian_velocity_constructor():
    geo = schwarzschild_integrator()
    x0 = np.array([0.0, 10.0, np.pi / 2, 0.0])
    u = geo.from_eulerian_velocity(x0, [0.5, 0.0, 0.0])
    assert geo.norm(x0, u) == pytest.approx(-1.0, abs=1e-12)
    k = geo.from_eulerian_velocity(x0, [0.0, 0.0, 3.0], kind="null")
    assert geo.norm(x0, k) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="< 1"):
        geo.from_eulerian_velocity(x0, [1.0, 0.0, 0.0])
