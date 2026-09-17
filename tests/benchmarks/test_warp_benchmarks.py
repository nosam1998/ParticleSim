"""Warp benchmarks from the design doc, Section 10."""

import numpy as np
import pytest
import sympy as sp

from particlesim.analysis.geodesics import GeodesicIntegrator
from particlesim.core.config import WarpAnalyzeConfig
from particlesim.core.grid import UniformGrid
from particlesim.scenarios.warp.analyze import analyze
from particlesim.scenarios.warp.metrics import SPATIAL, make_metric
from particlesim.symbolic.adm import FlatSliceADM

T0 = sp.Symbol("t", real=True)


def _grid(n=16):
    return UniformGrid([(-12.0, 12.0)] * 3, (n, n, n))


@pytest.mark.benchmark
def test_alcubierre_energy_density_matches_closed_form():
    """ρ = -(v²/32π) (y²+z²)/r_s² f'(r_s)² for Eulerian observers (Alcubierre 1994)."""
    m = make_metric("alcubierre", {"v_s": 2.0, "R": 5.0, "sigma": 2.0})
    grid = _grid(24)
    X = grid.coords()
    adm = FlatSliceADM([b.subs(T0, 0) for b in m.shift()], SPATIAL)
    rho = adm.compile(m.params)(*X)["energy_density"]
    rho_cf = m.closed_form_energy_density()(np.zeros_like(X[0]), *X)
    assert np.abs(rho - rho_cf).max() < 1e-12
    assert rho.max() <= 0.0
    assert rho.min() < 0.0


@pytest.mark.benchmark
def test_natario_has_zero_expansion():
    """Natário 2002: the Eulerian congruence has θ = -K = 0 everywhere."""
    m = make_metric("natario", {})
    grid = _grid(24)
    adm = FlatSliceADM([b.subs(T0, 0) for b in m.shift()], SPATIAL)
    out = adm.compile(m.params)(*grid.coords())
    assert np.abs(out["expansion"]).max() < 1e-12
    # Natário's drive still violates the WEC.
    assert out["energy_density"].min() < 0.0
    assert np.isfinite(out["energy_density"]).all()


@pytest.mark.benchmark
def test_full_symbolic_path_agrees_with_fast_path_for_alcubierre():
    """The Einstein-tensor path and the Hamiltonian-constraint path must agree in GR."""
    cfg = WarpAnalyzeConfig()
    cfg.metric.family = "alcubierre"
    cfg.grid.extent = [(-12.0, 12.0)] * 3
    cfg.grid.resolution = [12, 12, 12]
    cfg.analysis.full_stress_energy = True
    res = analyze(cfg)
    assert res.report["fast_vs_full_max_abs_diff"] < 1e-12
    ecs = res.report["energy_conditions"]
    assert ecs["WEC"]["min"] < 0 and ecs["NEC"]["min"] < 0
    assert ecs["eulerian"]["negative_energy"] < 0


@pytest.mark.benchmark
@pytest.mark.slow
def test_van_den_broeck_reduces_to_alcubierre():
    """Van Den Broeck 1999 at alpha_B = 0 is the Alcubierre metric."""
    cfg = WarpAnalyzeConfig()
    cfg.metric.family = "van_den_broeck"
    cfg.metric.params = {"alpha_B": 0.0}
    cfg.grid.extent = [(-12.0, 12.0)] * 3
    cfg.grid.resolution = [10, 10, 10]
    res = analyze(cfg)
    m = make_metric("alcubierre", {})
    X = res.grid.coords()
    rho_cf = m.closed_form_energy_density()(np.zeros_like(X[0]), *X)
    assert np.abs(res.fields["energy_density"] - rho_cf).max() < 1e-12


@pytest.mark.benchmark
def test_full_path_uses_cache_and_reports_kretschmann(tmp_path, monkeypatch):
    """Second identical analysis hits the kernel cache; Kretschmann is finite and nonzero."""
    monkeypatch.setenv("PARTICLESIM_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("PARTICLESIM_NO_CACHE", raising=False)
    cfg = WarpAnalyzeConfig()
    cfg.metric.family = "alcubierre"
    cfg.grid.extent = [(-12.0, 12.0)] * 3
    cfg.grid.resolution = [8, 8, 8]
    cfg.analysis.invariants = ["kretschmann"]
    first = analyze(cfg)
    second = analyze(cfg)
    assert first.timings["full_path_cache_hit"] == 0.0
    assert second.timings["full_path_cache_hit"] == 1.0
    k = first.fields["kretschmann"]
    assert np.isfinite(k).all() and np.abs(k).max() > 0
    np.testing.assert_allclose(second.fields["kretschmann"], k)


@pytest.mark.benchmark
def test_alcubierre_ship_horizon_at_f_equals_one_minus_inverse_speed():
    """Hiscock 1997: the ship's horizon sits where f(r_s) = 1 − 1/v (v > 1); none for v < 1."""
    fast = WarpAnalyzeConfig()
    fast.grid.extent = [(-12.0, 12.0)] * 3
    fast.grid.resolution = [8, 8, 8]
    fast.analysis.full_stress_energy = False
    fast.metric.params = {"v_s": 2.0, "R": 5.0, "sigma": 2.0}
    res = analyze(fast)
    hz = res.report["horizon"]
    assert hz["present"]
    # f = 1 − 1/v = 0.5 is reached at r_s = R to ~1e-9 for σR = 10.
    assert hz["front_radius"] == pytest.approx(5.0, abs=1e-3)
    assert hz["back_radius"] == pytest.approx(5.0, abs=1e-3)
    fast.metric.params = {"v_s": 0.5, "R": 5.0, "sigma": 2.0}
    res = analyze(fast)
    assert not res.report["horizon"]["present"]
    assert (res.fields["horizon_indicator"] > 0).all()


@pytest.mark.benchmark
def test_eulerian_tidal_tensor_on_alcubierre_wall(tmp_path, monkeypatch):
    monkeypatch.setenv("PARTICLESIM_CACHE_DIR", str(tmp_path))
    cfg = WarpAnalyzeConfig()
    cfg.grid.extent = [(-12.0, 12.0)] * 3
    cfg.grid.resolution = [8, 8, 8]
    cfg.analysis.tidal = True
    cfg.output.formats = ["json", "csv"]
    cfg.output.dir = str(tmp_path / "run")
    res = analyze(cfg)
    res.save(cfg.output.dir)
    tmax = res.report["tidal"]["max_eigenvalue_abs"]
    assert np.isfinite(tmax) and tmax > 0
    assert (tmp_path / "run" / "report.csv").exists()


@pytest.mark.benchmark
@pytest.mark.slow
def test_alcubierre_violates_the_ford_roman_quantum_inequality():
    """Pfenning and Ford 1997: an observer swept through an Alcubierre wall
    samples far more negative energy than a free scalar field may carry,
    unless the wall is near the Planck scale.

    The observer is Eulerian. With unit lapse those observers are geodesic
    (their acceleration is the gradient of ln alpha, which vanishes), so the
    worldline comes from the geodesic integrator and the proper time is the
    coordinate time. Outside the bubble the shape function's derivative is
    zero to double precision, so the density vanishes there and the default
    outside="zero" convention is exact.
    """
    from particlesim.analysis.quantum_inequality import check_ford_roman
    from particlesim.scenarios.warp.metrics import COORDS

    m = make_metric("alcubierre", {"v_s": 2.0, "R": 5.0, "sigma": 2.0})
    geo = GeodesicIntegrator(m.metric(), COORDS, m.params)
    start = np.array([0.0, 20.0, 5.0, 0.0])  # offset to y = R, where |rho| peaks
    u0 = geo.from_eulerian_velocity(start, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(u0, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    res = geo.integrate(start, u0, 22.0, n_out=2201, rtol=1e-10, atol=1e-12)
    assert res.success
    rho = m.closed_form_energy_density()(res.x[:, 0], res.x[:, 1], res.x[:, 2], res.x[:, 3])
    # The observer starts and ends in flat space, which is what makes
    # outside="zero" exact rather than an approximation.
    assert abs(rho[0]) < 1e-12 and abs(rho[-1]) < 1e-12
    tau = res.affine - res.affine[int(np.argmin(rho))]

    short = check_ford_roman(tau, rho, 0.1)
    long_ = check_ford_roman(tau, rho, 5.0)
    # Sampling briefly enough is permitted: the bound scales as tau0^-4.
    assert short.satisfied
    # Sampling over the crossing is not, by more than two orders of magnitude.
    assert not long_.satisfied
    assert long_.violation_factor > 100
    assert long_.sampled_energy < 0
