"""Warp benchmarks from the design doc, Section 10."""

import numpy as np
import pytest
import sympy as sp

from particlesim.analysis.geodesics import GeodesicIntegrator
from particlesim.core.config import WarpAnalyzeConfig
from particlesim.core.grid import UniformGrid
from particlesim.scenarios.warp.analyze import analyze, horizon_summary
from particlesim.scenarios.warp.metrics import SPATIAL, make_metric
from particlesim.symbolic.adm import FlatSliceADM
from particlesim.symbolic.curvature import lambdify_exprs

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


# --- Lentz 2021, and the Fell-Heisenberg objection to it --------------------


@pytest.mark.benchmark
def test_lentz_energy_density_matches_the_characteristic_product():
    """Lentz 2021: with beta_i = d_i phi, 16 pi rho = 8 kappa^2 Phi''(u) Phi''(w).

    Checking the closed form against the ADM machinery is what makes the WEC
    result below a statement about Lentz's construction rather than about a
    mistyped shift vector.
    """
    m = make_metric("lentz", {})
    grid = UniformGrid([(-16.0, 16.0)] * 3, (64, 64, 64))
    X = grid.coords()
    adm = FlatSliceADM([b.subs(T0, 0) for b in m.shift()], SPATIAL)
    rho = adm.compile(m.params)(*X)["energy_density"]
    rho_cf = m.closed_form_energy_density()(np.zeros_like(X[0]), *X)
    assert np.abs(rho - rho_cf).max() < 1e-12
    assert np.isfinite(rho).all()


@pytest.mark.benchmark
def test_lentz_rhomboid_lobes_alternate_in_sign():
    """The four diamonds: positive at the front and rear vertices, negative at the sides."""
    m = make_metric("lentz", {})
    rho = m.closed_form_energy_density()
    R = m.params[m.symbols["R"]]
    kappa = m.params[m.symbols["v_h"]] / np.sqrt(2)
    axial = [rho(0.0, s * R, 0.0, 0.0) for s in (-1.0, 1.0)]
    transverse = [rho(0.0, 0.0, 0.0, s * R / kappa) for s in (-1.0, 1.0)]
    assert min(axial) > 0.0
    assert max(transverse) < 0.0
    # The interior diamond |x| + kappa |z| < R is flat, so it carries no energy.
    assert abs(rho(0.0, 0.0, 0.0, 0.0)) < 1e-12 * min(axial)


@pytest.mark.benchmark
def test_lentz_violates_the_wec_per_fell_and_heisenberg():
    """Fell and Heisenberg 2021 (CQG 38, 155020): the positive-energy claim fails.

    Flat slices with a unit lapse make the Hamiltonian constraint exactly the
    Eulerian component of the full stress-energy -- the Alcubierre benchmark
    above pins the constraint path to the Einstein-tensor path to round-off --
    so this is the Fell-Heisenberg check on the stress-energy of the soliton.
    """
    cfg = WarpAnalyzeConfig()
    cfg.metric.family = "lentz"
    cfg.grid.extent = [(-16.0, 16.0)] * 3
    cfg.grid.resolution = [64, 64, 64]
    cfg.analysis.full_stress_energy = False
    res = analyze(cfg)
    eul = res.report["eulerian"]
    assert eul["min_density"] < 0.0  # the Fell-Heisenberg check
    assert eul["negative_fraction"] > 0.25
    # The negative lobes run as deep as the positive ones run high, and because
    # the shift is a pure gradient the two cancel: the total energy is zero.
    assert eul["min_density"] < -0.5 * eul["max_density"]
    assert abs(eul["total_energy"]) < 1e-2 * abs(eul["negative_energy"])
    assert "Fell and Heisenberg" in res.report["published_property"]
    assert np.isfinite(res.fields["energy_density"]).all()


# --- Bobrick and Martire 2021 ----------------------------------------------


@pytest.mark.benchmark
def test_bobrick_martire_subluminal_member_has_no_ship_frame_horizon():
    """Bobrick and Martire 2021: alpha = (1-f) + f/A >= 1-f > v_s (1-f) when v_s < 1."""
    grid = _grid(24)
    for A in (0.25, 1.0, 2.0, 8.0):
        m = make_metric("bobrick_martire", {"v_s": 0.5, "A": A})
        h_expr = m.horizon_indicator().subs(T0, 0)
        h = lambdify_exprs([h_expr], SPATIAL, m.params)(*grid.coords())[0]
        assert h.min() > 0.0
        assert horizon_summary(h_expr, m, grid)["present"] is False
    # Push the same class superluminal and Hiscock's horizon comes back at r_s = R.
    m = make_metric("bobrick_martire", {"v_s": 2.0, "A": 1.0, "R": 5.0, "sigma": 2.0})
    hz = horizon_summary(m.horizon_indicator().subs(T0, 0), m, grid)
    assert hz["present"]
    assert hz["front_radius"] == pytest.approx(5.0, abs=1e-3)


@pytest.mark.benchmark
def test_bobrick_martire_reduces_to_alcubierre_at_unit_interior_time_rate():
    """A = 1 switches the dynamic lapse off, leaving the Alcubierre metric."""
    a = make_metric("alcubierre", {"v_s": 2.0, "R": 5.0, "sigma": 2.0})
    b = make_metric("bobrick_martire", {"v_s": 2.0, "R": 5.0, "sigma": 2.0, "A": 1.0})
    X = _grid(16).coords()
    lapse = lambdify_exprs([b.lapse().subs(T0, 0)], SPATIAL, b.params)(*X)[0]
    np.testing.assert_allclose(lapse, 1.0, atol=1e-13)
    shift_a = lambdify_exprs([s.subs(T0, 0) for s in a.shift()], SPATIAL, a.params)(*X)
    shift_b = lambdify_exprs([s.subs(T0, 0) for s in b.shift()], SPATIAL, b.params)(*X)
    np.testing.assert_allclose(shift_b, shift_a, atol=1e-14)


@pytest.mark.benchmark
@pytest.mark.slow
def test_bobrick_martire_full_symbolic_path_runs_end_to_end():
    """The non-unit lapse forces the Einstein-tensor path; it must stay finite."""
    cfg = WarpAnalyzeConfig()
    cfg.metric.family = "bobrick_martire"
    cfg.grid.extent = [(-12.0, 12.0)] * 3
    cfg.grid.resolution = [8, 8, 8]
    res = analyze(cfg)
    assert "fast_vs_full_max_abs_diff" not in res.report  # not a fast-path family
    assert np.isfinite(res.fields["energy_density_full"]).all()
    assert res.report["horizon"]["present"] is False
    assert res.fields["horizon_indicator"].min() == pytest.approx(0.25, abs=1e-3)
    for name in ("NEC", "WEC", "SEC", "DEC"):
        assert np.isfinite(res.report["energy_conditions"][name]["min"])


# --- Fuchs et al 2024 -------------------------------------------------------


@pytest.mark.benchmark
def test_fuchs_shell_is_a_positive_mass_shell():
    """The static part of the constraint, R3/16pi, is positive and vanishes at the centre."""
    m = make_metric("fuchs", {})
    grid = UniformGrid([(-20.0, 20.0)] * 3, (40, 40, 40))
    shell = m.shell_energy_density()
    rho = shell(0.0, *grid.coords())
    assert rho.min() > 0.0
    assert grid.integrate(rho) > 0.0
    # Density ~ r^2 at the centre, so the passenger region is essentially flat.
    assert float(shell(0.0, 0.0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-14)


@pytest.mark.benchmark
@pytest.mark.slow
def test_fuchs_energy_density_is_non_negative_everywhere():
    """Fuchs et al 2024: a constant-velocity subluminal drive with no negative energy."""
    cfg = WarpAnalyzeConfig()
    cfg.metric.family = "fuchs"
    cfg.grid.extent = [(-12.0, 12.0)] * 3
    cfg.grid.resolution = [8, 8, 8]
    res = analyze(cfg)
    eul = res.report["eulerian"]
    assert eul["min_density"] >= 0.0
    assert eul["negative_fraction"] == 0.0
    assert eul["total_energy"] > 0.0
    assert res.report["horizon"]["present"] is False
    assert np.isfinite(res.fields["energy_density"]).all()
