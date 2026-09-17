"""Warp benchmarks from the design doc, Section 10."""

import numpy as np
import pytest
import sympy as sp

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
