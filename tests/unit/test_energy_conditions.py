import numpy as np

from particlesim.analysis import energy_conditions as ec


def _flat(N):
    g = np.zeros((4, 4, N))
    g[0, 0] = -1
    for i in range(1, 4):
        g[i, i] = 1
    return g


def test_dust_satisfies_all_conditions():
    N = 5
    g = _flat(N)
    T = np.zeros((4, 4, N))
    T[0, 0] = 1.0  # rest-frame dust, ρ = 1
    rep = ec.evaluate(T, g, cell_volume=1.0, n_directions=10)
    for r in rep.results.values():
        assert r.min_value >= -1e-12 and r.violating_fraction == 0.0
    assert rep.total_energy == 5.0


def test_negative_energy_violates_wec_and_nec():
    N = 3
    g = _flat(N)
    T = np.zeros((4, 4, N))
    T[0, 0] = -1.0
    rep = ec.evaluate(T, g, cell_volume=2.0, n_directions=10)
    assert rep.results["WEC"].min_value < 0 and rep.results["NEC"].min_value < 0
    assert rep.results["WEC"].violating_fraction == 1.0
    assert rep.negative_energy == -6.0


def test_stiff_radiation_like_fluid_satisfies_dec_but_vacuum_energy_violates_sec():
    N = 2
    g = _flat(N)
    # Radiation: ρ = 3p, p = 1/3.
    T = np.zeros((4, 4, N))
    T[0, 0] = 1.0
    for i in range(1, 4):
        T[i, i] = 1 / 3
    rep = ec.evaluate(T, g, cell_volume=1.0, n_directions=10)
    assert rep.results["DEC"].min_value >= -1e-12
    # Positive cosmological constant: T = -Λ g /(8π) → SEC violated.
    lam = 1.0
    Tv = -lam * g
    rep2 = ec.evaluate(Tv, g, cell_volume=1.0, n_directions=10)
    assert rep2.results["SEC"].min_value < 0
    assert rep2.results["WEC"].min_value >= -1e-12


def test_fibonacci_sphere_unit_vectors():
    d = ec.fibonacci_sphere(50)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0)
