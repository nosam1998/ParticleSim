import numpy as np
import sympy as sp

from particlesim.symbolic.curvature import Eulerian, MetricGeometry, lambdify_exprs

t, r, th, ph = sp.symbols("t r theta phi", positive=True)
M = sp.Symbol("M", positive=True)


def schwarzschild():
    f = 1 - 2 * M / r
    return sp.diag(-f, 1 / f, r**2, r**2 * sp.sin(th) ** 2), [t, r, th, ph]


def test_schwarzschild_is_vacuum_numerically():
    g, x = schwarzschild()
    geom = MetricGeometry(g, x)
    G = geom.einstein
    f = lambdify_exprs([G[a, b] for a in range(4) for b in range(4)], x, {M: 1.0})
    rng = np.random.default_rng(0)
    pts = (
        rng.uniform(0, 1, 5),
        rng.uniform(3, 10, 5),
        rng.uniform(0.3, 2.8, 5),
        rng.uniform(0, 6, 5),
    )
    assert np.abs(f(*pts)).max() < 1e-9


def test_schwarzschild_kretschmann():
    g, x = schwarzschild()
    geom = MetricGeometry(g, x, simplify=True)
    K = sp.simplify(geom.kretschmann)
    assert sp.simplify(K - 48 * M**2 / r**6) == 0


def test_eulerian_normal_for_adm_metric():
    tt, xx, yy, zz = sp.symbols("t x y z", real=True)
    a, b = sp.symbols("alpha beta", positive=True)
    # ds² = -α² dt² + (dx + β dt)² + dy² + dz²
    g = sp.Matrix([[-(a**2) + b**2, b, 0, 0], [b, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    e = Eulerian(g, [tt, xx, yy, zz])
    assert sp.simplify(e.lapse - a) == 0
    n = e.normal
    assert sp.simplify(n[0] - 1 / a) == 0 and sp.simplify(n[1] + b / a) == 0
    # n is unit timelike
    assert sp.simplify((n.T * g * n)[0, 0] + 1) == 0


def test_lambdify_broadcasts_constants():
    xx = sp.Symbol("x")
    f = lambdify_exprs([xx**2, sp.S.Zero, sp.pi], [xx])
    out = f(np.array([1.0, 2.0, 3.0]))
    assert out.shape == (3, 3)
    np.testing.assert_allclose(out[0], [1, 4, 9])
    np.testing.assert_allclose(out[1], 0)
    np.testing.assert_allclose(out[2], np.pi)
