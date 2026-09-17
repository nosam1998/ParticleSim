import numpy as np
import pytest

from particlesim.core.grid import UniformGrid, derivative


def test_coords_are_cell_centred_and_avoid_boundaries():
    g = UniformGrid([(-1.0, 1.0)], (4,))
    np.testing.assert_allclose(g.axis(0), [-0.75, -0.25, 0.25, 0.75])
    assert g.cell_volume == pytest.approx(0.5)


def test_ghost_zones_and_interior():
    g = UniformGrid([(0.0, 1.0), (0.0, 2.0)], (4, 8), ghost=2)
    X, Y = g.coords()
    assert X.shape == g.full_shape == (8, 12)
    assert g.interior(X).shape == (4, 8)
    assert g.integrate(np.ones(g.full_shape)) == pytest.approx(2.0)


@pytest.mark.parametrize("order", [2, 4, 6])
def test_derivative_convergence_order(order):
    errs = []
    for n in (64, 128):
        x = np.linspace(0, 2 * np.pi, n, endpoint=False)
        dx = x[1] - x[0]
        d = derivative(np.sin(x), 0, dx, order)
        interior = slice(order, -order)
        errs.append(np.abs(d[interior] - np.cos(x)[interior]).max())
    rate = np.log2(errs[0] / errs[1])
    assert rate == pytest.approx(order, abs=0.3)
