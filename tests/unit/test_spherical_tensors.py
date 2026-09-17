import numpy as np
import pytest

from particlesim.core.grid import kreiss_oliger
from particlesim.core.spherical import SphericalGrid
from particlesim.core.tensors import TensorField, minkowski


def test_no_grid_point_sits_on_the_origin():
    """1/r terms in spherical evolution make an origin point fatal."""
    for refine in (1.0, 8.0):
        g = SphericalGrid(r_max=10.0, n=64, refine=refine)
        r = g.radii()
        assert (r > 0).all()
        assert r[0] < g.r_max / 64  # first cell centre is inside the first cell


def test_uniform_grid_spacing_and_volume():
    g = SphericalGrid(r_max=2.0, n=100)
    assert g.dr == pytest.approx(0.02)
    # Volume of a sphere of radius 2 is 4/3 pi r^3 = 33.51.
    assert g.volume_integrate(np.ones(100)) == pytest.approx(4 / 3 * np.pi * 8, rel=1e-4)


def test_graded_grid_concentrates_resolution_at_the_centre():
    g = SphericalGrid(r_max=10.0, n=50, refine=10.0)
    w = g.spacings()
    assert w[-1] / w[0] == pytest.approx(10.0, rel=1e-9)
    assert w.sum() == pytest.approx(10.0)
    assert not g.uniform
    with pytest.raises(ValueError, match="no single spacing"):
        _ = g.dr


def test_ghost_radii_reflect_through_the_origin():
    g = SphericalGrid(r_max=1.0, n=10, ghost=2)
    full = g.full_radii()
    assert full.size == 14
    assert (full[:2] < 0).all()
    np.testing.assert_allclose(full[:2], -g.radii()[:2][::-1])


def test_parity_fill_even_and_odd():
    g = SphericalGrid(r_max=1.0, n=6, ghost=2)
    arr = np.zeros(10)
    arr[2:8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    even = g.apply_parity(arr, even=True)
    odd = g.apply_parity(arr, even=False)
    np.testing.assert_allclose(even[:2], [2.0, 1.0])
    np.testing.assert_allclose(odd[:2], [-2.0, -1.0])


def test_invalid_spherical_grids():
    with pytest.raises(ValueError):
        SphericalGrid(r_max=0.0, n=10)
    with pytest.raises(ValueError):
        SphericalGrid(r_max=1.0, n=0)
    with pytest.raises(ValueError):
        SphericalGrid(r_max=1.0, n=10, refine=0.5)


# --- Kreiss-Oliger dissipation -------------------------------------------


@pytest.mark.parametrize("order", [2, 4, 6])
def test_dissipation_annihilates_low_order_polynomials(order):
    """KO must not degrade the scheme's accuracy, so it kills polynomials
    below the stencil degree exactly."""
    x = np.linspace(0.0, 1.0, 40)
    dx = x[1] - x[0]
    r = order // 2 + 1
    for power in range(2 * r):
        q = kreiss_oliger(x**power, 0, dx, order, epsilon=1.0)
        assert np.abs(q[r:-r]).max() < 1e-6


@pytest.mark.parametrize("order", [2, 4, 6])
def test_dissipation_damps_the_nyquist_mode_at_the_expected_rate(order):
    """The shortest representable wave is what centred differences miss and
    KO exists to remove; the rate must be -epsilon/dx and the sign damping."""
    n = 41
    dx = 0.1
    eps = 0.3
    u = (-1.0) ** np.arange(n)
    q = kreiss_oliger(u, 0, dx, order, epsilon=eps)
    r = order // 2 + 1
    np.testing.assert_allclose(q[r:-r], -eps / dx * u[r:-r], rtol=1e-12)


def test_dissipation_is_zero_near_boundaries_and_validates_inputs():
    u = np.random.default_rng(0).normal(size=30)
    q = kreiss_oliger(u, 0, 0.1, 4, epsilon=0.1)
    assert np.all(q[:3] == 0) and np.all(q[-3:] == 0)
    with pytest.raises(ValueError):
        kreiss_oliger(u, 0, 0.1, order=3)
    with pytest.raises(ValueError):
        kreiss_oliger(u, 0, 0.1, 4, epsilon=-1.0)
    with pytest.raises(ValueError):
        kreiss_oliger(np.zeros(4), 0, 0.1, 4)


def test_dissipation_scales_as_dx_to_the_2r_minus_1_on_smooth_data():
    errs = []
    for n in (200, 400):
        x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        dx = x[1] - x[0]
        q = kreiss_oliger(np.sin(x), 0, dx, order=4, epsilon=1.0)
        errs.append(np.abs(q[3:-3]).max())
    # order 4 -> r = 3 -> amplitude ~ dx^5
    assert np.log2(errs[0] / errs[1]) == pytest.approx(5.0, abs=0.2)


# --- Tensor containers ----------------------------------------------------


def test_tensor_shape_and_symmetry_validation():
    good = TensorField("T", ("down", "down"), np.eye(4), symmetry="symmetric")
    assert good.rank == 2 and good.grid_shape == ()
    with pytest.raises(ValueError, match="does not start with"):
        TensorField("T", ("down", "down"), np.zeros((3, 3)))
    bad = np.zeros((4, 4))
    bad[0, 1] = 1.0
    with pytest.raises(ValueError, match="not symmetric"):
        TensorField("T", ("down", "down"), bad, symmetry="symmetric")
    with pytest.raises(ValueError, match='"up" or "down"'):
        TensorField("T", ("sideways",), np.zeros(4))


def test_contraction_requires_opposite_variance():
    up = TensorField("u", ("up",), np.array([1.0, 0.0, 0.0, 0.0]))
    down = TensorField("v", ("down",), np.array([-1.0, 0.0, 0.0, 0.0]))
    assert up.contract(down, 0, 0) == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="metric factor is missing"):
        up.contract(up, 0, 0)


def test_raise_and_lower_round_trip_on_minkowski():
    g = minkowski()
    ginv = minkowski(variance="up")
    v = TensorField("v", ("up",), np.array([2.0, 1.0, 0.0, 0.0]))
    lowered = v.lower_index(0, g)
    np.testing.assert_allclose(lowered.data, [-2.0, 1.0, 0.0, 0.0])
    back = lowered.raise_index(0, ginv)
    np.testing.assert_allclose(back.data, v.data)
    assert back.indices == ("up",)
    with pytest.raises(ValueError, match="already up"):
        v.raise_index(0, ginv)


def test_trace_needs_a_metric_only_when_variances_agree():
    mixed = TensorField("M", ("up", "down"), np.diag([1.0, 2.0, 3.0, 4.0]))
    assert mixed.trace() == pytest.approx(10.0)
    both_down = TensorField("T", ("down", "down"), np.eye(4), symmetry="symmetric")
    with pytest.raises(ValueError, match="needs a metric"):
        both_down.trace()
    # eta^{ab} delta_{ab} = -1 + 3 = 2
    assert both_down.trace(minkowski(variance="up")) == pytest.approx(2.0)


def test_tensor_fields_carry_grid_shapes():
    data = np.zeros((4, 4, 5, 6))
    t = TensorField("T", ("down", "down"), data, symmetry="symmetric")
    assert t.grid_shape == (5, 6)
    g = minkowski(grid_shape=(5, 6), variance="up")
    assert g.data.shape == (4, 4, 5, 6)
    assert t.trace(g).shape == (5, 6)
