"""Bounded kernels: exact polynomials, convergence at edges, and no wrap (#132)."""

import numpy as np
import pytest
import sympy as sp

from particlesim.solvers.nr.boundary import edge_derivative
from particlesim.symbolic import codegen


def kernel(order, degree, axis=0, backend="numpy", nonperiodic_axes=(0,)):
    symbol = sp.Symbol("du")
    return codegen.emit(
        {"out": symbol},
        {symbol: ("u", (axis,) * degree)},
        order=order,
        backend=backend,
        nonperiodic_axes=nonperiodic_axes,
    )


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("degree", [1, 2])
def test_polynomials_are_exact_including_both_edges(order, degree, backend):
    """Checks every moment up to the degree the stencil must reproduce."""
    if backend == "jax":
        pytest.importorskip("jax")
    derivative = kernel(order, degree, backend=backend)
    x = np.linspace(-0.5, 0.5, 17)
    h = x[1] - x[0]
    for power in range(order + degree):
        exact = (
            np.zeros_like(x)
            if power < degree
            else np.prod(range(power - degree + 1, power + 1)) * x ** (power - degree)
        )
        actual = derivative({"u": x**power}, (h,))["out"]
        np.testing.assert_allclose(actual, exact, atol=2e-11, rtol=2e-11)


@pytest.mark.benchmark
@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("degree", [1, 2])
def test_edge_error_converges_at_the_requested_order(order, degree):
    derivative = kernel(order, degree)
    errors = []
    for count in (17, 33, 65):
        x = np.linspace(0, 2, count)
        actual = derivative({"u": np.exp(x)}, (x[1] - x[0],))["out"]
        errors.append(np.max(np.abs(actual - np.exp(x))))
    measured = np.log2(np.array(errors[:-1]) / errors[1:])
    assert np.all(measured > order - 0.4), (errors, measured)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_mixed_derivatives_use_the_correct_topology_on_each_axis(backend):
    if backend == "jax":
        pytest.importorskip("jax")
    x = np.linspace(-1, 1, 9)
    y = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    symbol = sp.Symbol("dxy")
    mixed = codegen.emit(
        {"out": symbol},
        {symbol: ("u", (0, 1))},
        order=4,
        backend=backend,
        nonperiodic_axes=(0,),
    )
    out = mixed({"u": x[:, None] ** 2 * np.sin(y)}, (x[1] - x[0], y[1] - y[0]))["out"]
    np.testing.assert_allclose(out, 2 * x[:, None] * np.cos(y), atol=1e-4)


def test_the_far_edge_cannot_change_a_bounded_near_edge():
    bounded = kernel(4, 1)
    periodic = kernel(4, 1, nonperiodic_axes=())
    u = np.zeros(16)
    u[-1] = 1.0
    assert bounded({"u": u}, (1.0,))["out"][0] == 0.0
    assert periodic({"u": u}, (1.0,))["out"][0] != 0.0
    assert bounded.nonperiodic_axes == (0,)
    assert periodic.nonperiodic_axes == ()


def test_source_round_trip_preserves_boundary_topology():
    original = kernel(6, 2, axis=1, nonperiodic_axes=(1,))
    restored = codegen.from_source(original.source)
    assert restored.nonperiodic_axes == (1,)
    assert restored.summary()["nonperiodic_axes"] == (1,)
    x = np.linspace(0, 1, 13)
    result = restored({"u": np.broadcast_to(x**3, (2, 13))}, (1.0, x[1] - x[0]))["out"]
    np.testing.assert_allclose(result, np.broadcast_to(6 * x, (2, 13)), atol=1e-10)


def test_bounded_axes_outside_the_grid_are_rejected():
    derivative = kernel(4, 1, nonperiodic_axes=(2,))
    with pytest.raises(ValueError, match="outside the grid"):
        derivative({"u": np.ones(16)}, (1.0,))


@pytest.mark.parametrize("axes", [(-1,), (0.5,), (True,)])
def test_invalid_bounded_axes_are_rejected(axes):
    with pytest.raises(ValueError, match="non-negative integer"):
        kernel(4, 1, nonperiodic_axes=axes)


@pytest.mark.parametrize("degree", [1, 2])
def test_grids_too_small_for_the_edge_stencil_are_rejected(degree):
    derivative = kernel(6, degree)
    with pytest.raises(ValueError, match="at least"):
        derivative({"u": np.ones(6 + degree - 1)}, (1.0,))


def test_radiative_edge_derivative_is_sixth_order():
    """The old fourth-order fallback fails this polynomial by about 1e-3."""
    x = np.linspace(-1, 1, 17)
    exact = 6 * x**5
    actual = edge_derivative(x**6, -1, x[1] - x[0], order=6)
    np.testing.assert_allclose(actual, exact, atol=1e-12)
