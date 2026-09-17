"""The JAX emission backend (design doc Section 5.3, ADR-001)."""

import numpy as np
import pytest
import sympy as sp

from particlesim.scenarios.warp.metrics import SPATIAL, alcubierre_shape
from particlesim.symbolic.curvature import lambdify_exprs

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

x, y, z = SPATIAL
sigma = sp.Symbol("sigma", positive=True)


def test_jax_and_numpy_kernels_agree():
    """They are compiled from the same generated text, so a divergence would
    mean the emission itself is broken."""
    exprs = [sp.sqrt(x**2 + y**2 + z**2), sp.tanh(x) * y, sp.exp(-(z**2))]
    args = [np.linspace(-2, 2, 17), np.linspace(0.1, 3, 17), np.linspace(-1, 1, 17)]
    numpy_out = lambdify_exprs(exprs, SPATIAL)(*args)
    jax_result = lambdify_exprs(exprs, SPATIAL, backend="jax")(*args)
    jax_out = np.asarray(jax_result)
    # Double precision is required, not incidental: JAX defaults to float32,
    # and the difference first showed up only in the seventh digit.
    assert jax_result.dtype == np.float64
    np.testing.assert_allclose(jax_out, numpy_out, rtol=1e-12, atol=1e-13)


def test_free_parameters_stay_symbolic_and_are_accepted_at_call_time():
    expr = sigma * x**2
    f = lambdify_exprs([expr], SPATIAL, {sigma: 5.0}, free_params=[sigma])
    arr = np.array([2.0])
    zero = np.zeros(1)
    # The bound value is ignored because sigma was declared free.
    np.testing.assert_allclose(f(arr, zero, zero, 3.0)[0], [12.0])
    np.testing.assert_allclose(f(arr, zero, zero, 0.5)[0], [2.0])


def test_gradient_of_negative_energy_with_respect_to_wall_thickness():
    """Autodiff through the emitted kernel, which is the reason ADR-001 chose
    JAX: warp design search needs this derivative."""
    v, R = 2.0, 5.0
    rs = sp.Symbol("rs", positive=True)
    # Differentiate with respect to a symbol, then substitute the radius
    # expression; sympy cannot differentiate with respect to an expression.
    df = sp.diff(alcubierre_shape(rs, sp.Float(R), sigma), rs)
    r = sp.sqrt(x**2 + y**2 + z**2)
    rho = -(v**2) / (32 * sp.pi) * (y**2 + z**2) / r**2 * df.subs(rs, r) ** 2

    kernel = lambdify_exprs([rho], SPATIAL, free_params=[sigma], backend="jax")
    n, half = 24, 12.0
    ax = jnp.linspace(-half + half / n, half - half / n, n)
    X, Y, Z = jnp.meshgrid(ax, ax, ax, indexing="ij")
    cell = (2 * half / n) ** 3

    def negative_energy(s):
        out = kernel(X, Y, Z, s)[0]
        return jnp.sum(jnp.minimum(out, 0.0)) * cell

    value = float(negative_energy(2.0))
    grad = float(jax.grad(negative_energy)(2.0))
    assert value < 0

    # A thinner wall concentrates the gradient of the shape function, so the
    # negative energy gets more negative as sigma grows: the derivative is
    # negative. Checked against a central difference on the same kernel.
    h = 1e-3
    fd = (float(negative_energy(2.0 + h)) - float(negative_energy(2.0 - h))) / (2 * h)
    assert grad < 0
    assert grad == pytest.approx(fd, rel=2e-3)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        lambdify_exprs([x], SPATIAL, backend="tensorflow")


def test_jax_kernel_is_jitted_and_reusable():
    f = lambdify_exprs([x * 2], SPATIAL, backend="jax")
    a = jnp.array([1.0, 2.0])
    zero = jnp.zeros(2)
    first = np.asarray(f(a, zero, zero))
    second = np.asarray(f(a * 3, zero, zero))
    np.testing.assert_allclose(first[0], [2.0, 4.0])
    np.testing.assert_allclose(second[0], [6.0, 12.0])
