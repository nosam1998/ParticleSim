"""The BFSS matrix model's Monte Carlo, piece by piece and in the classical limit (issue #85)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particlesim.solvers.matrix.bfss import (
    BFSS,
    GRAVITY,
    RationalHMC,
    gamma_matrices,
    gravity_energy,
    multishift_cg,
    rational_approximation,
)


def _hermitian(rng, shape, size):
    a = rng.normal(size=shape + (size, size)) + 1j * rng.normal(size=shape + (size, size))
    h = (a + np.conj(np.swapaxes(a, -1, -2))) / 2
    return h - np.trace(h, axis1=-2, axis2=-1)[..., None, None] * np.eye(size) / size


def _dense(apply, shape):
    dim = int(np.prod(shape))
    columns = jax.vmap(lambda e: apply(e.reshape(shape)).reshape(-1))(jnp.eye(dim, dtype=complex))
    return np.asarray(columns).T


def test_the_black_zero_brane_coefficient_is_7_41():
    assert abs(GRAVITY - 7.4072) < 1e-4
    assert abs(gravity_energy(0.6, 5.58) - 1.2398) < 1e-4


def test_the_gamma_matrices_are_a_real_symmetric_clifford_algebra():
    g = gamma_matrices()
    assert g.shape == (9, 16, 16) and np.isrealobj(g)
    for i in range(9):
        assert np.allclose(g[i], g[i].T)
        for j in range(9):
            assert np.allclose(g[i] @ g[j] + g[j] @ g[i], 2 * (i == j) * np.eye(16))


def test_the_rational_approximations_hold_over_the_spectrum():
    for power in (-0.25, 0.125):
        a0, a, b, error = rational_approximation(power, 1e-3, 50.0, 24)
        x = np.geomspace(1e-3, 50.0, 777)
        value = a0 + np.sum(a[None] / (x[:, None] + b[None]), axis=1)
        assert error < 1e-7
        assert np.max(np.abs(value / x**power - 1)) <= error * 1.01


def test_multishift_cg_solves_every_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(120, 120)) + 1j * rng.normal(size=(120, 120))
    m = a.conj().T @ a / 120 + 0.05 * np.eye(120)
    b = rng.normal(size=120) + 1j * rng.normal(size=120)
    shifts = jnp.array([0.0, 0.1, 1.0, 10.0, 1e3])
    x, _ = multishift_cg(lambda v: jnp.asarray(m) @ v, jnp.asarray(b), shifts, tol=1e-12)
    for k, s in enumerate(np.asarray(shifts)):
        exact = np.linalg.solve(m + s * np.eye(120), b)
        assert np.max(np.abs(np.asarray(x[k]) - exact)) < 1e-9 * np.max(np.abs(exact))


def test_the_fermion_operator_has_the_structure_a_pfaffian_needs():
    """``Y`` Hermitian, the adjoint right, and ``L`` antisymmetric in ``Tr(psi chi)``."""
    rng = np.random.default_rng(0)
    model = BFSS(3, 2, 0.7)
    k = model.kernels
    x = k["fine"](*(jnp.asarray(v) for v in _split(_hermitian(rng, (9, model.coarse), 3))))
    alpha = jnp.asarray(rng.uniform(-1, 1, 3))
    shape = model.fermion_shape
    u, v = (jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape)) for _ in range(2))
    assert abs(jnp.vdot(u, k["yukawa"](x, v)) - jnp.vdot(k["yukawa"](x, u), v)) < 1e-12
    left = jnp.vdot(u, k["operator"](x, alpha, v))
    assert abs(left - jnp.vdot(k["adjoint"](x, alpha, u), v)) < 1e-12 * abs(left)

    def bilinear(p, c):  # sum_t Tr(p(t) c(t)) pairs mode r with -r
        return jnp.einsum("smab,smba->", p, c[:, ::-1])

    def full(p):
        return k["free"](alpha)[None] * p + k["yukawa"](x, p)

    lu, lv = full(u), full(v)
    assert abs(bilinear(u, lv) + bilinear(v, lu)) < 1e-12 * abs(bilinear(u, lv))


def _split(h):
    return np.real(h), np.imag(h)


def test_the_determinant_on_a_static_background_is_the_free_product():
    """Commuting ``X = diag(x, -x)``: ``prod_r ((2 pi r -+ theta)^2/beta^2 + 4|x|^2)^8``."""
    rng = np.random.default_rng(3)
    model = BFSS(2, 2, 0.7)
    k = model.kernels
    x = rng.normal(size=9) * 0.7
    theta = 0.9
    h = np.zeros((9, model.coarse, 2, 2), complex)
    h[:, :, 0, 0], h[:, :, 1, 1] = x[:, None], -x[:, None]
    fine = k["fine"](*(jnp.asarray(v) for v in _split(h)))
    alpha = jnp.asarray([theta / 2, -theta / 2])
    shape = model.fermion_shape
    dense = _dense(lambda p: k["free"](alpha)[None] * p + k["yukawa"](fine, p), shape)
    _, logdet = np.linalg.slogdet(dense)
    beta, r = model.beta, np.arange(-2, 2) + 0.5
    expected = 32 * np.sum(np.log(2 * np.pi * np.abs(r) / beta))
    for sign in (-1, 1):
        expected += 8 * np.sum(np.log(((2 * np.pi * r + sign * theta) / beta) ** 2 + 4 * x @ x))
    assert abs(logdet - expected) < 1e-10 * abs(expected)


def test_the_quartic_term_is_integrated_exactly():
    """On ``4 Lambda + 2`` points the truncated ``Tr [X, Y]^2`` is integrated without error."""
    rng = np.random.default_rng(4)
    model = BFSS(3, 3, 0.8)
    a, b = _split(_hermitian(rng, (9, model.coarse), 3))
    value = float(model.kernels["quartic"](jnp.asarray(a), jnp.asarray(b)))
    modes = np.asarray(model.kernels["modes"](jnp.asarray(a), jnp.asarray(b)))
    t = np.arange(200) / 200
    n = np.arange(-3, 4)
    xt = np.einsum("jn,inab->ijab", np.exp(2j * np.pi * np.outer(t, n)), modes)
    c = np.einsum("ijab,kjbc->ikjac", xt, xt)
    comm = c - np.swapaxes(c, 0, 1)
    direct = -(3 * model.beta / 4) * np.einsum("ikjab,ikjba->", comm, comm).real / 200
    assert abs(value - direct) < 1e-10 * abs(direct)


def test_the_force_is_the_derivative_of_the_pseudofermion_action():
    rng = np.random.default_rng(5)
    model = BFSS(3, 2, 0.8)
    sampler = RationalHMC(model, poles=28, md_poles=28, tol=1e-12, md_tol=1e-12, seed=2)
    A, B, alpha = sampler.start(2.0)
    phi = jnp.asarray(
        rng.normal(size=model.fermion_shape) + 1j * rng.normal(size=model.fermion_shape)
    )
    (fA, fB, fa), _ = sampler._fermion_force(A, B, alpha, phi)
    dA, dB, da = sampler.project(
        jnp.asarray(rng.normal(size=A.shape)),
        jnp.asarray(rng.normal(size=B.shape)),
        jnp.asarray(rng.normal(size=3)),
    )
    eps = 1e-5
    plus, _ = sampler._pseudofermion_action(A + eps * dA, B + eps * dB, alpha + eps * da, phi)
    minus, _ = sampler._pseudofermion_action(A - eps * dA, B - eps * dB, alpha - eps * da, phi)
    numeric = (float(plus) - float(minus)) / (2 * eps)
    analytic = float(jnp.sum(fA * dA) + jnp.sum(fB * dB) + jnp.sum(fa * da))
    assert abs(numeric - analytic) < 1e-6 * abs(numeric)


def test_the_pseudofermion_action_is_the_quarter_power():
    rng = np.random.default_rng(6)
    model = BFSS(2, 2, 0.8)
    sampler = RationalHMC(model, seed=3)
    A, B, alpha = sampler.start(2.0)
    phi = rng.normal(size=model.fermion_shape) + 1j * rng.normal(size=model.fermion_shape)
    x = model.kernels["fine"](A, B)
    dense = _dense(lambda p: model.kernels["normal"](x, alpha, p), model.fermion_shape)
    w, v = np.linalg.eigh(dense)
    assert w.min() > sampler.spectrum[0] and w.max() < sampler.spectrum[1]
    flat = phi.reshape(-1)
    exact = np.real(np.conj(flat) @ (v @ (w**-0.25 * (v.conj().T @ flat))))
    value, _ = sampler._pseudofermion_action(A, B, alpha, jnp.asarray(phi))
    assert abs(float(value) - exact) < 1e-7 * exact


@pytest.mark.slow
def test_at_high_temperature_the_energy_is_classical():
    """``E/N^2 -> 6 T (1 - 1/N^2)`` as ``T -> infinity``.

    Nine matrices, less the Gauss law's ``N^2 - 1`` momenta and gauge
    directions: ``8 (N^2 - 1)`` coordinates with a quartic potential give
    ``T/2 + T/4`` each. The corrections are ``O(T^-3/2)``. At ``N = 4``,
    ``T = 30``: 170.5 +- 2.2 against 168.75.
    """
    model = BFSS(4, 2, 30.0)
    sampler = RationalHMC(model, steps=5, substeps=4, seed=11)
    A, B, alpha = sampler.start(2.0)
    for _ in range(4):
        sampler.tune(A, B, alpha)
        for _ in range(20):
            A, B, alpha, *_ = sampler.trajectory(A, B, alpha)
    energies = []
    for _ in range(200):
        A, B, alpha, *_ = sampler.trajectory(A, B, alpha)
        energies.append(sampler.measure(A, B, alpha, noise=1)["energy"])
    blocks = np.asarray(energies).reshape(10, -1).mean(axis=1)
    mean, error = blocks.mean(), blocks.std(ddof=1) / np.sqrt(10)
    assert abs(mean - 6 * 30 * (1 - 1 / 16)) < max(3 * error, 0.03 * 168.75)
