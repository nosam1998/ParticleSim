"""Hu-Sawicki f(R): the scalaron and its multigrid solver (issue #79).

The solver is held to two references it shares nothing with:

- the linearised equation solved by FFT on the same 7-point stencil, which
  it must approach with a difference that is the nonlinearity itself --
  proportional to the density, decade after decade;
- a plane-symmetric slab solved as a one-dimensional boundary-value problem
  by ``scipy.integrate.solve_bvp``, which the three-dimensional multigrid
  must converge to at second order, screened or not.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_bvp

from particlesim.cosmo import fofr
from particlesim.cosmo.fofr import HUBBLE_LENGTH, HuSawicki, linear_scalaron, solve_scalaron
from particlesim.cosmo.nbody import Mesh, gaussian_field

F5 = HuSawicki(f_r0=1e-5, omega_m=0.3)


# --- the model ---------------------------------------------------------------


def test_the_background_and_the_compton_wavelength():
    """``fbar_R(1) = -f_R0``, and ``1/(a m) = (c/H_0) sqrt(6 f_R0 / Rbar_0)`` today: 7.6 Mpc/h."""
    assert F5.background_field(1.0) == pytest.approx(-1e-5, rel=1e-15)
    today = HUBBLE_LENGTH * np.sqrt(6 * 1e-5 / (3 * (0.3 + 4 * 0.7)))
    assert F5.compton_wavelength(1.0) == pytest.approx(today, rel=1e-14)
    assert F5.compton_wavelength(1.0) == pytest.approx(7.6147, abs=1e-4)
    # Deeper in the past the background curvature is higher and the field smaller.
    assert abs(F5.background_field(0.5)) < abs(F5.background_field(1.0))


def test_gravity_is_enhanced_by_a_third_inside_the_compton_wavelength():
    k = np.array([1e-6, 1.0 / F5.compton_wavelength(1.0), 1e3])
    assert F5.enhancement(k, 1.0) == pytest.approx([1.0, 1.0 + 1.0 / 6.0, 4.0 / 3.0], rel=1e-7)


@pytest.mark.parametrize("kwargs", [{"f_r0": 0.0}, {"f_r0": -1e-5}, {"omega_m": 0.0}])
def test_the_model_refuses_nonsense(kwargs):
    with pytest.raises(ValueError):
        HuSawicki(**kwargs)


# --- the pieces of the solver --------------------------------------------------


def test_each_cell_is_solved_by_the_cubics_positive_root():
    """``u^3 + p u + q`` with ``q < 0``, across both signs of ``p`` and both branches."""
    generator = np.random.default_rng(1)
    p = np.concatenate([generator.uniform(-50, 50, 2000), [1e8, -1e8, 0.0]])
    q = -3.0
    root = fofr._positive_root(p, q)
    assert np.all(root > 0.0)
    scale = np.abs(root) ** 3 + np.abs(p * root) + abs(q)
    assert np.max(np.abs(root**3 + p * root + q) / scale) < 1e-14
    # p = 1e8 is where Cardano's t1 + t2 would cancel: the root is -q/p.
    assert root[-3] == pytest.approx(3e-8, rel=1e-12)


def test_restriction_and_prolongation_keep_the_mean():
    field = np.random.default_rng(2).normal(size=(8, 8, 8))
    assert fofr._restrict(field).mean() == pytest.approx(field.mean(), abs=1e-15)
    assert fofr._prolong(field).mean() == pytest.approx(field.mean(), abs=1e-15)
    assert np.allclose(fofr._prolong(np.full((4, 4, 4), 2.5)), 2.5, rtol=0, atol=1e-15)


# --- the solver against its references -----------------------------------------


def test_a_uniform_universe_is_solved_by_the_background_before_any_cycle():
    solution = solve_scalaron(np.zeros((8, 8, 8)), F5, 0.7, 100.0)
    assert solution.cycles == 0
    assert np.allclose(solution.field, F5.background_field(0.7), rtol=1e-14)


def test_the_linear_limit_is_the_ffts_on_the_same_stencil():
    """The difference is the nonlinearity: ten times smaller for each decade in the density."""
    mesh = Mesh(size=256.0, cells=32)
    base = gaussian_field(mesh, lambda k: (k / 0.1) ** -1.5 * np.exp(-((k / 1.0) ** 2)), seed=2)
    base /= base.std()
    differences = []
    for amplitude in (1e-3, 1e-4, 1e-5):
        delta = amplitude * base
        solution = solve_scalaron(delta, F5, 1.0, mesh.size)
        linear = linear_scalaron(delta, F5, 1.0, mesh.size)
        differences.append(np.abs(solution.perturbation - linear).max() / np.abs(linear).max())
    assert differences[-1] < 2e-6
    for larger, smaller in zip(differences, differences[1:], strict=False):
        assert larger / smaller == pytest.approx(10.0, rel=0.05)


def test_a_v_cycle_cuts_the_residual_twentyfold_at_order_unity_density():
    mesh = Mesh(size=256.0, cells=32)
    delta = gaussian_field(mesh, lambda k: (k / 0.1) ** -1.5 * np.exp(-((k / 1.0) ** 2)), seed=3)
    delta = np.maximum(delta / delta.std(), -0.95)
    solution = solve_scalaron(delta, F5, 1.0, mesh.size, tolerance=1e-10)
    assert solution.residuals[-1] < 1e-10
    rates = [b / a for a, b in zip(solution.residuals, solution.residuals[1:], strict=False)]
    assert max(rates) < 0.05, rates
    # And it is nonlinear there: the field departs from the background by tens of per cent.
    assert np.abs(solution.perturbation).max() > 0.1 * abs(solution.background)


def _slab(size: float, amplitude: float = 4.5):
    """A smooth slab, ``delta = 3.6`` inside and ``-0.9`` outside, and its exact ``f_R``."""

    def shape(x):
        return 0.5 * (np.tanh((x - 0.4) / 0.02) - np.tanh((x - 0.6) / 0.02))

    mean = shape((np.arange(8192) + 0.5) / 8192).mean()

    def density(x):
        return amplitude * (shape(x) - mean)

    beta = (size / HUBBLE_LENGTH) ** 2 / 3.0
    c = float(F5.curvature(1.0)) * np.sqrt(F5.f_r0)
    background, matter = float(F5.curvature(1.0)), 3.0 * F5.omega_m

    def rhs(x, y):
        return np.vstack([y[1], beta * (c / np.sqrt(-y[0]) - background - matter * density(x))])

    x = np.linspace(0.0, 1.0, 4001)
    start = -((c / (background + matter * density(x))) ** 2)
    reference = solve_bvp(
        rhs,
        lambda ya, yb: np.array([ya[0] - yb[0], ya[1] - yb[1]]),
        x,
        np.vstack([start, np.gradient(start, x)]),
        tol=1e-10,
        max_nodes=1_000_000,
    )
    assert reference.status == 0, reference.message
    local_minimum = (
        -F5.f_r0 * (float(F5.curvature(1.0)) / (background + matter * density(0.5))) ** 2
    )
    return density, reference, local_minimum


def _solve_slab(density, size, n):
    x = (np.arange(n) + 0.5) / n
    cube = np.broadcast_to(density(x)[:, None, None], (n, n, n)).copy()
    solution = solve_scalaron(cube, F5, 1.0, size)
    field = solution.field.reshape(n, -1)
    assert np.ptp(field, axis=1).max() < 1e-10 * abs(solution.background)
    return x, field[:, 0], cube


def test_a_partly_screened_slab_converges_at_second_order_to_an_independent_solution():
    """128 Mpc/h box, a 26 Mpc/h slab: the 3-D multigrid against ``solve_bvp`` in 1-D.

    The centre sits 8.5% above the local minimum ``R(f_R) = 8 pi G rho`` --
    screened in part -- where linear theory puts it 14% *below*, a field
    value the nonlinear equation cannot reach.
    """
    size = 128.0
    density, reference, local_minimum = _slab(size)
    errors = []
    for n in (16, 32, 64):
        x, profile, cube = _solve_slab(density, size, n)
        errors.append(np.abs(profile - reference.sol(x)[0]).max() / F5.f_r0)
    orders = [np.log2(a / b) for a, b in zip(errors, errors[1:], strict=False)]
    assert all(order > 1.9 for order in orders), (errors, orders)
    assert errors[-1] < 1.5e-3
    assert reference.sol(0.5)[0] / local_minimum - 1 == pytest.approx(0.0849, abs=5e-4)
    linear = linear_scalaron(cube, F5, 1.0, size)[:, 0, 0] + F5.background_field(1.0)
    assert linear[len(linear) // 2] / local_minimum - 1 == pytest.approx(-0.137, abs=0.005)


def test_a_wide_slab_is_screened_to_its_local_minimum():
    """512 Mpc/h box, a 102 Mpc/h slab: inside, ``R(f_R)`` tracks the density; no fifth force.

    Linear theory would put the centre at 0.31 of the background field; the
    minimum the field cannot go below is 0.55.
    """
    size = 512.0
    density, reference, local_minimum = _slab(size)
    assert reference.sol(0.5)[0] / local_minimum - 1 == pytest.approx(0.0, abs=5e-4)
    x, profile, cube = _solve_slab(density, size, 32)
    assert profile[16] / local_minimum - 1 == pytest.approx(0.0, abs=2e-3)
    assert np.abs(profile - reference.sol(x)[0]).max() / F5.f_r0 < 0.02
    linear = linear_scalaron(cube, F5, 1.0, size)[:, 0, 0] + F5.background_field(1.0)
    assert linear[16] / F5.background_field(1.0) == pytest.approx(0.309, abs=0.002)
    assert local_minimum / F5.background_field(1.0) == pytest.approx(0.550, abs=0.001)


@pytest.mark.parametrize(
    "density, message",
    [
        (np.full((8, 8, 8), -1.0), "exceed -1"),
        (np.zeros((6, 6, 6)), "power-of-two"),
        (np.zeros((8, 8, 4)), "power-of-two"),
    ],
)
def test_the_solver_refuses_what_it_cannot_solve(density, message):
    with pytest.raises(ValueError, match=message):
        solve_scalaron(density, F5, 1.0, 100.0)
