"""The fully coupled theory in spherical symmetry, in the 3-D runs' gauge (issue #52).

:mod:`particlesim.theories.esgb_spherical` is the prediction the 3-D
modified-CCZ4 evolution is held to at strong coupling. These are its own
checks, each against something computed another way: the 3-metric's
curvature, a static hole shot from different equations, and the decoupling
limit's tortoise-coordinate evolution.
"""

from functools import cache

import numpy as np
import pytest
import sympy as sp

from particlesim.theories import esgb_spherical
from particlesim.theories.esgb_spherical import CoupledSpherical
from particlesim.theories.gauss_bonnet import DILATONIC, SCALARIZATION

#: Areal radii of the probes on the initial slice, and their isotropic radii (M = 1).
PROBES = np.array([2.5, 3.0])
ISOTROPIC = (PROBES - 1 + np.sqrt(PROBES**2 - 2 * PROBES)) / 2


@pytest.mark.slow
def test_the_lapse_equation_is_the_hamiltonian_constraint():
    """Varying the lapse gives ``e^(a + 2b) (R3 + K^2 - K_ij K^ij - 16 pi rho)`` at ``lambda = 0``.

    ``R3`` comes from :class:`~particlesim.symbolic.curvature.MetricGeometry`
    on the 3-metric, not from the action, and ``16 pi rho = 2 (K_phi^2 +
    D phi . D phi)`` is the scalar's energy density in this normalisation.
    :func:`~particlesim.theories.esgb_spherical.field_equations` also
    raises if a third derivative or an inverse power of the lapse survives,
    or if an evolution equation is not affine in the second time
    derivatives, so calling it checks those.
    """
    from particlesim.symbolic.curvature import MetricGeometry

    equations = esgb_spherical.field_equations()
    rho, theta, varphi = sp.symbols("rho theta varphi")
    a, b = sp.Function("a")(rho), sp.Function("b")(rho)
    metric = sp.diag(sp.exp(2 * a), sp.exp(2 * b), sp.exp(2 * b) * sp.sin(theta) ** 2)
    ricci = sp.simplify(MetricGeometry(metric, (rho, theta, varphi)).ricci_scalar)
    S = sp.Symbol
    A, B = sp.symbols("A B", positive=True)
    plain = ricci.subs(
        {
            sp.Derivative(a, (rho, 2)): S("a_02"),
            sp.Derivative(b, (rho, 2)): S("b_02"),
        }
    ).subs({sp.Derivative(a, rho): S("a_01"), sp.Derivative(b, rho): S("b_01")})
    plain = plain.subs({a: sp.log(A), b: sp.log(B)})
    ka, kb, pi, ph = S("ka"), S("kb"), S("pi"), S("ph_01")
    # K^rho_rho = -ka and K^theta_theta = K^varphi_varphi = -kb.
    traces = (ka + 2 * kb) ** 2 - (ka**2 + 2 * kb**2)
    expected = A * B**2 * (plain + traces - 2 * pi**2 - 2 * ph**2 / A**2)
    hamiltonian = equations["hamiltonian"].subs(S("lam2"), 0)
    assert sp.simplify(sp.expand(hamiltonian - expected)) == 0
    # The Gauss-Bonnet part is there, and is first order in f'.
    assert sp.diff(equations["hamiltonian"], S("lam2")) != 0
    assert sp.diff(equations["hamiltonian"], S("lam2"), S("f1"), S("f1")) == 0


def test_schwarzschild_is_static_to_fourth_order():
    """With no coupling, isotropic Schwarzschild in its static lapse has zero rates, on the grid."""
    rows = []
    for points in (150, 300, 600):
        model = CoupledSpherical(DILATONIC, 0.0, points=points)
        rates = model.rates(model.schwarzschild())
        region = (model.rho > 0.05) & (model.rho < 3.0)
        rows.append(max(float(np.max(np.abs(rates[k][region]))) for k in ("ka", "kb")))
        # phi and pi are untouched; a and b move only by the dissipation of a smooth profile.
        assert all(np.max(np.abs(rates[k])) == 0.0 for k in ("phi", "pi"))
        assert all(np.max(np.abs(rates[k])) < 1e-9 for k in ("a", "b"))
    orders = np.log2(np.array(rows[:-1]) / np.array(rows[1:]))
    print("Schwarzschild residuals", rows, "orders", orders)
    assert np.all(orders > 3.5), (rows, orders)


#: A scalarized hole at ``M = lambda/2`` and a dilatonic one at ``r_H = 6 lambda``.
STATIC_HOLES = {
    "scalarized": (SCALARIZATION, 0.8993, 0.40612252832039697),
    "dilatonic": (DILATONIC, 6.0, -0.12531528095715458),
}


@cache
def _hole(which):
    """The hole in isotropic coordinates, solved once per module: it is most of the cost."""
    from particlesim.solvers.nr.modified import IsotropicHole

    coupling, horizon, scalar = STATIC_HOLES[which]
    return IsotropicHole.solve(horizon, coupling, horizon_scalar=scalar)


def _static_residuals(which, coupling_squared=1.0, per=(150, 300)):
    """RMS of ``d_t k_a``, ``d_t k_b`` and ``d_t pi`` near the horizon, at each resolution.

    Over ``0.05 < rho < ln 1.6``, which is ``R_H`` to ``1.6 R_H``, the region
    the 3-D static test uses, less its first cells: there the spline the
    hole is carried in has a gap of ``1e-3`` across ``rho = 0``, and the
    residual of the first cell stops converging at 600 cells. Smooth
    non-solution data converges there.
    """
    coupling, _, _ = STATIC_HOLES[which]
    hole = _hole(which)
    rows = []
    for points in per:
        model = CoupledSpherical(
            coupling,
            coupling_squared,
            horizon=hole.isotropic_horizon,
            points=points,
            lapse=hole._splines[0],
        )
        rates = model.rates(model.from_isotropic(hole.log_conformal, hole.scalar))
        region = (model.rho > 0.05) & (model.rho < np.log(1.6))
        rows.append({k: float(np.sqrt(np.mean(rates[k][region] ** 2))) for k in ("ka", "kb", "pi")})
    return rows


@pytest.mark.parametrize("which", ["scalarized", "dilatonic"])
def test_a_static_hole_of_the_shooting_equations_is_static_here(which):
    """The hole of :func:`~particlesim.theories.gauss_bonnet.static_black_hole`, in this gauge.

    A third derivation: the static equations are the action reduced to one
    dimension and varied in areal coordinates. Mapped to isotropic ones with
    its own static lapse, the hole's rates here are truncation error, and
    fall at fourth order. Measured from 150 to 300 cells: 4.00 to 4.02 for
    the scalarized hole, 3.96 to 4.17 for the dilatonic one.
    """
    rows = _static_residuals(which)
    print(which, rows)
    for k in ("ka", "kb", "pi"):
        order = np.log2(rows[0][k] / rows[1][k])
        assert order > 3.5, (k, order, rows)


def test_without_the_coupling_the_hairy_hole_is_not_static_here():
    """The control: at ``lambda = 0`` the same data's rates do not fall with resolution."""
    rows = _static_residuals("scalarized", coupling_squared=0.0)
    for k in ("ka", "kb", "pi"):
        assert rows[0][k] / rows[1][k] < 1.1, (k, rows)
        assert rows[1][k] > 1e-2, (k, rows)


def test_with_the_metric_held_the_scalar_is_the_decoupling_limits():
    """``backreaction=False`` against :class:`~particlesim.theories.decoupling.DecouplingLimit`.

    The same scalar on the same hole, evolved in the isotropic radius here
    and in the tortoise coordinate there, from rest with the dilatonic
    coupling at ``lambda^2 = 0.3 M^2``. They share the time, because the
    lapse is the static one. Measured: within 2e-8 of the hair at ``t = 3M``.
    """
    from scipy.interpolate import CubicSpline

    from particlesim.theories.decoupling import DecouplingLimit, tortoise

    model = CoupledSpherical(DILATONIC, 0.3, points=300, backreaction=False)
    _, samples, _ = model.evolve(model.schwarzschild(), 3.0, radii=ISOTROPIC)
    reference = DecouplingLimit(DILATONIC, 0.3, 1.0, points=16001)
    _, _, psi = reference.evolve(lambda rstar, r: 0.0 * r, 3.0, probes=(0.0,))
    expected = CubicSpline(reference.tortoise, psi)(tortoise(PROBES)) / PROBES
    difference = np.abs(samples[-1] - expected) / np.abs(expected)
    print("decoupled here", samples[-1], "decoupling.py", expected, difference)
    assert np.all(difference < 1e-6), difference


def test_the_hamiltonian_constraint_converges_while_the_hair_grows():
    """Free evolution keeps the constraint at truncation error: fourth order, 150 to 600 cells."""
    norms, finals = [], []
    for points in (150, 300, 600):
        model = CoupledSpherical(DILATONIC, 0.3, points=points)
        _, samples, state = model.evolve(model.schwarzschild(), 3.0, radii=ISOTROPIC)
        rho = model.rho[2:-2]
        region = (rho > 0.05) & (rho < 4.0)
        norms.append(float(np.sqrt(np.mean(model.hamiltonian(state)[region] ** 2))))
        finals.append(samples[-1])
    orders = np.log2(np.array(norms[:-1]) / np.array(norms[1:]))
    print("constraint", norms, "orders", orders, "phi", finals)
    assert np.all(orders > 3.5), (norms, orders)
    assert np.max(np.abs(finals[0] - finals[-1])) < 1e-8


def test_the_metric_answer_is_fourth_order_in_lambda_and_raises_the_hair():
    """What the 3-D evolution is held to: the coupled hair against the decoupled hair, from rest.

    The dilatonic coupling, the scalar starting from nothing on
    Schwarzschild. The excess of the coupled over the decoupled hair is the
    metric's answer. It is relative order ``lambda^4``: divided by
    ``lambda^4`` it tends to a constant as the coupling falls. At
    ``lambda^2 = 0.3 M^2`` and ``t = 3M`` it is 9.1% at ``r = 2.5M`` and
    3.5% at ``3M``, in the direction that makes the hair larger.
    """
    excess = {}
    for coupling_squared in (0.3, 0.075, 0.0375):
        values = []
        for backreaction in (True, False):
            model = CoupledSpherical(
                DILATONIC, coupling_squared, points=150, backreaction=backreaction
            )
            _, samples, _ = model.evolve(model.schwarzschild(), 3.0, radii=ISOTROPIC)
            values.append(samples[-1])
        excess[coupling_squared] = values[0] / values[1] - 1
    print(excess)
    np.testing.assert_allclose(excess[0.3], [0.0909, 0.0349], atol=5e-4)
    scaled = {k: v / k**2 for k, v in excess.items()}
    # The leading coefficient, from the two weakest couplings; the next order is lambda^2.
    np.testing.assert_allclose(scaled[0.075], scaled[0.0375], rtol=0.02)
    assert np.all(excess[0.3] > 0)
