"""Einstein-scalar-Gauss-Bonnet at second order: the hair's pull on the hole (#52)."""

from __future__ import annotations

import pytest
import sympy as sp

from particlesim.theories.backreaction import (
    SecondOrder,
    hair_energy,
    hair_mass,
    lapse_correction,
    second_order_equations,
    static_horizon_scalar,
    static_mass,
    static_temperature,
)
from particlesim.theories.gauss_bonnet import (
    _PHI,
    DILATONIC,
    DilatonGaussBonnet,
    GaussBonnetCoupling,
    static_black_hole,
)

t, r = sp.symbols("t r", positive=True)
M, LAM2, SLOPE = sp.symbols("M lambda2 slope")
PHI = sp.Function("phi1")(t, r)
N = 1 - 2 * M / r
K = 4 * M * LAM2 * SLOPE

# The module's equations, written out in r rather than r*.
SCALAR = N * (N * PHI.diff(r, 2) + 2 * (r - M) / r**2 * PHI.diff(r)) + (
    12 * M**2 * LAM2 * SLOPE * N / r**6
)
MASS = r**2 / 2 * (N * PHI.diff(r) ** 2 + PHI.diff(t) ** 2 / N) + K / r**3 * (
    r * (r - 2 * M) * PHI.diff(r, 2) - (r - 3 * M) * PHI.diff(r)
)
SLICING = r * (PHI.diff(r) ** 2 + PHI.diff(t) ** 2 / N**2) + K / r**2 * (
    PHI.diff(r, 2) + PHI.diff(t, 2) / N**2
)
FLUX = r**2 * N * PHI.diff(t) * PHI.diff(r) + K * (
    (r - 2 * M) / r**2 * PHI.diff(t, r) - M / r**3 * PHI.diff(t)
)


def _static_hair():
    return SLOPE * LAM2 / (2 * M) * (1 / r + M / r**2 + 4 * M**2 / (3 * r**3))


@pytest.mark.slow
def test_the_equations_follow_from_the_action():
    """Varying the ``(t, r)``-reduced action, then expanding, gives the module's three equations.

    Below second order the metric equations vanish identically, and the
    ``f(0) G`` term drops out, being a total derivative. That is checked
    inside :func:`second_order_equations`.
    """
    derived = second_order_equations()
    assert sp.simplify(derived["scalar"] - SCALAR) == 0
    assert sp.simplify(derived["mass"] - MASS) == 0
    assert sp.simplify(derived["slicing"] - SLICING) == 0


def test_the_flux_is_what_the_constraint_and_the_scalar_equation_imply():
    """``d_t (d_r mu) = d_r (d_t mu)`` once ``phi_tt`` is the scalar equation's.

    This is the contracted Bianchi identity at second order. The flux also
    comes out of the action directly, by varying ``g_tr``. That derivation
    takes 14 minutes, so it is not repeated here. It agreed exactly.
    """
    residual = sp.diff(MASS, t) - sp.diff(FLUX, r)
    residual = residual.subs(sp.Derivative(PHI, (t, 2), r), sp.diff(SCALAR, r))
    residual = residual.subs(sp.Derivative(PHI, (t, 2)), SCALAR)
    assert sp.simplify(residual) == 0


def test_the_closed_forms_are_the_static_solution():
    """The static hair solves the scalar equation, and the closed forms integrate the metric's."""
    hair = _static_hair()
    assert sp.simplify(SCALAR.subs(PHI, hair).doit()) == 0
    mu = hair_energy(r, SLOPE, LAM2, M)
    delta = lapse_correction(r, SLOPE, LAM2, M)
    assert sp.simplify(sp.diff(mu, r) - MASS.subs(PHI, hair).doit()) == 0
    assert sp.simplify(sp.diff(delta, r) - SLICING.subs(PHI, hair).doit()) == 0
    assert mu.subs(r, 2 * M) == 0
    assert sp.simplify(sp.limit(mu, r, sp.oo) - hair_mass(SLOPE, LAM2, M)) == 0
    assert sp.limit(delta, r, sp.oo) == 0
    # The temperature: e^(delta_H) (1 - 2 mu'(r_H)) / 4 pi r_H, at the order kept.
    eps = sp.Symbol("epsilon")
    temperature = sp.exp(delta.subs(r, 2 * M)) * (1 - 2 * sp.diff(mu, r).subs(r, 2 * M))
    temperature = (temperature / (8 * sp.pi * M)).subs(LAM2, eps * LAM2)
    temperature = sp.series(temperature, eps, 0, 3).removeO().subs(eps, 1)
    assert sp.simplify(temperature - static_temperature(2 * M, SLOPE, LAM2)) == 0
    assert sp.simplify(hair.subs(r, 2 * M) - static_horizon_scalar(2 * M, SLOPE, LAM2)) == 0


def test_the_first_law_holds_at_second_order():
    """``dM = T dS``, with Wald's entropy ``pi r_H^2 + 4 pi lambda^2 f(phi_H)``, to ``lambda^4``.

    ``M`` is the far field of ``mu``, ``T`` comes from the horizon's
    ``delta`` and ``mu'``, and ``S`` from the horizon's scalar: three
    separate places. The temperature's coefficient is ``(-441 + 440)/60``,
    so a slip anywhere shows.
    """
    radius, eps, f0 = sp.symbols("r_H epsilon f0", positive=True)
    lam2 = eps * LAM2
    mass = static_mass(radius, SLOPE, lam2)
    area = sp.pi * radius**2
    wald = area + 4 * sp.pi * lam2 * (f0 + SLOPE * static_horizon_scalar(radius, SLOPE, lam2))

    def violation(temperature, entropy):
        first_law = sp.diff(mass, radius) - temperature * sp.diff(entropy, radius)
        return sp.simplify(sp.series(first_law, eps, 0, 3).removeO())

    assert violation(static_temperature(radius, SLOPE, lam2), wald) == 0
    assert violation(static_temperature(radius, SLOPE, lam2), area) != 0
    assert violation(1 / (4 * sp.pi * radius), wald) != 0


@pytest.mark.parametrize(
    ("coupling", "radius", "tolerance"),
    [
        (GaussBonnetCoupling("linear", -2 * _PHI), 8.0, 4e-3),
        (GaussBonnetCoupling("linear", -2 * _PHI), 16.0, 3e-4),
        (DILATONIC, 16.0, 0.06),
        (DILATONIC, 22.6, 0.03),
    ],
)
def test_the_mass_approaches_the_full_nonlinear_hole(coupling, radius, tolerance):
    """``static_black_hole`` shoots the full static equations, nonlinearly, from the horizon.

    Its mass less ``r_H/2``, over the second-order ``(49/40) f'(0)^2 / r_H^3``,
    with ``lambda = 1``:

    - linear coupling: 1.0027 at ``r_H = 8``, 1.00017 at 16. Nothing enters
      at ``lambda^6``, since ``f'' = 0``.
    - dilatonic, ``f'' = 4``: 1.047 at 16, 1.023 at 22.6. The excess times
      ``r_H^2`` holds at about 12, the ``lambda^6`` term.
    """
    hole = static_black_hole(radius, coupling, near=static_horizon_scalar(radius, -2.0, 1.0))
    ratio = (hole.mass - radius / 2) / (static_mass(radius, -2.0, 1.0) - radius / 2)
    assert abs(ratio - 1) < tolerance


def test_growing_hair_conserves_energy_and_settles_on_the_static_hole():
    """From ``phi = 0`` on Schwarzschild to 700 M, with ``f'(0) lambda^2 = M^2``.

    Measured at 1001 points, the same to 2e-7 as at 2001 and 4001:
    - the energy between the ends matches the time-integrated fluxes
      through them. The worst mismatch is when the front crosses the inner
      end, 2.3e-5 of the hair's 0.153, and it falls to 7e-6 and 1.9e-6 at
      2001 and 4001 points;
    - that energy settles on the static hair's, 49/320 less the tail
      beyond the outer end;
    - ``phi_H`` settles on 11/24, apart from a 1e-5 offset that the outer
      boundary lets in, which carries no energy;
    - radiated energy, horizon mass change and hair energy sum to zero.
    """
    growth = SecondOrder(1.0, 1.0, points=1001).run(700.0, every=50.0)
    hair = hair_mass(1.0, 1.0)
    assert growth.imbalance < 3e-5 * hair
    radius = SecondOrder(1.0, 1.0, points=1001).limit.radius
    static = hair_energy(radius[-1], 1.0, 1.0) - hair_energy(radius[0], 1.0, 1.0)
    assert abs(growth.inside[-1] - static) < 1e-5 * hair
    assert abs(growth.horizon_scalar[-1] - 11 / 24) < 5e-5
    assert abs(growth.radiated + growth.horizon_mass[-1] + hair) < 1e-5 * hair
    assert 0.0262 < growth.radiated < 0.0266
    assert -0.1800 < growth.horizon_mass[-1] < -0.1790


def test_the_hole_loses_area_and_gains_wald_entropy():
    """Growing hair costs the horizon 18.0 of area, and gains Wald's entropy 1.25.

    In units of ``f'(0)^2 lambda^4 / M^2``. The Gauss-Bonnet part of the
    flux through the horizon is ``-lambda^2 f'(0) phi_H / 2M``, negative,
    and outweighs the scalar's own. The area theorem needs the null energy
    condition, which the Gauss-Bonnet term does not keep. The radiated
    energy is the same, 0.02640, measured at ``r = 290`` or at 141.
    """
    near = SecondOrder(1.0, 1.0, outer=150.0, points=701).run(550.0, every=50.0)
    far = SecondOrder(1.0, 1.0, points=1001).run(700.0, every=50.0)
    assert abs(near.radiated / far.radiated - 1) < 1e-3
    for growth in (near, far):
        assert -18.2 < growth.area_change < -17.9
        assert 1.2 < growth.entropy_change < 1.3


def test_the_plugin_runs_and_scalarization_is_refused():
    """``string.eft4d.dgb`` has ``f'(0) = -2``; a coupling with ``f'(0) = 0`` has no hair here."""
    second = SecondOrder.from_theory(DilatonGaussBonnet(alpha=0.01), points=501)
    assert second.slope == -2.0 and second.coupling_squared == 0.01
    assert second.limit.coupling.expression == -2.0 * _PHI
    with pytest.raises(ValueError, match="not perturbative"):
        SecondOrder(0.0, 1.0)
