"""Einstein-scalar-Gauss-Bonnet in the decoupling limit, against four other answers (#52)."""

from __future__ import annotations

import numpy as np

from particlesim.theories.decoupling import (
    DecouplingLimit,
    areal_radius,
    growth_rate,
    static_hair,
    static_scalar,
    threshold,
    tortoise,
)
from particlesim.theories.gauss_bonnet import (
    _PHI,
    DILATONIC,
    QUADRATIC,
    SCALARIZATION,
    DilatonGaussBonnet,
    GaussBonnetCoupling,
    bifurcation_points,
)

LINEAR = GaussBonnetCoupling("linear", _PHI)


def _pulse(grid, radius):
    return 1e-3 * np.exp(-(((grid - 5.0) / 3.0) ** 2))


def test_the_tortoise_coordinate_inverts():
    r = np.array([2.001, 2.1, 3.0, 10.0, 250.0])
    assert np.allclose(areal_radius(tortoise(r)), r, rtol=1e-12)


def test_a_linear_coupling_relaxes_to_the_closed_form_hair():
    """``phi = (alpha lambda^2 / 2M)(1/r + M/r^2 + 4M^2/3r^3)``, from rest, by 200 M.

    Measured at 6.6e-04, 1.4e-04 and 6.2e-05 of the hair's size between
    ``r = 2.2`` and 30 M, at t = 100, 200 and 300 M: the relaxation's
    late-time tail.
    """
    limit = DecouplingLimit(LINEAR, 0.1, outer=400.0)
    _, _, psi = limit.evolve(lambda grid, radius: 0 * grid, 200.0)
    r = limit.radius
    window = (r > 2.2) & (r < 30)
    exact = static_hair(r, 1.0, 0.1)
    assert np.max(np.abs(psi / r - exact)[window]) < 3e-4 * np.max(np.abs(exact[window]))


def test_scalarization_starts_at_the_static_bifurcation_point():
    """The linear operator's lowest eigenvalue crosses zero at the static shooting's ``M/lambda``.

    Two different equations: the eigenvalue problem in the tortoise
    coordinate here, and the static ODE in ``r`` there. Measured 0.586980
    against 0.586972.
    """
    grid = {"inner": -60.0, "outer": 300.0, "points": 2001}
    assert abs(threshold(SCALARIZATION, **grid) - bifurcation_points(1)[0]) < 3e-5


def test_below_the_threshold_the_scalar_grows_at_the_bound_states_rate():
    """At ``M/lambda = 0.5``: the evolution grows at 0.0406690 and ``sqrt(-E_0)`` is 0.0406697.

    Above the threshold, at ``M/lambda = 0.71``, the same pulse decays.
    """
    times, samples, _ = DecouplingLimit(QUADRATIC, 4.0).evolve(_pulse, 300.0)
    late = times >= 200.0
    measured = np.polyfit(times[late], np.log(np.abs(samples[late, 0])), 1)[0]
    expected = growth_rate(QUADRATIC, 4.0)
    assert abs(measured / expected - 1) < 1e-3
    assert growth_rate(QUADRATIC, 2.0) == 0.0
    _, stable, _ = DecouplingLimit(QUADRATIC, 2.0).evolve(_pulse, 200.0, every=10.0)
    assert np.max(np.abs(stable[-5:, 0])) < 1e-3 * np.max(np.abs(stable[:, 0]))


def test_the_dilatonic_plugin_evolves_and_its_hair_is_the_linear_one_to_first_order():
    """``string.eft4d.dgb`` through :meth:`DecouplingLimit.from_theory`.

    ``f = e^(-2 phi)`` has ``f'(0) = -2``, so to first order the hair is
    the linear one with ``alpha = -2``. The rest is ``e^(-2 phi) - 1``,
    about ``2|phi|``: measured 1.4% at ``lambda^2 = 0.01`` and 18% at 0.1.
    """
    theory = DilatonGaussBonnet(alpha=0.01)
    limit = DecouplingLimit.from_theory(theory, outer=400.0)
    assert limit.coupling is DILATONIC and limit.coupling_squared == 0.01
    _, _, psi = limit.evolve(lambda grid, radius: 0 * grid, 150.0)
    r = limit.radius
    window = (r > 2.2) & (r < 30)
    linear = static_hair(r, -2.0, 0.01)
    relative = np.max(np.abs(psi / r - linear)[window]) / np.max(np.abs(linear[window]))
    assert 0.005 < relative < 0.03


def test_the_nonlinear_scalar_saturates_on_the_static_scalarized_profile():
    """Scalarization coupling at ``M/lambda = 0.5``, run to 600 M.

    The static decoupled scalar, found by shooting on the horizon value, has
    ``phi_H = 0.290674``. The growing mode saturates where ``f'`` turns
    over, and by 600 M the evolved profile agrees with the shooting to
    1.4e-05 from ``r = 2.5`` to 10.
    """
    solutions = static_scalar(SCALARIZATION, 4.0, search=(0.01, 0.6), samples=60)
    assert len(solutions) == 1
    horizon_value, profile = solutions[0]
    limit = DecouplingLimit(SCALARIZATION, 4.0, inner=-300.0, outer=500.0)
    _, _, psi = limit.evolve(_pulse, 600.0, every=10.0)
    r = limit.radius
    for radius in (2.5, 3.0, 4.0, 6.0, 10.0):
        k = int(np.argmin(np.abs(r - radius)))
        assert abs(psi[k] / r[k] - profile(r[k])) < 2e-4 * horizon_value, radius
