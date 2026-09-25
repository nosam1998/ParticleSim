"""Einstein-scalar-Gauss-Bonnet: scalarized black holes, and ADR-008 (issue #52)."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from particlesim.theories.base import IllPosedRun, require_well_posed
from particlesim.theories.emda import EMDA
from particlesim.theories.gauss_bonnet import (
    DILATONIC,
    QUADRATIC,
    SCALARIZATION,
    DilatonGaussBonnet,
    GaussBonnetCoupling,
    bifurcation_points,
    gauss_bonnet_invariant,
    horizon_slope,
    scalarized_branch,
    static_black_hole,
)
from particlesim.theories.limits import check_gr_limit
from particlesim.theories.registry import list_theories


@pytest.fixture(scope="module")
def near_unit_radius():
    """Three scalarized holes around ``r_H = lambda``, for derivatives along the branch."""
    return scalarized_branch([1.01, 1.0, 0.99])


def test_the_gauss_bonnet_invariant_of_schwarzschild():
    """``G = R_abcd R^abcd = 48 M^2 / r^6`` in vacuum, where ``R_ab = 0``."""
    t, r, theta, phi = sp.symbols("t r theta phi")
    M = sp.Symbol("M", positive=True)
    f = 1 - 2 * M / r
    metric = sp.diag(-f, 1 / f, r**2, r**2 * sp.sin(theta) ** 2)
    assert sp.simplify(gauss_bonnet_invariant(metric, [t, r, theta, phi]) - 48 * M**2 / r**6) == 0


def test_the_bifurcation_points_are_doneva_and_yazadjievs():
    """The acceptance's first half: where scalarized branches leave Schwarzschild.

    Doneva and Yazadjiev (2018) give ``M / lambda = 0.587, 0.226, 0.140`` for
    the first three. Here they are zero modes of the linearised scalar,
    found by shooting.
    """
    points = bifurcation_points(3)
    assert points == pytest.approx([0.587, 0.226, 0.140], abs=5e-4)
    assert points[0] == pytest.approx(0.58697, abs=2e-5)


def test_only_the_couplings_curvature_at_zero_sets_the_bifurcations():
    """The quadratic coupling shares ``f''(0) = 1``; doubling ``f''(0)`` scales them by ``sqrt(2)``.

    Only ``f''(0)`` enters the linearised scalar equation.
    """
    assert bifurcation_points(1, QUADRATIC) == pytest.approx(bifurcation_points(1), rel=1e-9)
    stronger = GaussBonnetCoupling("double", sp.Symbol("phi") ** 2)
    assert bifurcation_points(1, stronger)[0] == pytest.approx(
        np.sqrt(2) * bifurcation_points(1)[0], rel=1e-9
    )


def test_a_coupling_with_no_tachyon_does_not_scalarize():
    with pytest.raises(ValueError, match="bound state"):
        bifurcation_points(1, GaussBonnetCoupling("negative", -(sp.Symbol("phi") ** 2)))


def test_schwarzschild_is_a_solution_of_the_derived_equations():
    """``phi = 0`` with the scalarization coupling: the mass, temperature and entropy of GR."""
    hole = static_black_hole(1.3, horizon_scalar=0.0)
    assert hole.mass == pytest.approx(0.65, rel=1e-8)
    assert hole.temperature == pytest.approx(1 / (4 * np.pi * 1.3), rel=1e-6)
    assert hole.entropy == pytest.approx(np.pi * 1.3**2, rel=1e-12)
    assert abs(hole.charge) < 1e-12


def test_a_regular_horizon_needs_the_discriminant_positive():
    """``phi'_H`` is a root of a quadratic; it tends to the linear ``-3 f' / r_H^3``."""
    assert horizon_slope(0.4, 0.29, SCALARIZATION) is None  # f' near its maximum, r_H small
    small = 1e-4
    _, df, _ = SCALARIZATION(small)
    assert horizon_slope(1.0, small, SCALARIZATION) == pytest.approx(-3 * df, rel=1e-6)


def test_a_scalarized_hole_is_lighter_than_its_horizon_suggests(near_unit_radius):
    """At ``r_H = lambda``: ``phi_H = 0.3176`` and ``M = 0.5349``, above Schwarzschild's 0.5."""
    hole = near_unit_radius[1]
    assert hole.horizon_scalar == pytest.approx(0.31759, abs=1e-4)
    assert hole.mass == pytest.approx(0.53486, abs=1e-4)
    assert hole.charge == pytest.approx(0.14138, abs=1e-4)
    assert abs(hole.scalar[-1]) < 1e-3  # the scalar has decayed at the outer radius


def test_the_first_law_holds_along_the_branch(near_unit_radius):
    """``dM = T dS``, with Wald's entropy, not the area.

    ``M``, ``T`` and ``S`` come from three different places: the far field,
    the horizon's surface gravity and the horizon scalar. The first law
    ties them together only if the derived equations and the entropy
    formula are both right. At ``r_H = lambda`` it holds to 4e-5, the
    finite difference's own error. With the area alone, ``pi r_H^2``, it
    fails by a third.
    """
    outer, middle, inner = near_unit_radius
    dr = outer.horizon_radius - inner.horizon_radius
    dM = (outer.mass - inner.mass) / dr
    dS = (outer.entropy - inner.entropy) / dr
    assert dM == pytest.approx(middle.temperature * dS, rel=2e-4)
    area_only = 2 * np.pi * middle.horizon_radius
    assert abs(dM - middle.temperature * area_only) / dM > 0.3


def test_scalarized_holes_carry_more_entropy_than_schwarzschild(near_unit_radius):
    """The scalarized hole is favoured: more entropy than a GR hole of the same mass."""
    for hole in near_unit_radius:
        assert hole.entropy > hole.schwarzschild_entropy


def test_the_dilatonic_coupling_gives_every_hole_hair():
    """``f'(0) != 0`` for the string's own coupling, so there is no GR branch to leave.

    ``box phi = (lambda^2 / 2) e^(-2 phi) G`` is sourced with one sign, so the
    scalar is negative at the horizon. A regular horizon then needs
    ``r_H^4 > 96 e^(-4 phi_H)``, so these holes have a minimum size, about
    ``3.3 lambda``.
    """
    assert not DILATONIC.admits_general_relativity
    assert SCALARIZATION.admits_general_relativity
    hole = static_black_hole(6.0, DILATONIC)
    assert hole is not None
    assert hole.horizon_scalar < 0 and hole.charge < 0
    # At r_H = 3, 96 e^(-4 phi_H) < 81 needs phi_H > 0.04: no regular horizon for phi_H <= 0.
    assert all(horizon_slope(3.0, q, DILATONIC) is None for q in np.linspace(-1.0, 0.0, 11))


def test_the_dgb_plugin_is_registered_and_passes_the_gr_limit():
    assert list_theories()["string.eft4d.dgb"] is DilatonGaussBonnet
    report = check_gr_limit(DilatonGaussBonnet, check_action=False)
    assert report.checked and report.passed, report.reasons
    assert DilatonGaussBonnet.formulation == "order_reduced"
    predictions = DilatonGaussBonnet(alpha=0.2).observable_predictions()
    assert predictions["black_holes_have_scalar_hair"]
    assert not predictions["well_posed_evolution"]


def test_adr_008_refuses_a_strong_field_3d_run_with_an_order_reduced_plugin():
    theory = DilatonGaussBonnet(alpha=0.1)
    with pytest.raises(IllPosedRun, match="ADR-008"):
        require_well_posed(theory, dimensions=3, strong_field=True)
    require_well_posed(theory, dimensions=1, strong_field=True)
    require_well_posed(theory, dimensions=3, strong_field=False)


def test_the_3d_solvers_ask_before_building_anything():
    """The check runs first in ``Evolution.build`` and ``ccz4.build``, before any kernel is derived.

    At its GR limit a plugin is GR and is admitted. A standard-formulation
    plugin away from its limit is refused too, because the 3-D solvers
    evolve vacuum GR and would silently ignore it.
    """
    from particlesim.solvers.nr import ccz4
    from particlesim.solvers.nr.bssn import Evolution, admit_theory

    with pytest.raises(IllPosedRun):
        Evolution.build((0.1, 0.1, 0.1), theory=DilatonGaussBonnet(alpha=0.1))
    with pytest.raises(IllPosedRun):
        ccz4.build((0.1, 0.1, 0.1), theory=DilatonGaussBonnet(alpha=0.1))
    admit_theory(DilatonGaussBonnet())
    admit_theory(None)
    coupling = EMDA.couplings[0].name
    with pytest.raises(NotImplementedError, match="vacuum"):
        admit_theory(EMDA(**{coupling: 0.5}))


@pytest.mark.slow
@pytest.mark.benchmark
def test_the_first_law_holds_along_the_whole_branch():
    """From the bifurcation at ``r_H = 1.174 lambda`` down to ``r_H = 0.5 lambda``.

    The first law holds at every interior point. The entropy excess over
    Schwarzschild grows monotonically away from the bifurcation, and the
    branch starts from Schwarzschild's mass at the first bifurcation point.
    """
    radii = np.round(np.arange(1.17, 0.495, -0.01), 10)
    branch = scalarized_branch(radii)
    assert len(branch) == len(radii)
    assert branch[0].mass == pytest.approx(bifurcation_points(1)[0], abs=2e-3)
    r = np.array([h.horizon_radius for h in branch])
    M = np.array([h.mass for h in branch])
    T = np.array([h.temperature for h in branch])
    S = np.array([h.entropy for h in branch])
    law = np.abs(np.gradient(M, r) - T * np.gradient(S, r)) / np.abs(np.gradient(M, r))
    assert law[1:-1].max() < 2e-4
    excess = S / (4 * np.pi * M**2)
    assert np.all(np.diff(excess) > 0)
