"""The cosmological-constant plugin, which is also the authoring guide's example."""

import numpy as np
import pytest
import sympy as sp

from particlesim.theories import get_theory, list_theories
from particlesim.theories.gr_lambda import GRWithLambda
from particlesim.theories.limits import check_gr_limit


def test_registered_and_discoverable():
    assert "gr.lambda" in list_theories()
    theory = get_theory("gr.lambda", Lambda=0.1)
    assert theory.values["Lambda"] == 0.1
    assert theory.tier == "A" and theory.frame == "einstein"


def test_declared_gr_limit_actually_holds():
    """The guide tells authors to declare gr_limit; this is the check that
    makes the declaration mean something."""
    report = check_gr_limit(GRWithLambda)
    assert report.passed, report.reasons
    assert report.stress_energy_matches and report.lagrangian_matches


def test_de_sitter_vacuum_needs_no_matter():
    """With Lambda on the geometry side, G_ab = -Lambda g_ab is a vacuum."""
    lam = 0.3
    theory = get_theory("gr.lambda", Lambda=lam)
    g = sp.Matrix(4, 4, lambda i, j: sp.Symbol(f"g{min(i, j)}{max(i, j)}"))
    einstein = -lam * g
    T = sp.simplify(theory.effective_stress_energy(einstein, g))
    assert T.is_zero_matrix


def test_difference_from_gr_is_exactly_the_lambda_term():
    lam = 0.25
    g = sp.Matrix(4, 4, lambda i, j: sp.Symbol(f"g{min(i, j)}{max(i, j)}"))
    einstein = sp.Matrix(4, 4, lambda i, j: sp.Symbol(f"G{min(i, j)}{max(i, j)}"))
    with_lambda = get_theory("gr.lambda", Lambda=lam).effective_stress_energy(einstein, g)
    plain = get_theory("gr").effective_stress_energy(einstein, g)
    assert sp.simplify(with_lambda - plain - lam * g / (8 * sp.pi)).is_zero_matrix


def test_coupling_bounds_are_enforced():
    with pytest.raises(ValueError, match="outside bounds"):
        get_theory("gr.lambda", Lambda=50.0)


def test_observable_predictions_match_the_de_sitter_horizon():
    lam = 0.12
    pred = get_theory("gr.lambda", Lambda=lam).observable_predictions()
    assert pred["accelerating"] is True
    assert pred["de_sitter_horizon_radius"] == pytest.approx(np.sqrt(3.0 / lam))
    assert get_theory("gr.lambda", Lambda=-0.1).observable_predictions()["accelerating"] is False


def test_lagrangian_reduces_to_einstein_hilbert_at_zero_lambda():
    t, x, y, z = sp.symbols("t x y z", real=True)
    a = sp.Function("a")(t)
    metric = sp.diag(-(a**2), a**2, a**2, a**2)
    zero = get_theory("gr.lambda", Lambda=0.0).lagrangian(metric, [t, x, y, z])
    plain = get_theory("gr").lagrangian(metric, [t, x, y, z])
    assert sp.simplify(zero - plain) == 0
