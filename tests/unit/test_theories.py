import pytest
import sympy as sp

from particlesim.theories import TheoryStack, get_theory, list_theories
from particlesim.theories.base import Coupling, Theory


def test_registry_lists_gr():
    assert "gr" in list_theories()
    gr = get_theory("gr")
    assert gr.tier == "A" and gr.formulation == "standard"
    assert gr.gr_limit() == {}


def test_unknown_theory_raises():
    with pytest.raises(KeyError):
        get_theory("does.not.exist")


def test_gr_effective_stress_energy_is_G_over_8pi():
    gr = get_theory("gr")
    G = sp.Matrix(4, 4, lambda a, b: sp.Symbol(f"G{a}{b}"))
    T = gr.effective_stress_energy(G, sp.eye(4))
    assert sp.simplify(T[1, 2] - G[1, 2] / (8 * sp.pi)) == 0


class _Jordan(Theory):
    id = "test.jordan"
    frame = "jordan"
    couplings = [Coupling("k", 1.0, bounds=(0.0, 2.0))]


def test_stack_rejects_frame_mismatch():
    with pytest.raises(ValueError, match="frame"):
        TheoryStack(gravity=get_theory("gr"), em=_Jordan())


def test_coupling_bounds_and_unknown_names():
    assert _Jordan(k=1.5).values["k"] == 1.5
    with pytest.raises(ValueError):
        _Jordan(k=5.0)
    with pytest.raises(ValueError):
        _Jordan(zeta=1.0)
