"""Evolving ``W = e^(-2 phi)`` in place of ``phi`` (issue #48).

A puncture's ``phi`` diverges like the logarithm of the distance to it, and
the stencils next to it difference that. ``W`` vanishes like the distance
instead. The equations are not rewritten for it: ``phi``'s derivatives are
rebuilt from ``W``'s exactly, and the rate goes back as
``d_t W = -2 W d_t phi``. So on smooth data the two kernels can differ only
by how their stencils differ, which is truncation error, and the slow test
holds them to that.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from particlesim.solvers.nr import boundary, bssn
from particlesim.solvers.nr.puncture import NestedPuncture, trumpet_state
from particlesim.symbolic import bssn as symbolic_bssn


def test_the_state_names_swap_phi_for_w_and_nothing_else():
    names = symbolic_bssn.state_names("W")
    assert names[0] == "W" and "phi" not in names
    assert names[1:] == symbolic_bssn.STATE_NAMES[1:]
    assert symbolic_bssn.derivative_orders("W")["W"] == 2
    assert "phi" not in symbolic_bssn.derivative_orders("W")
    with pytest.raises(ValueError, match="not one of"):
        symbolic_bssn.state_names("chi")


def test_the_derivatives_of_phi_are_rebuilt_from_w_exactly():
    """``from_state`` on ``W`` gives what ``phi`` would have, symbolically."""
    x = sp.Symbol("x", real=True)
    phi = sp.sin(x) / 3 + x**2 / 5
    w = sp.exp(-2 * phi)
    state, derivatives, _ = symbolic_bssn.abstract_state("W")
    values = {state["W"]: w}
    for k in range(3):
        values[derivatives[f"d_W_{k}"]] = sp.diff(w, x) if k == 0 else 0
        for m in range(k, 3):
            values[derivatives[f"dd_W_{k}{m}"]] = sp.diff(w, x, 2) if k == m == 0 else 0
    variables, _, _ = symbolic_bssn.from_state(state, derivatives)
    assert sp.simplify(variables.d_phi[0].subs(values) - sp.diff(phi, x)) == 0
    assert sp.simplify(variables.dd_phi[0][0].subs(values) - sp.diff(phi, x, 2)) == 0
    assert sp.simplify(variables.conformal_exponent.subs(values) - sp.exp(-4 * phi)) == 0


def test_converting_to_w_and_back_is_the_identity():
    axis = np.linspace(-2.0, 2.0, 9) + 0.125
    grid = np.meshgrid(axis, axis, axis, indexing="ij")
    state = trumpet_state(grid, (0.0, 0.0, 0.0), 1.0, "numpy")
    w_state = bssn.with_conformal(state, "W", "numpy")
    assert "phi" not in w_state and float(np.min(w_state["W"])) > 0
    back = bssn.with_conformal(w_state, "phi", "numpy")
    assert np.allclose(back["phi"], state["phi"], rtol=0, atol=1e-13)
    assert np.allclose(bssn.conformal_exponent(w_state, "numpy"), state["phi"], atol=1e-13)
    # The physical slice is the same whichever the state carries.
    phys_phi = bssn.physical_slice_arrays(state, "numpy")
    phys_w = bssn.physical_slice_arrays(w_state, "numpy")
    for name, value in phys_phi.items():
        assert np.allclose(phys_w[name], value, rtol=1e-12, atol=0), name
    # And the radiative boundary knows what W tends to far away.
    assert boundary.ASYMPTOTIC["W"] == 1.0


@pytest.mark.slow
def test_the_w_kernel_is_the_phi_kernel_up_to_truncation():
    """On the trumpet, away from the puncture, the two agree at fourth order.

    Measured over ``2 < r < 3``: the largest difference in any rate is
    3.37e-05 at spacing ``M/4`` and 2.47e-06 at ``M/8``, a ratio of 13.6.
    """
    differences = []
    for n, h in ((32, 0.25), (64, 0.125)):
        axis = (np.arange(n) - n / 2 + 0.5) * h
        grid = np.meshgrid(axis, axis, axis, indexing="ij")
        state = trumpet_state(grid, (0.0, 0.0, 0.0), 1.0, "jax")
        rates_phi = bssn.Evolution.build((h,) * 3, advect=False).right_hand_side(state)
        w_state = bssn.with_conformal(state, "W")
        evolution = bssn.Evolution.build((h,) * 3, advect=False, conformal="W")
        rates_w = evolution.right_hand_side(w_state)
        radius = np.sqrt(sum(g**2 for g in grid))
        shell = (radius > 2.0) & (radius < 3.0)
        w = np.asarray(w_state["W"])
        worst = float(
            np.max(np.abs(np.asarray(rates_w["W"]) + 2 * w * np.asarray(rates_phi["phi"]))[shell])
        )
        for name, value in rates_phi.items():
            if name != "phi":
                gap = np.abs(np.asarray(rates_w[name]) - np.asarray(value))[shell]
                worst = max(worst, float(np.max(gap)))
        differences.append(worst)
    assert differences[0] < 1e-4, differences
    assert differences[0] / differences[1] > 10.0, differences


@pytest.mark.slow
def test_a_nested_puncture_can_carry_w():
    setup, states = NestedPuncture.build(n=32, extent=16.0, levels=3, conformal="W")
    assert all("W" in state and "phi" not in state for state in states)
    before = setup.diagnostics(states)
    after = setup.diagnostics(setup.step(states))
    assert after["finite"] and before["phi_max"] > 1.0
    assert abs(after["phi_max"] - before["phi_max"]) < 0.1
