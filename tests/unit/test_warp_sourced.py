"""Warp Mode W3, dynamical: a bubble evolved with its own source (issue #54)."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from particlesim.scenarios.warp.metrics import Alcubierre
from particlesim.solvers.nr.matter import PrescribedMatter, matter_rates, with_matter
from particlesim.solvers.warp.sourced import (
    SourcedBubble,
    comoving_alcubierre,
    eulerian_stress_energy,
    grid,
    interior,
)

BUBBLE_NAMES = ["trK", "phi"] + [f"At{i}{j}" for i in range(3) for j in range(i, 3)]
BUBBLE_NAMES += [f"Gt{i}" for i in range(3)] + [f"gt{i}{j}" for i in range(3) for j in range(i, 3)]


@pytest.fixture(scope="module")
def symbolic():
    return eulerian_stress_energy(comoving_alcubierre(0.5, 1.5, 1.0))


def test_the_source_is_alcubierres_energy_density(symbolic):
    """``rho = -(v^2 / 32 pi) (y^2 + z^2) f'^2 / r^2``, the closed form W1 already checks.

    The slices are the same in comoving coordinates, so the Eulerian
    observers and their energy density are too.
    """
    energy, _, _ = symbolic
    x, y, z = sp.symbols("x y z", real=True)
    closed = Alcubierre({"v_s": 0.5, "R": 1.5, "sigma": 1.0}).closed_form_energy_density()
    rng = np.random.default_rng(1)
    points = rng.uniform(-3, 3, size=(20, 3))
    got = sp.lambdify((x, y, z), energy)(*points.T)
    expected = closed(0.0, *points.T)
    assert np.allclose(got, expected, rtol=1e-10, atol=1e-14)
    assert np.all(got <= 1e-15)


def test_the_momentum_density_is_the_momentum_constraints(symbolic):
    """``8 pi S_i = d_j K^j_i - d_i K`` on flat slices, from the shift alone."""
    from particlesim.symbolic.adm import FlatSliceADM

    _, momentum, _ = symbolic
    t, x, y, z = sp.symbols("t x y z", real=True)
    shift = comoving_alcubierre(0.5, 1.5, 1.0)[0, 1]
    adm = FlatSliceADM([shift, 0, 0], [x, y, z])
    rng = np.random.default_rng(2)
    points = rng.uniform(-3, 3, size=(10, 3))
    for i in range(3):
        got = sp.lambdify((x, y, z), momentum[i])(*points.T)
        expected = sp.lambdify((x, y, z), adm.momentum_density[i])(*points.T)
        assert np.allclose(got, expected, rtol=1e-9, atol=1e-13)


def test_matter_terms_scale_and_leave_the_vacuum_constraints_alone():
    shape = (2, 2, 2)
    one, zero = np.ones(shape), np.zeros(shape)
    state = {"alpha": one, "phi": zero}
    for i in range(3):
        for j in range(i, 3):
            state[f"gt{i}{j}"] = one if i == j else zero
    stress = tuple(tuple((0.3 if i == j else 0.0) * one for j in range(3)) for i in range(3))
    matter = PrescribedMatter((0.1 * one, (0.2 * one, zero, zero), stress))
    energy, momentum, s = matter.scaled(0.5).at()
    assert np.allclose(energy, 0.05) and np.allclose(momentum[0], 0.1)
    rates = matter_rates(state, energy, momentum, s)
    assert np.allclose(rates["trK"], 4 * np.pi * (0.05 + 3 * 0.15))
    assert np.allclose(rates["Gt0"], -16 * np.pi * 0.1) and np.allclose(rates["B0"], rates["Gt0"])
    assert np.allclose(rates["At00"], 0.0)
    constraints = with_matter(
        {"hamiltonian": zero, **{f"momentum{i}": zero for i in range(3)}}, energy, momentum
    )
    assert np.allclose(constraints["hamiltonian"], -16 * np.pi * 0.05)


def _residual(points: int, source: bool) -> float:
    bubble = SourcedBubble(points=points)
    evolution = bubble.evolution(dissipation=0.0)
    state = bubble.state()
    rates = (
        evolution.right_hand_side(state) if source else evolution.geometry.right_hand_side(state)
    )
    coords, _ = grid(points, bubble.extent)
    mask = interior(coords, bubble.extent)
    return max(float(np.abs(np.asarray(rates[name])[mask]).max()) for name in BUBBLE_NAMES)


@pytest.mark.slow
@pytest.mark.benchmark
def test_with_its_own_source_the_bubble_is_stationary_at_fourth_order():
    """Every BSSN rate converges to zero with the source, and stays of order one without it.

    Measured: 1.3e-2, 4.3e-3, 1.6e-3 and 3.3e-4 at 20, 30, 40 and 60 points,
    local order approaching four. Without the source it is 0.43 at every
    resolution. First use derives the frozen-gauge BSSN kernel, about five
    minutes.
    """
    sourced = [_residual(n, True) for n in (30, 40, 60)]
    order = np.log(sourced[1] / sourced[2]) / np.log(60 / 40)
    assert order > 3.5
    assert _residual(30, False) > 100 * sourced[0]


@pytest.mark.slow
@pytest.mark.benchmark
def test_with_less_exotic_matter_the_bubble_comes_apart():
    """To ``t = 2`` at 30 points: the drift grows as the source is withdrawn.

    Measured: 0.017 with the whole source, 0.11 with 90% of it, and 1.06 with
    none. The first is the truncation error, and falls with resolution. The
    others are the bubble evolving away.
    """
    bubble = SourcedBubble(points=30)
    initial = bubble.state()
    coords, spacing = grid(bubble.points, bubble.extent)
    mask = interior(coords, bubble.extent)
    source = bubble.matter()
    drifts = []
    for fraction in (1.0, 0.9, 0.0):
        evolution = bubble.evolution(source.scaled(fraction))
        state, dt = dict(initial), 0.25 * spacing
        for _ in range(int(round(2.0 / dt))):
            state = evolution.step(state, dt)
        drifts.append(
            max(
                float(np.abs((state[k] - initial[k])[mask]).max())
                for k in ("trK", "phi", "At00", "Gt0")
            )
        )
    assert drifts[0] < 0.03
    assert drifts[1] > 4 * drifts[0]
    assert drifts[2] > 40 * drifts[0]
