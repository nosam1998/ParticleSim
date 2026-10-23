"""Composing the compiled kernel with its corrections must not change the rates."""

import numpy as np
import pytest

from particlesim.solvers.nr import bssn, puncture


@pytest.mark.slow
@pytest.mark.parametrize("advect", [False, True])
def test_composed_and_fused_right_hand_sides_agree(monkeypatch, advect):
    """On a CPU the kernel and its corrections are compiled apart; fused is the same arithmetic.

    Upwinding, dissipation and, with ``advect=True``, the driver's
    advection are all in play on an off-centre trumpet. The two agree to
    rounding. Leaving the corrections out, the bare kernel, differs at 1e-2.
    """
    import jax

    n, h = 16, 0.25
    axis = (np.arange(n) + 0.5) * h
    grid = np.meshgrid(axis, axis, axis, indexing="ij")
    state = bssn.with_conformal(
        puncture.trumpet_state(grid, (2.1, 1.9, 2.05), 1.0, "jax"), "W", "jax"
    )
    monkeypatch.setattr(bssn, "FUSE_RATES", False)
    monkeypatch.setattr(bssn, "_RATES", {})
    evolution = bssn.Evolution.build((h,) * 3, upwind=True, advect=advect, conformal="W")
    composed = evolution.right_hand_side(state)
    fused = jax.jit(evolution._rates)(dict(state))
    bare = evolution.kernel(dict(state), evolution.spacing)
    worst = 0.0
    for name in state:
        a, b = np.asarray(composed[name]), np.asarray(fused[name])
        assert np.max(abs(a - b)) <= 1e-12 * max(1.0, float(np.max(abs(b)))), name
        worst = max(worst, float(np.max(abs(a - np.asarray(bare[name])))))
    assert worst > 1e-3
