"""Upwinded advection, added outside the kernel.

The kernel's advection terms are ``beta^k`` times a centred derivative, and
linear in it. So replacing the centred derivative with a lopsided one is the
same as adding ``beta^k (D_lopsided - D_centred) f`` afterwards, which is what
:meth:`particlesim.solvers.nr.bssn.Evolution.right_hand_side` does when
``upwind=True``. These tests check that the correction really is that, and not
something that merely looks like dissipation.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from particlesim.solvers.nr import bssn


def _state(n: int):
    """Flat space with a wave in ``phi`` along ``x`` and a shift whose sign varies along ``y``."""
    state, spacing = bssn.gauge_wave(shape=(n, n, 4), amplitude=0.0, extent=1.0, backend="numpy")
    axis = np.arange(n) / n
    x, y, _ = np.meshgrid(axis, axis, np.arange(4) / 4, indexing="ij")
    out = {name: np.asarray(value, dtype=float) for name, value in state.items()}
    out["phi"] = 0.01 * np.sin(2 * np.pi * x)
    out["beta0"] = 0.3 * np.cos(2 * np.pi * y)
    return out, spacing


def _lopsided(field, axis, step, sign):
    """The fourth-order lopsided first derivative, written out directly."""

    def at(offset):
        return np.roll(field, -sign * offset, axis=axis)

    return sign * (-3 * at(-1) - 10 * at(0) + 18 * at(1) - 6 * at(2) + at(3)) / (12 * step)


def _centred(field, axis, step):
    def at(offset):
        return np.roll(field, -offset, axis=axis)

    return (at(-2) - 8 * at(-1) + 8 * at(1) - at(2)) / (12 * step)


def _evolutions(spacing):
    # The kernel the gauge-wave tests already derive, so this file costs no
    # derivation of its own.
    kwargs = dict(slicing="harmonic", shift_condition="frozen", dissipation=0.0)
    return (
        bssn.Evolution.build(spacing, **kwargs),
        bssn.Evolution.build(spacing, upwind=True, **kwargs),
    )


def test_the_correction_turns_the_centred_derivative_into_the_lopsided_one():
    """On ``phi``, where only ``beta^x d_x phi`` is non-zero, both signs of the shift.

    The difference between the upwinded and centred right-hand sides has to
    be ``beta^x (D_lopsided - D_centred) phi``, with the lopsided stencil
    leaning toward ``+x`` where ``beta^x > 0`` and toward ``-x`` where it is
    negative -- the direction the information comes from, since
    ``d_t f = beta^k d_k f`` moves ``f`` against the shift.
    """
    state, spacing = _state(16)
    centred, upwinded = _evolutions(spacing)
    difference = np.asarray(upwinded.right_hand_side(state)["phi"]) - np.asarray(
        centred.right_hand_side(state)["phi"]
    )
    shift = state["beta0"]
    step = spacing[0]
    lean = np.where(
        shift > 0,
        _lopsided(state["phi"], 0, step, +1),
        _lopsided(state["phi"], 0, step, -1),
    )
    expected = shift * (lean - _centred(state["phi"], 0, step))
    assert np.max(np.abs(expected)) > 1e-6
    assert np.max(np.abs(difference - expected)) < 1e-13 * np.max(np.abs(state["phi"])) / step


def test_the_correction_is_fourth_order_small():
    """A fifth difference over ``12 h``, so ``O(h^4)`` on smooth data: sixteen per halving.

    A wrong weight anywhere in the stencil leaves a lower-order remainder,
    and the ratio says which.
    """
    sizes = []
    for n in (16, 32, 64):
        state, spacing = _state(n)
        centred, upwinded = _evolutions(spacing)
        difference = np.asarray(upwinded.right_hand_side(state)["phi"]) - np.asarray(
            centred.right_hand_side(state)["phi"]
        )
        sizes.append(float(np.max(np.abs(difference))))
    ratios = [coarse / fine for coarse, fine in zip(sizes, sizes[1:], strict=False)]
    for ratio in ratios:
        assert 14.0 < ratio < 17.0, ratios


def test_upwinding_is_off_by_default_and_leaves_the_driver_alone():
    """The default is unchanged, and ``B^i`` -- which has no advection term -- gets nothing."""
    state, spacing = _state(16)
    centred, upwinded = _evolutions(spacing)
    assert centred.upwind is False
    plain = centred.right_hand_side(state)
    corrected = upwinded.right_hand_side(state)
    for name in ("B0", "B1", "B2"):
        assert np.array_equal(np.asarray(plain[name]), np.asarray(corrected[name])), name


def test_upwinding_is_refused_at_other_orders():
    """Only the fourth-order correction is written down; any other order is refused, not guessed."""
    state, spacing = _state(16)
    _, upwinded = _evolutions(spacing)
    with pytest.raises(ValueError, match="order 4 only"):
        replace(upwinded, order=2).right_hand_side(state)
