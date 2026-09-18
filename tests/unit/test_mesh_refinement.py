"""Fixed mesh refinement: the operators a hierarchy is built out of.

Two properties carry everything else. Restriction after prolongation is the
*identity*, exactly, because the grids are vertex-centred and a coarse
sample is a fine sample; and prolongation converges at the order of its
stencil, so refining does not cap the scheme's accuracy at the box edge.

The first is the one worth being strict about. It is exactly zero or the
grid convention is wrong somewhere, and a tolerance would hide precisely
the off-by-half that makes it not hold.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.solvers.nr import mesh


def _wave(n: int, axes: int = 1) -> np.ndarray:
    """A smooth periodic field with structure on the scale of the grid."""
    axis = np.linspace(0.0, 1.0, n, endpoint=False)
    field = np.sin(2 * np.pi * axis) * np.cos(4 * np.pi * axis)
    for _ in range(axes - 1):
        field = field[..., None]
    return field


# --- the operators ------------------------------------------------------


@pytest.mark.parametrize(("order", "target"), [(2, 4.0), (4, 16.0), (6, 64.0)])
def test_prolongation_converges_at_its_stencils_order(order, target):
    """Measured: 4.0, 15.8 and 63.3 at the finest pair tried.

    The midpoints are what is being tested -- the even points are copied and
    exact by construction -- so this is the accuracy of the interpolation
    and nothing else.
    """
    previous = None
    ratios = []
    for n in (16, 32, 64, 128):
        coarse = _wave(n)
        fine = np.linspace(0.0, 1.0, 2 * n, endpoint=False)
        exact = np.sin(2 * np.pi * fine) * np.cos(4 * np.pi * fine)
        error = float(np.max(np.abs(mesh.prolong_axis(coarse, 0, order=order) - exact)))
        if previous is not None:
            ratios.append(previous / error)
        previous = error
    assert ratios[-1] > 0.85 * target, ratios
    assert ratios[-1] < 1.1 * target, ratios


def test_restriction_undoes_prolongation_exactly():
    """Not to a tolerance: the coarse sample *is* a fine sample, copied."""
    field = _wave(32)[:, None, None] * np.ones((32, 8, 8))
    for order in sorted(mesh.MIDPOINT):
        restored = mesh.restrict(mesh.prolong(field, order=order))
        assert restored.shape == field.shape
        assert np.array_equal(restored, field), order


def test_prolongation_puts_the_coarse_values_on_the_even_points():
    """The property restriction relies on, checked on its own."""
    field = _wave(16)
    refined = mesh.prolong_axis(field, 0, order=4)
    assert refined.shape == (32,)
    assert np.array_equal(refined[0::2], field)


def test_midpoint_weights_sum_to_one():
    """Otherwise a constant would not survive interpolation."""
    for order, weights in mesh.MIDPOINT.items():
        assert sum(weights) == pytest.approx(1.0, abs=1e-15), order
        assert len(weights) == order


def test_a_constant_and_a_line_survive_every_stencil():
    """Exactness on low-order polynomials, which is what the order means."""
    for order in sorted(mesh.MIDPOINT):
        ramp = np.arange(32, dtype=float)
        refined = mesh.prolong_axis(ramp, 0, order=order)
        # A ramp on a periodic axis is only linear away from the wrap, so
        # the interior is what can be checked.
        interior = slice(2 * order, -2 * order)
        assert np.allclose(refined[interior], np.arange(64)[interior] / 2, atol=1e-12), order


def test_prolongation_rejects_an_order_it_has_no_weights_for():
    with pytest.raises(ValueError, match="order must be one of"):
        mesh.prolong_axis(_wave(16), 0, order=3)


def test_prolongation_refuses_an_axis_too_short_for_the_stencil():
    with pytest.raises(ValueError, match="too few"):
        mesh.prolong_axis(np.zeros(4), 0, order=6)


# --- the buffer ---------------------------------------------------------


def test_the_buffer_is_as_wide_as_a_runge_kutta_step_reaches():
    """Four stages times the widest radius, which is the dissipation's.

    Twelve for the default scheme. A derivative of fourth order reaches two
    points and the Kreiss-Oliger operator that goes with it reaches three,
    so the dissipation sets the width -- getting that backwards would leave
    four points of quiet rubbish inside every fine box.
    """
    assert mesh.buffer_width() == 12
    assert mesh.buffer_width(stencil_radius=2, dissipation_radius=3, stages=4) == 12
    assert mesh.buffer_width(stencil_radius=3, dissipation_radius=4, stages=4) == 16
    # The wider of the two wins, whichever it is.
    assert mesh.buffer_width(stencil_radius=5, dissipation_radius=3) == 20


# --- boxes --------------------------------------------------------------


def test_a_box_knows_its_fine_shape_and_where_it_sits():
    box = mesh.Box(origin=(8, 2, 2), shape=(16, 4, 4))
    assert box.ndim == 3
    assert box.fine_shape == (32, 8, 8)
    assert box.slices() == (slice(8, 24), slice(2, 6), slice(2, 6))
    assert box.spacing((0.5, 0.25, 0.25)) == (0.25, 0.125, 0.125)


def test_a_box_knows_whether_it_fits():
    assert mesh.Box(origin=(0, 0, 0), shape=(64, 8, 8)).contains((64, 8, 8))
    assert not mesh.Box(origin=(1, 0, 0), shape=(64, 8, 8)).contains((64, 8, 8))
    assert not mesh.Box(origin=(-1, 0, 0), shape=(8, 8, 8)).contains((64, 8, 8))


def test_a_box_must_be_consistent():
    with pytest.raises(ValueError, match="same length"):
        mesh.Box(origin=(0, 0), shape=(8, 8, 8))
    with pytest.raises(ValueError, match="positive"):
        mesh.Box(origin=(0, 0, 0), shape=(8, 0, 8))


# --- extract and inject -------------------------------------------------


def test_extract_prolongs_before_cutting_not_after():
    """The distinction this module exists to get right.

    Cutting the box out first and interpolating second would fill the box's
    edge from its own wrapped values instead of from the parent's real
    neighbours -- a stencil-radius of quiet error exactly where the buffer
    is supposed to be trustworthy. Checked by comparing against the whole
    parent prolonged, which is the answer by definition.
    """
    parent = _wave(32)[:, None, None] * np.ones((32, 8, 8))
    box = mesh.Box(origin=(4, 0, 0), shape=(8, 8, 8))
    whole = mesh.prolong(parent, order=4)
    assert np.array_equal(mesh.extract(parent, box), whole[8:24, :, :])


def test_inject_puts_a_box_back_without_touching_the_rest():
    parent = np.zeros((16, 4, 4))
    box = mesh.Box(origin=(4, 0, 0), shape=(4, 4, 4))
    fine = np.ones(box.fine_shape)
    out = mesh.inject(parent, fine, box)
    assert np.array_equal(out[box.slices()], np.ones((4, 4, 4)))
    assert float(np.sum(out)) == pytest.approx(4 * 4 * 4)
    # and the caller's array is untouched
    assert float(np.sum(parent)) == 0.0


def test_a_round_trip_through_a_box_is_the_identity_on_that_box():
    """extract then inject: what the hierarchy does every coarse step."""
    axis = np.linspace(0.0, 1.0, 32, endpoint=False)
    parent = np.sin(2 * np.pi * axis)[:, None, None] * np.ones((32, 4, 4))
    box = mesh.Box(origin=(8, 0, 0), shape=(8, 4, 4))
    out = mesh.inject(parent, mesh.extract(parent, box), box)
    assert np.allclose(out, parent, atol=1e-15)
