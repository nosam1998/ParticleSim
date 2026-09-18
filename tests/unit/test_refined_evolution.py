"""A two-level refinement hierarchy on the gauge wave.

The claim a refinement scheme has to support is narrow: putting a finer box
inside a coarse domain must not cost the scheme its order. Everything here
serves that, and the measurement took three tries to become one.

**What the harness got wrong twice, because it is instructive.** The first
version stripped a fixed twelve-point buffer and compared what was left,
which is the middle quarter of the box at ``n = 32`` and four fifths of it
at ``n = 128`` -- three different physical regions, and not a convergence
study at all. The second measured the Hamiltonian constraint over the whole
box including the buffer, where the values are prolonged parent data and the
constraint stencils wrap, so the norm was dominated by an edge that is not a
solution of anything and did not converge (ratios 0.9 and 1.4). Only a
*fixed physical window*, inside the buffer at the coarsest resolution, is
measuring the scheme.

One real bug came out of it: a buffer refilled once per fine step instead of
once per Runge-Kutta stage cost exactly one order, 8.5 against 13.8.

**What is not claimed.** The refined solution error is near fourth order
over the range tested -- 13.8, 16.2, 12.1 per halving, averaging 13.9, order
3.80 -- but the sequence is not a clean 16 and the asymptotic rate is not
established. The Hamiltonian constraint over the same window converges more
slowly still, around order two to three. Sixth-order prolongation lowers the
error and *worsens* the rate, which points at a floor the interpolation
error was masking rather than at the interpolation itself. The thresholds
here are set to what is measured, not to what fourth order would give.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.solvers.nr import bssn, mesh, refined

AMPLITUDE, EXTENT, FINAL = 0.1, 1.0, 0.25

#: A physical slab well inside the box at every resolution tested.
#:
#: The buffer is a fixed *number of points*, so it shrinks physically as the
#: grid refines; a window fixed in physical space is inside it at the
#: coarsest and therefore at all of them.
WINDOW = (0.35, 0.65)


def _hierarchy(n: int):
    """Coarse ``n`` along x, a box over the middle three quarters, refined."""
    coarse_state, spacing = bssn.gauge_wave(shape=(n, 8, 8), amplitude=AMPLITUDE, extent=EXTENT)
    coarse = bssn.Evolution.build(spacing, slicing="harmonic", shift_condition="frozen")
    box = mesh.Box(origin=(n // 8, 0, 0), shape=(3 * n // 4, 8, 8))
    hierarchy = refined.Hierarchy.build(coarse, box, (n, 8, 8))

    # Exact data on the fine grid rather than prolonged, so the comparison
    # measures the evolution and not the initial interpolation. Refinement is
    # isotropic, so the fine grid is (2n, 16, 16) over the same extent.
    fine_full, fine_spacing = bssn.gauge_wave(
        shape=(2 * n, 16, 16), amplitude=AMPLITUDE, extent=EXTENT
    )
    assert fine_spacing == pytest.approx(box.spacing(spacing))
    low = 2 * box.origin[0]
    high = 2 * (box.origin[0] + box.shape[0])
    fine_state = {name: value[low:high] for name, value in fine_full.items()}
    return hierarchy, coarse_state, fine_state, (low, high)


def _window(n: int, low: int, high: int, buffer: int) -> tuple[int, int]:
    total = 2 * n
    start = int(np.ceil(WINDOW[0] * total)) - low
    stop = int(np.floor(WINDOW[1] * total)) - low
    assert start >= buffer, (n, start, buffer)
    assert stop <= (high - low) - buffer, (n, stop, high - low, buffer)
    return start, stop


# --- the hierarchy's shape ----------------------------------------------


def test_the_fine_level_is_the_coarse_one_with_half_the_spacing():
    """One kernel serves both levels, which is why refinement is cheap.

    The emitted stencils take the spacing as an argument and difference with
    ``roll``, so they neither know nor care how many points there are.
    """
    hierarchy, _, _, _ = _hierarchy(32)
    assert hierarchy.fine.kernel is hierarchy.coarse.kernel
    assert hierarchy.fine.spacing == pytest.approx(
        tuple(value / mesh.RATIO for value in hierarchy.coarse.spacing)
    )
    assert hierarchy.fine.time_step == pytest.approx(hierarchy.coarse.time_step / mesh.RATIO)


def test_only_the_axes_the_box_does_not_span_are_buffered():
    """A box spanning a periodic axis is periodic along it: nothing to fill."""
    hierarchy, _, _, _ = _hierarchy(32)
    assert hierarchy.buffered_axes == (0,)
    assert hierarchy.buffer == 12

    coarse_state, spacing = bssn.gauge_wave(shape=(32, 8, 8), amplitude=0.1, extent=1.0)
    coarse = bssn.Evolution.build(spacing, slicing="harmonic", shift_condition="frozen")
    whole = mesh.Box(origin=(0, 0, 0), shape=(32, 8, 8))
    assert refined.Hierarchy.build(coarse, whole, (32, 8, 8)).buffered_axes == ()
    inner = mesh.Box(origin=(4, 1, 1), shape=(8, 4, 4))
    assert refined.Hierarchy.build(coarse, inner, (32, 8, 8)).buffered_axes == (0, 1, 2)


def test_a_box_that_does_not_fit_is_refused():
    """Rather than being filled from the wrap, which would look plausible."""
    _, spacing = bssn.gauge_wave(shape=(32, 8, 8), amplitude=0.1, extent=1.0)
    coarse = bssn.Evolution.build(spacing, slicing="harmonic", shift_condition="frozen")
    with pytest.raises(ValueError, match="does not fit"):
        refined.Hierarchy.build(coarse, mesh.Box(origin=(28, 0, 0), shape=(8, 8, 8)), (32, 8, 8))


# --- the time interpolant ------------------------------------------------


def test_hermite_hits_both_ends_and_their_slopes():
    """The interpolant is exact at the ends, which linear also is.

    What it adds is matching the *rates* there, which is what makes it
    fourth-order rather than second, and the property to check.
    """
    start = {"u": np.array([1.0, 2.0])}
    end = {"u": np.array([3.0, 5.0])}
    start_rate = {"u": np.array([0.5, -1.0])}
    end_rate = {"u": np.array([2.0, 0.25])}
    step = 0.5

    at_start = refined.hermite(start, start_rate, end, end_rate, 0.0, step)["u"]
    at_end = refined.hermite(start, start_rate, end, end_rate, 1.0, step)["u"]
    assert np.allclose(at_start, start["u"])
    assert np.allclose(at_end, end["u"])

    tiny = 1e-6
    slope_at_start = (
        refined.hermite(start, start_rate, end, end_rate, tiny, step)["u"] - start["u"]
    ) / (tiny * step)
    assert np.allclose(slope_at_start, start_rate["u"], atol=1e-4)


def test_hermite_reproduces_a_cubic_exactly():
    """A cubic in time is what the basis spans, so it is exact on one."""

    def cubic(t):
        return 1.0 + 2.0 * t - 3.0 * t**2 + 0.5 * t**3

    def rate(t):
        return 2.0 - 6.0 * t + 1.5 * t**2

    step = 0.25
    start = {"u": np.array([cubic(0.0)])}
    end = {"u": np.array([cubic(step)])}
    start_rate = {"u": np.array([rate(0.0)])}
    end_rate = {"u": np.array([rate(step)])}
    for theta in (0.25, 0.5, 0.75):
        got = refined.hermite(start, start_rate, end, end_rate, theta, step)["u"][0]
        assert got == pytest.approx(cubic(theta * step), abs=1e-14)


# --- the measurement -----------------------------------------------------


@pytest.mark.slow
@pytest.mark.benchmark
def test_refinement_does_not_cost_the_scheme_its_order():
    """A two-level gauge wave over a fixed physical window, near fourth order.

    Measured, with fourth-order prolongation:

        n      32         64         128        256
        err    1.429e-4   1.038e-5   6.394e-7   5.301e-8
        ratio             13.77      16.23      12.06

    Averaged over the three halvings that is 13.9 per halving, order 3.80.
    **The asymptotic order is not established**: the sequence is not a clean
    16, 16, 16, and pushing to 256 lowered the last ratio rather than
    confirming it. The assertion below is therefore that every halving beats
    third order comfortably, which is what the machinery has to deliver and
    what a regression would break -- not that the rate is exactly four.

    That matters because this is the test that caught the once-per-step
    buffer, which gave 8.48 and 8.90: one order lost, from stages two to four
    reading boundary values frozen at the step's start. A threshold tuned to
    16 would look more impressive and catch the same bug no better.

    Sixth-order prolongation lowers the error by about ten at n = 32 and
    makes the *rate* worse (14.0 then 3.1), which is the signature of a floor
    the interpolation error was masking -- most likely the coarse level's own
    error arriving through the buffer, since the fine level cannot be more
    accurate than the boundary data it is handed and that data is sixteen
    times worse at the same spacing. Consistent with the numbers, not
    established; order six at n = 256 was not run.
    """
    errors = []
    for n in (32, 64, 128):
        hierarchy, coarse_state, fine_state, (low, high) = _hierarchy(n)
        start, stop = _window(n, low, high, hierarchy.buffer)
        steps = max(1, int(round(FINAL / hierarchy.coarse.time_step)))
        _, out_fine = hierarchy.run(coarse_state, fine_state, steps, FINAL / steps)

        exact, _ = bssn.gauge_wave(
            shape=(2 * n, 16, 16), amplitude=AMPLITUDE, time=FINAL, extent=EXTENT
        )
        worst = 0.0
        for name, value in exact.items():
            got = np.asarray(out_fine[name])[start:stop]
            want = np.asarray(value[low:high])[start:stop]
            worst = max(worst, float(np.sqrt(np.mean((got - want) ** 2))))
        errors.append(worst)

    ratios = [coarse / fine for coarse, fine in zip(errors, errors[1:], strict=False)]
    # Comfortably better than third order at every halving, which is what
    # the once-per-step buffer failed (8.48, 8.90) and what a regression in
    # the buffer width or the time interpolant would break.
    assert all(ratio > 11.0 for ratio in ratios), ratios
    # And not better than fourth, which would mean the comparison is wrong.
    assert all(ratio < 20.0 for ratio in ratios), ratios


@pytest.mark.slow
def test_the_fine_level_agrees_with_the_coarse_one_where_they_overlap():
    """Restriction is injection, so the coarse points inside the box *are* fine ones.

    Checked after a step rather than at setup, which is where it could go
    wrong: the restriction happens at the end of the coarse step and a
    hierarchy that restricted before advancing, or from the wrong sub-step,
    would leave the two levels disagreeing here.
    """
    hierarchy, coarse_state, fine_state, _ = _hierarchy(32)
    out_coarse, out_fine = hierarchy.step(coarse_state, fine_state)
    inside = hierarchy.box.slices()
    for name in coarse_state:
        coarse_inside = np.asarray(out_coarse[name])[inside]
        restricted = mesh.restrict(np.asarray(out_fine[name]))
        assert np.allclose(coarse_inside, restricted, atol=0.0), name
