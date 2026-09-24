"""Teukolsky's quadrupole wave: is the data a solution, and would a slip show?

The wave solves the *linearised* Einstein equations, so the constraints on
it are not zero. They are second order in the amplitude, which makes a
correct implementation and a wrong one easy to tell apart: a sign error in
any of the dozen radial terms leaves a violation linear in the amplitude,
and the whole of this file is built around that one distinction.

Measured with the fourth-order constraint kernel, extent 8, ``λ = 1``, over
the central half of the box, ``|H| / a``:

    amplitude      n = 32     n = 64
    1e-2           24.4       24.5     quadratic, and resolution-independent
    1e-3           2.30       2.22
    1e-8           0.322      0.0220   linear part: truncation, x14.6

and for the linear part at ``t = 1``, where ``K_ij`` is not zero and the
momentum constraint has something to check, the orders from 32 to 128 points
are 3.82, 3.91, 3.96, 3.97 for ``H`` and 3.81, 3.91, 3.95, 3.98 for ``M``.

**The central half, not the whole box.** At ``t = 1`` the wave's tail at the
box edge is ``e^-9 H_5(3)``, about half the amplitude in ``C``, and the
wrapped stencil turns it into a jump. Over the whole box the same
measurement *grows* with resolution, as ``h^(-1/2)``, which is what a
point defect of fixed size does to an L2 norm of second derivatives.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.solvers.nr import bssn, teukolsky

EXTENT = 8.0


def _interior_constraints(n: int, amplitude: float, time: float = 0.0):
    """``|H|`` and ``|M|`` over the central half of the box, divided by the amplitude."""
    state, spacing = teukolsky.teukolsky_wave(
        shape=(n, n, n), amplitude=amplitude, width=1.0, extent=EXTENT, time=time
    )
    out = bssn.constraint_kernel()(bssn.physical_slice_arrays(state), tuple(spacing))
    inner = (slice(n // 4, n - n // 4),) * 3
    hamiltonian = np.asarray(out["hamiltonian"])[inner]
    momentum = sum(np.asarray(out[f"momentum{i}"])[inner] ** 2 for i in range(3))
    return (
        float(np.sqrt(np.mean(hamiltonian**2))) / amplitude,
        float(np.sqrt(np.mean(momentum))) / amplitude,
    )


# --- the closed form ------------------------------------------------------


@pytest.mark.parametrize("time", [0.0, 0.7, -1.3, 2.5])
def test_the_series_and_the_formula_agree_where_they_meet(time):
    """The two evaluations of ``A``, ``B`` and ``C`` either side of the switch.

    Forced both ways over the same radii. They agree to 1e-12 on values of
    order ten, which is the check on the series coefficients: a wrong
    ``c_q`` would put a jump in the metric at a quarter of a width.
    """
    radius = np.array([0.15, 0.2, 0.25, 0.3])
    for rate in (False, True):
        original = teukolsky.SERIES_RADIUS
        try:
            teukolsky.SERIES_RADIUS = 0.0
            formula = teukolsky.radial_functions(time, radius, 1.0, 1.0, rate=rate)
            teukolsky.SERIES_RADIUS = 1.0
            series = teukolsky.radial_functions(time, radius, 1.0, 1.0, rate=rate)
        finally:
            teukolsky.SERIES_RADIUS = original
        for name in "ABC":
            assert np.max(np.abs(formula[name] - series[name])) < 1e-10, (name, rate)


def test_the_coefficients_below_fifth_order_vanish():
    """Regularity at the origin, written out: no ``r^-4`` or ``r^-2`` term survives."""
    for name in teukolsky.TERMS:
        for q in range(5):
            assert teukolsky._series_coefficient(name, q) == pytest.approx(0.0, abs=1e-14)
        assert teukolsky._series_coefficient(name, 5) != 0.0


def test_the_rate_is_the_time_derivative():
    """``d/dt`` by shifting the profile's derivatives, against a difference in ``t``."""
    radius = np.linspace(0.01, 4.0, 60)
    step = 1e-5
    for time in (0.0, 0.8):
        ahead = teukolsky.radial_functions(time + step, radius, 1.0, 1.0)
        behind = teukolsky.radial_functions(time - step, radius, 1.0, 1.0)
        rate = teukolsky.radial_functions(time, radius, 1.0, 1.0, rate=True)
        for name in "ABC":
            difference = (ahead[name] - behind[name]) / (2 * step)
            assert np.max(np.abs(difference - rate[name])) < 1e-7


def test_the_perturbation_is_traceless_and_regular_at_the_origin():
    """Traceless everywhere, and the same tensor whichever way the origin is approached.

    The limit is ``A(0) (3 zz - δ)``, axisymmetric about the wave's axis.
    Approaching along the axis, the equator and two oblique directions at
    ``r = 1e-6`` gives it to eight figures, which is the check that the
    ``cos θ`` and ``sin^2 θ`` bookkeeping cancels where it has to.
    """
    rng = np.random.default_rng(3)
    points = [rng.uniform(-3, 3, 200) for _ in range(3)]
    h = teukolsky.perturbation(0.4, points, 1.0, 1.0)
    assert np.max(np.abs(h[0][0] + h[1][1] + h[2][2])) < 1e-12

    limits = []
    for direction in ((0, 0, 1), (1, 0, 0), (0.6, 0, 0.8), (0.3, 0.4, 0.866)):
        unit = np.asarray(direction) / np.linalg.norm(direction)
        h = teukolsky.perturbation(0.3, [np.array([1e-6 * c]) for c in unit], 1.0, 1.0)
        limits.append(np.array([[float(h[i][j][0]) for j in range(3)] for i in range(3)]))
    at_origin = teukolsky.perturbation(0.3, [np.zeros(1)] * 3, 1.0, 1.0)
    limits.append(np.array([[float(at_origin[i][j][0]) for j in range(3)] for i in range(3)]))
    for limit in limits[1:]:
        assert np.max(np.abs(limit - limits[0])) < 1e-6 * np.max(np.abs(limits[0]))
    assert limits[0][2, 2] == pytest.approx(-2 * limits[0][0, 0], rel=1e-9)


def test_the_data_is_time_symmetric_at_zero():
    """An outgoing wave minus an incoming one of an odd profile: ``K_ij = 0`` at ``t = 0``."""
    state, _ = teukolsky.teukolsky_wave(shape=(16, 16, 16), amplitude=1e-2, backend="numpy")
    for name, value in state.items():
        if name.startswith("At") or name == "trK":
            assert np.max(np.abs(value)) == 0.0, name
    later, _ = teukolsky.teukolsky_wave(
        shape=(16, 16, 16), amplitude=1e-2, time=1.0, backend="numpy"
    )
    assert np.max(np.abs(later["At22"])) > 1e-3


# --- the constraints --------------------------------------------------------


def test_the_constraints_are_second_order_in_the_amplitude():
    """The test that a slip in the formula cannot pass.

    Dividing by the amplitude, a quadratic violation falls by ten when the
    amplitude does, and a linear one stays put. At ``a = 1e-2`` and ``1e-3``
    on 32 points the ratio is 10.6 -- the quadratic term dominating the
    truncation error, which is itself linear in ``a``.
    """
    large, _ = _interior_constraints(32, 1e-2)
    small, _ = _interior_constraints(32, 1e-3)
    assert 8.0 < large / small < 12.0, (large, small)


def test_a_sign_error_in_the_formula_shows_as_a_linear_violation(monkeypatch):
    """What the previous test is for, demonstrated.

    Flip the sign of ``B`` as a whole -- the kind of slip a transcription of
    Teukolsky's paper makes, and one nothing else here would notice: each of
    ``A``, ``B`` and ``C`` is regular on its own, so the metric stays finite
    and traceless. Only the constraints see it, and they see it at first
    order: the ratio drops from ten to one.

    (A slip in a *single* term is louder still. It breaks the cancellation
    at the origin, the metric there goes as ``a / r^4``, and on 32 points
    its determinant goes negative.)
    """
    terms = dict(teukolsky.TERMS)
    terms["B"] = tuple((n, -weight) for n, weight in terms["B"])
    monkeypatch.setattr(teukolsky, "TERMS", terms)
    large, _ = _interior_constraints(32, 1e-2)
    small, _ = _interior_constraints(32, 1e-3)
    assert large / small < 1.5, (large, small)
    # And a hundred times the correct data's violation at a = 1e-3: 248 against 2.3.
    assert small > 50.0, small


@pytest.mark.slow
def test_the_linear_part_converges_at_fourth_order():
    """With the quadratic term out of the way, only truncation is left.

    ``a = 1e-8`` makes the second-order violation 1e-16 and leaves the
    stencil's error. At ``t = 1``, where the data carries ``K_ij`` and the
    momentum constraint is not trivially zero, 48 to 96 points gives order
    3.9 for both.
    """
    coarse = _interior_constraints(48, 1e-8, time=1.0)
    fine = _interior_constraints(96, 1e-8, time=1.0)
    for before, after in zip(coarse, fine, strict=True):
        assert np.log2(before / after) > 3.7, (coarse, fine)
