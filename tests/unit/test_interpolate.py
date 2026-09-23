"""Shared interpolation: midpoints in space, Hermite in time."""

import numpy as np
import pytest

from particlesim.core.interpolate import hermite


def test_hermite_is_exact_for_a_cubic():
    """Two values and two derivatives determine a cubic, so the interpolant
    reproduces one everywhere on the interval, not only at the ends."""

    def f(t):
        return 2.0 - 0.5 * t + 0.3 * t**2 - 0.11 * t**3

    def df(t):
        return -0.5 + 0.6 * t - 0.33 * t**2

    step = 0.7
    ends = [np.array([f(0.0)]), np.array([f(step)]), np.array([df(0.0)]), np.array([df(step)])]
    for fraction in (0.0, 0.1, 0.25, 0.5, 0.9, 1.0):
        got = hermite(*ends, step, fraction)[0]
        assert got == pytest.approx(f(fraction * step), abs=1e-14)


def test_hermite_is_fourth_order_where_linear_is_second():
    """The order the refinement hierarchy's boundary data depends on."""

    def errors(use_slopes: bool):
        out = []
        for step in (0.1, 0.05, 0.025):
            start, end = np.sin(0.3), np.sin(0.3 + step)
            if use_slopes:
                got = hermite(start, end, np.cos(0.3), np.cos(0.3 + step), step, 0.5)
            else:
                got = start + 0.5 * (end - start)
            out.append(abs(float(got) - np.sin(0.3 + 0.5 * step)))
        return out

    for use_slopes, expected in ((True, 4.0), (False, 2.0)):
        e = errors(use_slopes)
        orders = [np.log2(e[i] / e[i + 1]) for i in range(len(e) - 1)]
        assert all(abs(o - expected) < 0.2 for o in orders), f"slopes={use_slopes}: {orders}"
