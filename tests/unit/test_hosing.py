"""The electron hose instability in an ion channel (issue #36)."""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.scenarios.hosing import (
    HoseBeam,
    HoseChannel,
    asymptotic_exponent,
    growth_exponent,
)


def test_early_on_the_beam_bends_as_the_channel_behind_the_head_says():
    """``y_b = y_0 [1 - (k_b s)^2 cos(k_c xi) / 2] + O(s^4)``.

    At ``s = 0`` the channel behind the head is ``y_0 (1 - cos k_c xi)``.
    """
    model = HoseChannel(betatron=1.0, channel=10.0, length=10.0, slices=401)
    s = 1e-2
    profile = model.solve([s])[0]
    expected = 1 - 0.5 * s**2 * np.cos(10.0 * model.xi)
    assert np.abs(profile - expected).max() < 5e-8


def test_the_channel_response_is_integrated_at_fourth_order():
    """Against the Green's function in closed form, for a beam shaped ``cos(a xi)``."""
    k, a = 10.0, 0.3
    errors = []
    for slices in (101, 201, 401, 801):
        model = HoseChannel(channel=k, slices=slices)
        xi = model.xi
        exact = k**2 * (np.cos(a * xi) - np.cos(k * xi)) / (k**2 - a**2)
        errors.append(np.abs(model.channel_centroid(np.cos(a * xi)[None, :])[0] - exact).max())
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(orders > 3.9), orders


def test_the_growth_is_whittums_where_his_asymptotics_hold():
    """``ln |y_b|`` against ``(3 sqrt(3) / 4) (k_b s)^(2/3) (k_c xi)^(1/3)``: slope 0.994.

    Whittum's exponent needs two things: a large exponent, and ``k_b s`` small
    against ``k_c xi``. With ``k_c = 100`` and ``s = 20`` both hold. The
    exponent at the tail is 96, and the fitted slope is 1 to 0.6%. The two
    corrections pull opposite ways. At ``s = 5`` the ``1/exponent`` one
    dominates and the slope is 0.936. At ``s = 40`` the ``k_b s / k_c xi`` one
    does, and it is 1.018.
    """
    model = HoseChannel(betatron=1.0, channel=100.0, length=10.0, slices=4001)
    profiles = model.solve([5.0, 20.0])
    slopes = [
        growth_exponent(model.xi, profile, 1.0, 100.0, s)[0]
        for s, profile in zip((5.0, 20.0), profiles, strict=True)
    ]
    assert slopes[1] == pytest.approx(1.0, abs=0.01)
    assert slopes[0] < slopes[1]
    assert asymptotic_exponent(1.0, 100.0, 10.0, 20.0) == pytest.approx(95.71, abs=0.01)


def test_macroparticles_without_spread_are_the_centroid_model():
    """The equations are linear, so identical particles move as their centroid, to 1e-9."""
    model = HoseChannel(betatron=1.0, channel=10.0, length=10.0, slices=201)
    centroid = model.solve([5.0, 10.0])
    particles = HoseBeam(model, per_slice=8, spread=0.0).solve([5.0, 10.0])
    assert np.abs(particles - centroid).max() < 1e-9 * np.abs(centroid).max()


def test_a_spread_in_betatron_wavenumber_holds_the_hose_back():
    """Phase mixing. With ``k_c = 1`` the hose grows slowly enough for a spread to catch it.

    At ``s = 40`` the envelope's ``ln |y_b|`` falls monotonically with the
    spread: 15.3 without, 14.7, 12.1 and 6.5 at spreads of 0.1, 0.2 and
    0.5. The largest has all but stopped the instability.
    """
    model = HoseChannel(betatron=1.0, channel=1.0, length=10.0, slices=201)
    growth = [float(np.log(np.abs(model.solve([40.0])[0]).max()))]
    for spread in (0.1, 0.2, 0.5):
        profile = HoseBeam(model, per_slice=32, spread=spread).solve([40.0])[0]
        growth.append(float(np.log(np.abs(profile).max())))
    assert np.all(np.diff(growth) < 0), growth
    assert growth[0] - growth[-1] > 8.0
