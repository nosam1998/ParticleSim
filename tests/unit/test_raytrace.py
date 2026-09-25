"""Light rays through an Alcubierre bubble, and the view from inside (issue #55)."""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

from particlesim.analysis.raytrace import Bubble, camera_momenta, panorama, pinhole, trace
from particlesim.viz.warp_render import png_bytes, render, sky_colour


def _directions(n=40, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(3, n))
    return d / np.linalg.norm(d, axis=0)


def test_without_a_bubble_every_ray_goes_straight():
    d = _directions()
    result = trace(Bubble(v=0.0), d)
    np.testing.assert_allclose(result.sky, d, atol=1e-12)
    np.testing.assert_allclose(result.doppler, 1.0, rtol=1e-12)
    assert not result.trapped.any()


@pytest.mark.parametrize("v", [0.5, 0.99, 1.5])
def test_every_ray_arrives_shifted_by_one_minus_v_cos_alpha(v):
    """``K = p_t + v p_x`` is conserved, because the metric depends on ``t`` and ``x``
    only through ``x - v t``. With ``f = 1`` at the camera and 0 on the sky, that
    fixes each ray's frequency shift exactly, for any ``v``.

    The identity leans on the ray being null at the end, so it holds to the
    integration's accuracy. Near a horizon the shift is tiny and the relative
    error grows with it: at ``v = 1.5`` one ray arrives redshifted 500,000
    times and is right to ``1e-5``, converging at fourth order."""
    result = trace(Bubble(v=v), _directions(60, seed=1), step=0.005)
    ok = ~result.trapped
    alpha = -result.sky  # the light travels opposite to where it came from
    predicted = 1.0 - v * alpha[0]
    clear = ok & (predicted > 1e-3)
    np.testing.assert_allclose(result.doppler[clear], predicted[clear], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(result.doppler[ok], predicted[ok], rtol=1e-4)
    assert result.drift["K"] < 1e-13 and result.drift["L"] < 1e-13
    assert result.drift["H"] < 1e-9


def test_the_invariants_converge_at_fourth_order():
    d = _directions(30, seed=2)
    drift = [trace(Bubble(v=0.9), d, step=s).drift["H"] for s in (0.04, 0.02, 0.01)]
    assert drift[0] / drift[1] > 12 and drift[1] / drift[2] > 12


def test_the_tracer_agrees_with_the_christoffel_integrator():
    """An independent route: the geodesic equation from the symbolic metric's
    Christoffel symbols, integrated by DOP853, against Hamilton's equations by
    fourth-order Runge-Kutta. The camera sits off the axis, where the symbolic
    Christoffels are finite."""
    from particlesim.analysis.geodesics import GeodesicIntegrator
    from particlesim.scenarios.warp.metrics import COORDS, make_metric

    v, camera = 1.5, np.array([0.4, 0.3, 0.2])
    bubble = Bubble(v=v)
    metric = make_metric("alcubierre", {"v_s": v, "R": bubble.R, "sigma": bubble.sigma})
    geodesics = GeodesicIntegrator(metric.metric(), COORDS, metric.params)
    d = _directions(3, seed=1)
    ours = trace(bubble, d, step=0.0025, camera=tuple(camera))
    f = bubble.shape(np.linalg.norm(camera))

    def outside(x, _u):
        return np.sqrt((x[1] - v * x[0]) ** 2 + x[2] ** 2 + x[3] ** 2) - bubble.sky

    for i in range(d.shape[1]):
        u0 = np.array([-1.0, d[0, i] - v * f, d[1, i], d[2, i]])
        # Output is sampled, and the last sample before the stop is the one
        # returned, so it has to be dense enough to land past the wall.
        path = geodesics.integrate(
            np.array([0.0, *camera]),
            u0,
            60.0,
            n_out=120001,
            stop_when=outside,
            rtol=1e-12,
            atol=1e-13,
        )
        direction = path.u[-1][1:] / np.linalg.norm(path.u[-1][1:])
        np.testing.assert_allclose(direction, ours.sky[:, i], atol=1e-10)


def test_light_from_straight_ahead_is_blueshifted_by_one_plus_v():
    for v in (0.3, 2.0):
        result = trace(Bubble(v=v), np.array([[1.0], [0.0], [0.0]]), step=0.0025)
        np.testing.assert_allclose(result.sky[:, 0], [1.0, 0.0, 0.0], atol=1e-12)
        # 3e-8 at the default step, 16 times less for each halving: fourth order.
        assert result.doppler[0] == pytest.approx(1.0 + v, rel=1e-10)


def test_behind_a_superluminal_bubble_no_light_arrives():
    behind = np.array([[-1.0, -0.99], [0.0, 0.1], [0.0, 0.0]])
    behind = behind / np.linalg.norm(behind, axis=0)
    assert trace(Bubble(v=2.0), behind).trapped.all()
    assert not trace(Bubble(v=0.5), behind).trapped.any()


def test_the_view_is_symmetric_about_the_direction_of_travel():
    d = _directions(20, seed=3)
    angle = 0.7
    turn = np.array(
        [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    )
    bubble = Bubble(v=1.2)
    first, turned = trace(bubble, d), trace(bubble, turn @ d)
    ok = ~first.trapped
    np.testing.assert_allclose(turned.sky[:, ok], (turn @ first.sky)[:, ok], atol=1e-11)


def test_the_camera_frame_is_the_eulerian_observers():
    """The traced momentum is minus ``n - d``, lowered: it is null and has unit energy."""
    bubble = Bubble(v=1.7)
    d = _directions(10)
    q = camera_momenta(bubble, d)
    W = q[0] + bubble.v * q[1]  # f = 1 at the centre
    np.testing.assert_allclose(W, 1.0)  # the observer measures unit energy
    np.testing.assert_allclose(-(W**2) + (q[1:] ** 2).sum(axis=0), 0.0, atol=1e-15)


def test_view_directions():
    wide, narrow = panorama(8, 4), pinhole(8, 6, 90.0)
    for d in (wide, narrow):
        np.testing.assert_allclose(np.linalg.norm(d, axis=0), 1.0)
    assert np.all(narrow[0] > 0)
    middle = panorama(9, 5)[:, 2 * 9 + 4]
    np.testing.assert_allclose(middle, [1.0, 0.0, 0.0], atol=1e-12)  # the centre looks ahead


def _decode_png(data: bytes) -> tuple[int, int, bytes]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    at, idat, size = 8, b"", None
    while at < len(data):
        (length,) = struct.unpack(">I", data[at : at + 4])
        kind, body = data[at + 4 : at + 8], data[at + 8 : at + 8 + length]
        (crc,) = struct.unpack(">I", data[at + 8 + length : at + 12 + length])
        assert crc == zlib.crc32(kind + body) & 0xFFFFFFFF
        if kind == b"IHDR":
            size = struct.unpack(">II", body[:8])
        elif kind == b"IDAT":
            idat += body
        at += 12 + length
    return size[0], size[1], zlib.decompress(idat)


def test_the_render_paints_the_sky_and_blacks_out_the_trapped():
    still, flat = render(Bubble(v=0.0), 32, 16, doppler=False)
    expected = (sky_colour(panorama(32, 16)) * 255 + 0.5).astype(np.uint8).reshape(16, 32, 3)
    np.testing.assert_array_equal(still, expected)
    fast, result = render(Bubble(v=2.0), 32, 16)
    assert result.trapped.any()
    np.testing.assert_array_equal(fast.reshape(-1, 3)[result.trapped], 0)
    width, height, raw = _decode_png(png_bytes(fast))
    assert (width, height) == (32, 16) and len(raw) == 16 * (1 + 32 * 3)
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(16, 1 + 96)
    np.testing.assert_array_equal(rows[:, 1:].reshape(16, 32, 3), fast)
    with pytest.raises(ValueError, match="view"):
        render(Bubble(), 4, 4, view="fisheye")
