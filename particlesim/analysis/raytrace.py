"""Light rays through an Alcubierre bubble: what an observer inside sees (issue #55).

The metric is Alcubierre's, as :class:`~particlesim.scenarios.warp.metrics.Alcubierre`
writes it, with the bubble centre at ``x = v t``:

    ds^2 = -dt^2 + (dx - v f(r_s) dt)^2 + dy^2 + dz^2

Null geodesics are integrated in Hamiltonian form. With covariant momenta
``p_a`` and ``W = p_t + v f p_x``, the Hamiltonian ``H = (1/2) g^ab p_a p_b``
is

    H = (1/2) [ -W^2 + p_x^2 + p_y^2 + p_z^2 ]

and Hamilton's equations need only ``f`` and ``f'``, no Christoffel symbols:

    dt/dl = -W,    dx/dl = p_x - v f W,    dy/dl = p_y,    dz/dl = p_z
    dp_a/dl = v W p_x f'(r_s) d_a r_s

**Rays are traced backwards from the camera,** which is standard: a pixel
is a direction the light arrived from. The camera is the Eulerian observer
at the bubble centre, carried along at ``v``, looking out through a
spatial frame that is simply ``d_x, d_y, d_z``. For a flat-slice metric that
frame is orthonormal and orthogonal to the observer. A ray is followed until
it is ``R_sky`` from the bubble centre, where ``f`` is zero to double
precision and its direction no longer changes. The direction it came from
there is where its pixel looks on the sky.

**What does not depend on the integrator.** The metric depends on ``t`` and
``x`` only through ``x - v t``, so ``K = p_t + v p_x`` is conserved exactly.
At the camera ``f = 1`` and ``K`` is minus the energy the observer measures.
Far away ``f = 0``, the static observers are ``d_t``, and ``K`` fixes the
energy they measure. Every ray therefore arrives shifted by exactly

    E_camera / E_sky = 1 - v cos(alpha)

with ``alpha`` the angle between the ray's direction of travel on the sky
and ``+x``. That holds for any ``v``, superluminal included. Light from
straight ahead (``alpha = pi``) arrives blueshifted by ``1 + v``. The
tests hold the traced rays to it, and to the other two invariants:
- ``H = 0``, because the ray is null
- ``y p_z - z p_y``, the angular momentum about the axis of motion

Above ``v = 1`` some directions have no ray at all. A backward ray from them
stays in the wall, and those pixels are marked ``trapped`` rather than
painted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Bubble:
    """Alcubierre's bubble: speed ``v``, radius ``R`` and wall steepness ``sigma``."""

    v: float = 2.0
    R: float = 5.0
    sigma: float = 2.0

    def shape(self, r: np.ndarray) -> np.ndarray:
        """``f(r) = (tanh(sigma (r + R)) - tanh(sigma (r - R))) / (2 tanh(sigma R))``."""
        s, R = self.sigma, self.R
        return (np.tanh(s * (r + R)) - np.tanh(s * (r - R))) / (2.0 * np.tanh(s * R))

    def slope(self, r: np.ndarray) -> np.ndarray:
        """``f'(r)``, from ``d tanh(u)/du = 1 - tanh(u)^2``."""
        s, R = self.sigma, self.R
        plus, minus = np.tanh(s * (r + R)), np.tanh(s * (r - R))
        return s * ((1.0 - plus**2) - (1.0 - minus**2)) / (2.0 * np.tanh(s * R))

    @property
    def sky(self) -> float:
        """Where the sky starts: ``f`` and ``f'`` are below 1e-17 from here out."""
        return self.R + 20.0 / self.sigma


def _rates(bubble: Bubble, state: np.ndarray) -> np.ndarray:
    """Hamilton's equations for every ray; ``state`` is ``(8, N)``: ``t x y z p_t p_x p_y p_z``."""
    t, x, y, z, pt, px, py, pz = state
    v = bubble.v
    xs = x - v * t
    r = np.sqrt(xs * xs + y * y + z * z)
    f = bubble.shape(r)
    safe = np.where(r > 0, r, 1.0)
    slope = np.where(r > 0, bubble.slope(r) / safe, 0.0)  # f'(r) / r
    W = pt + v * f * px
    force = v * W * px * slope
    return np.array(
        [
            -W,
            px - v * f * W,
            py,
            pz,
            force * (-v * xs),
            force * xs,
            force * y,
            force * z,
        ]
    )


def camera_momenta(bubble: Bubble, directions: np.ndarray, shape: float = 1.0) -> np.ndarray:
    """Backward-traced covariant momenta for light arriving from ``directions``.

    ``directions`` is ``(3, N)``: unit vectors, in the Eulerian observer's
    frame, toward the sky position each pixel sees. The light itself travels
    along ``-d`` with unit energy, ``p^a = n^a - d^a`` with ``n = (1, v f, 0, 0)``,
    and the traced ray is its reverse, ``-p``. Lowered with the metric where
    ``f`` is ``shape`` (1 at the bubble centre), that is
    ``(1 - v f d_x, d_x, d_y, d_z)``.
    """
    d = np.asarray(directions, dtype=float)
    return np.array([1.0 - bubble.v * shape * d[0], d[0], d[1], d[2]])


@dataclass(frozen=True)
class RayTrace:
    """Where each pixel's light came from, and what happened to it on the way."""

    directions: np.ndarray  # (3, N): the camera's view directions
    sky: np.ndarray  # (3, N): the direction on the sky the light came from
    doppler: np.ndarray  # (N,): E_camera / E_sky
    trapped: np.ndarray  # (N,): no ray reached the sky
    drift: dict[str, float]  # the largest change of each invariant
    steps: int


def trace(
    bubble: Bubble,
    directions: np.ndarray,
    step: float = 0.01,
    max_steps: int = 200000,
    blueshift: float = 1e6,
    camera: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> RayTrace:
    """Trace a ray back from the bubble centre for each direction, to the sky.

    ``camera`` is the observer's position at ``t = 0``, the bubble centre by
    default. Classical fourth-order Runge-Kutta, with each ray's affine step set by how
    far it moves relative to the bubble, not by the affine parameter. At
    ``v = 2`` a ray crosses coordinates at up to three times the speed of
    light, and a step sized in the affine parameter alone jumps the whole
    wall: the first version of this did, and drifted ``H`` to ``3e8``. A step
    moves a ray at most ``step`` in the wall, ``0.3 |r - R|`` near it, and 2
    far from it. Each step is a fixed function of the state at its start, so
    every ray is still integrated at fourth order.
    """
    d = np.asarray(directions, dtype=float)
    d = d / np.linalg.norm(d, axis=0)
    n = d.shape[1]
    state = np.zeros((8, n))
    state[1:4] = np.asarray(camera, dtype=float)[:, None]
    state[4:] = camera_momenta(bubble, d, float(bubble.shape(np.linalg.norm(camera))))
    K0 = state[4] + bubble.v * state[5]
    L0 = state[2] * state[7] - state[3] * state[6]
    active = np.ones(n, dtype=bool)
    arrived = np.zeros(n, dtype=bool)
    taken = 0
    while active.any() and taken < max_steps:
        taken += 1
        s = state[:, active]
        k1 = _rates(bubble, s)
        r = np.sqrt((s[1] - bubble.v * s[0]) ** 2 + s[2] ** 2 + s[3] ** 2)
        # Coordinate speed relative to the moving bubble.
        speed = np.sqrt((k1[1] - bubble.v * k1[0]) ** 2 + k1[2] ** 2 + k1[3] ** 2)
        # f varies as exp(-2 sigma |r - R|) off the wall, and so does every
        # derivative Runge-Kutta's error is made of. A reach growing as
        # exp(0.4 sigma |r - R|) keeps the error per step falling off the wall.
        reach = np.minimum(step * np.exp(0.4 * bubble.sigma * np.abs(r - bubble.R)), 2.0)
        h = reach / np.maximum(speed, 1e-300)
        k2 = _rates(bubble, s + 0.5 * h * k1)
        k3 = _rates(bubble, s + 0.5 * h * k2)
        k4 = _rates(bubble, s + h * k3)
        s = s + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        state[:, active] = s
        r = np.sqrt((s[1] - bubble.v * s[0]) ** 2 + s[2] ** 2 + s[3] ** 2)
        done = r > bubble.sky
        # Traced back towards a horizon, a ray's energy grows without bound:
        # no light of finite energy on the sky reaches that pixel.
        lost = np.abs(s[4]) > blueshift
        index = np.flatnonzero(active)
        arrived[index[done & ~lost]] = True
        active[index[done | lost]] = False
    drift = {}
    t, x, y, z, pt, px, py, pz = state
    f = bubble.shape(np.sqrt((x - bubble.v * t) ** 2 + y**2 + z**2))
    W = pt + bubble.v * f * px
    # Each invariant's change, relative to the size of the ray's momentum.
    scale = np.maximum(np.abs(pt), 1.0)
    ok = arrived
    drift["H"] = float(
        (np.abs(0.5 * (-(W**2) + px**2 + py**2 + pz**2)) / scale**2)[ok].max(initial=0)
    )
    drift["K"] = float((np.abs(pt + bubble.v * px - K0) / scale)[ok].max(initial=0))
    drift["L"] = float((np.abs(y * pz - z * py - L0) / scale)[ok].max(initial=0))
    # Off the bubble the metric is flat, p^i = p_i, and the traced (reversed)
    # ray's spatial momentum points back along where the light came from.
    sky = np.array([px, py, pz])
    sky = sky / np.linalg.norm(sky, axis=0)
    # The light's energy for a static observer there is p_t of the traced ray.
    doppler = 1.0 / pt
    trapped = ~arrived
    doppler[trapped] = np.nan
    sky[:, trapped] = np.nan
    return RayTrace(d, sky, doppler, trapped, drift, taken)


def panorama(width: int, height: int) -> np.ndarray:
    """View directions for an equirectangular image: longitude from ``+x``, then latitude.

    The centre of the image looks straight ahead along ``+x``, the direction
    of travel, and the left and right edges look straight back.
    """
    longitude = (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi
    latitude = np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi
    lon, lat = np.meshgrid(longitude, latitude)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]).reshape(
        3, -1
    )


def pinhole(width: int, height: int, field_of_view: float = 100.0) -> np.ndarray:
    """View directions for a pinhole camera looking along ``+x``.

    ``field_of_view`` is the angle across the image's width, in degrees.
    """
    scale = np.tan(np.radians(field_of_view) / 2)
    u = ((np.arange(width) + 0.5) / width * 2 - 1) * scale
    w = (1 - (np.arange(height) + 0.5) / height * 2) * scale * height / width
    U, V = np.meshgrid(u, w)
    d = np.array([np.ones_like(U), U, V]).reshape(3, -1)
    return d / np.linalg.norm(d, axis=0)
