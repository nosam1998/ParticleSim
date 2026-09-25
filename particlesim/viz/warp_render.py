"""What an observer inside an Alcubierre bubble sees (issue #55).

Each pixel's direction is traced back through the bubble by
:mod:`particlesim.analysis.raytrace` to the patch of sky its light came from.
That patch is painted from a reference sky, so the picture's distortion is the
bubble's lensing and nothing else:
- a checkerboard in longitude and latitude
- warm hues ahead of the ship, cool ones behind
- a grid line every 30 degrees

The light's frequency shift, exactly ``1 - v cos(alpha)``, tints the pixel:
blue where the light arrives with more energy than it left with, red where it
arrives with less, and brighter as the Doppler factor to the fourth, as the
bolometric intensity transforms. Pixels no ray reaches, behind a
superluminal bubble's horizon, are black.

:func:`png_bytes` writes the image without matplotlib: the format is five
chunks and a zlib stream.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

from particlesim.analysis.raytrace import Bubble, RayTrace, panorama, pinhole, trace


def sky_colour(directions: np.ndarray) -> np.ndarray:
    """The reference sky's RGB in ``[0, 1]`` for each direction, shape ``(N, 3)``."""
    d = np.asarray(directions, dtype=float)
    lon = np.arctan2(d[1], d[0])
    lat = np.arcsin(np.clip(d[2], -1.0, 1.0))
    checker = ((np.floor(lon / (np.pi / 12)) + np.floor(lat / (np.pi / 12))) % 2).astype(float)
    ahead = 0.5 * (1.0 + np.cos(lon) * np.cos(lat))  # 1 straight ahead, 0 straight behind
    warm, cool = np.array([0.95, 0.55, 0.25]), np.array([0.25, 0.55, 0.95])
    base = ahead[:, None] * warm + (1.0 - ahead[:, None]) * cool
    shade = 0.55 + 0.45 * checker
    rgb = base * shade[:, None]
    # A thin line every 30 degrees in longitude and latitude.
    step = np.pi / 6
    meridian = np.abs(lon / step - np.round(lon / step)) * np.cos(lat)
    meridian[np.abs(lat) > np.radians(80.0)] = 1.0  # meridians converge; stop short of the poles
    near = np.minimum(meridian, np.abs(lat / step - np.round(lat / step)))
    rgb[near < 0.02] = 0.95
    return rgb


def shaded(trace_result: RayTrace, doppler: bool = True) -> np.ndarray:
    """RGB in ``[0, 1]`` for each pixel of a trace, shape ``(N, 3)``."""
    rgb = np.zeros((trace_result.sky.shape[1], 3))
    ok = ~trace_result.trapped
    rgb[ok] = sky_colour(trace_result.sky[:, ok])
    if doppler:
        D = trace_result.doppler[ok]
        blue, red = np.array([0.35, 0.55, 1.0]), np.array([1.0, 0.35, 0.25])
        shift = np.tanh(np.log(D))[:, None]  # -1 (red) .. 1 (blue)
        tint = np.where(shift > 0, blue, red)
        rgb[ok] = (1 - 0.5 * np.abs(shift)) * rgb[ok] + 0.5 * np.abs(shift) * tint
        rgb[ok] *= np.clip(D**4, 0.15, 3.0)[:, None] ** 0.25  # compressed D^4
    return np.clip(rgb, 0.0, 1.0)


def render(
    bubble: Bubble,
    width: int = 256,
    height: int = 128,
    view: str = "panorama",
    field_of_view: float = 100.0,
    doppler: bool = True,
    step: float = 0.02,
) -> tuple[np.ndarray, RayTrace]:
    """The view from the bubble's centre: an ``(height, width, 3)`` uint8 image and its trace.

    ``view`` is ``"panorama"``, which is equirectangular with straight ahead
    in the middle, or ``"pinhole"``, looking ahead across ``field_of_view``
    degrees.
    """
    if view == "panorama":
        directions = panorama(width, height)
    elif view == "pinhole":
        directions = pinhole(width, height, field_of_view)
    else:
        raise ValueError(f"view must be 'panorama' or 'pinhole', not {view!r}")
    result = trace(bubble, directions, step=step)
    image = (shaded(result, doppler) * 255 + 0.5).astype(np.uint8).reshape(height, width, 3)
    return image, result


def png_bytes(image: np.ndarray) -> bytes:
    """An 8-bit RGB PNG of an ``(height, width, 3)`` uint8 array."""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected an (height, width, 3) RGB array")
    height, width, _ = image.shape

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))  # filter 0
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows, 9))
        + chunk(b"IEND", b"")
    )
