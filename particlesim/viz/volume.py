"""Three-dimensional views of a field on a grid (issue #55).

Three routes, for three places a 3-D field is looked at:

**In a browser, with no graphics on the server.** :func:`volume_figure`
builds a Plotly figure with isosurfaces and three orthogonal slices. The
browser does the marching cubes in WebGL, so the served app and a static page
show a volume without the server ever rendering one. Only ``plotly`` is needed
for this.

**As geometry.** :func:`to_pyvista`, :func:`isosurface` and :func:`slices`
hand the field to PyVista (the ``3d`` extra) for contours, cuts and meshes that
can be measured or saved. None of them renders. VTK's off-screen rendering
needs EGL or OSMesa, which a server or a CI runner usually lacks; without them
it crashes rather than failing.

**As a file.** :func:`write_vti` writes VTK's XML ``ImageData`` format itself,
so ParaView, VisIt and PyVista open a run's fields with no VTK installed where
the file was written.

The fields are :class:`~particlesim.core.grid.UniformGrid` arrays, sampled at
cell centres, indexed ``[i, j, k]`` along ``x, y, z``.
"""

from __future__ import annotations

import base64
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


def _geometry(grid) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Origin (the first cell centre), spacing and point dimensions."""
    lower = np.array([lo for lo, _ in grid.extent], dtype=float)
    spacing = np.asarray(grid.spacing, dtype=float)
    return lower + 0.5 * spacing, spacing, tuple(int(n) for n in grid.shape)


def _check(fields: Mapping[str, np.ndarray], grid) -> None:
    for name, values in fields.items():
        if np.shape(values) != tuple(grid.shape):
            raise ValueError(f"{name} has shape {np.shape(values)}, the grid {tuple(grid.shape)}")


# --- as a file ---------------------------------------------------------------------


def _numbers(values) -> str:
    """Plain Python ``repr`` of each value, exact and free of numpy's type names."""
    return " ".join(repr(float(v)) for v in values)


def write_vti(path: str | Path, fields: Mapping[str, np.ndarray], grid) -> Path:
    """The fields as a VTK XML ``ImageData`` file (``.vti``), one point array each.

    Written by hand: base64-encoded little-endian doubles, each block preceded
    by its byte count as VTK's ``UInt32`` header, with ``x`` varying fastest.
    """
    _check(fields, grid)
    origin, spacing, dims = _geometry(grid)
    extent = " ".join(f"0 {n - 1}" for n in dims)
    arrays = []
    for name, values in fields.items():
        data = np.asarray(values, dtype="<f8").ravel(order="F").tobytes()
        encoded = base64.b64encode(struct.pack("<I", len(data)) + data).decode("ascii")
        arrays.append(
            f'        <DataArray type="Float64" Name="{name}" format="binary">{encoded}</DataArray>'
        )
    text = "\n".join(
        [
            '<?xml version="1.0"?>',
            '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" '
            'header_type="UInt32">',
            f'  <ImageData WholeExtent="{extent}" Origin="{_numbers(origin)}" '
            f'Spacing="{_numbers(spacing)}">',
            f'    <Piece Extent="{extent}">',
            f'      <PointData Scalars="{next(iter(fields), "")}">',
            *arrays,
            "      </PointData>",
            "    </Piece>",
            "  </ImageData>",
            "</VTKFile>",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    return path


# --- as geometry (PyVista) ---------------------------------------------------------


def to_pyvista(fields: Mapping[str, np.ndarray], grid):
    """A ``pyvista.ImageData`` holding the fields as point data."""
    import pyvista as pv

    _check(fields, grid)
    origin, spacing, dims = _geometry(grid)
    image = pv.ImageData(dimensions=dims, spacing=tuple(spacing), origin=tuple(origin))
    for name, values in fields.items():
        image.point_data[name] = np.asarray(values, dtype=float).ravel(order="F")
    return image


def isosurface(field: np.ndarray, grid, level: float) -> tuple[np.ndarray, np.ndarray]:
    """``(points, triangles)`` of the surface ``field = level``, by PyVista's contour."""
    surface = to_pyvista({"field": field}, grid).contour([float(level)], scalars="field")
    surface = surface.triangulate()
    faces = surface.faces.reshape(-1, 4)[:, 1:] if surface.n_cells else np.empty((0, 3), int)
    return np.asarray(surface.points), np.asarray(faces)


def slices(field: np.ndarray, grid, point: Sequence[float] | None = None):
    """The three orthogonal cuts through ``point`` (the centre by default), as PyVista meshes."""
    image = to_pyvista({"field": field}, grid)
    x, y, z = image.center if point is None else point
    return image.slice_orthogonal(x=x, y=y, z=z)


# --- in a browser (Plotly) ---------------------------------------------------------


def volume_figure(
    field: np.ndarray,
    grid,
    levels: Sequence[float] | None = None,
    title: str = "",
    label: str = "value",
    colorscale: str = "RdBu",
):
    """Isosurfaces of ``field`` at ``levels``, and its ``z = 0`` slice, as a Plotly figure.

    The default levels are a quarter, a half and three quarters of the way to
    the field's most negative value. For a warp bubble's energy density, that
    shows where the negative energy sits.
    """
    import plotly.graph_objects as go

    _check({"field": field}, grid)
    x, y, z = (np.asarray(c, dtype=float) for c in grid.coords())
    values = np.asarray(field, dtype=float)
    if levels is None:
        low = float(values.min())
        levels = [f * low for f in (0.75, 0.5, 0.25)] if low < 0 else [float(values.max()) / 2]
    levels = sorted(levels)
    bound = float(np.abs(values).max()) or 1.0
    iso = go.Isosurface(
        x=x.ravel(),
        y=y.ravel(),
        z=z.ravel(),
        value=values.ravel(),
        isomin=levels[0],
        isomax=levels[-1],
        surface=dict(count=len(levels)),
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorscale=colorscale,
        cmin=-bound,
        cmax=bound,
        opacity=0.7,
        colorbar=dict(title=label),
        name="isosurfaces",
    )
    middle = int(np.argmin(np.abs(z[0, 0, :])))
    slab = go.Surface(
        x=x[:, :, middle],
        y=y[:, :, middle],
        z=z[:, :, middle],
        surfacecolor=values[:, :, middle],
        colorscale=colorscale,
        cmin=-bound,
        cmax=bound,
        showscale=False,
        opacity=0.45,
        name="z slice",
    )
    figure = go.Figure(data=[iso, slab])
    figure.update_layout(
        title=title,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
        margin=dict(l=0, r=0, t=40 if title else 10, b=0),
        height=560,
    )
    return figure
