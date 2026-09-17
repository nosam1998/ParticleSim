"""HDF5 field output, time series, and XDMF sidecars (design doc Sections 5.2, 7).

Layout of a fields file:

    /fields/<name>      native (nx, ny, nz) C-order arrays, interior cells only
    /xdmf/<name>        optional float32 copies in (nz, ny, nx) order for ParaView
    attrs: schema_version, extent, shape, spacing, axis_names, plus user attrs

Time series are one resizable 2D dataset with a ``columns`` attribute.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import h5py
import numpy as np

from particlesim.core.grid import UniformGrid

SCHEMA_VERSION = 1


def save_fields(
    path: str | Path,
    fields: dict[str, np.ndarray],
    grid: UniformGrid,
    attrs: dict[str, Any] | None = None,
    xdmf: bool = False,
) -> Path:
    """Write ``fields`` (each with trailing grid axes) to an HDF5 file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["extent"] = json.dumps(grid.extent)
        f.attrs["shape"] = json.dumps(list(grid.shape))
        f.attrs["spacing"] = json.dumps(list(grid.spacing))
        f.attrs["axis_names"] = json.dumps(list(grid.axis_names))
        for k, v in (attrs or {}).items():
            f.attrs[k] = v if isinstance(v, (int, float, str, bool)) else json.dumps(v, default=str)
        g = f.create_group("fields")
        for name, arr in fields.items():
            data = grid.interior(np.asarray(arr))
            g.create_dataset(name, data=data, compression="gzip", compression_opts=4)
        if xdmf:
            gx = f.create_group("xdmf")
            for name, arr in fields.items():
                data = grid.interior(np.asarray(arr))
                if data.ndim == grid.ndim:
                    gx.create_dataset(name, data=np.ascontiguousarray(data.T, dtype=np.float32))
                elif data.ndim == grid.ndim + 1 and data.shape[0] == grid.ndim:
                    # Vector field (3, nx, ny, nz) -> (nz, ny, nx, 3)
                    vec = np.moveaxis(data, 0, -1)
                    vec = np.ascontiguousarray(np.transpose(vec, (2, 1, 0, 3)), dtype=np.float32)
                    gx.create_dataset(name, data=vec)
    if xdmf:
        write_xdmf(path, grid, fields)
    return path


def load_fields(path: str | Path) -> tuple[dict[str, np.ndarray], UniformGrid, dict[str, Any]]:
    with h5py.File(path, "r") as f:
        if int(f.attrs["schema_version"]) != SCHEMA_VERSION:
            raise ValueError(f"unsupported fields schema version {f.attrs['schema_version']}")
        grid = UniformGrid(
            [tuple(e) for e in json.loads(f.attrs["extent"])],
            tuple(json.loads(f.attrs["shape"])),
            axis_names=tuple(json.loads(f.attrs["axis_names"])),
        )
        fields = {name: ds[()] for name, ds in f["fields"].items()}
        skip = {"schema_version", "extent", "shape", "spacing", "axis_names"}
        attrs = {k: v for k, v in f.attrs.items() if k not in skip}
    return fields, grid, attrs


def write_xdmf(
    h5_path: str | Path,
    grid: UniformGrid,
    fields: dict[str, np.ndarray],
    xdmf_path: Path | None = None,
) -> Path:
    """Write an XDMF 3 sidecar describing ``/xdmf/<name>`` datasets on a uniform grid."""
    if grid.ndim != 3:
        raise ValueError("XDMF sidecar supports 3D grids only")
    h5_path = Path(h5_path)
    xdmf_path = xdmf_path or h5_path.with_suffix(".xmf")
    nx, ny, nz = grid.shape
    dx, dy, dz = grid.spacing
    x0, y0, z0 = (lo for lo, _ in grid.extent)

    root = ET.Element("Xdmf", Version="3.0")
    domain = ET.SubElement(root, "Domain")
    gridel = ET.SubElement(domain, "Grid", Name="fields", GridType="Uniform")
    ET.SubElement(
        gridel, "Topology", TopologyType="3DCoRectMesh", Dimensions=f"{nz + 1} {ny + 1} {nx + 1}"
    )
    geom = ET.SubElement(gridel, "Geometry", GeometryType="ORIGIN_DXDYDZ")
    o = ET.SubElement(geom, "DataItem", Dimensions="3", NumberType="Float", Format="XML")
    o.text = f"{z0} {y0} {x0}"
    d = ET.SubElement(geom, "DataItem", Dimensions="3", NumberType="Float", Format="XML")
    d.text = f"{dz} {dy} {dx}"
    for name, arr in fields.items():
        data = grid.interior(np.asarray(arr))
        if data.ndim == 3:
            atype, dims = "Scalar", f"{nz} {ny} {nx}"
        elif data.ndim == 4 and data.shape[0] == 3:
            atype, dims = "Vector", f"{nz} {ny} {nx} 3"
        else:
            continue
        attr = ET.SubElement(gridel, "Attribute", Name=name, AttributeType=atype, Center="Cell")
        item = ET.SubElement(
            attr, "DataItem", Dimensions=dims, NumberType="Float", Precision="4", Format="HDF"
        )
        item.text = f"{h5_path.name}:/xdmf/{name}"
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(xdmf_path, xml_declaration=True, encoding="utf-8")
    return xdmf_path


class TimeSeriesWriter:
    """Append rows of named scalars to a resizable HDF5 dataset."""

    def __init__(self, path: str | Path, columns: list[str], dataset: str = "series"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.columns = list(columns)
        self.dataset = dataset
        self._file = h5py.File(self.path, "a")
        if dataset in self._file:
            self._ds = self._file[dataset]
            stored = json.loads(self._ds.attrs["columns"])
            if stored != self.columns:
                raise ValueError(f"existing columns {stored} differ from {self.columns}")
        else:
            self._ds = self._file.create_dataset(
                dataset, shape=(0, len(columns)), maxshape=(None, len(columns)), dtype="f8"
            )
            self._ds.attrs["columns"] = json.dumps(self.columns)

    def append(self, **values: float) -> None:
        missing = set(self.columns) - set(values)
        if missing:
            raise ValueError(f"missing columns {sorted(missing)}")
        n = self._ds.shape[0]
        self._ds.resize((n + 1, len(self.columns)))
        self._ds[n] = [float(values[c]) for c in self.columns]

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> TimeSeriesWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_time_series(path: str | Path, dataset: str = "series") -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        ds = f[dataset]
        cols = json.loads(ds.attrs["columns"])
        data = ds[()]
    return {c: data[:, i] for i, c in enumerate(cols)}


def flatten_report(report: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dictionaries into ``a.b.c`` keys; lists become JSON strings."""
    out: dict[str, Any] = {}
    for k, v in report.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_report(v, key + "."))
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps(v, default=str)
        else:
            out[key] = v
    return out


def export_reduced_csv(report: dict[str, Any], path: str | Path) -> Path:
    """Write the flattened scalar observables of a report as a two-column CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = flatten_report(report)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in flat.items():
            w.writerow([k, v])
    return path
