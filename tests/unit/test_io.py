import json
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from particlesim.core.grid import UniformGrid
from particlesim.core.io import (
    TimeSeriesWriter,
    load_fields,
    read_time_series,
    save_fields,
    write_xdmf,
)


def test_fields_round_trip_with_ghosts(tmp_path):
    grid = UniformGrid([(-1.0, 1.0), (0.0, 2.0), (0.0, 1.0)], (4, 5, 6), ghost=1)
    X, Y, Z = grid.coords()
    fields = {"rho": X * Y + Z, "S": np.stack([X, Y, Z])}
    path = save_fields(tmp_path / "f.h5", fields, grid, attrs={"family": "test", "n": 3})
    loaded, g2, attrs = load_fields(path)
    assert g2.shape == grid.shape and g2.extent == grid.extent
    np.testing.assert_array_equal(loaded["rho"], grid.interior(fields["rho"]))
    assert loaded["S"].shape == (3, 4, 5, 6)
    assert attrs["family"] == "test" and int(attrs["n"]) == 3


def test_xdmf_sidecar_references_datasets(tmp_path):
    grid = UniformGrid([(-1.0, 1.0)] * 3, (3, 4, 5))
    X, Y, Z = grid.coords()
    fields = {"rho": X, "S": np.stack([X, Y, Z])}
    h5 = save_fields(tmp_path / "f.h5", fields, grid, xdmf=True)
    xmf = h5.with_suffix(".xmf")
    root = ET.parse(xmf).getroot()
    topo = root.find(".//Topology")
    assert topo.get("Dimensions") == "6 5 4"  # nz+1 ny+1 nx+1
    names = {a.get("Name"): a for a in root.findall(".//Attribute")}
    assert set(names) == {"rho", "S"}
    assert names["S"].get("AttributeType") == "Vector"
    assert names["rho"].find("DataItem").text.endswith(":/xdmf/rho")
    # Standalone writer with an explicit path works too.
    out = write_xdmf(h5, grid, fields, tmp_path / "other.xmf")
    assert out.exists()


def test_time_series_append_and_read(tmp_path):
    path = tmp_path / "ts.h5"
    with TimeSeriesWriter(path, ["step", "time", "mass"]) as w:
        for i in range(5):
            w.append(step=i, time=0.1 * i, mass=1.0)
    with TimeSeriesWriter(path, ["step", "time", "mass"]) as w:
        w.append(step=5, time=0.5, mass=0.9)
    ts = read_time_series(path)
    assert ts["step"].tolist() == [0, 1, 2, 3, 4, 5]
    assert ts["mass"][-1] == 0.9


def test_export_reduced_csv_flattens_nested_report(tmp_path):
    import csv

    from particlesim.core.io import export_reduced_csv, flatten_report

    report = {"a": 1, "b": {"c": 2.5, "d": {"e": "x"}}, "f": [1, 2]}
    flat = flatten_report(report)
    assert flat == {"a": 1, "b.c": 2.5, "b.d.e": "x", "f": "[1, 2]"}
    path = export_reduced_csv(report, tmp_path / "r.csv")
    rows = list(csv.reader(path.open()))
    assert rows[0] == ["key", "value"] and ["b.d.e", "x"] in rows


def test_units_and_dimensions_round_trip(tmp_path):
    from particlesim.core import units as u
    from particlesim.core.io import SCHEMA_VERSION

    grid = UniformGrid([(-1.0, 1.0)] * 3, (4, 4, 4))
    geo = u.geometric_solar_mass()
    path = save_fields(
        tmp_path / "f.h5",
        {"energy_density": np.ones((4, 4, 4))},
        grid,
        unit_system=geo,
        dimensions={"energy_density": u.ENERGY_DENSITY},
    )
    _, _, meta = load_fields(path)
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["unit_system"].name == geo.name
    assert meta["unit_system"].length_m == pytest.approx(geo.length_m)
    assert meta["dimensions"]["energy_density"] == tuple(float(e) for e in u.ENERGY_DENSITY)


def test_units_are_unrecorded_not_guessed_when_not_supplied(tmp_path):
    grid = UniformGrid([(-1.0, 1.0)] * 3, (4, 4, 4))
    path = save_fields(tmp_path / "f.h5", {"rho": np.zeros((4, 4, 4))}, grid)
    _, _, meta = load_fields(path)
    assert meta["unit_system"] is None
    assert meta["dimensions"] == {}


def test_version_1_checkpoint_migrates_forward_without_inventing_units(tmp_path):
    """A v1 file recorded no units and nothing can recover them, so the
    migration must say unknown rather than assume a system."""
    import h5py

    grid = UniformGrid([(-2.0, 2.0)] * 3, (3, 3, 3))
    path = tmp_path / "old.h5"
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = 1
        f.attrs["extent"] = json.dumps(grid.extent)
        f.attrs["shape"] = json.dumps(list(grid.shape))
        f.attrs["spacing"] = json.dumps(list(grid.spacing))
        f.attrs["axis_names"] = json.dumps(list(grid.axis_names))
        f.attrs["family"] = "alcubierre"
        f.create_group("fields").create_dataset("rho", data=np.full((3, 3, 3), 2.0))

    fields, g, meta = load_fields(path)
    np.testing.assert_allclose(fields["rho"], 2.0)
    assert g.shape == (3, 3, 3)
    assert meta["family"] == "alcubierre"
    assert meta["migrated_from"] == 1
    assert meta["unit_system"] is None
    assert any("unknown, not assumed" in n for n in meta["migration_notes"])


def test_future_schema_version_is_refused_rather_than_misread(tmp_path):
    import h5py

    from particlesim.core.io import SCHEMA_VERSION, SchemaMigrationError

    grid = UniformGrid([(-1.0, 1.0)] * 3, (2, 2, 2))
    path = tmp_path / "future.h5"
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION + 5
        f.attrs["extent"] = json.dumps(grid.extent)
        f.attrs["shape"] = json.dumps(list(grid.shape))
        f.attrs["axis_names"] = json.dumps(list(grid.axis_names))
        f.create_group("fields")
    with pytest.raises(SchemaMigrationError, match="newer than this build"):
        load_fields(path)


def test_missing_migration_is_an_error_not_a_silent_pass(monkeypatch, tmp_path):
    from particlesim.core import io

    monkeypatch.setattr(io, "MIGRATIONS", {})
    with pytest.raises(io.SchemaMigrationError, match="no migration registered"):
        io._apply_migrations(1, {})
