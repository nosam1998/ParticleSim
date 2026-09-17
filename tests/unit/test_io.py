from xml.etree import ElementTree as ET

import numpy as np

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
