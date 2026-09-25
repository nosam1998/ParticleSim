"""A warp bubble under a modified theory, live, and its 3-D views (issue #55)."""

from __future__ import annotations

import base64
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import sympy as sp

from particlesim.analysis import energy_conditions as ec
from particlesim.core.grid import UniformGrid
from particlesim.scenarios.warp.analyze import theory_stress_energy
from particlesim.scenarios.warp.live import live_warp, ready_families, split_theories
from particlesim.theories import get_theory
from particlesim.viz.volume import volume_figure, write_vti

# --- the split, applied as arithmetic --------------------------------------------


def _random_pair(rng, n=5):
    G = rng.normal(size=(4, 4, n))
    G = G + np.swapaxes(G, 0, 1)
    g = np.repeat(np.diag([-1.0, 1.0, 1.0, 1.0])[:, :, None], n, axis=2)
    g = g + 0.1 * (lambda m: m + np.swapaxes(m, 0, 1))(rng.normal(size=(4, 4, n)))
    return G, g


def test_a_theorys_split_applied_to_numbers_is_its_split():
    """GR+Lambda's own ``effective_stress_energy``, on each point's numbers, against
    the vectorised evaluation that the warp analysis now uses."""
    rng = np.random.default_rng(3)
    G, g = _random_pair(rng)
    theory = get_theory("gr.lambda", Lambda=0.3)
    T = theory_stress_energy(theory, G, g)
    for n in range(G.shape[-1]):
        exact = theory.effective_stress_energy(sp.Matrix(G[:, :, n]), sp.Matrix(g[:, :, n]))
        np.testing.assert_allclose(T[:, :, n], np.array(exact, dtype=float), rtol=1e-13)


def test_a_theory_with_general_relativitys_split_gets_exactly_its_array():
    rng = np.random.default_rng(4)
    G, g = _random_pair(rng)
    general = G / (8 * np.pi)
    for theory_id in ("gr", "string.eft4d.dilaton", "string.eft4d.emda"):
        assert theory_stress_energy(get_theory(theory_id), G, g, general) is general
    # Lambda = 0 is general relativity too.
    assert theory_stress_energy(get_theory("gr.lambda", Lambda=0.0), G, g, general) is general


def test_a_split_that_needs_more_than_the_geometry_is_refused():
    class Needy:
        id = "needy"

        def effective_stress_energy(self, einstein, metric):
            return einstein * sp.Symbol("x")

    G, g = _random_pair(np.random.default_rng(5))
    with pytest.raises(ValueError, match="symbolic path"):
        theory_stress_energy(Needy(), G, g)


def test_the_theories_with_a_numeric_split_are_found():
    usable = split_theories()
    assert {"gr", "gr.lambda"} <= set(usable)
    assert "lqg.lqc" not in usable  # Tier B: no split of a 4-metric at all


# --- energy conditions and rounding ------------------------------------------------


def test_rounding_is_not_a_violation_but_a_violation_is():
    """``Lambda g_ab / 8 pi`` on flat space has ``T_ab k^a k^b = 0`` for every null ``k``.

    Evaluated, it is ``+-1e-18``. Counting the negative half of that called most
    of an Alcubierre bubble's vacuum a violation of the null energy condition.
    A genuine negative energy density is still counted.
    """
    n = 200
    rng = np.random.default_rng(6)
    g = np.repeat(np.diag([-1.0, 1.0, 1.0, 1.0])[:, :, None], n, axis=2)
    # A boosted frame makes the null vectors' components unequal, as in a bubble.
    g[0, 1] = g[1, 0] = rng.uniform(-0.3, 0.3, n)
    g[1, 1] = 1.0
    g[0, 0] = -1.0 + g[0, 1] ** 2
    lam = 0.05
    T = lam * g / (8 * np.pi)
    report = ec.evaluate(T, g, 1.0, ("NEC", "WEC"))
    nec = report.results["NEC"]
    assert np.abs(nec.pointwise_min).max() < 1e-16 and np.mean(nec.pointwise_min < 0) > 0.1
    assert nec.violating_fraction == 0.0 and nec.integrated_violation == 0.0
    # Positive Lambda counted as geometry leaves negative Eulerian matter: a real violation.
    assert report.results["WEC"].violating_fraction == 1.0

    dust = np.zeros_like(T)
    dust[0, 0] = -0.01  # negative energy density, genuinely
    assert ec.evaluate(dust, g, 1.0, ("WEC",)).results["WEC"].violating_fraction == 1.0


# --- the live warp view -------------------------------------------------------------


@pytest.fixture(scope="module")
def general_relativity():
    return live_warp("alcubierre", "gr", resolution=16)


def test_lambda_shifts_the_density_and_leaves_the_null_condition_alone(general_relativity):
    """Under GR+Lambda the bubble needs ``(G + Lambda g) / 8 pi``: its Eulerian density
    moves by ``-Lambda / 8 pi`` everywhere, and ``T_ab k^a k^b`` cannot move, since
    ``g_ab k^a k^b = 0``. No cosmological constant rescues the null energy condition."""
    base = general_relativity
    for lam in (0.2, -0.3):
        shifted = live_warp("alcubierre", "gr.lambda", {"Lambda": lam}, resolution=16)
        change = shifted.fields["energy_density"] - base.fields["energy_density"]
        np.testing.assert_allclose(change, -lam / (8 * np.pi), rtol=1e-12)
        np.testing.assert_allclose(
            shifted.fields["NEC_min"], base.fields["NEC_min"], rtol=0, atol=1e-15
        )
        assert shifted.report["energy_conditions"]["NEC"]["integrated_violation"] == pytest.approx(
            base.report["energy_conditions"]["NEC"]["integrated_violation"], rel=1e-12
        )
        # The weak condition moves with Lambda: a negative one fills in the tails.
        wec = shifted.report["energy_conditions"]["WEC"]["violating_fraction"]
        assert (wec < 0.9) if lam < 0 else (wec == 1.0)


def test_a_coupling_change_derives_nothing(general_relativity):
    """The geometry is on disk after the first look, so a new coupling costs one evaluation."""
    result = live_warp("alcubierre", "gr.lambda", {"Lambda": 0.123}, resolution=16)
    assert result.seconds < 5.0
    assert result.report["theory"]["gravity"]["couplings"] == {"Lambda": 0.123}


def test_the_live_families_are_the_ones_ready():
    families = ready_families()
    assert families[0] == "alcubierre" or "alcubierre" in families
    assert set(families) <= {
        "alcubierre",
        "natario",
        "van_den_broeck",
        "lentz",
        "bobrick_martire",
        "fuchs",
    }


# --- volume views -----------------------------------------------------------------


def _sphere(n=20):
    grid = UniformGrid([(-1.0, 1.0)] * 3, (n, n, n))
    x, y, z = grid.coords()
    return grid, x**2 + y**2 + z**2, x


def test_a_vti_file_holds_the_fields_where_vtk_expects_them(tmp_path):
    grid, radius2, x = _sphere(8)
    path = write_vti(tmp_path / "f.vti", {"r2": radius2, "x": x}, grid)
    root = ET.parse(path).getroot()
    image = root.find("ImageData")
    assert image.get("WholeExtent") == "0 7 0 7 0 7"
    np.testing.assert_allclose([float(v) for v in image.get("Origin").split()], [-0.875] * 3)
    np.testing.assert_allclose([float(v) for v in image.get("Spacing").split()], [0.25] * 3)
    arrays = {a.get("Name"): a.text for a in root.iter("DataArray")}
    raw = base64.b64decode(arrays["x"])
    (size,) = struct.unpack("<I", raw[:4])
    values = np.frombuffer(raw[4:], dtype="<f8")
    assert size == 8 * 512 and values.size == 512
    # x varies fastest, as VTK's point order has it.
    np.testing.assert_array_equal(values, x.ravel(order="F"))


def test_pyvista_reads_the_file_and_contours_the_field(tmp_path):
    pv = pytest.importorskip("pyvista")
    from particlesim.viz.volume import isosurface, slices

    grid, radius2, _ = _sphere(24)
    read = pv.read(write_vti(tmp_path / "f.vti", {"r2": radius2}, grid))
    np.testing.assert_array_equal(read.point_data["r2"], radius2.ravel(order="F"))
    points, triangles = isosurface(radius2, grid, 0.5)
    radius = np.sqrt((points**2).sum(axis=1))
    assert radius.mean() == pytest.approx(np.sqrt(0.5), rel=0.01) and triangles.shape[1] == 3
    assert slices(radius2, grid).n_blocks == 3


def test_the_browser_figure_shows_the_negative_energy():
    pytest.importorskip("plotly")
    grid, radius2, _ = _sphere(12)
    field = -np.exp(-4 * radius2)
    figure = volume_figure(field, grid, title="t", label="ρ")
    iso, slab = figure.data
    assert iso.type == "isosurface" and slab.type == "surface"
    assert iso.isomin == pytest.approx(0.75 * field.min()) and iso.isomax == pytest.approx(
        0.25 * field.min()
    )
    assert re.search("ρ", str(iso.colorbar.title.text))
    with pytest.raises(ValueError, match="shape"):
        volume_figure(field[:-1], grid)
