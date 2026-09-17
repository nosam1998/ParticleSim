import numpy as np
import pytest

from particlesim.scenarios.warp.metrics import make_metric
from particlesim.viz.widgets import (
    INTERACTIVE_FAMILIES,
    explorer_state,
    warp_explorer,
)

widgets = pytest.importorskip("ipywidgets")


def test_explorer_state_matches_the_closed_form():
    """The slider path must agree with the published Alcubierre density, or
    the panel is showing something other than the physics."""
    state = explorer_state("alcubierre", v_s=2.0, R=5.0, sigma=2.0, n=32, half_width=12.0)
    m = make_metric("alcubierre", {"v_s": 2.0, "R": 5.0, "sigma": 2.0})
    X, Y, Z = state.grid.coords()
    expected = m.closed_form_energy_density()(np.zeros_like(X), X, Y, Z)
    np.testing.assert_allclose(state.energy_density, expected, atol=1e-13)


def test_summary_reports_expected_quantities():
    s = explorer_state(n=32).summary()
    assert s["min_density"] < 0 < s["negative_fraction"] < 1
    assert s["negative_energy"] <= s["total_energy"] + 1e-12
    assert s["max_density"] <= 1e-12  # Alcubierre density is non-positive


def test_natario_is_offered_and_has_zero_expansion():
    assert set(INTERACTIVE_FAMILIES) >= {"alcubierre", "natario"}
    s = explorer_state("natario", n=32).summary()
    assert s["max_abs_expansion"] < 1e-12


def test_curved_slice_families_are_refused_with_a_reason():
    """Van Den Broeck is not flat-sliced, so the constraint shortcut would be
    silently wrong rather than merely slow."""
    with pytest.raises(ValueError, match="flat slices"):
        explorer_state("van_den_broeck", n=16)
    with pytest.raises(KeyError, match="unknown family"):
        explorer_state("warp_factor_nine", n=16)


def test_slice_is_the_midplane():
    state = explorer_state(n=32)
    assert state.slice_z0.shape == (32, 32)
    np.testing.assert_allclose(state.slice_z0, state.energy_density[:, :, 16])


def test_widget_builds_and_updates_on_change():
    import matplotlib

    matplotlib.use("Agg")
    panel = warp_explorer(n=32)
    sliders = [w for w in panel.children[0].children[0].children]
    names = [getattr(w, "description", "") for w in sliders]
    assert {"family", "v_s", "R", "sigma", "grid"} <= set(names)
    readout = panel.children[1]
    before = readout.value
    assert "negative energy" in before
    # Moving a slider must change the reported numbers, not just redraw.
    v = next(w for w in sliders if w.description == "v_s")
    v.value = 3.0
    assert readout.value != before


def test_example_notebook_runs_top_to_bottom():
    """The notebook is an entry point for new users, so a broken cell is a
    broken front door. Executed here rather than trusted."""
    import json
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    nb_path = Path(__file__).resolve().parents[2] / "examples" / "warp_explorer.ipynb"
    nb = json.loads(nb_path.read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code, "notebook has no code cells"
    namespace: dict = {}
    for i, cell in enumerate(code):
        source = "".join(cell["source"])
        exec(compile(source, f"{nb_path.name}[cell {i}]", "exec"), namespace)  # noqa: S102
    assert "explorer_state" in namespace
