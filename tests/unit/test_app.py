"""The served app: runs and modified theories, live (issue #83)."""

from __future__ import annotations

import json
import socket
import time
import urllib.request

import numpy as np
import pytest

from particlesim.viz.app import (
    _frame,
    fofr_enhancement,
    fofr_figure,
    hypothesis_page,
    run_dashboards,
)

# --- the computations, without a browser ---------------------------------------


def test_the_live_fofr_view_is_the_linear_theory_the_nbody_was_checked_against():
    """F5 today: the 22% ``P(k)`` enhancement at ``k = 0.3`` that #180 measured
    the N-body against, and no enhancement on scales far above the Compton
    wavelength."""
    result = fofr_enhancement(1e-5, 0.3, 1.0, np.array([1e-3, 0.3]))
    assert result["power"][0] == pytest.approx(1.0, abs=1e-4)
    assert result["power"][1] == pytest.approx(1.2225, abs=1e-3)
    np.testing.assert_allclose(result["power"], result["growth"] ** 2)
    # 1 / (a m) is 7.6 Mpc/h for F5 today.
    assert 1.0 / result["compton"] == pytest.approx(7.6, abs=0.1)


def test_the_enhancement_grows_with_the_field_and_vanishes_without_it():
    k = np.array([0.1, 0.5])
    weak, strong = (fofr_enhancement(f, wavenumbers=k)["power"] for f in (1e-7, 1e-4))
    assert np.all(strong > weak) and np.all(weak > 1.0)
    assert fofr_enhancement(1e-12, wavenumbers=k)["power"] == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(ValueError, match="scale factor"):
        fofr_enhancement(1e-5, scale=0.0)


def test_the_fofr_figure_is_drawn_without_pyplot():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    fig = fofr_figure(1e-6, 0.25, 0.5)
    assert plt.get_fignums() == before  # nothing left in pyplot's registry
    assert "0.25" in fig.axes[0].get_title() and len(fig.axes[0].lines) == 3


def test_a_theory_plugins_report_is_computed_and_framed():
    pytest.importorskip("matplotlib")
    page = hypothesis_page("lqg.lqc")
    assert "Hypothesis report: lqg.lqc" in page and "data:image/png;base64," in page
    framed = _frame(page)
    assert framed.startswith("<iframe srcdoc=") and "sandbox" in framed
    assert "<h1>" not in framed  # the page is escaped into the attribute


def test_runs_are_found_and_given_dashboards(tmp_path):
    for name in ("a", "nested/b"):
        run = tmp_path / name
        run.mkdir(parents=True)
        manifest = {"config": {"scenario": "warp.analyze"}, "seed": 0}
        (run / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "not_a_run").mkdir()
    found = run_dashboards(tmp_path)
    assert list(found) == ["a", "nested/b"]
    assert all(page.is_file() for page in found.values())
    assert run_dashboards(None) == {}


# --- the app, served --------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_the_app_builds_with_its_three_tabs(tmp_path):
    pytest.importorskip("panel")
    from particlesim.viz.app import build_app

    app = build_app(tmp_path)
    assert [title for title, _ in zip(app._names, app, strict=True)] == [
        "Runs",
        "Modified gravity, live",
        "Theory plugins, live",
    ]


def test_the_app_is_served(tmp_path):
    """``particlesim serve``'s server answers with the Bokeh document."""
    pytest.importorskip("panel")
    from particlesim.viz.app import serve

    port = _free_port()
    server = serve(tmp_path, port=port, threaded=True)
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
                    page = response.read().decode()
                    status = response.status
                break
            except OSError:
                time.sleep(0.1)
        else:
            pytest.fail("the server never answered")
        assert status == 200
        assert "<title>ParticleSim</title>" in page and "bokeh" in page.lower()
    finally:
        server.stop()
