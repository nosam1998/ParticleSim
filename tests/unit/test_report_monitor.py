import base64
import re

import numpy as np
import pytest

from particlesim.core.io import read_time_series
from particlesim.core.monitor import MonitorAbort, make_monitor
from particlesim.viz.plots import warp_figures
from particlesim.viz.report import render_html_report


def test_monitor_records_prints_and_writes_series(tmp_path, capsys):
    path = tmp_path / "ts.h5"
    with make_monitor(["step", "residual"], path=path, every=2) as mon:
        for i in range(5):
            mon.record(step=i, residual=1e-6 * i)
    out = capsys.readouterr().err
    assert "step" in out and "residual" in out
    # every=2 throttles printing but every row is still recorded.
    assert len(re.findall(r"^\s+\d", out, flags=re.M)) == 3
    assert len(mon.rows) == 5
    assert mon.series("step") == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert read_time_series(path)["residual"][-1] == pytest.approx(4e-6)


def test_guard_aborts_a_run_that_has_already_gone_wrong(capsys):
    mon = make_monitor(["step", "constraint"], guards={"constraint": 1.0})
    mon.record(step=0, constraint=0.1)
    with pytest.raises(MonitorAbort, match="constraint"):
        mon.record(step=1, constraint=5.0)
    # The breaching row is kept, so the trace shows what happened.
    assert len(mon.rows) == 2


def test_guard_uses_absolute_value():
    mon = make_monitor(["c"], guards={"c": 1.0})
    with pytest.raises(MonitorAbort):
        mon.record(c=-3.0)


def test_monitor_rejects_missing_columns():
    mon = make_monitor(["a", "b"])
    with pytest.raises(ValueError, match="missing"):
        mon.record(a=1.0)


def test_html_report_is_self_contained(tmp_path):
    report = {
        "family": "alcubierre",
        "eulerian": {"total_energy": -5.65, "min_density": -0.039},
        "energy_conditions": {"WEC": {"min": -1.07, "violating_fraction": 0.81}},
    }
    png = warp_figures(
        {"energy_density": np.random.default_rng(0).normal(size=(8, 8, 8))},
        [(-1.0, 1.0)] * 3,
    )
    path = render_html_report(report, tmp_path / "report.html", png)
    text = path.read_text()
    # No network: every image is inline and there are no external references.
    assert "data:image/png;base64," in text
    assert not re.search(r'(src|href)=["\']https?://', text)
    assert "alcubierre" in text and "-5.65" in text
    assert "prefers-color-scheme" in text
    # Negative numbers are marked so a sign change is visible at a glance.
    assert "num bad" in text


def test_html_report_escapes_untrusted_strings(tmp_path):
    report = {"family": "<script>alert(1)</script>"}
    text = render_html_report(report, tmp_path / "r.html").read_text()
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;" in text


def test_figures_are_valid_pngs(tmp_path):
    figs = warp_figures({"energy_density": np.zeros((6, 6, 6))}, [(-1.0, 1.0)] * 3)
    assert figs
    for png in figs.values():
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        assert base64.b64encode(png)


def test_report_renders_list_sections_not_just_scalars():
    """Regression: list-valued fields were dropped, which on a hypothesis
    report card meant the verdict itself never reached the page."""
    import tempfile
    from pathlib import Path

    report = {
        "theory": "test.x",
        "confirmed": ["bounce", "singularity_resolved"],
        "contradicted": [],
        "battery": [
            {"scenario": "flrw_collapse", "outcome": "turning_point", "max_density": 0.41},
            {"scenario": "flrw_expansion", "outcome": "ran_to_t_max", "max_density": 1e-3},
        ],
    }
    with tempfile.TemporaryDirectory() as d:
        text = render_html_report(report, Path(d) / "r.html").read_text()
    assert "Confirmed" in text and "bounce" in text
    # An empty list still gets a section: "nothing contradicted" is a result.
    assert "Contradicted" in text and "none" in text
    # A list of dicts becomes a table with a column per key.
    assert "Battery" in text
    assert "<th>scenario</th>" in text and "flrw_collapse" in text
    assert "turning_point" in text
