"""Figures for collapse runs and hypothesis comparisons."""

import numpy as np
import pytest

from particlesim.analysis import penrose
from particlesim.scenarios.singularity.harness import collapse_solutions, evaluate, write_report
from particlesim.theories import get_theory
from particlesim.viz.singularity_views import (
    bounce_comparison,
    collapse_spacetime_diagram,
    hypothesis_figures,
    invariants_vs_time,
    penrose_diagram,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_spacetime_diagram_renders():
    times = np.linspace(0.0, 5.0, 12)
    radii = np.linspace(0.1, 10.0, 30)
    compactness = np.outer(np.linspace(0.0, 0.9, 12), np.exp(-radii / 4))
    horizons = [None] * 6 + [2.0] * 6
    png = collapse_spacetime_diagram(times, radii, compactness, horizons)
    assert png.startswith(PNG_MAGIC)


def test_spacetime_diagram_without_a_horizon_track():
    times = np.linspace(0.0, 1.0, 4)
    radii = np.linspace(0.1, 3.0, 8)
    png = collapse_spacetime_diagram(times, radii, np.zeros((4, 8)))
    assert png.startswith(PNG_MAGIC)
    # An all-None track is the same as none at all, and must not raise.
    assert collapse_spacetime_diagram(times, radii, np.zeros((4, 8)), [None] * 4)


def test_invariants_plot_handles_zeros_and_many_decades():
    """A logarithmic axis must survive exact zeros in the data, which occur
    whenever a run starts from flat initial data."""
    t = np.linspace(0.0, 1.0, 50)
    series = {"kretschmann": np.concatenate([[0.0], np.logspace(-10, 12, 49)])}
    png = invariants_vs_time(t, series)
    assert png.startswith(PNG_MAGIC)


def test_bounce_comparison_renders_both_curves():
    t = np.linspace(0.0, 10.0, 40)
    png = bounce_comparison(t, np.exp(-t), t, 0.3 + np.exp(-t))
    assert png.startswith(PNG_MAGIC)


def test_penrose_diagram_renders_labelled_curves():
    t = np.linspace(-20.0, 20.0, 30)
    big_t, x = penrose.schwarzschild(t, np.full_like(t, 2.0), 1.0)
    sing_t, sing_x = penrose.schwarzschild(t, np.full_like(t, 1e-9), 1.0)
    png = penrose_diagram({"horizon": (x, big_t), "singularity": (sing_x, sing_t)})
    assert png.startswith(PNG_MAGIC)


def test_collapse_solutions_gives_a_baseline_to_compare_against():
    """The corrected run alone proves nothing; the pair is the evidence."""
    baseline, corrected = collapse_solutions(get_theory("lqg.lqc"))
    assert baseline.outcome == "singularity"
    assert corrected.outcome == "turning_point"
    assert corrected.min_scale_factor > baseline.min_scale_factor


def test_hypothesis_figures_for_a_bouncing_theory():
    theory = get_theory("lqg.lqc")
    card = evaluate(theory)
    baseline, corrected = collapse_solutions(theory)
    figs = hypothesis_figures(card, baseline, corrected)
    assert set(figs) == {
        "Collapse with and without the correction",
        "Density during collapse",
    }
    for png in figs.values():
        assert png.startswith(PNG_MAGIC)


def test_hypothesis_figures_are_empty_without_solutions():
    card = evaluate(get_theory("gr"))
    assert hypothesis_figures(card) == {}


def test_written_report_is_self_contained(tmp_path):
    import re

    theory = get_theory("lqg.lqc")
    path = write_report(evaluate(theory), tmp_path / "card.html", theory)
    text = path.read_text()
    assert "lqg.lqc" in text
    assert "data:image/png;base64," in text
    assert not re.search(r'(src|href)=["\']https?://', text)


def test_cli_writes_a_report(tmp_path, capsys):
    from particlesim.cli import main

    out = tmp_path / "report.html"
    assert main(["hypothesis", "lqg.lqc", "--report", str(out)]) == 0
    assert out.exists()
    assert "SURVIVED" in capsys.readouterr().out
    assert out.stat().st_size > 10_000


def test_compactness_colour_scale_is_pinned_to_its_physical_range():
    """Autoscaling would make a weak-field run look near-horizon, which is the
    single most misleading thing this plot could do."""
    import matplotlib

    matplotlib.use("Agg")
    weak = collapse_spacetime_diagram(
        np.linspace(0, 1, 4), np.linspace(0.1, 3, 8), np.full((4, 8), 0.01)
    )
    strong = collapse_spacetime_diagram(
        np.linspace(0, 1, 4), np.linspace(0.1, 3, 8), np.full((4, 8), 0.99)
    )
    # Different data must give visibly different images under a fixed scale.
    assert weak != strong
    assert pytest.approx(len(weak), rel=2.0) == len(strong)
