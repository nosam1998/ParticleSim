"""Static dashboards for runs and benchmark runs (issue #83)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from particlesim.cli import main
from particlesim.viz.dashboard import (
    BenchmarkDashboard,
    Outcome,
    documented_benchmarks,
    read_outcomes,
    render_benchmark_dashboard,
    write_outcomes,
    write_run_dashboard,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "docs" / "benchmarks.md"

# The smallest valid PNG: one transparent pixel.
PIXEL = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000105fe02fea70000000049454e44ae426082"
)


# --- what the documentation says -------------------------------------------------


def test_every_test_the_documentation_names_exists():
    """``docs/benchmarks.md`` names tests by function. A rename that leaves the
    documentation behind makes the dashboard show a stale row, so this guards
    the join the dashboard depends on."""
    documented = documented_benchmarks(BENCHMARKS.read_text(encoding="utf-8"))
    assert len(documented) > 90
    source = "\n".join(p.read_text() for p in (ROOT / "tests").rglob("test_*.py"))
    defined = set(re.findall(r"^def (test_\w+)", source, re.MULTILINE))
    assert sorted(set(documented) - defined) == []


def test_documented_rows_carry_their_section_and_columns():
    documented = documented_benchmarks(BENCHMARKS.read_text(encoding="utf-8"))
    (row,) = documented["test_alcubierre_energy_density_matches_closed_form"]
    assert row.section == "Closed-form and exact results"
    assert row.cells["Benchmark"] == "Alcubierre Eulerian energy density"
    assert row.cells["Tolerance"] == "1e-12, achieved 1e-15"


def test_the_table_reader_follows_the_documentations_conventions():
    markdown = "\n".join(
        [
            "## Section A",
            "| Benchmark | Reference | Tolerance | Test |",
            "|---|---|---|---|",
            "| first | r1 | 1e-3 | `test_one` and `test_two` |",
            "| second | r2 | 1e-4 | same |",
            "| third | r3 | 1e-5 | same test |",
            "",
            "| k | measured |",  # no Test column: not a benchmark table
            "|---|---|",
            "| 1 | `test_not_this` |",
            "### Section B",
            "| Benchmark | Test |",
            "|:--|--:|",
            "| fourth | `test_three` |",
        ]
    )
    documented = documented_benchmarks(markdown)
    assert set(documented) == {"test_one", "test_two", "test_three"}
    assert [r.cells["Benchmark"] for r in documented["test_one"]] == ["first", "second", "third"]
    assert documented["test_three"][0].section == "Section B"


# --- the benchmark dashboard -----------------------------------------------------


def _outcomes() -> list[Outcome]:
    return [
        Outcome("tests/a.py::test_one", "test_one", "passed", 0.5, "Checks one."),
        Outcome("tests/a.py::test_two[x]", "test_two", "failed", 1.25, message="assert 1 == 2"),
        Outcome("tests/b.py::test_new", "test_new", "skipped", 0.0, message="needs yt"),
    ]


def test_the_dashboard_joins_results_with_the_documentation(tmp_path):
    markdown = (
        "## S\n| Benchmark | Reference | Tolerance | Measured | Test |\n|---|---|---|---|---|\n"
    )
    markdown += "| Thing <b> | ref `x` | **5%** | −1.7% | `test_one` |\n"
    markdown += "| Other | r | t | m | `test_two` |\n| Gone | r | t | m | `test_slow` |\n"
    dashboard = BenchmarkDashboard(_outcomes(), documented_benchmarks(markdown))
    assert dashboard.counts == {"failed": 1, "error": 0, "passed": 1, "skipped": 1}
    assert [o.name for o in dashboard.undocumented] == ["test_new"]
    assert dashboard.not_run == ["test_slow"]
    page = dashboard.html(generated="fixed")
    # Failures lead, and the documentation's cells arrive escaped, code and bold kept.
    assert page.index("Failing") < page.index("Results")
    assert "Thing &lt;b&gt;" in page and "<code>x</code>" in page and "<strong>5%</strong>" in page
    assert "assert 1 == 2" in page and "needs yt" in page
    assert "<script" not in page and "http" not in page.replace("http-equiv", "")


def test_outcomes_round_trip_through_json_and_are_validated(tmp_path):
    path = write_outcomes(tmp_path / "r.json", _outcomes(), {"python": "3.11"})
    outcomes, meta = read_outcomes(path)
    assert outcomes == _outcomes() and meta == {"python": "3.11"}
    with pytest.raises(ValueError, match="outcome"):
        Outcome("n", "n", "xfailed", 0.0)
    name = "test_alcubierre_energy_density_matches_closed_form"
    real = [Outcome(f"tests/unit/test_warp.py::{name}", name, "passed", 0.1)]
    page = render_benchmark_dashboard(real, tmp_path / "d.html", documentation=BENCHMARKS)
    text = page.read_text()
    assert "Alcubierre 1994" in text and "Closed-form and exact results" in text


def test_the_plugin_records_a_benchmark_run(pytester):
    """A real pytest session, run inside this one, with ``--dashboard``.

    Each outcome category is here: a pass, a failure, a skip, a setup error,
    and a parametrized test. There is also an unmarked test, which the
    dashboard leaves out.
    """
    pytester.makeini("[pytest]\nmarkers =\n    benchmark: a benchmark\n")
    pytester.makepyfile(
        test_sample='''
        import pytest

        @pytest.fixture
        def broken():
            raise RuntimeError("fixture broke")

        @pytest.mark.benchmark
        def test_passes():
            """Reproduces something.

            More detail that is not the summary."""

        @pytest.mark.benchmark
        def test_fails():
            assert 1 == 2

        @pytest.mark.benchmark
        def test_skips():
            pytest.skip("needs a GPU")

        @pytest.mark.benchmark
        def test_errors(broken):
            pass

        @pytest.mark.benchmark
        @pytest.mark.parametrize("n", [1, 2])
        def test_each(n):
            pass

        def test_not_a_benchmark():
            pass
        '''
    )
    docs = pytester.path / "bench.md"
    docs.write_text("## S\n| Benchmark | Test |\n|---|---|\n| passing thing | `test_passes` |\n")
    result = pytester.runpytest(
        "-p",
        "particlesim.viz.pytest_plugin",
        "--dashboard",
        "out/dash.html",
        "--dashboard-docs",
        str(docs),
    )
    result.assert_outcomes(passed=4, failed=1, skipped=1, errors=1)
    result.stdout.fnmatch_lines(["*benchmark dashboard:*dash.html (6 benchmarks)*"])
    outcomes, meta = read_outcomes(pytester.path / "out" / "dash.json")
    by_id = {o.nodeid.split("::")[1]: o for o in outcomes}
    assert {k: o.outcome for k, o in by_id.items()} == {
        "test_passes": "passed",
        "test_fails": "failed",
        "test_skips": "skipped",
        "test_errors": "error",
        "test_each[1]": "passed",
        "test_each[2]": "passed",
    }
    assert by_id["test_passes"].summary == "Reproduces something."
    assert by_id["test_each[2]"].name == "test_each"
    assert "fixture broke" in by_id["test_errors"].message
    assert by_id["test_skips"].message == "needs a GPU"
    assert "benchmark" in by_id["test_fails"].markers and meta["exit status"] == 1
    page = (pytester.path / "out" / "dash.html").read_text()
    assert "passing thing" in page and "test_not_a_benchmark" not in page


def test_without_the_option_the_plugin_does_nothing(pytester):
    pytester.makepyfile("def test_x():\n    pass\n")
    result = pytester.runpytest("-p", "particlesim.viz.pytest_plugin")
    result.assert_outcomes(passed=1)
    assert not list(pytester.path.rglob("*.html"))


# --- the run dashboard -----------------------------------------------------------


def _run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run / "sub").mkdir(parents=True)
    manifest = {
        "particlesim_version": "0.1.0",
        "config": {"scenario": "warp.analyze", "seed": 3},
        "config_hash": "abc",
        "seed": 3,
        "git_commit": "deadbeef",
        "packages": {"numpy": "2.0"},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    report = {"family": "natario", "expansion": {"max_abs": 1e-15}, "violations": ["NEC"]}
    (run / "report.json").write_text(json.dumps(report))
    (run / "energy_density.png").write_bytes(PIXEL)
    (run / "report.html").write_text("<html></html>")
    (run / "sub" / "fields.npz").write_bytes(b"\0" * 2048)
    return run


def test_a_run_dashboard_shows_what_the_run_wrote(tmp_path):
    run = _run_dir(tmp_path)
    page = write_run_dashboard(run).read_text()
    assert "warp.analyze: run" in page and "deadbeef" in page and "natario" in page
    assert "data:image/png;base64," in page and "energy_density.png" in page
    assert "<a href='report.html'>report.html</a>" in page
    assert "sub/fields.npz" in page and "2.0 KB" in page
    assert "&quot;seed&quot;: 3" in page  # the configuration, escaped
    # Rewriting it does not list the dashboard itself.
    assert "dashboard.html" not in write_run_dashboard(run).read_text()


def test_a_directory_without_a_manifest_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        write_run_dashboard(tmp_path)


def test_every_run_gets_a_dashboard_and_the_command_rebuilds_them(tmp_path, capsys):
    cfg = {
        "scenario": "warp.analyze",
        "metric": {"family": "natario"},
        "grid": {"extent": [[-8, 8]] * 3, "resolution": [6, 6, 6]},
        "analysis": {"full_stress_energy": False},
        "output": {"dir": str(tmp_path / "out"), "formats": ["json"]},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert main(["run", str(path)]) == 0
    dashboard = tmp_path / "out" / "dashboard.html"
    assert "natario" in dashboard.read_text()
    assert "dashboard:" in capsys.readouterr().err

    dashboard.unlink()
    results = write_outcomes(tmp_path / "bench.json", _outcomes(), {})
    assert main(["dashboard", str(tmp_path / "out"), str(results), "--docs", str(BENCHMARKS)]) == 0
    assert dashboard.is_file() and (tmp_path / "bench.html").is_file()
    assert main(["dashboard", str(path)]) == 2
