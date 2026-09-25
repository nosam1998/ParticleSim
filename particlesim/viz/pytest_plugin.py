"""A pytest plugin that writes a benchmark dashboard for the run (issue #83).

The repository's ``conftest.py`` loads it. With ``--dashboard PATH``, every
test marked ``benchmark`` is recorded. At the end of the session, PATH gets
the HTML dashboard of :mod:`particlesim.viz.dashboard`, and the results go as
JSON beside it, at PATH with ``.json``.

A test's outcome is its worst phase's:
- a failure in setup or teardown is an error
- a skip anywhere is a skip
- a failure in the call is a failure

Those are the categories pytest's own summary counts. Nothing here imports
pytest, so the package does not depend on it. The recorder is registered per
session, so a pytest run inside a pytest run, as the plugin's own tests do,
keeps its results apart.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from particlesim.viz.dashboard import Outcome, render_benchmark_dashboard, write_outcomes


def pytest_addoption(parser) -> None:
    group = parser.getgroup("particlesim")
    group.addoption(
        "--dashboard",
        metavar="PATH",
        default=None,
        help="write an HTML dashboard of the benchmark tests run, and its JSON beside it",
    )
    group.addoption(
        "--dashboard-docs",
        metavar="PATH",
        default="docs/benchmarks.md",
        help="the benchmark documentation to join results with (default: docs/benchmarks.md)",
    )


def pytest_configure(config) -> None:
    if config.getoption("dashboard", None):
        config.pluginmanager.register(Recorder(config), "particlesim-dashboard")


def _message(report) -> str:
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:  # a skip: (file, line, reason)
        return str(longrepr[2]).removeprefix("Skipped: ")
    crash = getattr(getattr(longrepr, "reprcrash", None), "message", None)
    text = (crash or str(longrepr or "")).strip()
    return text.splitlines()[0][:300] if text else ""


class Recorder:
    """Collects one session's benchmark results and writes them at its end."""

    def __init__(self, config) -> None:
        self.config = config
        self.items: dict[str, tuple[str, str, tuple[str, ...]]] = {}
        self.results: dict[str, list] = {}

    def _resolve(self, option: str) -> Path:
        path = Path(self.config.getoption(option))
        return path if path.is_absolute() else Path(self.config.rootpath) / path

    def pytest_collection_modifyitems(self, session, config, items) -> None:
        for item in items:
            if item.get_closest_marker("benchmark") is None:
                continue
            doc = (getattr(getattr(item, "function", None), "__doc__", None) or "").strip()
            summary = " ".join(doc.split("\n\n", 1)[0].split())
            markers = tuple(sorted({m.name for m in item.iter_markers()}))
            name = getattr(item, "originalname", None) or item.name.split("[", 1)[0]
            self.items[item.nodeid] = (name, summary, markers)

    def pytest_runtest_logreport(self, report) -> None:
        if report.nodeid not in self.items:
            return
        entry = self.results.setdefault(report.nodeid, ["passed", 0.0, ""])
        entry[1] += report.duration
        if entry[0] in ("failed", "error", "skipped"):
            return
        if report.outcome == "failed":
            entry[0] = "failed" if report.when == "call" else "error"
            entry[2] = _message(report)
        elif report.outcome == "skipped":
            entry[0] = "skipped"
            entry[2] = _message(report)

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        outcomes = [
            Outcome(nodeid, name, outcome, duration, summary, markers, message)
            for nodeid, (name, summary, markers) in self.items.items()
            if nodeid in self.results
            for outcome, duration, message in [self.results[nodeid]]
        ]
        path = self._resolve("dashboard")
        docs = self._resolve("dashboard_docs")
        meta = {
            "selection": self.config.getoption("markexpr", "") or "all benchmarks collected",
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "exit status": int(exitstatus),
        }
        write_outcomes(path.with_suffix(".json"), outcomes, meta)
        render_benchmark_dashboard(
            outcomes, path, documentation=docs if docs.is_file() else None, meta=meta
        )
        reporter = self.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(f"benchmark dashboard: {path} ({len(outcomes)} benchmarks)")
