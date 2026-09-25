"""Static HTML dashboards: one per run, and one per benchmark run (issue #83).

Both are single self-contained files in :mod:`particlesim.viz.report`'s
style, with figures inlined and no external requests. They stay readable as a
CI artifact, from an archive, or attached to a message, with no server.

**A benchmark dashboard** is what a benchmark run found, set beside what
``docs/benchmarks.md`` says each benchmark is. The pytest plugin in
:mod:`particlesim.viz.pytest_plugin` records every test marked ``benchmark``:
its outcome, its duration and the first paragraph of its docstring.
:func:`documented_benchmarks` reads the tables in ``docs/benchmarks.md`` that
name tests, and :func:`render_benchmark_dashboard` joins the two by test name.
Each result shows its reference and tolerance, and the value the documentation
last recorded.

The join is also a check. A benchmark that ran but is named in no table, and a
table row naming a test that did not run, are both listed, so the
documentation cannot drift from the suite unnoticed.

**A run dashboard** is a run directory seen at a glance:
- the manifest's provenance
- the report
- every figure the run wrote
- the files, with their sizes

:func:`write_run_dashboard` writes it as ``dashboard.html`` beside them, and
``particlesim run`` does so after every run.
"""

from __future__ import annotations

import base64
import html
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from particlesim.viz.report import _CSS, _flat_sections, _fmt, _list_table, _rows

#: Status colours and badges, on top of the report's palette.
_DASHBOARD_CSS = """
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
         gap: 10px; margin: 16px 0 8px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
        padding: 10px 12px; }
.card .n { font-size: 1.5rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.card .l { color: var(--muted); font-size: 0.85rem; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 0.8rem;
         font-weight: 600; white-space: nowrap; }
.passed { background: #d8f3dc; color: #1b5e20; }
.failed, .error { background: #ffdad6; color: #8c1d18; }
.skipped { background: #eceef2; color: #4a4d57; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) .passed { background: #1f3b26; color: #a6e3b0; }
  :root:not([data-theme="light"]) .failed, :root:not([data-theme="light"]) .error {
    background: #4a1f1c; color: #ffb4ab; }
  :root:not([data-theme="light"]) .skipped { background: #2c2e36; color: #c4c6d0; }
}
.scroll { overflow-x: auto; }
td, th { overflow-wrap: anywhere; }
tbody th { width: 32%; }
.card .n { overflow-wrap: anywhere; font-size: 1.25rem; }
td code, li code { font-size: 0.85em; }
td.why { color: var(--muted); font-size: 0.85rem; }
"""

_STATUSES = ("failed", "error", "passed", "skipped")


# --- what docs/benchmarks.md says ------------------------------------------------


@dataclass(frozen=True)
class Documented:
    """One row of a ``docs/benchmarks.md`` table that names a test."""

    test: str
    section: str
    cells: dict[str, str]


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def documented_benchmarks(markdown: str) -> dict[str, list[Documented]]:
    """Every table row in ``markdown`` that names a test, by test name.

    A table counts when its header has a ``Test`` column. The cell's
    backquoted ``test_*`` names are the tests the row documents. ``same``,
    ``same test`` or a cell with no test name carries the row above's names
    down, as the tables write it. ``section`` is the nearest heading above the
    table.
    """
    found: dict[str, list[Documented]] = {}
    section = ""
    header: list[str] | None = None
    previous: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            section = stripped.lstrip("#").strip()
            header = None
            continue
        if not stripped.startswith("|"):
            header = None
            continue
        cells = _split_row(stripped)
        if header is None:
            header = cells if any(c.lower() == "test" for c in cells) else []
            previous = []
            continue
        if not header or all(re.fullmatch(r":?-+:?", c) for c in cells):
            continue
        row = dict(zip(header, cells, strict=False))
        test_cell = next((v for k, v in row.items() if k.lower() == "test"), "")
        names = re.findall(r"`(test_\w+)`", test_cell) or previous
        previous = names
        for name in names:
            found.setdefault(name, []).append(Documented(name, section, row))
    return found


# --- what a benchmark run found --------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    """One benchmark test's result in one run."""

    nodeid: str
    name: str
    outcome: str
    duration: float
    summary: str = ""
    markers: tuple[str, ...] = ()
    message: str = ""

    def __post_init__(self) -> None:
        if self.outcome not in _STATUSES:
            raise ValueError(f"outcome must be one of {_STATUSES}, got {self.outcome!r}")


def write_outcomes(path: str | Path, outcomes: Iterable[Outcome], meta: Mapping[str, Any]) -> Path:
    """The run's results as JSON, which :func:`read_outcomes` reads back."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": dict(meta), "outcomes": [asdict(o) for o in outcomes]}
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def read_outcomes(path: str | Path) -> tuple[list[Outcome], dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    outcomes = [
        Outcome(**{**o, "markers": tuple(o.get("markers", ()))}) for o in payload["outcomes"]
    ]
    return outcomes, payload.get("meta", {})


# --- rendering -------------------------------------------------------------------


def _inline(markdown: str) -> str:
    """A table cell's inline Markdown as HTML: code and bold, escaped first.

    Single asterisks are left alone, since the tables use them for
    multiplication as often as for emphasis.
    """
    text = html.escape(markdown)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)


def _cell(row: Documented | None, *names: str) -> str:
    if row is None:
        return ""
    for key, value in row.cells.items():
        if key.lower() in names:
            return _inline(value)
    return ""


def _page(title: str, body: Sequence[str], generated: str | None = None) -> str:
    stamp = generated or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return "\n".join(
        [
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'>",
            f"<title>{html.escape(title)}</title>",
            f"<style>{_CSS}{_DASHBOARD_CSS}</style></head><body><main>",
            f"<h1>{html.escape(title)}</h1>",
            f"<p class='sub'>Generated {html.escape(stamp)}</p>",
            *body,
            "</main></body></html>",
        ]
    )


def _card(number: object, label: str) -> str:
    return f"<div class='card'><div class='n'>{number}</div><div class='l'>{label}</div></div>"


@dataclass
class BenchmarkDashboard:
    """A benchmark run joined with its documentation, ready to render."""

    outcomes: list[Outcome]
    documented: dict[str, list[Documented]]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def counts(self) -> dict[str, int]:
        return {s: sum(o.outcome == s for o in self.outcomes) for s in _STATUSES}

    @property
    def undocumented(self) -> list[Outcome]:
        """Benchmarks that ran and that no table in the documentation names."""
        return [o for o in self.outcomes if o.name not in self.documented]

    @property
    def not_run(self) -> list[str]:
        """Tests the documentation names that did not run here.

        A slow benchmark in a fast run lands here. That is expected, and the
        page says so; a name that no longer exists at all lands here too.
        """
        ran = {o.name for o in self.outcomes}
        return sorted(name for name in self.documented if name not in ran)

    def html(self, title: str = "ParticleSim benchmarks", generated: str | None = None) -> str:
        counts = self.counts
        body = ["<div class='cards'>"]
        body.append(_card(len(self.outcomes), "benchmarks run"))
        for status in _STATUSES:
            if counts[status] or status in ("passed", "failed"):
                body.append(_card(counts[status], status))
        body.append(_card(len(self.outcomes) - len(self.undocumented), "documented"))
        body.append(_card(f"{sum(o.duration for o in self.outcomes):.0f} s", "total time"))
        body.append("</div>")
        if self.meta:
            body.append(f"<table><tbody>{_rows(self.meta)}</tbody></table>")

        order = {status: i for i, status in enumerate(_STATUSES)}
        ranked = sorted(self.outcomes, key=lambda o: (order[o.outcome], o.nodeid))
        failing = [o for o in ranked if o.outcome in ("failed", "error")]
        if failing:
            body.append("<h2>Failing</h2>")
            body.append(self._table(failing))
        body.append("<h2>Results</h2>")
        body.append(self._table(ranked))

        body.append("<h2>Not named in docs/benchmarks.md</h2>")
        if self.undocumented:
            body.append(
                "<ul>"
                + "".join(
                    f"<li><code>{html.escape(o.nodeid)}</code></li>" for o in self.undocumented
                )
                + "</ul>"
            )
        else:
            body.append("<p class='sub'>none: every benchmark that ran is documented</p>")
        body.append("<h2>Documented, not run here</h2>")
        body.append(
            "<p class='sub'>Slow benchmarks run on main and by hand, so a fast run lists "
            "them here.</p>"
        )
        if self.not_run:
            body.append(
                "<details><summary>"
                f"{len(self.not_run)} tests</summary><ul>"
                + "".join(f"<li><code>{html.escape(n)}</code></li>" for n in self.not_run)
                + "</ul></details>"
            )
        else:
            body.append("<p class='sub'>none</p>")
        return _page(title, body, generated)

    def _table(self, outcomes: Sequence[Outcome]) -> str:
        head = (
            "<tr><th>Result</th><th>Test</th><th>Benchmark</th><th>Reference</th>"
            "<th>Tolerance</th><th>Documented</th><th>Time</th></tr>"
        )
        rows = []
        for o in outcomes:
            documented = self.documented.get(o.name, [])
            first = documented[0] if documented else None
            what = _cell(first, "benchmark") or html.escape(o.summary)
            section = f"<div class='sub'>{html.escape(first.section)}</div>" if first else ""
            why = f"<div class='why'>{html.escape(o.message)}</div>" if o.message else ""
            rows.append(
                "<tr>"
                f"<td><span class='badge {o.outcome}'>{o.outcome}</span></td>"
                f"<td><code>{html.escape(o.nodeid.split('::', 1)[-1])}</code>{why}</td>"
                f"<td>{what}{section}</td>"
                f"<td>{_cell(first, 'reference', 'expected', 'theory')}</td>"
                f"<td>{_cell(first, 'tolerance')}</td>"
                f"<td>{_cell(first, 'measured', 'achieved', 'result')}</td>"
                f"<td class='num'>{o.duration:.2f} s</td>"
                "</tr>"
            )
        table = f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"
        return f"<div class='scroll'>{table}</div>"


def render_benchmark_dashboard(
    outcomes: Sequence[Outcome],
    path: str | Path,
    documentation: str | Path | None = None,
    meta: Mapping[str, Any] | None = None,
    title: str = "ParticleSim benchmarks",
) -> Path:
    """Write the dashboard for one benchmark run.

    ``documentation`` is ``docs/benchmarks.md``: a ``Path`` to it, or its
    text as a ``str``. Without it, the page shows the results alone.
    """
    if documentation is None:
        documented: dict[str, list[Documented]] = {}
    elif isinstance(documentation, Path):
        documented = documented_benchmarks(documentation.read_text(encoding="utf-8"))
    else:
        documented = documented_benchmarks(documentation)
    page = BenchmarkDashboard(list(outcomes), documented, dict(meta or {})).html(title)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path


# --- a run directory -------------------------------------------------------------

#: Manifest fields shown on a run dashboard, in order.
_PROVENANCE = (
    "particlesim_version",
    "config_hash",
    "seed",
    "git_commit",
    "git_dirty",
    "python",
    "platform",
    "created_at",
)


def _size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def write_run_dashboard(run_dir: str | Path, title: str | None = None) -> Path:
    """``dashboard.html`` for a run directory, from whatever the run wrote.

    It reads ``manifest.json`` and ``report.json`` if present, inlines every
    ``.png``, and lists every file. A run that wrote less gets a smaller page,
    not an error. A directory with no manifest is refused, because a dashboard
    that cannot say what produced the results is not one.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{run_dir} has no manifest.json; is it a run directory?")
    manifest = json.loads(manifest_path.read_text())
    config = manifest.get("config", {})
    scenario = config.get("scenario", "run")
    title = title or f"{scenario}: {run_dir.name}"

    commit = str(manifest.get("git_commit") or "unknown")[:10]
    if manifest.get("git_dirty"):
        commit += " (modified)"
    body: list[str] = ["<div class='cards'>"]
    body.append(_card(html.escape(scenario), "scenario"))
    body.append(_card(html.escape(str(manifest.get("created_at", "?"))[:10]), "created"))
    body.append(_card(html.escape(commit), "commit"))
    body.append(_card(html.escape(str(manifest.get("seed", "?"))), "seed"))
    body.append("</div>")

    report_path = run_dir / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        for heading, content in _flat_sections(report):
            if isinstance(content, list):
                if content and all(isinstance(i, dict) for i in content):
                    section = _list_table(content)
                else:
                    section = "<ul>" + "".join(f"<li>{_fmt(i)}</li>" for i in content) + "</ul>"
            elif content:
                section = f"<table><tbody>{_rows(content)}</tbody></table>"
            else:
                continue
            body.append(f"<h2>{html.escape(heading)}</h2>{section}")

    figures = sorted(run_dir.glob("*.png"))
    if figures:
        body.append("<h2>Figures</h2>")
        for figure in figures:
            data = base64.b64encode(figure.read_bytes()).decode("ascii")
            body.append(
                f"<figure><img alt='{html.escape(figure.stem)}' src='data:image/png;base64,{data}'>"
                f"<figcaption>{html.escape(figure.name)}</figcaption></figure>"
            )

    provenance = {k: manifest[k] for k in _PROVENANCE if k in manifest}
    body.append("<h2>Provenance</h2>")
    body.append(f"<table><tbody>{_rows(provenance)}</tbody></table>")
    if manifest.get("packages"):
        body.append(f"<table><tbody>{_rows(manifest['packages'])}</tbody></table>")

    files = sorted(p for p in run_dir.rglob("*") if p.is_file() and p.name != "dashboard.html")
    body.append("<h2>Files</h2><table><tbody>")
    for p in files:
        name = html.escape(str(p.relative_to(run_dir)))
        link = f"<a href='{name}'>{name}</a>" if p.suffix == ".html" else f"<code>{name}</code>"
        body.append(f"<tr><td>{link}</td><td class='num'>{_size(p.stat().st_size)}</td></tr>")
    body.append("</tbody></table>")
    body.append(
        "<details><summary>Configuration</summary><pre>"
        + html.escape(json.dumps(config, indent=2, default=str))
        + "</pre></details>"
    )
    path = run_dir / "dashboard.html"
    path.write_text(_page(title, body), encoding="utf-8")
    return path
