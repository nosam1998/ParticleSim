"""Self-contained HTML report for a run (design doc Sections 5.7, 7).

The page embeds its figures as data URIs and carries no external requests,
so a report stays readable from a results directory, an archive, or an
email attachment years later, with no server and no network.
"""

from __future__ import annotations

import base64
import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_CSS = """
:root {
  color-scheme: light dark;
  --bg: #ffffff; --fg: #1b1b1f; --muted: #5c5f6b;
  --line: #e3e4e8; --card: #f7f7f9; --accent: #2d5bd7; --warn: #b3261e;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #15161a; --fg: #e8e9ed; --muted: #a0a3ad;
    --line: #2c2e36; --card: #1d1f25; --accent: #8ab0ff; --warn: #ff8a80;
  }
}
:root[data-theme="dark"] {
  --bg: #15161a; --fg: #e8e9ed; --muted: #a0a3ad;
  --line: #2c2e36; --card: #1d1f25; --accent: #8ab0ff; --warn: #ff8a80;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 24px 16px 64px; background: var(--bg); color: var(--fg);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
main { max-width: 900px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 4px; }
h2 { font-size: 1.1rem; margin: 32px 0 10px; padding-bottom: 6px;
     border-bottom: 1px solid var(--line); }
.sub { color: var(--muted); font-size: 0.9rem; margin: 0 0 8px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 16px; font-size: 0.92rem; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; }
td.num { font-variant-numeric: tabular-nums; }
.bad { color: var(--warn); font-weight: 600; }
figure { margin: 0 0 20px; background: var(--card); border: 1px solid var(--line);
         border-radius: 10px; padding: 10px; }
figure img { width: 100%; height: auto; display: block; border-radius: 6px; }
figcaption { color: var(--muted); font-size: 0.85rem; margin-top: 8px; }
pre { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
      padding: 12px; overflow-x: auto; font-size: 0.82rem; }
ul { margin: 6px 0 16px; padding-left: 22px; }
li { margin: 3px 0; }
details { margin-top: 12px; }
summary { cursor: pointer; color: var(--accent); }
"""


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value == 0:
            return "0"
        return f"{value:.6g}"
    if value is None:
        return "n/a"
    return html.escape(str(value))


def _rows(mapping: dict[str, Any]) -> str:
    out = []
    for k, v in mapping.items():
        cls = "num"
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v < 0:
            cls = "num bad"
        out.append(f"<tr><th>{html.escape(str(k))}</th><td class='{cls}'>{_fmt(v)}</td></tr>")
    return "\n".join(out)


def _list_table(rows: list[dict[str, Any]]) -> str:
    """Render a list of dictionaries as a table with a column per key."""
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    body = []
    for row in rows:
        cells = []
        for c in columns:
            value = row.get(c)
            cls = "num"
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0:
                cls = "num bad"
            cells.append(f"<td class='{cls}'>{_fmt(value)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _bullets(items: list[Any]) -> str:
    if not items:
        return "<p class='sub'>none</p>"
    return "<ul>" + "".join(f"<li>{_fmt(i)}</li>" for i in items) + "</ul>"


def _flat_sections(report: dict[str, Any]) -> list[tuple[str, Any]]:
    """Split a report into renderable sections.

    Lists are sections in their own right. Dropping them, which an earlier
    version did by filtering for non-dict non-list values, silently removed
    every list-valued field from the page. On a hypothesis report card those
    fields are the verdict itself: what was confirmed, what was contradicted,
    what went untested. A report page that omits its own conclusion is worse
    than no page, because it looks complete.
    """
    scalars = {k: v for k, v in report.items() if not isinstance(v, (dict, list))}
    sections: list[tuple[str, Any]] = []
    if scalars:
        sections.append(("Overview", scalars))
    for key, value in report.items():
        title = key.replace("_", " ").title()
        if isinstance(value, dict):
            flat = {k: v for k, v in value.items() if not isinstance(v, dict)}
            nested = {k: v for k, v in value.items() if isinstance(v, dict)}
            if flat:
                sections.append((title, flat))
            for sub, subval in nested.items():
                sections.append((f"{key} / {sub}".replace("_", " ").title(), subval))
        elif isinstance(value, list):
            sections.append((title, value))
    return sections


def render_html_report(
    report: dict[str, Any],
    path: str | Path,
    figures: dict[str, bytes] | None = None,
    title: str = "ParticleSim run report",
) -> Path:
    """Write a self-contained HTML report. Figures are PNG bytes, embedded inline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body><main>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p class='sub'>Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}</p>",
    ]

    for heading, content in _flat_sections(report):
        if isinstance(content, list):
            # An empty list is still rendered, because "nothing was
            # contradicted" is a result and a missing section is not.
            if content and all(isinstance(i, dict) for i in content):
                body = _list_table(content)
            else:
                body = _bullets(content)
            parts.append(f"<h2>{html.escape(heading)}</h2>{body}")
            continue
        if not content:
            continue
        parts.append(
            f"<h2>{html.escape(heading)}</h2><table><tbody>{_rows(content)}</tbody></table>"
        )

    if figures:
        parts.append("<h2>Figures</h2>")
        for caption, png in figures.items():
            b64 = base64.b64encode(png).decode("ascii")
            parts.append(
                f"<figure><img alt='{html.escape(caption)}' src='data:image/png;base64,{b64}'>"
                f"<figcaption>{html.escape(caption)}</figcaption></figure>"
            )

    parts.append(
        "<details><summary>Full report JSON</summary><pre>"
        + html.escape(json.dumps(report, indent=2, default=str))
        + "</pre></details>"
    )
    parts.append("</main></body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
