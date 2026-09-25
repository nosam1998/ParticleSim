"""The served app: runs, and modified theories computed live (issue #83).

This is the design document's second delivery family (Section 5.7): a Panel app
served from the published Docker image, and the one place a modified theory's
results are computed while you watch. ``particlesim serve`` starts it, and it
has three tabs.

**Runs.** Every run directory under ``--runs`` gets its dashboard shown
(:mod:`particlesim.viz.dashboard`), written first if the run predates it.

**Modified gravity, live.** Hu-Sawicki ``f(R)`` with ``n = 1``, from
:mod:`particlesim.cosmo.fofr`. Sliders set ``|f_R0|``, ``Omega_m`` and the
scale factor, and linear growth is integrated again for each wavenumber on
every change:
- ``D_f(R) / D_LCDM`` against ``k``
- its square, the ``P(k)`` enhancement
- the scale where the scalaron's Compton wavelength cuts in

**Theory plugins, live.** Any installed theory plugin can be scored against
the singularity battery, with the report card and its figures that
``particlesim hypothesis`` writes. That runs the harness itself, not a replay.

Panel is an optional dependency (``pip install particlesim[serve]``). Nothing
here imports it until the app is built. The computations are plain functions,
so they are tested without a browser.
"""

from __future__ import annotations

import html
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

#: The wavenumbers the live f(R) view integrates, in h/Mpc.
WAVENUMBERS = np.geomspace(1e-3, 1.0, 60)


def fofr_enhancement(
    f_r0: float, omega_m: float = 0.3, scale: float = 1.0, wavenumbers: np.ndarray = WAVENUMBERS
) -> dict[str, np.ndarray | float]:
    """Linear growth in Hu-Sawicki ``f(R)`` relative to LCDM, per wavenumber.

    Returns the growth ratio ``D_f(R) / D_LCDM``, its square (the ``P(k)``
    enhancement), and the Compton wavenumber ``a m`` at ``scale``. Above that
    wavenumber the fifth force acts and the ratio climbs towards its
    small-scale limit, ``4/3`` in ``mu``.
    """
    from particlesim.cosmo.fofr import HuSawicki, growth_ratio

    if not 0.0 < scale <= 1.0:
        raise ValueError("the scale factor must lie in (0, 1]")
    model = HuSawicki(f_r0=float(f_r0), omega_m=float(omega_m))
    k = np.asarray(wavenumbers, dtype=float)
    ratio = np.asarray(growth_ratio(model, k, scale), dtype=float)
    compton = float(1.0 / model.compton_wavelength(scale))
    return {"k": k, "growth": ratio, "power": ratio**2, "compton": compton}


def fofr_figure(f_r0: float, omega_m: float = 0.3, scale: float = 1.0):
    """The live f(R) view as a matplotlib figure, made without pyplot."""
    from matplotlib.figure import Figure

    result = fofr_enhancement(f_r0, omega_m, scale)
    fig = Figure(figsize=(7.5, 4.2), layout="constrained")
    ax = fig.add_subplot()
    ax.semilogx(result["k"], result["power"] - 1.0, lw=2, label="P(k) enhancement")
    ax.semilogx(result["k"], result["growth"] - 1.0, lw=1.5, ls="--", label="growth D")
    ax.axvline(result["compton"], color="0.5", lw=1, ls=":", label="Compton scale a m")
    ax.set_xlabel("k  [h / Mpc]")
    ax.set_ylabel("f(R) / LCDM − 1")
    ax.set_title(f"Hu–Sawicki n = 1, |f_R0| = {f_r0:.1e}, Ω_m = {omega_m:.2f}, a = {scale:.2f}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    return fig


@lru_cache(maxsize=32)
def hypothesis_page(theory_id: str) -> str:
    """The report card for ``theory_id``, as a self-contained HTML page.

    The same page ``particlesim hypothesis THEORY --report`` writes. Cached,
    because a plugin's card does not change while the app runs.
    """
    from particlesim.scenarios.singularity.harness import evaluate, write_report
    from particlesim.theories import get_theory

    theory = get_theory(theory_id)
    card = evaluate(theory)
    with tempfile.TemporaryDirectory() as scratch:
        path = write_report(card, Path(scratch) / "report.html", theory)
        return path.read_text(encoding="utf-8")


def run_dashboards(runs: str | Path | None) -> dict[str, Path]:
    """Every run under ``runs``, by relative name, with its dashboard written if missing."""
    from particlesim.viz.dashboard import write_run_dashboard

    if runs is None:
        return {}
    root = Path(runs)
    found: dict[str, Path] = {}
    for manifest in sorted(root.rglob("manifest.json")):
        run = manifest.parent
        page = run / "dashboard.html"
        if not page.is_file():
            write_run_dashboard(run)
        found[str(run.relative_to(root)) or run.name] = page
    return found


def _frame(page: str, height: str = "82vh") -> str:
    """A whole HTML page inside an iframe, so its styles stay its own."""
    return (
        f'<iframe srcdoc="{html.escape(page, quote=True)}" '
        f'style="width:100%;height:{height};border:0" sandbox></iframe>'
    )


def build_app(runs: str | Path | None = None) -> Any:
    """The three-tab app. A new one per browser session, as Panel serves it."""
    import panel as pn

    pn.extension(sizing_mode="stretch_width")

    # --- runs
    dashboards = run_dashboards(runs)
    if dashboards:
        chooser = pn.widgets.Select(label="Run", options=list(dashboards))
        shown = pn.bind(lambda name: pn.pane.HTML(_frame(dashboards[name].read_text())), chooser)
        runs_tab = pn.Column(chooser, shown)
    else:
        where = f"under {html.escape(str(runs))}" if runs else "(start with --runs DIR)"
        runs_tab = pn.pane.Markdown(f"No run directories {where}.")

    # --- f(R), live
    exponent = pn.widgets.FloatSlider(
        label="log10 |f_R0|", start=-7.0, end=-4.0, step=0.25, value=-5.0
    )
    omega_m = pn.widgets.FloatSlider(label="Ω_m", start=0.15, end=0.5, step=0.01, value=0.3)
    scale = pn.widgets.FloatSlider(label="scale factor a", start=0.2, end=1.0, step=0.05, value=1.0)

    def fofr_view(e: float, om: float, a: float):
        return pn.pane.Matplotlib(fofr_figure(10.0**e, om, a), dpi=110, tight=True)

    fofr_tab = pn.Column(
        pn.pane.Markdown(
            "Linear growth in Hu–Sawicki f(R) against ΛCDM. It is integrated again "
            "for each wavenumber whenever a slider moves (`particlesim.cosmo.fofr`)."
        ),
        pn.Row(exponent, omega_m, scale),
        pn.bind(fofr_view, exponent, omega_m, scale),
    )

    # --- theory plugins, live
    from particlesim.theories import list_theories

    theories = sorted(list_theories())
    selected = pn.widgets.Select(
        label="Theory plugin", options=theories, value="lqg.lqc" if "lqg.lqc" in theories else None
    )

    def card_view(theory_id: str | None):
        if not theory_id:
            return pn.pane.Markdown("Choose a plugin.")
        try:
            return pn.pane.HTML(_frame(hypothesis_page(theory_id)))
        except Exception as error:  # noqa: BLE001 - a plugin's failure is its result here
            return pn.pane.Alert(f"{theory_id}: {error}", alert_type="danger")

    theory_tab = pn.Column(
        pn.pane.Markdown(
            "A plugin scored against the singularity battery, computed now. This is "
            "the report `particlesim hypothesis` writes."
        ),
        selected,
        pn.bind(card_view, selected),
    )

    return pn.Tabs(
        ("Runs", runs_tab),
        ("Modified gravity, live", fofr_tab),
        ("Theory plugins, live", theory_tab),
        dynamic=True,
    )


def serve(
    runs: str | Path | None = None,
    port: int = 5006,
    address: str = "localhost",
    allow_websocket_origin: list[str] | None = None,
    show: bool = False,
    threaded: bool = False,
):
    """Serve :func:`build_app` with Panel's server; blocks unless ``threaded``."""
    import panel as pn

    return pn.serve(
        {"/": lambda: build_app(runs)},
        port=port,
        address=address,
        allow_websocket_origin=allow_websocket_origin or [f"localhost:{port}", f"127.0.0.1:{port}"],
        show=show,
        threaded=threaded,
        title="ParticleSim",
    )
