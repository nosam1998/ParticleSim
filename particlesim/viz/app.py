"""The served app: runs, and modified theories computed live (issue #83).

This is the design document's second delivery family (Section 5.7): a Panel app
served from the published Docker image, and the one place a modified theory's
results are computed while you watch. ``particlesim serve`` starts it, and it
has four tabs.

**Runs.** Every run directory under ``--runs`` gets its dashboard shown
(:mod:`particlesim.viz.dashboard`), written first if the run predates it.

**Modified gravity, live.** Hu-Sawicki ``f(R)`` with ``n = 1``, from
:mod:`particlesim.cosmo.fofr`. Sliders set ``|f_R0|``, ``Omega_m`` and the
scale factor, and linear growth is integrated again for each wavenumber on
every change:
- ``D_f(R) / D_LCDM`` against ``k``
- its square, the ``P(k)`` enhancement
- the scale where the scalaron's Compton wavelength cuts in

**Warp, live.** A warp bubble analysed under a chosen theory
(:mod:`particlesim.scenarios.warp.live`, issue #55). Each of the theory's
couplings gets a slider, and releasing one recomputes the matter the bubble
needs. The view shows its energy density as isosurfaces and a slice in 3-D
(:mod:`particlesim.viz.volume`), and the null and weak energy conditions.
Under GR+Lambda, the weak energy condition moves with ``Lambda`` and the null
one does not.

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


def warp_summary(result) -> str:
    """The live warp result's numbers, as a small HTML table."""
    report = result.report
    rows = [
        ("theory", html.escape(report["theory"]["gravity"]["id"])),
        (
            "couplings",
            html.escape(
                ", ".join(
                    f"{k} = {v:g}" for k, v in report["theory"]["gravity"]["couplings"].items()
                )
                or "none"
            ),
        ),
        ("Eulerian energy, total", f"{report['eulerian']['total_energy']:.6g}"),
        ("Eulerian energy, negative part", f"{report['eulerian']['negative_energy']:.6g}"),
        ("most negative density", f"{report['eulerian']['min_density']:.6g}"),
    ]
    for name in ("NEC", "WEC"):
        condition = report["energy_conditions"][name]
        rows.append(
            (
                f"{name} violated at",
                f"{100 * condition['violating_fraction']:.2f}% of points, "
                f"integral {condition['integrated_violation']:.6g}",
            )
        )
    rows.append(("computed in", f"{result.seconds:.2f} s on {result.grid.shape[0]}³ points"))
    body = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<table style='border-collapse:collapse'>{body}</table>"


def _frame(page: str, height: str = "82vh") -> str:
    """A whole HTML page inside an iframe, so its styles stay its own."""
    return (
        f'<iframe srcdoc="{html.escape(page, quote=True)}" '
        f'style="width:100%;height:{height};border:0" sandbox></iframe>'
    )


def build_app(runs: str | Path | None = None) -> Any:
    """The three-tab app. A new one per browser session, as Panel serves it."""
    import panel as pn

    pn.extension("plotly", sizing_mode="stretch_width")

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

    # --- a warp bubble under a theory, live
    from particlesim.scenarios.warp.live import (
        QUICK_FAMILY,
        live_warp,
        ready_families,
        split_theories,
    )
    from particlesim.theories import list_theories
    from particlesim.viz.volume import volume_figure

    usable = list(split_theories())
    family = pn.widgets.Select(label="Warp metric", options=ready_families(), value=QUICK_FAMILY)
    gravity = pn.widgets.Select(
        label="Theory", options=usable, value="gr.lambda" if "gr.lambda" in usable else usable[0]
    )
    couplings = pn.Column()
    active: list[tuple[str, Any]] = []  # (coupling name, its slider)
    warp_view = pn.Column()

    def recompute(*_) -> None:
        values = {name: slider.value for name, slider in active}
        try:
            result = live_warp(family.value, gravity.value, values)
        except Exception as error:  # noqa: BLE001 - shown in the page, where it belongs
            warp_view[:] = [pn.pane.Alert(f"{family.value} under {gravity.value}: {error}")]
            return
        figure = volume_figure(
            result.fields["energy_density"],
            result.grid,
            title=f"{family.value}: Eulerian energy density the bubble needs",
            label="ρ",
        )
        warp_view[:] = [pn.pane.HTML(warp_summary(result)), pn.pane.Plotly(figure, height=580)]

    def rebuild(*_) -> None:
        active.clear()
        for coupling in list_theories()[gravity.value].couplings:
            low, high = coupling.bounds or (coupling.default - 1.0, coupling.default + 1.0)
            if not np.isfinite([low, high]).all():
                low, high = coupling.default - 1.0, coupling.default + 1.0
            slider = pn.widgets.FloatSlider(
                label=coupling.name,
                start=float(low),
                end=float(high),
                step=(float(high) - float(low)) / 200,
                value=float(coupling.default),
            )
            slider.param.watch(recompute, "value_throttled")
            active.append((coupling.name, slider))
        sliders = [slider for _, slider in active]
        couplings[:] = sliders or [pn.pane.Markdown("This theory has no couplings.")]
        recompute()

    speed = pn.widgets.FloatSlider(
        label="bubble speed v (c = 1)", start=0.0, end=3.0, step=0.05, value=1.5
    )

    def inside_view(v: float):
        from particlesim.analysis.raytrace import Bubble
        from particlesim.viz.warp_render import png_bytes, render

        image, _ = render(Bubble(v=float(v)), 256, 128, step=0.03)
        return pn.pane.PNG(png_bytes(image), sizing_mode="scale_width")

    gravity.param.watch(rebuild, "value")
    family.param.watch(recompute, "value")
    rebuild()
    warp_tab = pn.Column(
        pn.pane.Markdown(
            "The matter a warp bubble needs under the chosen theory, recomputed when a "
            "coupling is released. A metric's geometry is derived once. The list holds "
            "Alcubierre, which takes seconds, and any family already derived: a full-path "
            "`particlesim run` of a family adds it. After that, every theory and "
            "coupling costs about a second."
        ),
        pn.Row(family, gravity),
        couplings,
        warp_view,
        pn.pane.Markdown(
            "### The view from inside\n"
            "An Alcubierre bubble's passenger, looking out: light traced back through "
            "the bubble to the sky it came from (`particlesim.analysis.raytrace`), tinted "
            "by its exact frequency shift, `1 − v cos α`. Straight ahead is in the "
            "middle. Above `v = 1` no light reaches the passenger from behind, and those "
            "pixels are black."
        ),
        speed,
        pn.bind(inside_view, speed),
    )

    # --- theory plugins, live

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
        ("Warp, live", warp_tab),
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
