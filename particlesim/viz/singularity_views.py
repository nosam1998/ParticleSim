"""Figures for collapse runs and hypothesis comparisons (design doc 5.7).

What a collapse run needs to show is not field values but causal structure:
did a horizon form, how fast did curvature grow, and does a correction change
the answer. These views are built around those three questions.

One rule runs through all of them. A quantity that spans many orders of
magnitude is drawn on a logarithmic axis, and a quantity that changes sign is
drawn on a diverging scale centred at zero. Mixing those up is how a plot
comes to say the opposite of the data: a linear axis on a curvature invariant
shows a flat line and then a wall, hiding the power law that is the actual
result.
"""

from __future__ import annotations

import io
from collections.abc import Sequence

import numpy as np


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    _pyplot().close(fig)
    return buf.getvalue()


def collapse_spacetime_diagram(
    times: Sequence[float],
    radii: np.ndarray,
    compactness: np.ndarray,
    horizon_radii: Sequence[float | None] | None = None,
) -> bytes:
    """Compactness on the time-radius plane, with the horizon track over it.

    ``compactness`` is ``(n_times, n_radii)``. The colour scale runs from zero
    to one because that is the physical range of ``2m/r``: letting it
    autoscale would make a weak-field run look like a near-horizon one.
    """
    plt = _pyplot()
    times = np.asarray(times, dtype=float)
    compactness = np.asarray(compactness, dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    mesh = ax.pcolormesh(
        radii,
        times,
        np.clip(compactness, 0.0, 1.0),
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        shading="auto",
    )
    if horizon_radii is not None:
        pairs = [(t, r) for t, r in zip(times, horizon_radii, strict=True) if r is not None]
        if pairs:
            ax.plot(
                [r for _, r in pairs],
                [t for t, _ in pairs],
                color="cyan",
                lw=2.0,
                label="2m/r = 1",
            )
            ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("r")
    ax.set_ylabel("t")
    ax.set_title("Compactness 2m/r", fontsize=10)
    fig.colorbar(mesh, ax=ax, label="2m/r")
    return _to_png(fig)


def invariants_vs_time(
    times: Sequence[float],
    series: dict[str, Sequence[float]],
    ylabel: str = "invariant",
) -> bytes:
    """Curvature invariants against time on a logarithmic axis.

    Logarithmic because these grow by many orders of magnitude, and a linear
    axis would show a flat line followed by a wall, hiding the power law that
    is the result people actually want from this plot.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for name, values in series.items():
        v = np.abs(np.asarray(values, dtype=float))
        v = np.where(v > 0, v, np.nan)
        ax.semilogy(times, v, lw=1.6, label=name)
    ax.set_xlabel("t")
    ax.set_ylabel(f"|{ylabel}|")
    ax.grid(alpha=0.25, which="both")
    if len(series) > 1:
        ax.legend(fontsize=8)
    return _to_png(fig)


def bounce_comparison(
    baseline_t: Sequence[float],
    baseline_a: Sequence[float],
    corrected_t: Sequence[float],
    corrected_a: Sequence[float],
    labels: tuple[str, str] = ("general relativity", "hypothesis"),
) -> bytes:
    """Scale factor under general relativity and under a correction, together.

    The comparison is the point: a bounce means nothing on its own, since an
    integrator that stops early also produces a smallest scale factor. Shown
    beside the uncorrected run on identical axes, the difference is either
    visible or it is not there.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.semilogy(baseline_t, np.maximum(baseline_a, 1e-30), lw=1.8, label=labels[0])
    ax.semilogy(corrected_t, np.maximum(corrected_a, 1e-30), lw=1.8, ls="--", label=labels[1])
    ax.set_xlabel("t")
    ax.set_ylabel("scale factor a")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    ax.set_title("Collapse with and without the correction", fontsize=10)
    return _to_png(fig)


def penrose_diagram(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str = "Penrose diagram",
) -> bytes:
    """Draw labelled curves in conformal coordinates with equal axes.

    Equal aspect is not a style choice here. The entire content of the diagram
    is which lines are at forty-five degrees, and a stretched axis destroys
    exactly that.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    for label, (x, t) in curves.items():
        ax.plot(x, t, lw=1.8, label=label)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("T")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    return _to_png(fig)


def hypothesis_figures(card, baseline_solution=None, corrected_solution=None) -> dict[str, bytes]:
    """The standard figure set for a hypothesis report card."""
    figures: dict[str, bytes] = {}
    if baseline_solution is not None and corrected_solution is not None:
        figures["Collapse with and without the correction"] = bounce_comparison(
            baseline_solution.t,
            baseline_solution.a,
            corrected_solution.t,
            corrected_solution.a,
            labels=("general relativity", card.theory_id),
        )
        figures["Density during collapse"] = invariants_vs_time(
            corrected_solution.t,
            {
                "general relativity": np.interp(
                    corrected_solution.t, baseline_solution.t, baseline_solution.density
                ),
                card.theory_id: corrected_solution.density,
            },
            ylabel="density",
        )
    return figures
