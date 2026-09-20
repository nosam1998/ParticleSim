"""Views for lattice runs (design doc Section 5.7, Milestone 6).

Four pictures, each drawn because a number for the same thing is either
missing the point or actively misleading. All of them take stored results --
a :class:`~particlesim.solvers.lattice.euclidean.Chain` or a
:class:`~particlesim.scenarios.cosmo.lattice.GrowthReport` -- and none of
them runs anything.

**The observable series** carries both error estimates, not one. The error
from summing the autocorrelation function and the error from binning use
different information, so drawing them together is a check rather than a
decoration: when they disagree, one of them is wrong and the picture says so
before a table would.

**The autocorrelation function** is drawn with the window that was actually
summed to, marked. ``tau_int`` is a number; where the sum stopped is a
judgement, and a reader cannot audit it from the number alone. A curve still
well above zero at the cut is a run that was too short.

**The binning curve** is the picture with the most to say. The error rises
with bin size and flattens once the bins are long compared with the
correlation time, and the plateau is the honest error. Whether the curve has
*reached* a plateau is the question, and a reported ``tau_int = 9.5`` says
nothing about whether the run was long enough to measure 9.5. The tail also
goes noisy as the bin count falls, which is visible here and invisible in any
summary.

**The growth spectrum** exists because of what
:mod:`particlesim.scenarios.cosmo.lattice` found. Past a dynamic range of
about ``1/eps`` the quiet modes stop reporting themselves and start
reporting round-off from the loudest one, at a fixed relative amplitude near
``1e-15``. On a plot of relative amplitude that is unmistakable -- a flat
floor of modes lying on a horizontal line -- and in a column of growth rates
it looks like a slow resonance. The floored modes are marked rather than
dropped, because the reader should see how many there are.
"""

from __future__ import annotations

import io

import numpy as np

from particlesim.solvers.lattice.euclidean import (
    binned_errors,
    integrated_autocorrelation,
    plateau_error,
)


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


def _values(source) -> np.ndarray:
    """A measured series, from a stored chain or a bare array."""
    values = np.asarray(getattr(source, "values", source), dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"a series of at least two measurements is expected, got {values.shape}")
    return values


def observable_series(source, title: str = "observable", reference: float | None = None) -> bytes:
    """The measurement against sweep, with both error estimates on it.

    ``reference`` draws the exact value where one is known, which is what
    turns the picture from a trace into a comparison. The two error bands
    come from independent estimators and are drawn together on purpose.
    """
    values = _values(source)
    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(7.5, 4.0))
    sweeps = np.arange(values.size)

    axes.plot(sweeps, values, lw=0.4, color="0.7", label="measurement")
    running = np.cumsum(values) / (sweeps + 1.0)
    axes.plot(sweeps, running, lw=1.6, color="C0", label="running mean")

    mean = float(np.mean(values))
    tau = integrated_autocorrelation(values)
    naive = float(np.std(values) / np.sqrt(values.size))
    correlated = naive * np.sqrt(2.0 * tau)
    axes.axhspan(
        mean - correlated, mean + correlated, color="C0", alpha=0.18, label="autocorrelation error"
    )
    try:
        binning = plateau_error(values)
        axes.axhspan(mean - binning, mean + binning, color="C2", alpha=0.18, label="binning error")
    except ValueError:
        binning = float("nan")
    if reference is not None:
        axes.axhline(reference, color="C3", lw=1.2, ls="--", label="exact")

    annotation = f"mean {mean:.6g}\ntau_int {tau:.2f}\nerror {correlated:.3g}"
    if np.isfinite(binning):
        annotation += f"\nbinning {binning:.3g}"
    for name in ("acceptance", "exchange_average", "sign_changes"):
        if hasattr(source, name):
            annotation += f"\n{name} {getattr(source, name):.6g}"
    axes.text(
        0.99,
        0.02,
        annotation,
        transform=axes.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )
    axes.set_xlabel("sweep")
    axes.set_ylabel(title)
    axes.set_title(title)
    axes.legend(fontsize=8, loc="upper right")
    return _to_png(figure)


def autocorrelation_function(source, title: str = "autocorrelation", lags: int = 200) -> bytes:
    """``rho(t)`` with the summation window marked.

    The window is where the Madras-Sokal criterion stopped the sum. A curve
    still well above zero there is a run too short to have measured its own
    correlation time, which no single number reports.
    """
    values = _values(source)
    centred = values - values.mean()
    variance = float(np.dot(centred, centred) / centred.size)
    reach = int(min(lags, centred.size // 4))
    if variance == 0.0 or reach < 2:
        correlations = np.zeros(max(reach, 2))
    else:
        correlations = np.array(
            [1.0]
            + [
                float(np.dot(centred[:-lag], centred[lag:]) / (centred.size - lag) / variance)
                for lag in range(1, reach)
            ]
        )

    tau = integrated_autocorrelation(values)
    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(7.0, 4.0))
    axes.axhline(0.0, color="0.6", lw=0.8)
    axes.plot(np.arange(correlations.size), correlations, lw=1.4, color="C0")
    window = 6.0 * tau
    if window < correlations.size:
        axes.axvline(window, color="C3", lw=1.2, ls="--", label=f"window 6 tau = {window:.1f}")
        axes.legend(fontsize=8)
    axes.set_xlabel("lag (sweeps)")
    axes.set_ylabel("rho")
    axes.set_title(f"{title}   tau_int = {tau:.2f}")
    return _to_png(figure)


def binning_curve(source, title: str = "binning analysis") -> bytes:
    """Error on the mean against bin size, with both estimates marked.

    The plateau is the honest error. Reaching one is the question this
    picture answers and a ``tau_int`` cannot: a curve still climbing at the
    largest usable bin has not converged, however confident the number
    attached to it looks.
    """
    values = _values(source)
    sizes, errors = binned_errors(values)
    tau = integrated_autocorrelation(values)
    naive = float(np.std(values) / np.sqrt(values.size))

    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(7.0, 4.0))
    axes.plot(sizes, errors, "o-", lw=1.4, ms=4, color="C0", label="binned error")
    axes.axhline(naive, color="0.5", lw=1.0, ls=":", label=f"naive {naive:.3g}")
    axes.axhline(
        naive * np.sqrt(2.0 * tau),
        color="C3",
        lw=1.2,
        ls="--",
        label=f"naive x sqrt(2 tau) = {naive * np.sqrt(2.0 * tau):.3g}",
    )
    axes.axhline(
        plateau_error(values), color="C2", lw=1.2, label=f"plateau {plateau_error(values):.3g}"
    )
    axes.set_xscale("log", base=2)
    axes.set_xlabel("bin size (sweeps)")
    axes.set_ylabel("error on the mean")
    axes.set_title(title)
    axes.legend(fontsize=8, loc="lower right")
    return _to_png(figure)


def growth_spectrum(report, title: str = "growth spectrum") -> bytes:
    """Measured growth against Floquet, with the round-off floor shown.

    Two panels, because the second explains the first. The upper one is the
    comparison; the lower is each mode's amplitude relative to the loudest,
    where the floored modes appear as a flat line near machine epsilon. A
    mode on that line is reporting round-off from the resonance rather than
    itself, and its rate in the upper panel is meaningless however steady it
    looks.
    """
    from particlesim.scenarios.cosmo.lattice import RESOLUTION_FLOOR

    measured = np.asarray(report.measured, dtype=float)
    predicted = np.asarray(report.predicted, dtype=float)
    relative = np.asarray(report.relative_amplitude, dtype=float)
    resolvable = np.asarray(report.resolvable, dtype=bool)
    modes = np.arange(measured.size)

    plt = _pyplot()
    figure, (upper, lower) = plt.subplots(
        2, 1, figsize=(7.5, 6.0), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    upper.axhline(0.0, color="0.6", lw=0.8)
    upper.plot(modes, predicted, "-", lw=1.6, color="C3", label="Floquet")
    upper.plot(modes[resolvable], measured[resolvable], "o", ms=5, color="C0", label="measured")
    if np.any(~resolvable):
        upper.plot(
            modes[~resolvable],
            measured[~resolvable],
            "x",
            ms=6,
            color="0.55",
            label="below the floor",
        )
    upper.set_ylabel("growth rate")
    upper.set_title(f"{title}   dynamic range {report.dynamic_range:.2g}")
    upper.legend(fontsize=8)

    lower.semilogy(modes, np.maximum(relative, 1e-20), "o-", ms=3, lw=1.0, color="C0")
    lower.axhline(
        RESOLUTION_FLOOR, color="C3", lw=1.2, ls="--", label=f"floor {RESOLUTION_FLOOR:.0e}"
    )
    lower.axhspan(1e-20, RESOLUTION_FLOOR, color="C3", alpha=0.12)
    lower.set_xlabel("mode")
    lower.set_ylabel("amplitude / loudest")
    lower.legend(fontsize=8, loc="lower left")
    return _to_png(figure)


__all__ = [
    "autocorrelation_function",
    "binning_curve",
    "growth_spectrum",
    "observable_series",
]
