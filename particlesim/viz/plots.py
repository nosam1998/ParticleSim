"""Standard figures, rendered to PNG bytes for embedding (design doc Section 5.7)."""

from __future__ import annotations

import io

import numpy as np


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt = _pyplot()
    plt.close(fig)
    return buf.getvalue()


def diverging_slice(
    data: np.ndarray,
    extent: list[tuple[float, float]],
    title: str,
    label: str,
    axis: int = 2,
) -> bytes:
    """A mid-plane slice on a symmetric diverging scale centred on zero.

    Centring matters: an energy-density map on an uncentred scale makes a
    sign change look like a magnitude change.
    """
    plt = _pyplot()
    k = data.shape[axis] // 2
    sl = np.take(data, k, axis=axis)
    lim = float(np.abs(sl).max()) or 1.0
    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    im = ax.imshow(
        sl.T,
        origin="lower",
        extent=[*extent[0], *extent[1]],
        cmap="RdBu",
        vmin=-lim,
        vmax=lim,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label=label)
    return _to_png(fig)


def line_cut(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str) -> bytes:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(x, y, lw=1.6)
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    return _to_png(fig)


def warp_figures(
    fields: dict[str, np.ndarray], extent: list[tuple[float, float]]
) -> dict[str, bytes]:
    """The standard warp figure set from an analyzer's fields."""
    figs: dict[str, bytes] = {}
    rho = fields.get("energy_density")
    if rho is not None:
        figs["Energy density, z = 0"] = diverging_slice(
            rho, extent, "Eulerian energy density", "rho"
        )
        n = rho.shape
        ax_y = np.linspace(extent[1][0], extent[1][1], n[1])
        figs["Energy density along y at x = 0"] = line_cut(
            ax_y, rho[n[0] // 2, :, n[2] // 2], "Cut along y", "y", "rho"
        )
    for name, label in (
        ("WEC_min", "WEC minimum"),
        ("NEC_min", "NEC minimum"),
        ("horizon_indicator", "horizon indicator"),
        ("kretschmann", "Kretschmann"),
    ):
        arr = fields.get(name)
        if arr is not None and arr.ndim == 3:
            figs[f"{label}, z = 0"] = diverging_slice(arr, extent, label, name)
    return figs
