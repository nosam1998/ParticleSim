"""Views for particle-in-cell runs (design doc Section 5.7, Milestone 2).

Phase space is the picture that matters. A plot of density against position
shows a two-stream instability as a wobble; the same run in ``(x, v)`` shows
the two beams, the vortices they roll into, and the moment trapping stops
the growth, which is the thing the run was for.

Beyond about a hundred thousand macro-particles a scatter plot is a solid
block of ink and stops carrying information, so :func:`phase_space` switches
to a two-dimensional histogram past a threshold rather than drawing a
misleading picture faster.

Emittance and spectra are here too, defined the way an accelerator would
define them, because a number called emittance that is not the usual one is
worse than no number at all.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

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


# --- beam diagnostics -------------------------------------------------------


@dataclass(frozen=True)
class BeamMoments:
    """Second moments of a bunch in one plane, and what follows from them."""

    mean_position: float
    mean_momentum: float
    rms_position: float
    rms_momentum: float
    correlation: float
    emittance: float

    @property
    def divergence(self) -> float:
        """RMS angle, ``sigma_p / <p_longitudinal>`` is the caller's business;
        this is the momentum spread itself."""
        return self.rms_momentum


def beam_moments(position, momentum, weight=None) -> BeamMoments:
    """RMS emittance in one transverse plane.

    ``eps = sqrt(<x^2><p^2> - <x p>^2)`` with the means subtracted first.
    This is the *normalized* emittance when ``momentum`` is ``gamma beta``,
    which is what :class:`~particlesim.solvers.pic.particles.Species` carries,
    and it is the quantity that is conserved under acceleration. Passing a
    velocity instead gives the geometric emittance, which is not, and the
    difference is a factor of ``gamma beta`` that silently grows.

    The determinant is not evaluated as ``<x^2><p^2> - <x p>^2``. For a well
    collimated beam those two are nearly equal and the difference loses most
    of its digits: on a perfectly chirped beam, where the true emittance is
    zero, that form returns 4e-8 rather than something near machine epsilon.

    Instead ``p`` is regressed on ``x`` and the residual spread taken, since
    ``det = <x^2> var(p - (<xp>/<x^2>) x)`` identically. A perfect chirp then
    leaves residuals that are zero to round-off, and the emittance comes out
    at 1e-16 where it belongs.
    """
    x = np.asarray(position, dtype=float)
    p = np.asarray(momentum, dtype=float)
    if x.shape != p.shape:
        raise ValueError("position and momentum must have the same shape")
    if x.size < 2:
        raise ValueError("need at least two particles for a second moment")
    w = np.ones_like(x) if weight is None else np.asarray(weight, dtype=float)
    total = w.sum()
    if total <= 0:
        raise ValueError("total weight must be positive")

    mx = float((w * x).sum() / total)
    mp = float((w * p).sum() / total)
    dx, dp = x - mx, p - mp
    xx = float((w * dx * dx).sum() / total)
    pp = float((w * dp * dp).sum() / total)
    xp = float((w * dx * dp).sum() / total)
    if xx > 0.0:
        residual = dp - (xp / xx) * dx
        determinant = xx * float((w * residual * residual).sum() / total)
    else:
        determinant = xx * pp - xp * xp
    determinant = max(determinant, 0.0)
    return BeamMoments(
        mean_position=mx,
        mean_momentum=mp,
        rms_position=float(np.sqrt(xx)),
        rms_momentum=float(np.sqrt(pp)),
        correlation=xp,
        emittance=float(np.sqrt(determinant)),
    )


def energy_spectrum(momentum, mass: float = 1.0, bins: int = 64, weight=None):
    """Kinetic-energy histogram, ``(gamma - 1) m`` per particle.

    Returns bin centres and weights rather than a figure, so the same numbers
    can be plotted, fitted or asserted on.
    """
    p = np.asarray(momentum, dtype=float)
    if p.ndim == 1:
        gamma = np.sqrt(1.0 + p**2)
    else:
        gamma = np.sqrt(1.0 + np.sum(p**2, axis=-1))
    energy = (gamma - 1.0) * mass
    w = None if weight is None else np.asarray(weight, dtype=float)
    counts, edges = np.histogram(energy, bins=bins, weights=w)
    return 0.5 * (edges[:-1] + edges[1:]), counts


# --- figures ----------------------------------------------------------------


def phase_space(
    position,
    momentum,
    title: str = "phase space",
    xlabel: str = "x",
    ylabel: str = "u",
    scatter_limit: int = 100_000,
    bins: int = 240,
) -> bytes:
    """``(x, u)`` for one species, as points or as a density.

    Past ``scatter_limit`` particles the scatter becomes a solid block and a
    histogram carries more, so the switch happens on its own rather than
    leaving the caller to notice the plot has stopped meaning anything.
    """
    x = np.asarray(position, dtype=float).ravel()
    u = np.asarray(momentum, dtype=float).ravel()
    if x.shape != u.shape:
        raise ValueError("position and momentum must have the same number of entries")
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    if x.size <= scatter_limit:
        size = max(0.4, 6.0 - np.log10(max(x.size, 1)))
        ax.scatter(x, u, s=size, linewidths=0, alpha=0.5, color="#2d5bd7")
    else:
        ax.hist2d(x, u, bins=bins, cmap="magma")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} ({x.size} macro-particles)")
    ax.grid(alpha=0.2)
    return _to_png(fig)


def field_snapshot(x, fields: dict[str, np.ndarray], title: str = "fields") -> bytes:
    """Several field components against position on one axis."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for label, values in fields.items():
        ax.plot(np.asarray(x), np.asarray(values), lw=1.2, label=label)
    ax.set_xlabel("x")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    if len(fields) > 1:
        ax.legend(fontsize=8)
    return _to_png(fig)


def spectrum_plot(centres, counts, title: str = "energy spectrum") -> bytes:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.step(np.asarray(centres), np.asarray(counts), where="mid", color="#b3261e")
    ax.set_xlabel("kinetic energy")
    ax.set_ylabel("weight")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    return _to_png(fig)


def growth_history(
    times, amplitudes, rate: float | None = None, title: str = "mode growth"
) -> bytes:
    """Mode amplitude against time on a log axis, with an optional slope.

    Drawing the predicted rate through the data is the point: an exponential
    on a log axis is a straight line, and whether the measurement lies on the
    predicted one is visible at a glance in a way a fitted number is not.
    """
    plt = _pyplot()
    t = np.asarray(times, dtype=float)
    a = np.abs(np.asarray(amplitudes, dtype=float))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.semilogy(t, np.maximum(a, 1e-300), lw=1.2, color="#2d5bd7", label="measured")
    if rate is not None:
        anchor = int(len(t) * 0.2)
        reference = a[anchor] * np.exp(rate * (t - t[anchor]))
        ax.semilogy(t, reference, "--", lw=1.0, color="#b3261e", label=f"exp({rate:.3f} t)")
        ax.legend(fontsize=8)
    ax.set_xlabel("t")
    ax.set_ylabel("|mode amplitude|")
    ax.set_title(title)
    ax.grid(alpha=0.2, which="both")
    return _to_png(fig)
