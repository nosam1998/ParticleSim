"""Views for cosmology runs (design doc Section 5.7, Milestone 3).

Four pictures, each of which answers a question the numbers alone do not.

**The expansion history** shows whether a run bounced, and a bounce is the
one thing a table of numbers understates: the scale factor's minimum is a
sharp cusp and the expansion rate jumps through zero, so the picture says
in one glance what three columns say in three.

**The Hubble diagram** puts several cosmologies on the same axes, because
the interesting question about a model is never what its ``H(z)`` is but
how far it is from another model's.

**The potential landscape with the trajectory on it** is the picture that
makes an inflationary model comprehensible. A plateau, a pivot fifty-five
e-folds from the end, and the field falling off the edge into the minimum
are three separate numbers in a report and one shape here.

**The primordial spectra** are where the model meets the observation, with
the tilt and the tensor ratio annotated so the plot carries its own
summary.

The expansion-history view takes anything with ``time``, ``scale_factor``
and ``hubble`` arrays, which is every run in :mod:`particlesim.cosmo` --
the theory-driven background, an ekpyrotic contraction, a dilaton-driven
branch -- rather than one dataclass. They are the same picture and there is
no reason for three functions.
"""

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
    _pyplot().close(fig)
    return buf.getvalue()


def expansion_history(
    run, title: str = "expansion history", logarithmic: bool = False, time_label: str = "t"
) -> bytes:
    """``a(t)`` and ``H(t)``, with the turning point marked if there is one.

    ``run`` is anything carrying ``time``, ``scale_factor`` and ``hubble``.
    A bounce is found from the sign change of ``H`` rather than from the
    minimum of ``a``, because a run that merely ended while still
    contracting also has its smallest scale factor at the end.
    """
    time = np.asarray(run.time, dtype=float)
    scale = np.asarray(run.scale_factor, dtype=float)
    hubble = np.asarray(run.hubble, dtype=float)
    if not (time.shape == scale.shape == hubble.shape):
        raise ValueError(
            f"time, scale_factor and hubble must have the same shape, got "
            f"{time.shape}, {scale.shape}, {hubble.shape}"
        )

    plt = _pyplot()
    fig, (upper, lower) = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    upper.plot(time, scale, lw=1.6)
    upper.set_ylabel("scale factor $a$")
    if logarithmic:
        upper.set_yscale("log")
    upper.grid(alpha=0.3)

    crossings = np.where((hubble[:-1] < 0.0) & (hubble[1:] >= 0.0))[0]
    if crossings.size:
        index = int(crossings[0])
        span = hubble[index + 1] - hubble[index]
        weight = 0.0 if span == 0.0 else -hubble[index] / span
        at = time[index] + weight * (time[index + 1] - time[index])
        for axis in (upper, lower):
            axis.axvline(at, color="crimson", lw=0.9, ls="--")
        upper.annotate(
            "bounce",
            (at, scale[index]),
            textcoords="offset points",
            xytext=(8, 10),
            color="crimson",
        )

    lower.plot(time, hubble, lw=1.6, color="tab:orange")
    lower.axhline(0.0, color="0.5", lw=0.8)
    lower.set_ylabel("expansion rate $H$")
    lower.set_xlabel(time_label)
    lower.grid(alpha=0.3)
    upper.set_title(title)
    fig.tight_layout()
    return _to_png(fig)


def hubble_diagram(
    cosmologies,
    labels=None,
    redshifts=None,
    title: str = "expansion rate and distance",
) -> bytes:
    """``H(z)`` and the comoving distance for one or several cosmologies.

    Both panels are logarithmic in redshift, because the range that matters
    spans four decades: a supernova sample lives near ``z = 1`` and the CMB
    at ``z = 1100``, and a linear axis shows one or the other.
    """
    if not isinstance(cosmologies, (list, tuple)):
        cosmologies = [cosmologies]
    if not cosmologies:
        raise ValueError("give at least one cosmology")
    if labels is None:
        labels = [f"model {index + 1}" for index in range(len(cosmologies))]
    if len(labels) != len(cosmologies):
        raise ValueError(f"got {len(cosmologies)} cosmologies and {len(labels)} labels")
    if redshifts is None:
        redshifts = np.logspace(-2.0, 3.0, 160)
    redshifts = np.asarray(redshifts, dtype=float)

    plt = _pyplot()
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for cosmology, label in zip(cosmologies, labels, strict=True):
        left.loglog(redshifts, cosmology.hubble(redshifts), lw=1.5, label=label)
        right.loglog(redshifts, cosmology.comoving_distance(redshifts), lw=1.5, label=label)
    left.set_xlabel("redshift $z$")
    left.set_ylabel("$H(z)$ [km/s/Mpc]")
    right.set_xlabel("redshift $z$")
    right.set_ylabel("comoving distance [Mpc]")
    for axis in (left, right):
        axis.grid(alpha=0.3, which="both")
        if len(cosmologies) > 1:
            axis.legend(fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    return _to_png(fig)


def potential_landscape(
    potential,
    run=None,
    efolds_remaining: float | None = None,
    field_range: tuple[float, float] | None = None,
    title: str | None = None,
) -> bytes:
    """``V(phi)`` with the inflaton's trajectory on it.

    Given a run, the field values it actually visited are drawn over the
    potential, the end of inflation is where the trajectory stops, and
    ``efolds_remaining`` marks the pivot. The field range defaults to the
    trajectory's own, padded, rather than to a fixed interval: a plateau
    potential plotted on the wrong range is a horizontal line.
    """
    plt = _pyplot()
    fig, (upper, lower) = plt.subplots(
        2, 1, figsize=(7.2, 6.0), gridspec_kw={"height_ratios": [2, 1]}
    )

    if field_range is None:
        if run is None:
            low, high = potential.domain
            span = 10.0 if not np.isfinite(high - low) else (high - low)
            field_range = (max(low, 0.0) + 1e-3 * span, min(high, max(low, 0.0) + span))
        else:
            values = np.asarray(run.field, dtype=float)
            pad = 0.08 * (float(values.max()) - float(values.min()) + 1e-12)
            field_range = (float(values.min()) - pad, float(values.max()) + pad)
    fields = np.linspace(field_range[0], field_range[1], 600)
    inside = potential.contains(fields)
    upper.plot(fields[inside], potential.value(fields[inside]), lw=1.6, color="0.4")
    upper.set_ylabel(r"$V(\phi)$")
    upper.grid(alpha=0.3)

    if run is not None:
        trajectory = np.asarray(run.field, dtype=float)
        upper.plot(trajectory, potential.value(trajectory), lw=2.4, color="tab:blue", alpha=0.85)
        upper.plot(
            [trajectory[-1]],
            [float(potential.value(trajectory[-1]))],
            "s",
            color="crimson",
            ms=6,
            label=r"$\epsilon = 1$",
        )
        if efolds_remaining is not None:
            pivot = run.at_efolds_remaining(efolds_remaining)
            field = float(run.state(pivot)[0])
            upper.plot(
                [field],
                [float(potential.value(field))],
                "o",
                color="tab:green",
                ms=7,
                label=f"pivot, $N = {efolds_remaining:g}$",
            )
        upper.legend(fontsize=9)
        lower.semilogy(run.efolds, run.epsilon, lw=1.6, label=r"$\epsilon_H$")
        lower.axhline(1.0, color="crimson", lw=0.9, ls="--")
        lower.set_xlabel("e-folds $N$")
        lower.set_ylabel(r"$\epsilon_H$")
        lower.grid(alpha=0.3, which="both")
        lower.legend(fontsize=9)
    else:
        lower.semilogy(fields[inside], potential.epsilon(fields[inside]), lw=1.6)
        lower.axhline(1.0, color="crimson", lw=0.9, ls="--")
        lower.set_xlabel(r"$\phi$")
        lower.set_ylabel(r"$\epsilon_V$")
        lower.grid(alpha=0.3, which="both")

    upper.set_title(title or type(potential).__name__)
    fig.tight_layout()
    return _to_png(fig)


def primordial_spectra(spectrum, title: str = "primordial power spectra") -> bytes:
    """``P_R(k)`` and ``P_t(k)`` with the tilt and the tensor ratio annotated.

    The wavenumbers are drawn relative to the pivot, because the absolute
    scale of an inflationary comoving wavenumber is not observable without
    the post-inflationary expansion history -- the same reason the
    amplitude does not transfer to a Boltzmann code.
    """
    wavenumbers = np.asarray(spectrum.wavenumbers, dtype=float) / spectrum.pivot
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.loglog(wavenumbers, spectrum.scalar_power, "o-", lw=1.5, label=r"$P_\mathcal{R}$")
    axis.loglog(wavenumbers, spectrum.tensor_power, "s-", lw=1.5, label=r"$P_t$")
    axis.set_xlabel(r"$k/k_*$")
    axis.set_ylabel("power")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=9)
    axis.set_title(
        f"{title}\n$n_s = {spectrum.spectral_index:.4f}$,  "
        f"$r = {spectrum.tensor_to_scalar:.4g}$,  "
        f"$\\alpha_s = {spectrum.running:.2e}$,  "
        f"$N = {spectrum.efolds_remaining:g}$",
        fontsize=10,
    )
    fig.tight_layout()
    return _to_png(fig)
