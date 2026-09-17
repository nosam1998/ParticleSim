"""Views for linear-perturbation runs (design doc Sections 3.1, 5.7).

Two plots, both of which are conventions rather than data. A CMB spectrum
is almost always drawn as ``l(l+1)C_l/2pi`` in microkelvin squared against
a logarithmic multipole axis, and a matter power spectrum log-log in
``(h/Mpc, (Mpc/h)^3)``; the raw ``C_l`` and ``P(k)`` that
:mod:`particlesim.cosmo.linear` brings back are in neither of those forms.
The conversion lives here rather than in the adapter so that what comes out
of CLASS is what CLASS computed.

These functions take a :class:`~particlesim.cosmo.linear.LinearSpectra` and
nothing else, so they work on a stored run and do not need CLASS installed.
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


def angular_power(
    spectra,
    keys: tuple[str, ...] = ("tt",),
    title: str = "CMB angular power spectrum",
    mark_first_peak: bool = True,
) -> bytes:
    """``l(l+1)C_l/2pi`` in microkelvin squared, on a logarithmic multipole axis.

    The monopole and dipole are dropped: ``l = 0`` and ``l = 1`` are not
    observable and the ``l(l+1)`` prefactor makes them zero anyway, so
    keeping them only puts the axis origin somewhere misleading.

    The polarisation spectra are added as separate axes rather than on the
    temperature scale, because ``EE`` is two orders below ``TT`` and drawing
    them together on a linear scale hides one of them.
    """
    if not keys:
        raise ValueError("give at least one spectrum key to plot")
    missing = [key for key in keys if key not in spectra.cl]
    if missing:
        raise KeyError(f"no {missing} in this run; have {sorted(spectra.cl)}")

    plt = _pyplot()
    fig, axes = plt.subplots(len(keys), 1, figsize=(7.5, 3.2 * len(keys)), squeeze=False)
    visible = spectra.ell >= 2
    for axis, key in zip(axes[:, 0], keys, strict=True):
        power = spectra.band_power(key)[visible]
        axis.semilogx(spectra.ell[visible], power, lw=1.4)
        axis.set_ylabel(rf"$\ell(\ell+1)C_\ell^{{{key.upper()}}}/2\pi\ [\mu K^2]$")
        axis.grid(alpha=0.3)
        if key == "te":
            axis.axhline(0.0, color="0.5", lw=0.8)
        if mark_first_peak and key == "tt":
            multipole, height = spectra.first_peak()
            axis.plot([multipole], [height], "o", color="crimson", ms=5)
            axis.annotate(
                f"$\\ell_1 = {multipole}$",
                (multipole, height),
                textcoords="offset points",
                xytext=(8, -4),
                color="crimson",
            )
    axes[-1, 0].set_xlabel(r"multipole $\ell$")
    axes[0, 0].set_title(f"{title} ({spectra.backend})")
    fig.tight_layout()
    return _to_png(fig)


def matter_power(
    spectra, title: str = "linear matter power spectrum", hubble_parameter: float | None = None
) -> bytes:
    """``P(k)`` log-log, in CLASS's units unless ``hubble_parameter`` is given.

    CLASS returns ``P(k)`` in ``Mpc^3`` against ``k`` in ``1/Mpc``. Passing
    ``h`` converts to the ``(Mpc/h)^3`` against ``h/Mpc`` convention that
    galaxy surveys quote, which moves the curve by ``h^3`` and the axis by
    ``h`` -- a factor of three in amplitude that is a real source of
    confusion, so the axis labels say which one is being drawn.
    """
    if spectra.wavenumber is None or spectra.matter_power is None:
        raise ValueError(
            "this run has no matter power spectrum: include mPk in the outputs and "
            "set a non-zero max_wavenumber"
        )
    wavenumber = np.asarray(spectra.wavenumber, dtype=float)
    power = np.asarray(spectra.matter_power, dtype=float)
    if hubble_parameter is not None:
        wavenumber = wavenumber / hubble_parameter
        power = power * hubble_parameter**3
        xlabel = r"$k\ [h/\mathrm{Mpc}]$"
        ylabel = r"$P(k)\ [(\mathrm{Mpc}/h)^3]$"
    else:
        xlabel = r"$k\ [1/\mathrm{Mpc}]$"
        ylabel = r"$P(k)\ [\mathrm{Mpc}^3]$"

    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.loglog(wavenumber, power, lw=1.4)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.3, which="both")
    sigma = spectra.derived.get("sigma8")
    suffix = "" if sigma is None else rf"  ($\sigma_8 = {sigma:.4f}$)"
    axis.set_title(f"{title}{suffix}")
    fig.tight_layout()
    return _to_png(fig)
