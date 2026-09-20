"""Lattice views, and the binning analysis behind two of them.

Issue #67's acceptance is that the views render from stored data. They do:
every test here builds a :class:`Chain` or a ``GrowthReport`` from arrays and
draws it, and nothing in the file runs a simulation.

The numeric half is the binning analysis. It gives a *second* estimate of the
error on a mean, from different information than summing the autocorrelation
function, so the two agreeing is a check on both -- and the shape of the
curve answers a question neither number can: whether the run was long enough
to have measured its own correlation time.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.scenarios.cosmo.lattice import RESOLUTION_FLOOR, GrowthReport
from particlesim.solvers.lattice.euclidean import (
    MINIMUM_BINS,
    Chain,
    binned_errors,
    integrated_autocorrelation,
    plateau_error,
)
from particlesim.viz.lattice_views import (
    autocorrelation_function,
    binning_curve,
    growth_spectrum,
    observable_series,
)

PNG = b"\x89PNG"


def correlated(rho: float, size: int = 60000, seed: int = 0) -> np.ndarray:
    """An order-one autoregressive series, whose ``tau`` is known exactly."""
    noise = np.random.default_rng(seed).normal(size=size)
    series = np.zeros(size)
    for index in range(1, size):
        series[index] = rho * series[index - 1] + noise[index]
    return series


def stored_chain(rho: float = 0.8) -> Chain:
    values = correlated(rho)
    return Chain(
        values=values,
        energy_changes=np.random.default_rng(1).normal(size=values.size) * 0.01,
        acceptance=0.99,
    )


def stored_report() -> GrowthReport:
    """A report built from arrays, with a floored tail -- no evolution."""
    modes = 24
    predicted = np.where(np.arange(modes) < 5, 0.2, 0.0)
    measured = predicted + np.where(np.arange(modes) < 5, 1e-4, 0.06)
    relative = np.where(np.arange(modes) < 5, 1.0, 5e-16)
    return GrowthReport(
        frequencies_squared=np.linspace(0.0, 16.0, modes),
        measured=measured,
        predicted=predicted,
        relative_amplitude=relative,
        dynamic_range=2.0e15,
    )


# --- the binning analysis --------------------------------------------------


def test_the_binning_plateau_is_the_correlated_error():
    """``plateau = naive x sqrt(2 tau_int)``, on a process whose ``tau`` is exact.

    Two estimators of the same quantity from different information. They
    agree to a few percent from uncorrelated data up to ``rho = 0.9``, where
    the correlated error is four times the naive one.
    """
    for rho in (0.0, 0.5, 0.8, 0.9):
        series = correlated(rho)
        naive = float(np.std(series) / np.sqrt(series.size))
        summed = naive * np.sqrt(2.0 * integrated_autocorrelation(series))
        assert plateau_error(series) == pytest.approx(summed, rel=0.10)


def test_the_binning_curve_rises_from_the_naive_error_and_flattens():
    """Flat for uncorrelated data, climbing then level for correlated.

    The shape is the diagnostic. A curve still climbing at the largest usable
    bin has not converged, however confident the ``tau_int`` attached to it
    looks.
    """
    sizes, flat = binned_errors(correlated(0.0))
    assert np.max(flat) / flat[0] < 1.15

    sizes, rising = binned_errors(correlated(0.9))
    assert rising[0] < rising[len(rising) // 2]
    assert np.max(rising) / rising[0] > 3.0
    assert sizes[0] == 1
    assert np.all(np.diff(sizes) > 0)


def test_binning_refuses_a_series_too_short_to_read():
    """Below sixteen bins the spread of bin means is itself noise."""
    with pytest.raises(ValueError, match="at least 16 samples"):
        binned_errors(np.arange(MINIMUM_BINS - 1, dtype=float))


# --- the views, from stored data only --------------------------------------


def test_every_view_renders_from_a_stored_chain():
    chain = stored_chain()
    for png in (
        observable_series(chain, title="plaquette"),
        autocorrelation_function(chain),
        binning_curve(chain),
    ):
        assert png.startswith(PNG)
        assert len(png) > 5000


def test_the_series_view_takes_a_reference_and_a_bare_array():
    """A trace becomes a comparison once the exact value is on it."""
    assert observable_series(stored_chain(), reference=0.4464).startswith(PNG)
    assert observable_series(correlated(0.5)).startswith(PNG)


def test_the_growth_spectrum_renders_from_a_report_built_by_hand():
    """The acceptance, literally: stored arrays in, a picture out.

    The report here is constructed rather than measured, so the view is
    demonstrably reading data and not running anything.
    """
    png = growth_spectrum(stored_report(), title="preheating")
    assert png.startswith(PNG)
    assert len(png) > 5000


def test_the_report_the_spectrum_draws_has_a_floored_tail():
    """What the lower panel is for, asserted on the data behind it.

    Nineteen of the twenty-four modes sit below the resolution floor at a
    relative amplitude of ``5e-16``, and every one of them carries a
    plausible-looking growth rate of 0.06 where the prediction is zero.
    """
    report = stored_report()
    assert np.count_nonzero(~report.resolvable) == 19
    assert np.all(report.relative_amplitude[~report.resolvable] < RESOLUTION_FLOOR)
    assert np.all(report.measured[~report.resolvable] > 0.0)
    assert np.all(report.predicted[~report.resolvable] == 0.0)


def test_a_spectrum_with_nothing_floored_still_renders():
    """The view must not depend on there being a failure to show."""
    report = stored_report()
    report.relative_amplitude = np.ones_like(report.relative_amplitude)
    assert np.all(report.resolvable)
    assert growth_spectrum(report).startswith(PNG)


# --- degenerate inputs -----------------------------------------------------


def test_a_constant_series_does_not_break_the_autocorrelation_view():
    """Zero variance has no correlation function, and the view says so quietly."""
    constant = np.full(5000, 0.25)
    assert autocorrelation_function(constant).startswith(PNG)
    assert integrated_autocorrelation(constant) == 0.5


def test_a_series_too_short_to_plot_is_refused():
    with pytest.raises(ValueError, match="at least two measurements"):
        observable_series(np.array([1.0]))
    with pytest.raises(ValueError, match="at least two measurements"):
        autocorrelation_function(np.zeros((4, 4)))


def test_the_series_view_survives_a_chain_too_short_to_bin():
    """Short runs still draw; only the binning band is left off.

    A view that raised on a short chain would be useless for exactly the
    runs a reader most wants to look at before committing to a long one.
    """
    short = Chain(
        values=np.linspace(0.0, 1.0, MINIMUM_BINS - 2),
        energy_changes=np.zeros(MINIMUM_BINS - 2),
        acceptance=1.0,
    )
    assert observable_series(short).startswith(PNG)
