"""Critical-collapse threshold search (design doc Section 5.4, Milestone 1)."""

import numpy as np
import pytest

from particlesim.analysis.critical_collapse import (
    InitialDataTooStrong,
    ScalingFit,
    bisect_threshold,
    evolve_to_verdict,
    fit_scaling,
    run_amplitude,
)
from particlesim.core.spherical import SphericalGrid
from particlesim.solvers.nr.spherical import ScalarCollapse, gaussian_pulse

# A thin shell: width / r0 = 1/8, so the initial slice stays weak right up
# to threshold. The reference threshold below was measured on this family.
R0, WIDTH, RMAX = 4.0, 0.5, 10.0
REFERENCE_PSTAR = 8.470419e-4  # n = 400, bracketed to a relative width of 6e-7


def shell(sim: ScalarCollapse, amplitude: float):
    return gaussian_pulse(sim.grid, amplitude=amplitude, r0=R0, width=WIDTH, ingoing=True)


def sim_factory(n: int):
    return lambda: ScalarCollapse(SphericalGrid(r_max=RMAX, n=n), courant=0.25)


def test_fit_recovers_choptuiks_exponent_from_ideal_data():
    """The fit itself must be right before a measurement through it means
    anything, so it is checked against the law it is looking for."""
    eps = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3)
    peaks = tuple(e ** (-2 * 0.374) for e in eps)
    fit = fit_scaling(eps, peaks)
    assert not fit.saturated
    assert fit.gamma == pytest.approx(0.374, abs=1e-6)
    assert fit.slope == pytest.approx(-0.748, abs=1e-6)


def test_fit_refuses_an_exponent_when_the_peaks_have_plateaued():
    """A plateau is the absence of a measurement, not a noisy one.

    These are the peaks actually measured on a uniform grid at n = 400: two
    decades in 1 - p/p* and five per cent in the peak. Choptuik's exponent
    predicts a factor of 5.6 per decade, so fitting a line through this and
    reporting the slope would be reporting the grid spacing.
    """
    eps = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3)
    measured = (9.1118e3, 9.1432e3, 9.1713e3, 9.5939e3, 9.5857e3)
    fit = fit_scaling(eps, measured)
    assert fit.saturated
    assert fit.gamma is None
    assert fit.slope is not None  # still reported, so the plateau is visible
    assert "adaptive mesh refinement" in fit.note.lower()
    assert "no exponent" in str(fit)


def test_fit_needs_at_least_three_points():
    fit = fit_scaling((1e-1, 1e-2), (1.0, 5.6))
    assert fit.gamma is None and not fit.saturated
    assert "at least three" in fit.note


def test_fit_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="same length"):
        fit_scaling((1e-1, 1e-2), (1.0,))


def test_a_genuine_scaling_run_is_not_called_saturated():
    """The saturation guard must not fire on a real measurement.

    Ideal data with realistic scatter still swings far more than the
    threshold, so a code that did resolve the critical solution would be
    reported rather than suppressed.
    """
    rng = np.random.default_rng(0)
    eps = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3)
    peaks = tuple(e ** (-0.748) * float(rng.normal(1.0, 0.05)) for e in eps)
    fit = fit_scaling(eps, peaks)
    assert not fit.saturated
    assert fit.gamma == pytest.approx(0.374, abs=0.03)


def test_strong_initial_data_is_refused_rather_than_measured():
    """The trap this search exists to avoid.

    A fat shell saturates 2m/r on the initial slice, so a lapse-based
    classifier fires at t = 0 and a bisection converges on the initial data.
    The symptom is a threshold identical at every resolution, which looks
    like robustness. Refusing the data is the only way to make that visible.
    """
    sim = ScalarCollapse(SphericalGrid(r_max=12.0, n=200))
    fat = gaussian_pulse(sim.grid, amplitude=7.8e-3, r0=5.0, width=1.0, ingoing=True)
    with pytest.raises(InitialDataTooStrong, match="initial slice"):
        evolve_to_verdict(sim, fat, t_end=1.0)


def test_weak_initial_data_is_accepted():
    sim = ScalarCollapse(SphericalGrid(r_max=RMAX, n=150))
    run = evolve_to_verdict(sim, shell(sim, 1e-4), t_end=1.0)
    assert run.initial_compactness < 0.1
    assert run.verdict == "subcritical"


def test_a_bracket_that_does_not_straddle_the_threshold_is_refused():
    """An unverified bracket makes every iteration after it meaningless while
    still returning a confident-looking number."""
    make = sim_factory(120)
    with pytest.raises(ValueError, match="does not collapse"):
        bisect_threshold(make, shell, 1e-5, 5e-5, t_end=6.0, iterations=1)
    with pytest.raises(ValueError, match="already collapses"):
        bisect_threshold(make, shell, 2e-3, 3e-3, t_end=12.0, iterations=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_the_collapse_threshold_is_sharp_and_resolution_stable():
    """The measurable half of the Choptuik benchmark.

    The exponent needs adaptive refinement (see ``fit_scaling``), but the
    threshold itself does not: it is a bisection on a binary outcome, and it
    converges. Pinning it is what catches a solver that has stopped being
    able to collapse at all -- which is exactly what three separate bugs in
    the spherical evolution had done, each of them silently.

    The reference value comes from an n = 400 search bracketed to a relative
    width of 6e-7. A coarser grid must land near it, and either side of it
    must behave oppositely.
    """
    make = sim_factory(150)
    threshold = bisect_threshold(make, shell, 5e-4, 1e-3, t_end=16.0, iterations=8, resolution=150)
    assert threshold.relative_width < 5e-3
    assert threshold.estimate == pytest.approx(REFERENCE_PSTAR, rel=0.02)

    below = run_amplitude(make, shell, threshold.lower * 0.98, t_end=16.0)
    above = run_amplitude(make, shell, threshold.upper * 1.02, t_end=16.0)
    assert below.verdict == "subcritical"
    assert above.verdict == "supercritical"
    # Below threshold the field disperses and 2m/r stays modest. Above it,
    # 2m/r runs up toward one -- either far enough to be recorded, or far
    # enough that the constraint solve refuses to step past it, which on a
    # coarse grid happens mid-stage and leaves the recorded value short.
    assert below.peak_compactness < 0.7
    assert above.peak_compactness > 0.75 or above.slicing_refused
    assert above.peak_compactness > 1.4 * below.peak_compactness


def test_scaling_fit_is_printable_either_way():
    good = ScalingFit((1e-1,), (1.0,), -0.748, 0.374, False, "")
    assert "0.374" in str(good)
    bad = ScalingFit((1e-1,), (1.0,), -0.01, None, True, "plateau")
    assert "plateau" in str(bad)
