"""Critical-collapse threshold search (design doc Section 5.4, Milestone 1)."""

import numpy as np
import pytest

from particlesim.analysis.critical_collapse import (
    InitialDataTooStrong,
    ScalingFit,
    Threshold,
    bisect_threshold,
    evolve_to_verdict,
    fit_scaling,
    run_amplitude,
    subcritical_scaling,
)
from particlesim.core.spherical import SphericalGrid
from particlesim.solvers.nr.spherical import ScalarCollapse, gaussian_pulse

# A thin shell: width / r0 = 1/8, so the initial slice stays weak right up
# to threshold. The reference threshold below was measured on this family.
R0, WIDTH, RMAX = 4.0, 0.5, 10.0
# n = 400, bracketed to a relative width of 4.5e-7. 8.482933e-4 at n = 800.
REFERENCE_PSTAR = 8.483530e-4


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

    These are peaks once measured on a uniform grid at n = 400: two decades
    in 1 - p/p* and five per cent in the peak. Choptuik's exponent predicts a
    factor of 5.6 per decade, so fitting a line through this and reporting
    the slope would have been reporting a defect -- which it was, though not
    the one first blamed. The plateau was an origin instability holding a
    grid-scale curvature, not the grid failing to resolve the critical
    solution; see test_uniform_grid_resolves_the_scaling_law_it_once_missed.
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
    width of 4.5e-7. A coarser grid must land near it, and either side of it
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


@pytest.mark.slow
@pytest.mark.benchmark
def test_uniform_grid_resolves_the_scaling_law_it_once_missed():
    """Peak curvature grows toward threshold, as Choptuik's law says it must.

    At this resolution it used not to: three decades in ``1 - p/p*`` moved
    the peak by six per cent, around 9e3, and the plateau was read as the
    grid's dynamic range running out. It was an origin instability -- the
    ``Pi`` equation written in a form that makes energy at the origin, which
    held a grid-scale curvature there near threshold and made mass while it
    did (test_spherical_origin.py). With it fixed the peaks grow by a factor
    of four per decade here.

    This pins that the law is seen, not the exponent's value. The peaks
    converge with resolution down to ``1 - p/p* = 3e-3`` and grow with it
    closer than that -- 3.0e3, 3.4e3 and 5.1e3 at 1e-3 on 400, 800 and 1600
    cells -- so the grid does run out of dynamic range, only much later than
    the plateau suggested, and an exponent needs the refinement of #111.
    """
    make = sim_factory(400)
    threshold = Threshold(REFERENCE_PSTAR, REFERENCE_PSTAR, resolution=400, evaluations=0)
    fit = subcritical_scaling(make, shell, threshold, epsilons=(1e-2, 3e-3, 1e-3), t_end=10.0)
    assert len(fit.epsilons) == 3
    assert not fit.saturated
    assert max(fit.peaks) / min(fit.peaks) > 3.5
    assert 0.25 < fit.gamma < 0.45


def test_scaling_fit_is_printable_either_way():
    good = ScalingFit((1e-1,), (1.0,), -0.748, 0.374, False, "")
    assert "0.374" in str(good)
    bad = ScalingFit((1e-1,), (1.0,), -0.01, None, True, "plateau")
    assert "plateau" in str(bad)
