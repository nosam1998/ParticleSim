"""Critical-collapse threshold search (design doc Section 5.4, Milestone 1).

Choptuik 1993 (PRL 70, 9) found that a one-parameter family of imploding
scalar data has a sharp threshold ``p*`` between dispersal and black-hole
formation, and that near it the subcritical peak curvature obeys

    max |R| ~ (p* - p)^(-2 gamma),    gamma = 0.374

with the same exponent for every family. Measuring ``gamma`` is a strong
test of a spherical evolution code, because the exponent comes from a
self-similar solution nobody puts into the initial data.

This module runs the search and fits the law, and :class:`ScalingFit`
refuses an exponent when the peaks have plateaued. ``docs/benchmarks.md``
has the measurements, and the story of a plateau that was not what it
seemed.

Two guards here exist because both failures are easy to walk into and
neither announces itself.

**The initial slice must be weak.** A collapse classifier watches the
lapse. If the initial data is already compact enough to collapse the lapse
by itself, the classifier fires on ``t = 0`` and the bisection converges on
a property of the initial data rather than of the evolution. The symptom is
a threshold that is identical at every resolution, which reads like
robustness and is the opposite. :func:`evolve_to_verdict` refuses such data.

**The shell must be thin compared with its radius.** Compactness on the
initial slice is roughly ``width / r0`` times compactness at focus, so a
thin shell reaches threshold while ``t = 0`` is still weak and a fat one
never does: it saturates ``2m/r`` on the initial slice first, and then every
amplitude the family can express is rejected by the guard above.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from particlesim.analysis.spherical_diagnostics import ricci_scalar
from particlesim.solvers.nr.spherical import (
    PolarSlicingBreakdown,
    ScalarCollapse,
    SphericalState,
)

SUBCRITICAL = "subcritical"
SUPERCRITICAL = "supercritical"


class InitialDataTooStrong(ValueError):
    """Raised when the initial slice would trip the collapse classifier itself."""


@dataclass(frozen=True)
class CollapseRun:
    """The outcome of one evolution."""

    amplitude: float
    verdict: str
    peak_ricci: float
    peak_time: float
    peak_compactness: float
    initial_compactness: float
    slicing_refused: bool = False

    @property
    def collapsed(self) -> bool:
        return self.verdict == SUPERCRITICAL


@dataclass(frozen=True)
class Threshold:
    """A bracket on ``p*``: the largest subcritical and smallest supercritical
    amplitude the search confirmed."""

    lower: float
    upper: float
    resolution: int
    evaluations: int

    @property
    def estimate(self) -> float:
        return 0.5 * (self.lower + self.upper)

    @property
    def relative_width(self) -> float:
        return (self.upper - self.lower) / self.estimate


@dataclass(frozen=True)
class ScalingFit:
    """A fit of ``ln max|R|`` against ``ln(1 - p/p*)``.

    ``gamma`` is ``None`` whenever the peaks saturated. The critical
    solution is discretely self-similar with echoing period ``Delta = 3.44``
    in the logarithm of scale, so each successive echo lives on a region
    ``exp(3.44) = 31`` times smaller than the last. A uniform grid spanning
    ``r_max`` in steps of ``dr`` carries a fixed dynamic range, and once the
    self-similar structure falls below ``dr`` the peak curvature stops
    tracking the solution and reports what the grid can represent instead.
    Reporting an exponent fitted through that plateau would be reporting the
    grid spacing.

    A plateau is also what an instability looks like. Every uniform-grid
    measurement here once saturated near ``9e3`` at 400 cells, and it was
    not the dynamic range: the solver was making energy at the origin and
    holding a grid-scale curvature there. Fixed, the peaks grow toward
    threshold as the law says. The guard cannot tell the two apart, and
    does not need to: either way there is no exponent in the data.

    What it cannot catch is the opposite failure, peaks that still grow but
    are not converged. On the same grid they converge with resolution only
    down to ``1 - p/p* = 3e-3``; closer, they grow with refinement, and a
    line through them returns a number that is not the exponent. Only a
    comparison across resolutions shows that.
    """

    epsilons: tuple[float, ...]
    peaks: tuple[float, ...]
    slope: float | None
    gamma: float | None
    saturated: bool
    note: str

    CHOPTUIK_GAMMA = 0.374

    def __str__(self) -> str:
        if self.gamma is None:
            return f"no exponent: {self.note}"
        return f"gamma = {self.gamma:.4f} (Choptuik {self.CHOPTUIK_GAMMA})"


def evolve_to_verdict(
    sim: ScalarCollapse,
    state: SphericalState,
    t_end: float,
    lapse_threshold: float = 1e-4,
    inner_radius: float = 1.5,
    min_initial_lapse: float = 0.5,
) -> CollapseRun:
    """Evolve until the lapse collapses, the slicing gives out, or time runs out.

    ``inner_radius`` bounds where the peak curvature is looked for. The
    critical solution lives at the centre; the outer boundary carries the
    reflection the Sommerfeld condition fails to absorb, and letting that
    into the maximum is how boundary error gets reported as physics.
    """
    a0, alpha0 = sim.solve_metric(state.Phi, state.Pi)
    initial_compactness = float((1.0 - 1.0 / a0**2).max())
    if float(alpha0.min()) < min_initial_lapse:
        raise InitialDataTooStrong(
            f"the initial slice already has min(alpha) = {alpha0.min():.3e} and "
            f"2m/r = {initial_compactness:.3f}. A collapse classifier would fire "
            "on t = 0, so a threshold search on this family would converge on "
            "the initial data rather than the evolution. Use a thinner shell "
            "(smaller width / r0), which reaches threshold while t = 0 is weak."
        )

    peak, peak_time = 0.0, 0.0
    # A refined solver steps its finest level many times per step of its
    # base, and near threshold the peak is shorter than a base step; one
    # that can report what its finest level passed through is asked to.
    watch = getattr(sim, "watch_ricci", None)
    if watch is not None:
        watch(inner_radius)
    peak_compactness = initial_compactness
    verdict, refused = SUBCRITICAL, False
    try:
        for _ in range(int(t_end / sim.dt)):
            state = sim.step(state, sim.dt)
            a, alpha = sim.solve_metric(state.Phi, state.Pi)
            # Recomputed every step: an adaptive solver's radii change as its
            # levels come and go.
            inner = sim.r < inner_radius
            value = float(np.abs(ricci_scalar(a, state.Phi, state.Pi))[inner].max())
            if watch is not None:
                value = max(value, sim.ricci_between_steps())
            if value > peak:
                peak, peak_time = value, state.t
            peak_compactness = max(peak_compactness, float((1.0 - 1.0 / a**2).max()))
            if float(alpha.min()) < lapse_threshold:
                verdict = SUPERCRITICAL
                break
    except PolarSlicingBreakdown:
        # Polar-areal coordinates do not cover a trapped region, so the
        # integration refusing to step past 2m/r = 1 is collapse reported in
        # the only way this slicing can report it.
        #
        # The refusal happens inside a Runge-Kutta stage, before the step it
        # belongs to is accepted, so ``peak_compactness`` is the last value
        # from a completed step and undershoots. On a coarse grid the gap is
        # wide. ``slicing_refused`` is therefore the reliable signal that
        # 2m/r reached one, and the recorded compactness is a lower bound.
        verdict, refused = SUPERCRITICAL, True

    return CollapseRun(
        amplitude=float("nan"),
        verdict=verdict,
        peak_ricci=peak,
        peak_time=peak_time,
        peak_compactness=peak_compactness,
        initial_compactness=initial_compactness,
        slicing_refused=refused,
    )


PulseFamily = Callable[[ScalarCollapse, float], SphericalState]


def run_amplitude(
    make_sim: Callable[[], ScalarCollapse],
    family: PulseFamily,
    amplitude: float,
    t_end: float,
    **kw,
) -> CollapseRun:
    sim = make_sim()
    run = evolve_to_verdict(sim, family(sim, amplitude), t_end, **kw)
    return CollapseRun(
        amplitude=amplitude,
        verdict=run.verdict,
        peak_ricci=run.peak_ricci,
        peak_time=run.peak_time,
        peak_compactness=run.peak_compactness,
        initial_compactness=run.initial_compactness,
        slicing_refused=run.slicing_refused,
    )


def bisect_threshold(
    make_sim: Callable[[], ScalarCollapse],
    family: PulseFamily,
    lower: float,
    upper: float,
    t_end: float,
    iterations: int = 14,
    resolution: int = 0,
    **kw,
) -> Threshold:
    """Bisect between a subcritical ``lower`` and a supercritical ``upper``.

    The bracket is verified rather than assumed. A bracket whose ends do not
    straddle the threshold makes every subsequent iteration meaningless while
    still returning a confident-looking number.
    """
    evaluations = 2
    if run_amplitude(make_sim, family, lower, t_end, **kw).collapsed:
        raise ValueError(f"lower bracket {lower:g} already collapses; lower it")
    if not run_amplitude(make_sim, family, upper, t_end, **kw).collapsed:
        raise ValueError(f"upper bracket {upper:g} does not collapse; raise it")

    for _ in range(iterations):
        mid = 0.5 * (lower + upper)
        evaluations += 1
        if run_amplitude(make_sim, family, mid, t_end, **kw).collapsed:
            upper = mid
        else:
            lower = mid
    return Threshold(lower=lower, upper=upper, resolution=resolution, evaluations=evaluations)


def subcritical_scaling(
    make_sim: Callable[[], ScalarCollapse],
    family: PulseFamily,
    threshold: Threshold,
    epsilons: tuple[float, ...] = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3),
    t_end: float = 16.0,
    saturation_ratio: float = 2.0,
    **kw,
) -> ScalingFit:
    """Fit ``max|R|`` against ``1 - p/p*`` and say whether the fit means anything.

    Saturation is declared when the peaks move by less than ``saturation_ratio``
    while ``eps`` moves by more than a decade. Choptuik's exponent predicts a
    factor of ``10^0.748 = 5.6`` per decade, so a plateau is not a noisy
    measurement of the exponent, it is the absence of one.
    """
    eps_used: list[float] = []
    peaks: list[float] = []
    for eps in epsilons:
        run = run_amplitude(make_sim, family, threshold.upper * (1.0 - eps), t_end, **kw)
        if run.collapsed:
            # Above threshold after all: p* is known only to a bracket, so
            # the smallest epsilons can fall on the wrong side of it.
            continue
        eps_used.append(eps)
        peaks.append(run.peak_ricci)

    return fit_scaling(tuple(eps_used), tuple(peaks), saturation_ratio)


def fit_scaling(
    epsilons: tuple[float, ...],
    peaks: tuple[float, ...],
    saturation_ratio: float = 2.0,
) -> ScalingFit:
    """Fit an exponent to measured peaks, or explain why there isn't one.

    Separated from the evolution so the decision can be tested on numbers
    rather than on a four-minute run.
    """
    if len(epsilons) != len(peaks):
        raise ValueError("epsilons and peaks must have the same length")
    if len(epsilons) < 3:
        return ScalingFit(
            epsilons,
            peaks,
            None,
            None,
            False,
            f"only {len(epsilons)} subcritical points; need at least three to fit",
        )

    span = max(epsilons) / min(epsilons)
    swing = max(peaks) / min(peaks)
    slope = float(np.polyfit(np.log(epsilons), np.log(peaks), 1)[0])

    if span > 10.0 and swing < saturation_ratio:
        return ScalingFit(
            epsilons,
            peaks,
            slope,
            None,
            True,
            f"peak curvature moved by only {swing:.2f}x while 1 - p/p* moved by "
            f"{span:.0f}x. The self-similar structure has fallen below the grid "
            "spacing, so the peak reports what the grid can represent rather "
            "than the solution. Adaptive mesh refinement is what closes this, "
            "not a longer run or a smaller epsilon",
        )
    return ScalingFit(
        epsilons,
        peaks,
        slope,
        -slope / 2.0,
        False,
        f"fitted over {span:.0f}x in 1 - p/p*, peaks spanning {swing:.1f}x",
    )
