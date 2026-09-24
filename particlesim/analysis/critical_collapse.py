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
from scipy.optimize import least_squares

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

    **The law is not a straight line.** Because the critical solution echoes,
    the peak a subcritical run reaches depends on where in the echo it
    leaves, so ``ln max|R|`` carries a periodic ripple on top of the power
    law, of period ``Delta / (2 gamma)`` in ``ln(1 - p/p*)`` -- two decades --
    and here of amplitude 0.3 (Hod and Piran 1997, PRD 55, 440). A line
    through three decades of refined data read 0.367 to 0.391 depending on
    where the decades began. When the data span more than a period, the
    ripple is fitted along with the line, and ``delta`` is the echoing period
    it implies.
    """

    epsilons: tuple[float, ...]
    peaks: tuple[float, ...]
    slope: float | None
    gamma: float | None
    saturated: bool
    note: str
    delta: float | None = None

    CHOPTUIK_GAMMA = 0.374
    CHOPTUIK_DELTA = 3.44

    def __str__(self) -> str:
        if self.gamma is None:
            return f"no exponent: {self.note}"
        out = f"gamma = {self.gamma:.4f} (Choptuik {self.CHOPTUIK_GAMMA})"
        if self.delta is not None:
            out += f", Delta = {self.delta:.3f} (Choptuik {self.CHOPTUIK_DELTA})"
        return out


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
    periodic = fit_fine_structure(epsilons, peaks)
    if periodic is not None:
        gamma, delta = periodic
        return ScalingFit(
            epsilons,
            peaks,
            slope,
            gamma,
            False,
            f"power law and echo ripple fitted over {span:.0f}x in 1 - p/p*, peaks "
            f"spanning {swing:.1f}x",
            delta,
        )
    return ScalingFit(
        epsilons,
        peaks,
        slope,
        -slope / 2.0,
        False,
        f"fitted over {span:.0f}x in 1 - p/p*, peaks spanning {swing:.1f}x",
    )


def fit_fine_structure(
    epsilons: tuple[float, ...], peaks: tuple[float, ...], minimum_points: int = 6
) -> tuple[float, float] | None:
    """``(gamma, Delta)`` from ``ln max|R| = c - 2 gamma x + A sin(2 pi x / T + phi)``.

    ``x = ln(1 - p/p*)``. The ripple's period is ``T = Delta / (2 gamma)``,
    so fitting it gives the echoing period as well as a slope the ripple no
    longer biases. ``None`` unless the data span more than one period at the
    expected ``Delta`` and hold at least ``minimum_points``: fewer, and a
    sine has the freedom to fit the noise.
    """
    x, y = np.log(np.asarray(epsilons, float)), np.log(np.asarray(peaks, float))
    expected = ScalingFit.CHOPTUIK_DELTA / (2.0 * ScalingFit.CHOPTUIK_GAMMA)
    if len(x) < minimum_points or np.ptp(x) < expected:
        return None
    slope, intercept = np.polyfit(x, y, 1)

    def residual(q):
        c, gamma, amplitude, period, phase = q
        return y - (c - 2.0 * gamma * x + amplitude * np.sin(2.0 * np.pi * x / period + phase))

    best = None
    for period in np.linspace(0.6, 1.6, 11) * expected:
        for phase in np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False):
            start = [intercept, -slope / 2.0, 0.1, period, phase]
            bounds = (
                [-np.inf, 0.0, 0.0, 0.4 * expected, -np.inf],
                [np.inf, 2.0, 2.0, 2.5 * expected, np.inf],
            )
            trial = least_squares(residual, start, bounds=bounds)
            if best is None or trial.cost < best.cost:
                best = trial
    _, gamma, _, period, _ = best.x
    return float(gamma), float(2.0 * gamma * period)


@dataclass(frozen=True)
class EchoFit:
    """The echoing of the central field, read against central proper time.

    ``extrema`` are the proper times of the alternating extrema of ``phi`` at
    the centre, ``values`` the field there. Discrete self-similarity puts them
    at ``tau_k = tau* - C exp(-k Delta / 2)``: the field changes sign every
    half period. ``ratios`` are successive gaps divided, ``exp(Delta / 2)``
    each in the limit.
    """

    delta: float
    accumulation: float
    extrema: tuple[float, ...]
    values: tuple[float, ...]
    ratios: tuple[float, ...]

    @property
    def echoes(self) -> float:
        """Full periods spanned: two extrema per echo."""
        return (len(self.extrema) - 1) / 2.0


def central_proper_time(t: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """``tau = integral alpha dt`` at the centre, by the trapezoid rule."""
    t, alpha = np.asarray(t, float), np.asarray(alpha, float)
    return np.concatenate([[0.0], np.cumsum(0.5 * (alpha[1:] + alpha[:-1]) * np.diff(t))])


def central_field(t: np.ndarray, alpha: np.ndarray, a: np.ndarray, Pi: np.ndarray) -> np.ndarray:
    """``phi`` at the centre from ``d(phi)/dt = alpha Pi / a``, starting from zero."""
    rate = np.asarray(alpha, float) * np.asarray(Pi, float) / np.asarray(a, float)
    t = np.asarray(t, float)
    return np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])


def echo_period(tau: np.ndarray, phi: np.ndarray, floor: float = 0.25) -> EchoFit:
    """Fit ``Delta`` to the alternating extrema of ``phi(tau)`` above ``floor``.

    ``floor`` keeps the fit to the critical regime: the approach and the
    departure hold smaller extrema that do not echo. At least three are
    needed; the fit is least squares on all of them.
    """
    tau, phi = np.asarray(tau, float), np.asarray(phi, float)
    picked: list[int] = []
    sign = 0
    for k, value in enumerate(phi):
        s = int(np.sign(value)) if abs(value) > floor else 0
        if s == 0:
            continue
        if s != sign:
            picked.append(k)
            sign = s
        elif abs(value) > abs(phi[picked[-1]]):
            picked[-1] = k
    if len(picked) < 3:
        raise ValueError(f"need three alternating extrema above {floor}, found {len(picked)}")
    taus = tau[picked]
    gaps = np.diff(taus)
    ratios = gaps[:-1] / gaps[1:]
    k = np.arange(len(taus))
    guess = 2.0 * np.log(max(ratios[-1], 1.0 + 1e-9))
    start = taus[-1] + gaps[-1] / (np.exp(guess / 2.0) - 1.0)

    def residual(q):
        accumulation, log_scale, delta = q
        return taus - (accumulation - np.exp(log_scale - k * delta / 2.0))

    accumulation, _, delta = least_squares(
        residual, [start, np.log(max(start - taus[0], 1e-300)), guess]
    ).x
    return EchoFit(
        float(delta), float(accumulation), tuple(taus), tuple(phi[picked]), tuple(ratios)
    )
