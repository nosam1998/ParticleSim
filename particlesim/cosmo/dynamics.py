"""Theory-driven background evolution (design doc Sections 3.1, 4.3, M3).

Where :mod:`particlesim.cosmo.background` answers what an observer measures,
this answers what the scale factor does when a theory plugin replaces the
Friedmann equation. The plugin supplies one function,

    H^2 = f(rho)

through ``reduced_equations("flrw")``, and everything else follows. General
relativity is ``f = (8 pi / 3) rho``; effective loop quantum cosmology is
``f = (8 pi / 3) rho (1 - rho / rho_c)``, which passes through zero at
``rho_c`` and turns the big bang into a bounce.

Units are geometric, ``G = c = 1``, with densities in Planck units. That is
the convention the plugins are written in, and converting to observational
units here would mean guessing which convention a plugin meant.

**The evolution is second order, not first.** Taking ``H = +-sqrt(f)``
directly looks simpler and cannot cross a bounce: at the bounce ``f = 0``, so
the square root has a branch point exactly where the interesting thing
happens, and an integrator either stops there or picks a sign at random.
Differentiating the constraint instead, with the continuity equation
``rho' = -3 H (1 + w) rho``, gives

    H' = -(3/2) (1 + w) rho f'(rho)

which at a loop-quantum bounce is ``+4 pi (1+w) rho_c``: finite, positive,
and the statement that the universe turns around. The constraint is then
monitored rather than imposed, so a drift in it is visible as a number
instead of hidden by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

#: ``8 pi / 3``, the coefficient of the general-relativistic Friedmann equation.
FRIEDMANN = 8.0 * np.pi / 3.0


def general_relativity_hubble_squared(rho):
    return FRIEDMANN * np.asarray(rho, dtype=float)


def _hubble_squared_of(theory):
    if theory is None:
        return general_relativity_hubble_squared
    return theory.reduced_equations("flrw")


def bounce_density(
    theory, lower: float = 1e-12, upper: float = 1e6, tolerance: float = 1e-14
) -> float | None:
    """Density at which ``H^2`` passes through zero, or ``None`` if it never does.

    Found by root-finding on the plugin's own ``H^2(rho)`` rather than by
    watching an evolution, so the answer is exact to the solver's tolerance
    and does not depend on a step size. A theory without a bounce -- general
    relativity, where ``H^2`` is proportional to ``rho`` and positive for all
    positive ``rho`` -- returns ``None`` rather than a large number that
    would look like a bounce nobody could reach.
    """
    f = _hubble_squared_of(theory)
    at_lower = float(f(lower))
    if at_lower <= 0.0:
        raise ValueError(
            f"H^2 is already non-positive at rho = {lower:g}, so there is no "
            "expanding branch to bounce from. Lower the bracket"
        )
    if float(f(upper)) > 0.0:
        return None
    return float(brentq(lambda rho: float(f(rho)), lower, upper, xtol=tolerance))


@dataclass(frozen=True)
class BackgroundRun:
    """A scale-factor history and what it did."""

    time: np.ndarray
    scale_factor: np.ndarray
    hubble: np.ndarray
    density: np.ndarray
    constraint_drift: float
    equation_of_state: float
    bounce_time: float | None = None
    bounce_density: float | None = None
    reached_singularity: bool = False

    @property
    def bounced(self) -> bool:
        """Did the expansion rate change sign?

        Asked of ``H`` rather than of the scale factor's minimum, because a
        run that merely ended while still contracting also has its smallest
        scale factor at the end.
        """
        return bool(np.any(self.hubble < 0) and np.any(self.hubble > 0))

    @property
    def minimum_scale_factor(self) -> float:
        return float(self.scale_factor.min())

    @property
    def maximum_density(self) -> float:
        """Largest density among the *sampled* points.

        Sample-limited, and on a bouncing run that matters: the density peaks
        sharply at the turning point and an output grid coarse enough to span
        the whole run understates the peak by tens of per cent. Use
        :attr:`bounce_density`, which comes from the event that located
        ``H = 0`` exactly, whenever the question is how high the density got.
        """
        return float(self.density.max())

    def summary(self) -> dict[str, float | bool]:
        return {
            "bounced": self.bounced,
            "bounce_time": self.bounce_time,
            "bounce_density": self.bounce_density,
            "reached_singularity": self.reached_singularity,
            "minimum_scale_factor": self.minimum_scale_factor,
            "maximum_density": self.maximum_density,
            "constraint_drift": self.constraint_drift,
            "duration": float(self.time[-1] - self.time[0]),
        }


def evolve(
    theory=None,
    equation_of_state: float = 0.0,
    density: float = 1e-6,
    scale_factor: float = 1.0,
    contracting: bool = True,
    duration: float | None = None,
    points: int = 2001,
    rtol: float = 1e-11,
    atol: float = 1e-14,
    hubble_ceiling: float = 1e3,
) -> BackgroundRun:
    """Integrate a theory's FLRW dynamics through whatever it does.

    ``density`` is the density at ``scale_factor``, and the fluid's density
    follows ``rho ~ a^(-3(1+w))`` from there, which is the continuity
    equation's exact solution for a constant equation of state.

    ``contracting`` starts on the collapsing branch, which is the only way to
    reach a bounce: an expanding universe's density falls and never gets near
    a critical one.

    ``duration`` defaults to ``2.5 / |H|`` at the starting density, which is
    set by the physics rather than picked. A collapse from low density takes
    ``(2/(3(1+w))) / |H|`` to reach any bounce -- two thirds of a Hubble time
    for matter, a third for a stiff fluid -- so that default clears it for
    every equation of state and leaves room to watch the expansion afterwards.
    A fixed duration is the trap: at a starting density of 1e-6 the Hubble
    time is 345, and a plausible-looking duration of 1 moves the scale factor
    by three parts in a thousand and finds no bounce at all.

    The derivative of ``f`` is taken by central differences on a relative
    step, because the plugin contract supplies ``f`` and not ``f'``. A
    relative step of 1e-6 makes that derivative accurate to about 1e-11,
    which is below the integrator's tolerance and so is not what limits the
    answer.
    """
    f = _hubble_squared_of(theory)
    exponent = -3.0 * (1.0 + equation_of_state)
    coefficient = density * scale_factor ** (-exponent)

    def density_of(a):
        return coefficient * a**exponent

    def derivative_of_f(rho):
        step = 1e-6 * max(abs(rho), 1e-30)
        return (float(f(rho + step)) - float(f(rho - step))) / (2.0 * step)

    initial = float(f(density))
    if initial < 0:
        raise ValueError(
            f"H^2 = {initial:g} at the starting density {density:g}, which is "
            "past this theory's bounce: start at a lower density"
        )
    hubble0 = np.sqrt(initial) * (-1.0 if contracting else 1.0)
    if duration is None:
        duration = 2.5 / abs(hubble0)

    def right_hand_side(_t, state):
        a, hubble = state
        rho = density_of(a)
        rate = -1.5 * (1.0 + equation_of_state) * rho * derivative_of_f(rho)
        return [a * hubble, rate]

    def turning_point(_t, state):
        return state[1]

    turning_point.direction = 1.0  # contracting to expanding only

    # General relativity's collapse reaches a = 0 in finite time, and an
    # integrator run into it fails with a step-size message rather than
    # reporting the physics. A terminal event stops it cleanly, so a crunch
    # is an outcome the run carries instead of an exception.
    floor = 1e-8 * scale_factor

    def crunch(_t, state):
        return state[0] - floor

    crunch.terminal = True
    crunch.direction = -1.0

    # A floor on the scale factor is not enough on its own, and the reason
    # is arithmetic rather than physical. For a radiation collapse
    # ``rho ~ a^-4``, so ``a = 1e-8`` means ``|H| ~ 1e13`` and a dynamical
    # time of 1e-13 -- at a cosmic time of order a hundred, that step is
    # below the spacing between neighbouring doubles and the integrator
    # fails with a step-size message before the event can fire. Only ``w =
    # 0`` survives it. A ceiling on ``|H|`` is reached first and stops the
    # run where the classical description has run out anyway.
    #
    # The default is three orders above anything a bounce reaches: the
    # loop-quantum ``H^2 = (8 pi/3) rho (1 - rho/rho_c)`` peaks at
    # ``rho = rho_c/2``, which is ``|H| = 0.93`` at ``rho_c = 0.41``, so a
    # ceiling of 1e3 cannot cut a bounce short.
    def curvature(_t, state):
        return abs(state[1]) - hubble_ceiling

    curvature.terminal = True
    curvature.direction = 1.0

    times = np.linspace(0.0, duration, points)
    solution = solve_ivp(
        right_hand_side,
        (0.0, duration),
        [scale_factor, hubble0],
        t_eval=times,
        rtol=rtol,
        atol=atol,
        method="DOP853",
        events=[turning_point, crunch, curvature],
    )
    if not solution.success:
        raise RuntimeError(f"the background integration failed: {solution.message}")

    a = solution.y[0]
    hubble = solution.y[1]
    rho = density_of(a)
    expected = np.array([float(f(value)) for value in rho])
    scale = max(float(np.abs(expected).max()), 1e-300)
    drift = float(np.abs(hubble**2 - expected).max() / scale)
    crossings = solution.t_events[0]
    turned = solution.y_events[0]
    at_bounce = float(density_of(turned[0][0])) if len(turned) else None
    return BackgroundRun(
        time=solution.t,
        scale_factor=a,
        hubble=hubble,
        density=rho,
        constraint_drift=drift,
        equation_of_state=float(equation_of_state),
        bounce_time=float(crossings[0]) if len(crossings) else None,
        bounce_density=at_bounce,
        reached_singularity=bool(len(solution.t_events[1]) or len(solution.t_events[2])),
    )
