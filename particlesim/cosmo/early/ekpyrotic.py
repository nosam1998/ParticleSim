"""Ekpyrotic slow contraction (design doc Section 3.1).

A scalar field with a steep *negative* exponential potential in a
contracting universe. The defining solution is a power law,

    a ~ (-t)^p,    p = 2/c^2,    epsilon_H = c^2/2

for ``V = -V_0 exp(-c phi)`` in reduced Planck units, and the whole point
of the model is that ``p`` is *small*: the universe contracts by a tiny
factor over many Hubble times, which is what lets the ekpyrotic component
outgrow the shear.

The derivation is worth having in full, because every number this module
reports is a consequence of it. Put ``phi = (2/c) ln(-t) + phi_0`` and
``a = (-t)^p`` into the two equations of motion:

    3H^2 = (1/2) phi_dot^2 + V        (Friedmann)
    phi_ddot + 3 H phi_dot + V_phi = 0   (field)

With ``H = p/t``, ``phi_dot = 2/(ct)`` and ``V = -A/t^2`` where
``A = V_0 exp(-c phi_0)``, the field equation gives ``A = (2 - 6p)/c^2``
and the Friedmann constraint then forces ``3p^2 = 6p/c^2``, so
``p = 2/c^2``.

**The existence condition is the ekpyrotic condition.** ``A`` has to be
positive for the potential to be negative, and ``A = (2 - 6p)/c^2 > 0``
means ``p < 1/3``, which is ``epsilon > 3``. So the scaling solution exists
exactly when the contraction dilutes anisotropy rather than amplifying it
(see :mod:`particlesim.cosmo.early.diagnostics`), and a shallower potential
has no such solution at all rather than a marginal one. This module refuses
``c^2 <= 6`` for that reason.

**The evolution is second order and the constraint is monitored.** ``H``
could be had algebraically from the Friedmann equation, but the square root
has a branch point where ``(1/2) phi_dot^2 + V`` passes through zero --
which is exactly what a bounce would be -- so taking it would prejudge the
question. Integrating ``H_dot = -(1/2) phi_dot^2`` instead leaves the
constraint as an independent check, and its drift is reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

#: Steepness below which there is no ekpyrotic scaling solution.
#:
#: ``c^2 = 6`` is ``epsilon = 3`` is ``w = 1``: a stiff fluid, which grows
#: exactly as fast as the shear it is meant to beat.
CRITICAL_STEEPNESS_SQUARED = 6.0


@dataclass(frozen=True)
class EkpyroticPotential:
    """``V = -scale * exp(-steepness * phi)``.

    Deliberately *not* a :class:`~particlesim.cosmo.potentials.Potential`.
    That base class exists for inflation and its slow-roll parameters --
    ``epsilon_V = (1/2)(V'/V)^2`` and the rest -- are derived on the
    assumption ``V > 0``. Here ``V < 0`` throughout, the field is in
    fast roll rather than slow roll, and every one of those quantities
    would return a number that means nothing. The two useful parameters
    have closed forms instead.
    """

    scale: float = 1.0
    steepness: float = 10.0

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        if self.steepness**2 <= CRITICAL_STEEPNESS_SQUARED:
            raise ValueError(
                f"steepness^2 = {self.steepness**2:g} is not above "
                f"{CRITICAL_STEEPNESS_SQUARED:g}, so this potential has no ekpyrotic "
                "scaling solution: the amplitude A = (2 - 6p)/c^2 that the field "
                "equation demands would be negative, and a contraction this shallow "
                "amplifies anisotropy rather than diluting it"
            )

    @property
    def epsilon(self) -> float:
        """``epsilon_H = c^2/2``, constant along the scaling solution."""
        return 0.5 * self.steepness**2

    @property
    def scaling_exponent(self) -> float:
        """``p = 2/c^2`` in ``a ~ (-t)^p``."""
        return 2.0 / self.steepness**2

    @property
    def equation_of_state(self) -> float:
        """``w = (2/3) epsilon - 1``, large and positive for a steep potential."""
        return 2.0 * self.epsilon / 3.0 - 1.0

    def value(self, phi):
        return -self.scale * np.exp(-self.steepness * np.asarray(phi, dtype=float))

    def gradient(self, phi):
        return self.steepness * self.scale * np.exp(-self.steepness * np.asarray(phi, dtype=float))

    def summary(self) -> dict[str, float]:
        return {
            "steepness": self.steepness,
            "epsilon": self.epsilon,
            "scaling_exponent": self.scaling_exponent,
            "equation_of_state": self.equation_of_state,
        }


def scaling_solution(potential: EkpyroticPotential, time):
    """Exact ``(a, phi, phi_dot, H)`` on the attractor at ``time < 0``.

    Normalised so that ``a = 1`` at ``t = -1``. The field offset is fixed by
    the field equation rather than chosen: ``phi_0`` is whatever makes
    ``V_0 exp(-c phi_0) = (2 - 6p)/c^2``.
    """
    time = np.asarray(time, dtype=float)
    if np.any(time >= 0.0):
        raise ValueError("the ekpyrotic scaling solution lives at t < 0, before the crunch")
    exponent = potential.scaling_exponent
    steepness = potential.steepness
    amplitude = (2.0 - 6.0 * exponent) / steepness**2
    offset = -math.log(amplitude / potential.scale) / steepness
    span = -time
    return (
        span**exponent,
        (2.0 / steepness) * np.log(span) + offset,
        2.0 / (steepness * time),
        exponent / time,
    )


@dataclass(frozen=True)
class ContractionRun:
    """A contracting scalar-field background in cosmic time."""

    potential: EkpyroticPotential
    time: np.ndarray
    scale_factor: np.ndarray
    field: np.ndarray
    velocity: np.ndarray
    hubble: np.ndarray
    constraint_drift: float
    reached_ceiling: bool = False
    solution: object = field(repr=False, default=None)

    @property
    def epsilon(self) -> np.ndarray:
        """``epsilon_H = -H_dot/H^2 = (1/2) phi_dot^2 / H^2``."""
        return 0.5 * self.velocity**2 / self.hubble**2

    @property
    def density(self) -> np.ndarray:
        return 0.5 * self.velocity**2 + self.potential.value(self.field)

    @property
    def pressure(self) -> np.ndarray:
        return 0.5 * self.velocity**2 - self.potential.value(self.field)

    @property
    def equation_of_state(self) -> np.ndarray:
        return self.pressure / self.density

    @property
    def singular_time(self) -> float:
        """Time the contraction is heading for, from the last sample.

        On the attractor ``H = p/(t - t_s)``, so ``t_s = t - 1/(epsilon H)``.
        It is zero for a run started on the scaling solution, and *earlier*
        than zero for one started with excess kinetic energy: a perturbation
        off the attractor is absorbed by shifting the crunch, not by
        changing the power law, which is the sharpest statement of what kind
        of attractor this is.
        """
        return float(self.time[-1] - 1.0 / (self.epsilon[-1] * self.hubble[-1]))

    @property
    def contraction_factor(self) -> float:
        """``a_initial / a_final``, larger than one for a contraction."""
        return float(self.scale_factor[0] / self.scale_factor[-1])

    def summary(self) -> dict[str, float]:
        return {
            "contraction_factor": self.contraction_factor,
            "hubble_times": float(abs(self.hubble[-1]) * (self.time[-1] - self.time[0])),
            "final_epsilon": float(self.epsilon[-1]),
            "final_equation_of_state": float(self.equation_of_state[-1]),
            "singular_time": self.singular_time,
            "reached_ceiling": self.reached_ceiling,
            "constraint_drift": self.constraint_drift,
        }


def evolve_contraction(
    potential: EkpyroticPotential,
    start: float = -1e6,
    stop: float = -1.0,
    field_value: float | None = None,
    velocity: float | None = None,
    points: int = 2001,
    curvature_ceiling: float = 1.0,
    rtol: float = 1e-12,
    atol: float = 1e-20,
) -> ContractionRun:
    """Integrate the contraction from ``start`` to ``stop``, both negative.

    Initial data defaults to the scaling solution at ``start``; supply
    ``field_value`` or ``velocity`` to start off the attractor, which is how
    a run that demonstrates the attractor is set up. ``H`` is initialised
    from the Friedmann constraint on the contracting branch, so a
    perturbed start is a consistent solution of the constraint and not
    merely a nearby set of numbers.

    A perturbed start brings the crunch forward, so the run may reach it
    before ``stop``. The integration then stops at ``curvature_ceiling``,
    where ``|H|`` reaches one in reduced Planck units and the classical
    description has run out, and ``reached_ceiling`` says so. **A floor on
    the scale factor would not work here**, which is worth knowing: with
    ``a ~ (t_s - t)^p`` and ``p = 0.02``, reaching ``a = 1e-8`` needs
    ``t_s - t ~ 1e-400``. The scale factor barely moves while the curvature
    diverges, so the curvature is the thing to watch. An integrator run
    into the crunch without such an event fails with a step-size message
    and reports physics as a crash.

    ``atol`` is very small on purpose: ``H`` is of order ``1e-8`` at the
    start of a slow contraction, so an absolute tolerance anywhere near it
    would control the answer instead of ``rtol``. What accuracy remains
    shows up as a shift in :attr:`ContractionRun.singular_time` -- 8e-7 at
    ``rtol = 1e-11`` -- rather than as an error in the exponent, which
    holds to 2e-8, because a shift of the crunch time is the scaling
    family's zero mode.
    """
    if not (start < stop < 0.0):
        raise ValueError(
            f"need start < stop < 0, got start = {start:g}, stop = {stop:g}: the "
            "ekpyrotic phase runs from the far past up to the crunch at t = 0"
        )
    exact = scaling_solution(potential, start)
    phi = float(exact[1]) if field_value is None else float(field_value)
    rate = float(exact[2]) if velocity is None else float(velocity)
    energy = 0.5 * rate**2 + float(potential.value(phi))
    if energy <= 0.0:
        raise ValueError(
            f"the initial kinetic energy does not exceed the depth of the potential "
            f"(rho = {energy:g}), so 3H^2 would be negative: this is not a state the "
            "contracting branch can be started from"
        )
    hubble = -math.sqrt(energy / 3.0)

    def right_hand_side(_t, state):
        _, value, rate, hubble = state
        return [
            hubble,
            rate,
            -3.0 * hubble * rate - float(potential.gradient(value)),
            -0.5 * rate**2,
        ]

    def ceiling(_t, state):
        return state[3] + curvature_ceiling

    ceiling.terminal = True
    ceiling.direction = -1.0

    times = np.linspace(start, stop, int(points))
    solution = solve_ivp(
        right_hand_side,
        (start, stop),
        [0.0, phi, rate, hubble],
        t_eval=times,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
        events=ceiling,
    )
    if not solution.success:
        raise RuntimeError(f"the contraction integration failed: {solution.message}")
    if solution.t.size < 2:
        raise RuntimeError(
            f"the contraction reached the curvature ceiling within one output step "
            f"of {start:g}: the initial data is far enough off the attractor that "
            "there is no classical contraction to report"
        )

    log_scale, fields, velocities, hubbles = solution.y
    density = 0.5 * velocities**2 + potential.value(fields)
    residual = 3.0 * hubbles**2 - density
    scale = max(float(np.abs(3.0 * hubbles**2).max()), 1e-300)
    return ContractionRun(
        potential=potential,
        time=solution.t,
        scale_factor=np.exp(log_scale),
        field=fields,
        velocity=velocities,
        hubble=hubbles,
        constraint_drift=float(np.abs(residual).max() / scale),
        reached_ceiling=bool(solution.t_events[0].size),
        solution=solution,
    )
