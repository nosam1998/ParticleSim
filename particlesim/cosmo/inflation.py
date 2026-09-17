"""Inflationary backgrounds: slow roll and exact evolution (Section 3.1, C1).

Two descriptions of the same physics live here, and the point of having
both is that the cheap one is checkable against the expensive one.

**Slow roll** is algebra on the potential: ``epsilon_V``, ``eta_V`` and the
e-fold integral ``N = integral V/V' dphi``. It costs one quadrature and is
what the analytic predictions in the literature are quoted from.

**Exact evolution** integrates the field equation with the number of
e-folds as the time coordinate,

    phi'' = -(3 - epsilon) (phi' + V_phi / V),    epsilon = (1/2) phi'^2

which is the full nonlinear system with nothing dropped -- the slow-roll
attractor is where it ends up, not an assumption it makes. Three properties
earn e-folds their place as the time coordinate:

* ``H`` never appears as a variable. It is algebraic, ``H^2 = V/(3 -
  epsilon)``, so the Friedmann constraint is solved identically at every
  step instead of drifting.
* The integration interval is the quantity the answer is quoted against.
  "Fifty-five e-folds before the end" is a coordinate value, not something
  to be searched for afterwards in a time series.
* Inflation ends at ``epsilon = 1`` exactly, which is a terminal event on a
  state variable rather than a threshold on a derived one.

The cost is that the equations do not describe what happens after inflation:
at ``epsilon = 3`` the algebraic ``H`` diverges. Every run here stops at
``epsilon = 1``, well before that.

Units are reduced Planck, ``M_p = 1``, as in :mod:`particlesim.cosmo.potentials`.
The scale factor is normalised to ``a = 1`` at the start of a run, so
``ln a = N`` and the comoving Hubble rate is ``ln(aH) = N + ln H``. Nothing
observable depends on that choice: a wavenumber is only ever compared with
``aH`` from the same run.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from particlesim.cosmo.potentials import MultiFieldPotential, Potential

#: Default quadrature order for the e-fold integral.
EFOLD_ORDER = 96
#: Largest number of e-folds a background run is integrated for.
MAX_EFOLDS = 1000.0


class InflationNeverEnds(ValueError):
    """Raised when ``epsilon_V`` never reaches one on the potential's domain.

    An exponential potential has constant ``epsilon_V``; a plateau that
    flattens out forever has a decreasing one. Neither has an end of
    inflation, so anything defined by counting e-folds backwards from it has
    no answer, and saying so is better than returning the edge of a search
    bracket.
    """


@lru_cache(maxsize=32)
def _gauss_legendre(order: int):
    return np.polynomial.legendre.leggauss(order)


def _quadrature(f, lower: float, upper: float, order: int = EFOLD_ORDER):
    """``integral_lower^upper f`` by fixed-order Gauss-Legendre."""
    nodes, weights = _gauss_legendre(order)
    half = 0.5 * (upper - lower)
    middle = 0.5 * (upper + lower)
    return float(np.sum(f(middle + half * nodes) * weights) * half)


@dataclass(frozen=True)
class SlowRoll:
    """Slow-roll parameters and the observables they predict.

    The relations are the standard first-order ones,

        n_s = 1 - 6 epsilon + 2 eta,   r = 16 epsilon,   n_t = -2 epsilon,
        alpha_s = 16 epsilon eta - 24 epsilon^2 - 2 xi^2

    and they are *first order*: the corrections are ``O(epsilon^2)``, which
    at sixty e-folds is a few parts in ten thousand -- small, but the same
    size as the difference between two inflationary models. Where a number
    has to be right beyond that, use the mode evolution in
    :mod:`particlesim.cosmo.perturbations` instead of this.
    """

    epsilon: float
    eta: float
    xi_squared: float
    field_value: float
    hubble: float

    @property
    def spectral_index(self) -> float:
        return 1.0 - 6.0 * self.epsilon + 2.0 * self.eta

    @property
    def tensor_to_scalar(self) -> float:
        return 16.0 * self.epsilon

    @property
    def tensor_index(self) -> float:
        return -2.0 * self.epsilon

    @property
    def running(self) -> float:
        return 16.0 * self.epsilon * self.eta - 24.0 * self.epsilon**2 - 2.0 * self.xi_squared

    @property
    def scalar_amplitude(self) -> float:
        """``P_R = H^2 / (8 pi^2 epsilon)`` at horizon crossing."""
        return self.hubble**2 / (8.0 * math.pi**2 * self.epsilon)

    @property
    def tensor_amplitude(self) -> float:
        """``P_t = 2 H^2 / pi^2``."""
        return 2.0 * self.hubble**2 / math.pi**2

    def summary(self) -> dict[str, float]:
        return {
            "field": self.field_value,
            "epsilon": self.epsilon,
            "eta": self.eta,
            "spectral_index": self.spectral_index,
            "tensor_to_scalar": self.tensor_to_scalar,
            "running": self.running,
            "scalar_amplitude": self.scalar_amplitude,
        }


def slow_roll(potential: Potential, phi: float) -> SlowRoll:
    """Slow-roll parameters of ``potential`` at ``phi``."""
    value = float(potential.value(phi))
    if value <= 0.0:
        raise ValueError(
            f"V = {value:g} at phi = {phi:g} is not positive, so there is no "
            "inflating background here"
        )
    epsilon = float(potential.epsilon(phi))
    return SlowRoll(
        epsilon=epsilon,
        eta=float(potential.eta(phi)),
        xi_squared=float(potential.xi_squared(phi)),
        field_value=float(phi),
        hubble=math.sqrt(value / (3.0 - epsilon)),
    )


def _bracket_towards(excess, lower: float, target: float, name: str) -> float:
    """Root of ``excess`` between ``lower`` and a point where it diverges.

    Used where ``epsilon_V`` blows up at a boundary -- the quadratic
    minimum of a potential, or a zero of ``V`` inside the domain -- so the
    root cannot be bracketed by evaluating at the boundary itself. The
    bracket closes on it geometrically instead.
    """
    gap = target - lower
    for shrink in (1e-3, 1e-6, 1e-9, 1e-12):
        probe = target - shrink * gap
        if excess(probe) > 0.0:
            return float(brentq(excess, lower, probe, xtol=1e-14, rtol=1e-15))
    raise InflationNeverEnds(
        f"epsilon_V stays below one all the way to phi = {target:g} for {name}: "
        "this potential has no end of inflation on the side the field rolls"
    )


def end_of_inflation(potential: Potential, start: float | None = None) -> float:
    """Field value where ``epsilon_V = 1``, downhill from ``start``.

    The search follows the rolling direction rather than assuming one, so
    it works for a quadratic rolling towards the origin and for natural
    inflation rolling away from it.

    On a bounded domain it scans towards the edge, with the probes graded so
    that they cluster against it, and stops at the first one where
    ``epsilon_V`` exceeds one or where ``V`` has gone non-positive.
    Both cases have to be handled and they are different: ``epsilon_V``
    diverges at a quadratic minimum sitting exactly on the domain edge, and
    it diverges at a zero of ``V`` *inside* the domain -- which is what a
    truncated plateau potential has, and what a search that only refined
    against the edge would walk straight past, reporting no end of
    inflation for a potential that has one.

    On an unbounded domain the bracket doubles outward, and failing to find
    a root raises :class:`InflationNeverEnds` rather than returning the last
    probe.
    """
    start = potential.typical_field if start is None else float(start)
    name = type(potential).__name__
    if not bool(potential.contains(start)):
        raise ValueError(f"phi = {start:g} is outside the domain {potential.domain} of {name}")
    if float(potential.epsilon(start)) >= 1.0:
        raise ValueError(
            f"epsilon_V = {float(potential.epsilon(start)):g} at the starting point "
            f"phi = {start:g}, which is already past the end of inflation"
        )

    direction = potential.roll_direction(start)
    low, high = potential.domain
    edge = high if direction > 0 else low

    def excess(phi: float) -> float:
        return float(potential.epsilon(phi)) - 1.0

    def positive(phi: float) -> bool:
        value = float(potential.value(phi))
        return value > 0.0 and math.isfinite(value)

    if math.isfinite(edge):
        fractions = sorted(
            {*np.linspace(0.0, 1.0, 65)[1:-1], *(1.0 - 10.0 ** -np.arange(3.0, 13.0))}
        )
        previous = start
        for fraction in fractions:
            probe = start + fraction * (edge - start)
            if not positive(probe):
                zero = float(
                    brentq(
                        lambda phi: float(potential.value(phi)),
                        previous,
                        probe,
                        xtol=1e-14,
                        rtol=1e-15,
                    )
                )
                return _bracket_towards(excess, previous, zero, name)
            if excess(probe) > 0.0:
                return float(brentq(excess, previous, probe, xtol=1e-14, rtol=1e-15))
            previous = probe
        raise InflationNeverEnds(
            f"epsilon_V stays below one all the way to phi = {edge:g}, the edge of "
            f"{name}'s domain: this potential has no end of inflation"
        )

    step = 0.1 * max(1.0, abs(start))
    probe = start
    for _ in range(200):
        probe += direction * step
        if not positive(probe):
            raise InflationNeverEnds(
                f"V has run out at phi = {probe:g} without epsilon_V reaching one: "
                f"{name} has no end of inflation"
            )
        if excess(probe) > 0.0:
            return float(brentq(excess, probe - direction * step, probe, xtol=1e-14, rtol=1e-15))
        step *= 1.5
    raise InflationNeverEnds(
        f"epsilon_V stays below one out to phi = {probe:g} on an unbounded domain: "
        f"{name} has no end of inflation (an exponential potential has constant "
        "epsilon_V by construction)"
    )


def efolds(
    potential: Potential,
    phi: float,
    phi_end: float | None = None,
    order: int = EFOLD_ORDER,
) -> float:
    """Slow-roll e-folds from ``phi`` to the end of inflation.

    ``N = |integral_phi^phi_end V/V' dphi|``. The absolute value is taken
    rather than tracked with a sign, which is correct exactly as long as
    ``V'`` keeps one sign over the interval -- and that is checked, because
    an interval straddling a turning point would otherwise return the
    difference of two e-fold counts as if it were their sum.
    """
    phi_end = end_of_inflation(potential, phi) if phi_end is None else float(phi_end)

    def integrand(x):
        return potential.value(x) / potential.gradient(x)

    nodes, _ = _gauss_legendre(order)
    half = 0.5 * (phi_end - phi)
    slopes = potential.gradient(0.5 * (phi_end + phi) + half * nodes)
    if np.ptp(np.sign(slopes)) != 0.0:
        raise ValueError(
            f"V' changes sign between phi = {phi:g} and phi = {phi_end:g}, so the "
            "e-fold integral spans a turning point of the potential and its "
            "magnitude is not the number of e-folds along a trajectory"
        )
    return abs(_quadrature(integrand, phi, phi_end, order))


def field_at_efolds(
    potential: Potential,
    number: float,
    phi_end: float | None = None,
    order: int = EFOLD_ORDER,
) -> float:
    """Field value ``number`` slow-roll e-folds before the end of inflation."""
    if number <= 0.0:
        raise ValueError(f"number of e-folds must be positive, got {number}")
    phi_end = end_of_inflation(potential) if phi_end is None else float(phi_end)
    direction = -potential.roll_direction(phi_end + 0.0)
    low, high = potential.domain
    edge = high if direction > 0 else low

    def excess(phi: float) -> float:
        return efolds(potential, phi, phi_end, order) - number

    if math.isfinite(edge):
        gap = abs(edge - phi_end)
        for shrink in (1e-2, 1e-4, 1e-8, 1e-12):
            probe = edge - direction * shrink * gap
            if excess(probe) > 0.0:
                return float(brentq(excess, phi_end, probe, xtol=1e-12, rtol=1e-14))
        raise InflationNeverEnds(
            f"{type(potential).__name__} accumulates fewer than {number:g} e-folds "
            f"between the end of inflation at phi = {phi_end:g} and the edge of its "
            f"domain at phi = {edge:g}"
        )

    step = 0.1 * max(1.0, abs(phi_end))
    probe = phi_end
    for _ in range(200):
        probe += direction * step
        if excess(probe) > 0.0:
            return float(brentq(excess, probe - direction * step, probe, xtol=1e-12, rtol=1e-14))
        step *= 1.5
    raise InflationNeverEnds(
        f"{type(potential).__name__} accumulates fewer than {number:g} e-folds out to "
        f"phi = {probe:g}"
    )


def slow_roll_prediction(
    potential: Potential, efolds_remaining: float, order: int = EFOLD_ORDER
) -> SlowRoll:
    """Slow-roll observables ``efolds_remaining`` e-folds before the end."""
    phi = field_at_efolds(potential, efolds_remaining, order=order)
    return slow_roll(potential, phi)


def _acceleration(potential: Potential, phi, velocity):
    """``phi''`` from the exact field equation in e-fold time."""
    epsilon = 0.5 * velocity**2
    return -(3.0 - epsilon) * (velocity + potential.gradient(phi) / potential.value(phi))


@dataclass(frozen=True)
class InflationRun:
    """An exact single-field background history, sampled and interpolable.

    The sampled arrays are for plotting and for reading off summaries; the
    dense solution behind :meth:`state` is what the mode equations use,
    because a mode needs the background at whatever e-fold its own
    integrator asks for and interpolating a stored grid would put the
    background's sampling error into the power spectrum.
    """

    potential: Potential
    efolds: np.ndarray
    field: np.ndarray
    velocity: np.ndarray
    hubble: np.ndarray
    epsilon: np.ndarray
    ended: bool
    solution: object = field(repr=False, default=None)

    @property
    def total_efolds(self) -> float:
        return float(self.efolds[-1])

    def state(self, number):
        """``(phi, dphi/dN)`` at e-fold ``number``, by dense interpolation."""
        values = self.solution.sol(np.asarray(number, dtype=float))
        return values[0], values[1]

    def epsilon_at(self, number):
        _, velocity = self.state(number)
        return 0.5 * velocity**2

    def hubble_at(self, number):
        phi, velocity = self.state(number)
        return np.sqrt(self.potential.value(phi) / (3.0 - 0.5 * velocity**2))

    def acceleration_at(self, number):
        phi, velocity = self.state(number)
        return _acceleration(self.potential, phi, velocity)

    def log_comoving_hubble(self, number):
        """``ln(aH)``, with ``a = exp(N)`` from the start of the run."""
        number = np.asarray(number, dtype=float)
        return number + np.log(self.hubble_at(number))

    def efolds_remaining(self, number):
        return self.total_efolds - np.asarray(number, dtype=float)

    def at_efolds_remaining(self, number: float) -> float:
        """The e-fold coordinate ``number`` e-folds before the end of the run."""
        value = self.total_efolds - float(number)
        if value < float(self.efolds[0]):
            raise ValueError(
                f"the run covers {self.total_efolds - float(self.efolds[0]):.3g} e-folds, "
                f"so it does not reach back {number:g} e-folds before its end. Start the "
                "background earlier"
            )
        return value

    def summary(self) -> dict[str, float | bool]:
        return {
            "total_efolds": self.total_efolds,
            "ended": self.ended,
            "initial_field": float(self.field[0]),
            "final_field": float(self.field[-1]),
            "final_epsilon": float(self.epsilon[-1]),
            "initial_hubble": float(self.hubble[0]),
        }


def evolve_inflation(
    potential: Potential,
    phi: float,
    velocity: float | None = None,
    max_efolds: float = MAX_EFOLDS,
    samples: int = 2001,
    rtol: float = 1e-11,
    atol: float = 1e-13,
) -> InflationRun:
    """Integrate the exact single-field background from ``phi``.

    ``velocity`` defaults to the slow-roll attractor ``-V'/V``. Starting off
    the attractor is allowed and is how a run with an initial kinetic
    transient is set up; the attractor is reached within an e-fold or so,
    which is the reason the margin in front of a pivot exists.

    Integration stops at ``epsilon = 1`` -- the end of inflation -- as a
    terminal event, so the final e-fold count is the solver's root of a
    state variable rather than the first sample past a threshold.
    """
    phi = float(phi)
    if not bool(potential.contains(phi)):
        raise ValueError(
            f"phi = {phi:g} is outside the domain {potential.domain} of {type(potential).__name__}"
        )
    if velocity is None:
        velocity = -float(potential.gradient(phi) / potential.value(phi))
    if 0.5 * velocity**2 >= 1.0:
        raise ValueError(
            f"epsilon = {0.5 * velocity**2:g} at the start, so the initial state is not inflating"
        )

    def right_hand_side(_n, state):
        return [state[1], float(_acceleration(potential, state[0], state[1]))]

    def inflation_ends(_n, state):
        return 0.5 * state[1] ** 2 - 1.0

    inflation_ends.terminal = True
    inflation_ends.direction = 1.0

    solution = solve_ivp(
        right_hand_side,
        (0.0, float(max_efolds)),
        [phi, float(velocity)],
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
        events=inflation_ends,
    )
    if not solution.success:
        raise RuntimeError(f"background integration failed: {solution.message}")

    ended = bool(solution.t_events[0].size)
    final = float(solution.t_events[0][0]) if ended else float(solution.t[-1])
    grid = np.linspace(0.0, final, int(samples))
    states = solution.sol(grid)
    fields, velocities = states[0], states[1]
    epsilon = 0.5 * velocities**2
    hubble = np.sqrt(potential.value(fields) / (3.0 - epsilon))
    return InflationRun(
        potential=potential,
        efolds=grid,
        field=fields,
        velocity=velocities,
        hubble=hubble,
        epsilon=epsilon,
        ended=ended,
        solution=solution,
    )


def run_to_end(
    potential: Potential,
    efolds_remaining: float = 55.0,
    margin: float = 8.0,
    **kwargs,
) -> InflationRun:
    """Background run that reaches back ``efolds_remaining + margin`` e-folds.

    The starting field is placed by the slow-roll e-fold integral, then the
    exact equations decide where the end actually is -- so the pivot is
    defined by the *numerical* e-fold count, not by the estimate used to
    choose the initial condition. The two differ by a fraction of an e-fold,
    which matters at the fourth digit of ``n_s``.

    The margin is what a mode needs to start well inside the horizon before
    the pivot crosses it. Eight e-folds puts ``k/aH`` near three thousand
    there, and :func:`particlesim.cosmo.perturbations.mode_power` checks it
    has the room it asked for rather than silently starting late.
    """
    phi = field_at_efolds(potential, efolds_remaining + margin)
    return evolve_inflation(potential, phi, **kwargs)


@dataclass(frozen=True)
class MultiFieldRun:
    """An exact multi-field background history. Fields index the last axis."""

    potential: MultiFieldPotential
    efolds: np.ndarray
    field: np.ndarray
    velocity: np.ndarray
    hubble: np.ndarray
    epsilon: np.ndarray
    ended: bool
    solution: object = field(repr=False, default=None)

    @property
    def total_efolds(self) -> float:
        return float(self.efolds[-1])

    def state(self, number):
        values = self.solution.sol(np.asarray(number, dtype=float))
        half = values.shape[0] // 2
        return values[:half], values[half:]

    def hubble_at(self, number):
        phi, velocity = self.state(number)
        epsilon = 0.5 * np.sum(velocity**2, axis=0)
        return np.sqrt(self.potential.value(np.moveaxis(phi, 0, -1)) / (3.0 - epsilon))

    def summary(self) -> dict[str, float | bool]:
        return {
            "total_efolds": self.total_efolds,
            "ended": self.ended,
            "fields": int(self.field.shape[-1]),
            "final_epsilon": float(self.epsilon[-1]),
        }


def evolve_fields(
    potential: MultiFieldPotential,
    phi,
    velocity=None,
    max_efolds: float = MAX_EFOLDS,
    samples: int = 2001,
    rtol: float = 1e-11,
    atol: float = 1e-13,
    stop: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> MultiFieldRun:
    """Integrate the exact multi-field background.

    The equation is the same one the single-field case solves, with the
    gradient now a vector and ``epsilon`` the sum over fields:

        phi_i'' = -(3 - epsilon) (phi_i' + V_i / V),   epsilon = (1/2) sum_j phi_j'^2

    Nothing about it is a slow-roll or a straight-trajectory assumption, so
    a turning trajectory -- the thing multi-field inflation exists to
    describe -- is integrated as it stands.
    """
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    if phi.shape != (potential.fields,):
        raise ValueError(f"expected {potential.fields} initial field values, got shape {phi.shape}")
    if velocity is None:
        velocity = -potential.gradient(phi) / potential.value(phi)
    velocity = np.atleast_1d(np.asarray(velocity, dtype=float))
    if 0.5 * float(np.sum(velocity**2)) >= 1.0:
        raise ValueError("the initial state is not inflating: epsilon >= 1")

    count = potential.fields

    def right_hand_side(_n, state):
        fields, velocities = state[:count], state[count:]
        epsilon = 0.5 * float(np.sum(velocities**2))
        gradient = potential.gradient(fields)
        acceleration = -(3.0 - epsilon) * (velocities + gradient / potential.value(fields))
        return np.concatenate([velocities, acceleration])

    if stop is None:

        def inflation_ends(_n, state):
            return 0.5 * float(np.sum(state[count:] ** 2)) - 1.0

        inflation_ends.direction = 1.0
    else:

        def inflation_ends(_n, state):
            return float(stop(state[:count], state[count:]))

        inflation_ends.direction = 0.0

    inflation_ends.terminal = True

    solution = solve_ivp(
        right_hand_side,
        (0.0, float(max_efolds)),
        np.concatenate([phi, velocity]),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
        events=inflation_ends,
    )
    if not solution.success:
        raise RuntimeError(f"background integration failed: {solution.message}")

    ended = bool(solution.t_events[0].size)
    final = float(solution.t_events[0][0]) if ended else float(solution.t[-1])
    grid = np.linspace(0.0, final, int(samples))
    states = solution.sol(grid)
    fields = states[:count].T
    velocities = states[count:].T
    epsilon = 0.5 * np.sum(velocities**2, axis=-1)
    hubble = np.sqrt(potential.value(fields) / (3.0 - epsilon))
    return MultiFieldRun(
        potential=potential,
        efolds=grid,
        field=fields,
        velocity=velocities,
        hubble=hubble,
        epsilon=epsilon,
        ended=ended,
        solution=solution,
    )
