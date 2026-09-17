"""Pre-big-bang dilaton-driven cosmology (design doc Sections 3.1, 4.4).

The lowest-order string-frame effective action for gravity and the dilaton,

    S = -(1/2 lambda_s^(d-1)) integral d^(d+1)x sqrt(-g) e^(-phi) [R + (grad phi)^2]

has a homogeneous isotropic vacuum solution that *super*-inflates: the
curvature grows while the universe expands, driven by the dilaton rather
than by a potential. In ``d`` spatial dimensions,

    a ~ |t|^(-1/sqrt(d)),      e^phi ~ |t|^(-(1 + sqrt(d)))

on the branch that runs from ``t = -infinity`` up to a curvature
singularity at ``t = 0``. For ``d = 3`` those exponents are ``-0.5774``
and ``-2.7321``. Both the curvature and the string coupling diverge as
``t -> 0``, which is the whole problem with the scenario: the graceful exit
needs the ``alpha'`` corrections this module does not have.

**The shifted dilaton is what makes the equations tractable.** With
``phi_bar = phi - d ln a`` the vacuum equations are just

    phi_bar_dot^2 = d H^2,      H_dot = H phi_bar_dot

Rather than integrate that reduced pair -- whose constraint is satisfied
identically, so monitoring it would prove nothing -- this module integrates
the unreduced system in ``(a, phi)``,

    H_dot = H phi_dot - d H^2
    phi_ddot = d H phi_dot - d(d-1) H^2

and monitors the independent constraint

    C = phi_dot^2 - 2 d H phi_dot + d(d-1) H^2

which the evolution conserves rather than imposes. Its drift is reported.

**Scale-factor duality is exact, and is tested as such.** The map

    a -> 1/a,     phi -> phi - 2 d ln a

leaves ``phi_bar`` alone and sends ``H -> -H``, and substituting it into
the three equations above leaves each of them unchanged term by term. So a
solution's dual is a solution, exactly, and :func:`dual` plus a residual
check is a machine-precision test of the whole system rather than of one
number. Composing duality with time reversal gives the four branches of
pre-big-bang cosmology: expanding or contracting, coupling growing or
decaying.

Truncation: tree level in ``alpha'`` and lowest order in the string
coupling, vacuum only -- no matter sources, no Kalb-Ramond axion, no
higher-curvature corrections, and therefore no graceful exit. The module
describes the dilaton-driven phase and says nothing about how it ends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class DilatonVacuum:
    """The tree-level dilaton-driven background in ``dimension`` spatial dimensions.

    ``branch`` is the sign of ``phi_bar_dot / (sqrt(d) H)``. With
    ``branch = +1`` and ``t < 0`` the solution expands with growing
    curvature and growing coupling -- the pre-big-bang branch. The other
    sign, and the time reverse of each, give the remaining three.
    """

    dimension: int = 3
    branch: int = 1

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError(f"dimension must be at least 1, got {self.dimension}")
        if self.branch not in (1, -1):
            raise ValueError(f"branch must be +1 or -1, got {self.branch}")

    @property
    def root_dimension(self) -> float:
        return math.sqrt(self.dimension)

    @property
    def scale_exponent(self) -> float:
        """``p`` in ``a ~ |t|^p``: negative, so the expansion accelerates."""
        return -1.0 / (self.branch * self.root_dimension)

    @property
    def coupling_exponent(self) -> float:
        """``q`` in ``e^phi ~ |t|^q``."""
        return -1.0 - self.dimension / (self.branch * self.root_dimension)

    @property
    def superinflating(self) -> bool:
        """Does ``|H|`` grow along the solution?"""
        return self.scale_exponent < 0.0

    def velocity_from_hubble(self, hubble):
        """``phi_dot`` on this branch, from the constraint at a given ``H``.

        The constraint is quadratic in ``phi_dot``, and its two roots are
        the two branches: ``phi_dot = H (d +- sqrt(d))``.
        """
        return np.asarray(hubble, dtype=float) * (
            self.dimension + self.branch * self.root_dimension
        )

    def scaling_solution(self, time):
        """Exact ``(a, phi, phi_dot, H)`` at ``time < 0``, with ``a = 1`` at ``t = -1``."""
        time = np.asarray(time, dtype=float)
        if np.any(time >= 0.0):
            raise ValueError(
                "the dilaton-driven branch runs at t < 0, up to the singularity at t = 0"
            )
        span = -time
        hubble = -1.0 / (self.branch * self.root_dimension * time)
        scale = span**self.scale_exponent
        phi = self.coupling_exponent * np.log(span)
        return scale, phi, self.velocity_from_hubble(hubble), hubble

    def constraint(self, hubble, velocity):
        """``C = phi_dot^2 - 2 d H phi_dot + d(d-1) H^2``, zero on a solution."""
        hubble = np.asarray(hubble, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        d = self.dimension
        return velocity**2 - 2.0 * d * hubble * velocity + d * (d - 1.0) * hubble**2

    def residuals(self, hubble, velocity, hubble_rate, acceleration):
        """Residuals of the two evolution equations, zero on a solution."""
        hubble = np.asarray(hubble, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        d = self.dimension
        return (
            np.asarray(hubble_rate, dtype=float) - (hubble * velocity - d * hubble**2),
            np.asarray(acceleration, dtype=float)
            - (d * hubble * velocity - d * (d - 1.0) * hubble**2),
        )

    def summary(self) -> dict[str, float | bool | int]:
        return {
            "dimension": self.dimension,
            "branch": self.branch,
            "scale_exponent": self.scale_exponent,
            "coupling_exponent": self.coupling_exponent,
            "superinflating": self.superinflating,
        }


@dataclass(frozen=True)
class DilatonRun:
    """A dilaton-driven background in cosmic time."""

    background: DilatonVacuum
    time: np.ndarray
    scale_factor: np.ndarray
    dilaton: np.ndarray
    velocity: np.ndarray
    hubble: np.ndarray
    constraint_drift: float
    solution: object = field(repr=False, default=None)

    @property
    def shifted_dilaton(self) -> np.ndarray:
        """``phi_bar = phi - d ln a``, invariant under scale-factor duality."""
        return self.dilaton - self.background.dimension * np.log(self.scale_factor)

    @property
    def coupling(self) -> np.ndarray:
        """``e^phi``, the string coupling squared."""
        return np.exp(self.dilaton)

    @property
    def hubble_rate(self) -> np.ndarray:
        """``H_dot`` from the equation of motion, not by differencing."""
        d = self.background.dimension
        return self.hubble * self.velocity - d * self.hubble**2

    @property
    def acceleration(self) -> np.ndarray:
        d = self.background.dimension
        return d * self.hubble * self.velocity - d * (d - 1.0) * self.hubble**2

    def summary(self) -> dict[str, float]:
        return {
            "expansion": float(self.scale_factor[-1] / self.scale_factor[0]),
            "coupling_growth": float(np.exp(self.dilaton[-1] - self.dilaton[0])),
            "curvature_growth": float(abs(self.hubble[-1] / self.hubble[0])),
            "constraint_drift": self.constraint_drift,
        }


def evolve_dilaton(
    background: DilatonVacuum | None = None,
    start: float = -1e4,
    stop: float = -1.0,
    hubble: float | None = None,
    points: int = 2001,
    rtol: float = 1e-12,
    atol: float = 1e-20,
) -> DilatonRun:
    """Integrate the unreduced dilaton-gravity system from ``start`` to ``stop``.

    Initial data is taken from the exact solution at ``start`` unless
    ``hubble`` is given, in which case ``phi_dot`` follows from the
    constraint on the chosen branch -- so the initial state solves the
    constraint by construction and the drift that is reported afterwards is
    the integrator's, not the initial data's.
    """
    background = DilatonVacuum() if background is None else background
    if not (start < stop < 0.0):
        raise ValueError(
            f"need start < stop < 0, got start = {start:g}, stop = {stop:g}: the "
            "dilaton-driven branch runs up to the singularity at t = 0"
        )
    exact = background.scaling_solution(start)
    hubble0 = float(exact[3]) if hubble is None else float(hubble)
    velocity0 = float(background.velocity_from_hubble(hubble0))
    phi0 = float(exact[1])
    d = background.dimension

    def right_hand_side(_t, state):
        _, _, velocity, hubble = state
        return [
            hubble,
            velocity,
            d * hubble * velocity - d * (d - 1.0) * hubble**2,
            hubble * velocity - d * hubble**2,
        ]

    times = np.linspace(start, stop, int(points))
    solution = solve_ivp(
        right_hand_side,
        (start, stop),
        [0.0, phi0, velocity0, hubble0],
        t_eval=times,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(f"the dilaton integration failed: {solution.message}")

    log_scale, dilaton, velocity, hubble_values = solution.y
    residual = background.constraint(hubble_values, velocity)
    scale = max(float(np.abs(velocity**2).max()), 1e-300)
    return DilatonRun(
        background=background,
        time=solution.t,
        scale_factor=np.exp(log_scale),
        dilaton=dilaton,
        velocity=velocity,
        hubble=hubble_values,
        constraint_drift=float(np.abs(residual).max() / scale),
        solution=solution,
    )


def dual(run: DilatonRun) -> DilatonRun:
    """The scale-factor dual of a run: ``a -> 1/a``, ``phi -> phi - 2 d ln a``.

    An exact symmetry of the tree-level equations, and the one that has no
    analogue in general relativity. The dual of an expanding,
    super-inflating pre-big-bang solution is a contracting one -- which is
    why the scenario has four branches and why the same solution describes
    both sides of the bounce it cannot yet compute.

    The returned run carries the same ``constraint_drift``, because the
    constraint is invariant term by term under the map: the dual is not a
    fresh integration and does not accumulate its own error.
    """
    d = run.background.dimension
    log_scale = np.log(run.scale_factor)
    return DilatonRun(
        background=DilatonVacuum(dimension=d, branch=-run.background.branch),
        time=run.time,
        scale_factor=1.0 / run.scale_factor,
        dilaton=run.dilaton - 2.0 * d * log_scale,
        velocity=run.velocity - 2.0 * d * run.hubble,
        hubble=-run.hubble,
        constraint_drift=run.constraint_drift,
        solution=run.solution,
    )
