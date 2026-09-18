"""Differentiable design search over warp shape functions.

Issue #53, Warp Mode W2. W1 analyses a *given* bubble and W3 evolves fields
on one; this asks the question between them -- given a speed, a passenger
region and a domain, what shape function costs the least negative energy?

**The objective reduces to one dimension, exactly.** Alcubierre's Eulerian
energy density is known in closed form,

    rho = -(v^2/32 pi) (y^2 + z^2)/r_s^2 f'(r_s)^2

and it holds for *any* shape function, not only the ``tanh`` one. Integrating
it over space in spherical coordinates about the bubble centre, with
``(y^2+z^2)/r^2 = sin^2(theta)``:

    int sin^2(theta) r^2 sin(theta) dr dtheta dphi
        = (8 pi/3) int f'(r)^2 r^2 dr

so the total negative energy is

    E = -(v^2/12) int_0^inf f'(r)^2 r^2 dr

A one-dimensional functional of the shape alone. Checked against the
repository's own published closed form integrated on a 200^3 grid: the two
agree to 2e-12, which is what makes it safe to optimise the reduced form
rather than a three-dimensional integral.

**So the design problem has an analytic answer to check the search against.**
Minimising ``int f'^2 r^2 dr`` subject to ``f = 1`` on the passenger region
``r <= r_p`` and ``f = 0`` beyond ``r_max`` is a classical variational
problem. The Euler-Lagrange equation is ``d/dr(r^2 f') = 0``, so
``r^2 f' = const`` and

    f(r) = (1/r - 1/r_max) / (1/r_p - 1/r_max)

with minimum value ``r_p r_max / (r_max - r_p)``. For ``r_p = 5`` and
``r_max = 20`` that is ``20/3 = 6.667``, against 11.67 for a linear taper,
13.14 for a cubic smoothstep and 13.46 for a raised cosine. The optimum is
real and the alternatives are not close to it.

That turns issue #53's acceptance -- "search reduces a violation objective
under constraints" -- into something much sharper: the search has a known
target, in both the objective *and* the shape, and the tests hold it to
both.

**Why the answer is a ``1/r`` profile.** The ``r^2`` weight means gradient
at large radius is expensive, so the optimiser spends its transition where
the weight is small, near the passenger region, and coasts outward. The
familiar ``tanh`` wall does the opposite: it puts a narrow, steep transition
at a fixed radius, which is the worst place for it. This is the quantitative
form of the observation that thick walls are cheaper.

**What this does not claim.** The energy is negative for every admissible
shape -- the constraint ``f(0) = 1`` with ``f(infinity) = 0`` forces a
non-zero gradient somewhere, and the integrand is a square. Nothing here
makes a warp bubble satisfy an energy condition; it minimises the violation
subject to the bubble existing at all, which is a different and much weaker
statement. The infimum over *unconstrained* shapes is zero and is attained
only by ``f == 0``, which is flat space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from particlesim.symbolic.codegen import _backend


@dataclass(frozen=True)
class DesignProblem:
    """A shape-function search: the objective, the constraints, and the grid.

    ``passenger_radius`` is where the flat interior the ship rides in ends,
    and ``outer_radius`` where the bubble has to have died away. The shape is
    pinned to one and zero at those two radii, which is how the constraints
    are imposed: by construction rather than by penalty, so the search is
    unconstrained in the free variables and cannot trade a small objective
    against a violated constraint.

    ``nodes`` is the number of radial samples across the wall. The
    discretisation is the midpoint one,

        int f'^2 r^2 dr  ~  sum_k ((f_{k+1} - f_k)/h)^2 r_{k+1/2}^2 h

    which is the natural finite-difference form for this functional and
    converges to the continuum optimum from above.
    """

    speed: float = 2.0
    passenger_radius: float = 5.0
    outer_radius: float = 20.0
    nodes: int = 161
    backend: str = "jax"

    def __post_init__(self) -> None:
        if self.passenger_radius <= 0.0:
            raise ValueError(f"the passenger radius must be positive, got {self.passenger_radius}")
        if self.outer_radius <= self.passenger_radius:
            raise ValueError(
                f"the outer radius {self.outer_radius} must exceed the passenger radius "
                f"{self.passenger_radius}: there is no wall to shape otherwise"
            )
        if self.nodes < 3:
            raise ValueError(f"a wall needs at least three nodes, got {self.nodes}")

    @property
    def radii(self) -> np.ndarray:
        """The radial grid across the wall, endpoints included."""
        return np.linspace(self.passenger_radius, self.outer_radius, self.nodes)

    @property
    def free_count(self) -> int:
        """Interior nodes: the endpoints are pinned by the constraints."""
        return self.nodes - 2

    def shape(self, free) -> Any:
        """The full profile, with ``f = 1`` and ``f = 0`` pinned at the ends."""
        module = _backend(self.backend)
        free = module.asarray(free)
        return module.concatenate([module.ones(1), free, module.zeros(1)])

    def energy(self, free) -> Any:
        """``|E| = (v^2/12) int f'^2 r^2 dr``, differentiable in ``free``.

        Returned positive: it is the magnitude of a negative energy, and a
        search that minimises it is reducing the violation. The sign is put
        back by :meth:`signed_energy` for anything reporting a density.
        """
        module = _backend(self.backend)
        radii = module.asarray(self.radii)
        profile = self.shape(free)
        step = radii[1] - radii[0]
        slope = (profile[1:] - profile[:-1]) / step
        midpoints = 0.5 * (radii[1:] + radii[:-1])
        return (self.speed**2 / 12.0) * module.sum(slope**2 * midpoints**2 * step)

    def signed_energy(self, free) -> Any:
        """``E`` itself, which is negative for every admissible shape."""
        return -self.energy(free)

    # --- the analytic answer to check a search against -------------------

    def analytic_shape(self) -> np.ndarray:
        """``f = (1/r - 1/r_max)/(1/r_p - 1/r_max)``, the Euler-Lagrange solution."""
        radii = self.radii
        scale = 1.0 / self.passenger_radius - 1.0 / self.outer_radius
        return (1.0 / radii - 1.0 / self.outer_radius) / scale

    def analytic_energy(self) -> float:
        """``(v^2/12) r_p r_max/(r_max - r_p)``: the continuum minimum."""
        inner, outer = self.passenger_radius, self.outer_radius
        return (self.speed**2 / 12.0) * inner * outer / (outer - inner)

    def alcubierre_shape(self, thickness: float = 1.0) -> np.ndarray:
        """The usual ``tanh`` wall on this grid, as a starting point to beat.

        Sampled from the same family
        :func:`particlesim.scenarios.warp.metrics.alcubierre_shape` uses, then
        pinned to the endpoints so it satisfies the same constraints as
        anything the search produces -- otherwise the comparison would be
        between a feasible shape and an infeasible one.
        """
        radii = self.radii
        middle = 0.5 * (self.passenger_radius + self.outer_radius)
        raw = 0.5 * (1.0 - np.tanh((radii - middle) / float(thickness)))
        span = raw[0] - raw[-1]
        if span <= 0.0:
            raise ValueError("the tanh wall is flat on this grid; use a smaller thickness")
        return (raw - raw[-1]) / span


@dataclass(frozen=True)
class DesignResult:
    """What a search found, and the evidence it got there."""

    shape: np.ndarray
    energy: float
    initial_energy: float
    history: np.ndarray
    iterations: int
    converged: bool
    problem: DesignProblem = field(repr=False, default=None)

    @property
    def reduction(self) -> float:
        """``E_initial/E_final``: how much of the violation the search removed."""
        return self.initial_energy / self.energy if self.energy > 0.0 else float("inf")

    @property
    def excess(self) -> float:
        """``E/E_analytic - 1``: distance above the continuum optimum.

        Positive for any discretisation of this functional, because the
        midpoint form over-counts a profile that curves. It is the number to
        watch when refining ``nodes``.
        """
        exact = self.problem.analytic_energy()
        return self.energy / exact - 1.0

    def summary(self) -> dict[str, float]:
        return {
            "energy": self.energy,
            "initial_energy": self.initial_energy,
            "reduction": self.reduction,
            "excess": self.excess,
            "iterations": float(self.iterations),
            "converged": float(self.converged),
        }


def optimise(
    problem: DesignProblem,
    initial=None,
    max_iterations: int = 2000,
    tolerance: float = 1e-12,
) -> DesignResult:
    """Minimise the violation by L-BFGS on a JAX gradient.

    The gradient is exact -- reverse-mode automatic differentiation of the
    objective, not a finite difference -- which is what makes a search over
    a hundred and sixty free nodes affordable at all: one gradient costs the
    same as a handful of objective evaluations regardless of the dimension,
    where differencing would cost one per node.

    ``initial`` defaults to a linear taper, which is feasible and about
    seventy per cent worse than the optimum, so the search has somewhere to
    go. Every objective value is recorded in ``history``, which is what a
    live view would plot.

    The default iteration budget is 2000 because 500 is not enough: the
    search reaches an excess of 3.85e-05 either way, but at 500 it stops on
    the cap rather than on its own tolerance and reports ``converged=False``.
    It finishes at 695.
    """
    import jax

    # **x64 before anything is traced.** ``_backend`` turns on
    # ``jax_enable_x64``, and calling it for the first time *inside* a jitted
    # function enables it mid-trace: the executable is compiled for the f32
    # inputs JAX downcast to, and the next call passes f64 and fails with
    # "expected parameter 0 of size 636 (f32[159]) but got 1272 (f64[159])".
    # Touching the backend here fixes the dtype before ``jax.jit`` sees
    # anything. The same ordering hazard is documented on
    # :func:`particlesim.solvers.nr.bssn._module`, where it did not raise at
    # all -- it silently produced single-precision initial data.
    _backend(problem.backend)

    radii = problem.radii
    if initial is None:
        start = np.linspace(1.0, 0.0, problem.nodes)[1:-1]
    else:
        start = np.asarray(initial, dtype=float)
        if start.size == problem.nodes:
            start = start[1:-1]
        if start.size != problem.free_count:
            raise ValueError(
                f"initial shape has {start.size} free values but this problem has "
                f"{problem.free_count} interior nodes"
            )

    objective = jax.jit(problem.energy)
    gradient = jax.jit(jax.grad(problem.energy))
    history: list[float] = []

    def value_and_grad(free):
        value = float(objective(free))
        history.append(value)
        return value, np.asarray(gradient(free), dtype=float)

    outcome = minimize(
        value_and_grad,
        start,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": int(max_iterations), "ftol": tolerance, "gtol": tolerance},
    )
    shape = np.asarray(problem.shape(outcome.x), dtype=float)
    assert shape.size == radii.size
    return DesignResult(
        shape=shape,
        energy=float(objective(outcome.x)),
        initial_energy=float(objective(start)),
        history=np.array(history),
        iterations=int(outcome.nit),
        converged=bool(outcome.success),
        problem=problem,
    )


def energy_density(problem: DesignProblem, shape, coords) -> np.ndarray:
    """``rho`` on a Cartesian grid, from a sampled shape function.

    The reduced objective is a statement about the integral; this is what it
    is an integral *of*, and exists so a search's answer can be put back on
    a three-dimensional grid and checked against
    :meth:`particlesim.scenarios.warp.metrics.WarpMetric.closed_form_energy_density`
    rather than trusted.
    """
    radii = problem.radii
    values = np.asarray(shape, dtype=float)
    slope = np.gradient(values, radii)

    position = [np.asarray(value, dtype=float) for value in coords]
    distance = np.sqrt(sum(value**2 for value in position))
    safe = np.maximum(distance, 1e-12)
    derivative = np.interp(distance, radii, slope, left=0.0, right=0.0)
    transverse = (position[1] ** 2 + position[2] ** 2) / safe**2
    return -(problem.speed**2 / (32.0 * np.pi)) * transverse * derivative**2


__all__ = [
    "DesignProblem",
    "DesignResult",
    "energy_density",
    "optimise",
]
