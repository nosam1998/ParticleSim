"""Differentiable design search: does it find the shape that is known to be best?

Issue #53 asks only that a search "reduces a violation objective under
constraints". For the Alcubierre family that is a much weaker statement than
what is available, because the objective reduces exactly to a
one-dimensional functional whose minimiser is a classical Euler-Lagrange
problem with a closed-form answer. So the tests hold the search to the
*known optimum*, in both the objective and the shape, rather than to having
gone downhill.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.scenarios.warp.design import DesignProblem, energy_density, optimise
from particlesim.scenarios.warp.metrics import make_metric

SPEED = 2.0


@pytest.fixture(scope="module")
def problem() -> DesignProblem:
    return DesignProblem(speed=SPEED, passenger_radius=5.0, outer_radius=20.0, nodes=161)


@pytest.fixture(scope="module")
def search(problem) -> object:
    return optimise(problem)


# --- the objective is the physics ---------------------------------------


@pytest.mark.benchmark
def test_the_reduced_objective_is_the_published_energy_integrated():
    """``E = -(v^2/12) int f'^2 r^2 dr`` against the closed form on a 3-D grid.

    This is what licenses optimising a one-dimensional functional instead of
    a volume integral. The comparison is against the repository's own
    ``closed_form_energy_density`` -- Alcubierre's published Eulerian
    density, itself checked to 1e-15 elsewhere -- summed over a 200^3 grid.
    Measured agreement: 2.1e-12 relative.

    The reduction is exact rather than approximate because
    ``(y^2+z^2)/r_s^2 = sin^2(theta)`` and the angular integral is just
    ``8 pi/3``; nothing about the shape function enters it.
    """
    metric = make_metric("alcubierre", {"v_s": SPEED, "R": 5.0, "sigma": 1.0})
    density = metric.closed_form_energy_density()

    count, half = 200, 30.0
    axis = (np.arange(count) + 0.5) * (2 * half / count) - half
    mesh = np.meshgrid(axis, axis, axis, indexing="ij")
    volume = float(np.sum(density(0.0, *mesh))) * (2 * half / count) ** 3

    import sympy as sp

    from particlesim.scenarios.warp.metrics import alcubierre_shape

    radius = sp.Symbol("r", positive=True)
    slope = sp.lambdify(
        radius, sp.diff(alcubierre_shape(radius, sp.Integer(5), sp.Integer(1)), radius), "numpy"
    )
    line = np.linspace(1e-6, 60.0, 400001)
    reduced = -(SPEED**2 / 12.0) * np.trapezoid(slope(line) ** 2 * line**2, line)

    assert volume == pytest.approx(reduced, rel=1e-9)


def test_the_energy_is_negative_for_every_admissible_shape(problem):
    """Nothing here makes a warp bubble satisfy an energy condition.

    ``f(r_p) = 1`` with ``f(r_max) = 0`` forces a non-zero gradient and the
    integrand is a square, so the signed energy is strictly negative whatever
    the search returns. The module minimises the *violation*, which is a much
    weaker statement than removing it, and this pins the difference.
    """
    generator = np.random.default_rng(20260918)
    for _ in range(5):
        free = generator.uniform(0.0, 1.0, problem.free_count)
        assert float(problem.energy(free)) > 0.0
        assert float(problem.signed_energy(free)) < 0.0


# --- the analytic optimum ------------------------------------------------


def test_the_analytic_optimum_really_is_one(problem):
    """Euler-Lagrange against the alternatives, on the same constraints.

    ``d/dr(r^2 f') = 0`` gives ``f = A + B/r``, worth ``r_p r_max/(r_max -
    r_p)`` -- 6.667 in units where the prefactor is dropped. Measured against
    three other admissible profiles on the same grid: linear 11.67, cubic
    smoothstep 13.14, raised cosine 13.46. The optimum is not marginal.
    """
    radii = problem.radii
    inner, outer = problem.passenger_radius, problem.outer_radius

    def functional(values):
        return float(np.trapezoid(np.gradient(values, radii) ** 2 * radii**2, radii))

    span = (radii - inner) / (outer - inner)
    candidates = {
        "analytic": problem.analytic_shape(),
        "linear": 1.0 - span,
        "smoothstep": 1.0 - (3 * span**2 - 2 * span**3),
        "cosine": 0.5 * (1.0 + np.cos(np.pi * span)),
    }
    values = {name: functional(shape) for name, shape in candidates.items()}
    exact = inner * outer / (outer - inner)

    assert values["analytic"] == pytest.approx(exact, rel=1e-3)
    for name in ("linear", "smoothstep", "cosine"):
        assert values[name] > 1.5 * values["analytic"], (name, values)


def test_the_analytic_shape_satisfies_the_constraints(problem):
    shape = problem.analytic_shape()
    assert shape[0] == pytest.approx(1.0, abs=1e-12)
    assert shape[-1] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(shape) < 0.0), "the wall falls away monotonically"


# --- the search ----------------------------------------------------------


@pytest.mark.benchmark
def test_the_search_finds_the_known_optimum(problem, search):
    """Issue #53's acceptance, met against a closed form rather than a trend.

    From a linear taper the objective falls 3.889 to 2.2223 against an exact
    2.22222 -- an excess of 3.8e-05 -- and the *shape* matches the
    Euler-Lagrange solution to 5.1e-06. Reducing the objective is the
    acceptance; landing on the analytic answer is the stronger claim, and it
    is the one that would catch an objective that was subtly the wrong
    functional.
    """
    assert search.converged, search.summary()
    assert search.energy < search.initial_energy
    assert search.energy == pytest.approx(problem.analytic_energy(), rel=1e-3)
    assert 0.0 < search.excess < 1e-3, search.excess
    assert np.max(np.abs(search.shape - problem.analytic_shape())) < 1e-4


@pytest.mark.benchmark
def test_the_search_beats_the_usual_tanh_wall(problem):
    """A factor of 7.8, and the reason is where the wall is put.

    The ``tanh`` profile costs 17.39 on this problem against the optimum's
    2.222. The ``r^2`` weight makes gradient expensive at large radius, so
    the cheap thing to do is transition close to the passenger region and
    coast outward -- which is the quantitative form of the observation that
    thick walls are cheaper, and the opposite of what a fixed-radius
    ``tanh`` wall does.
    """
    start = problem.alcubierre_shape(thickness=1.0)
    result = optimise(problem, initial=start)
    assert result.initial_energy > 15.0
    assert result.reduction > 5.0, result.summary()
    assert result.energy == pytest.approx(problem.analytic_energy(), rel=1e-3)


def test_the_optimum_respects_every_constraint(problem, search):
    """Passenger region, outer boundary, and positivity -- checked, not imposed.

    The endpoints are pinned by construction, so those two are structural.
    Staying within ``[0, 1]`` and falling monotonically are *not* imposed
    anywhere: the search is free to overshoot and simply does not, which is
    worth checking rather than assuming, because a shape that dipped below
    zero would be a bubble that reversed and would still score well.
    """
    shape = search.shape
    assert shape[0] == pytest.approx(1.0, abs=1e-12), "the passenger region stays flat"
    assert shape[-1] == pytest.approx(0.0, abs=1e-12), "and the bubble dies by r_max"
    assert float(shape.min()) >= -1e-9
    assert float(shape.max()) <= 1.0 + 1e-9
    assert np.all(np.diff(shape) <= 1e-9), "monotone, though nothing required it to be"


@pytest.mark.slow
@pytest.mark.benchmark
def test_the_excess_over_the_continuum_optimum_is_second_order(problem):
    """What the residual 3.8e-05 actually is: the grid, not the optimiser.

    The midpoint discretisation of ``int f'^2 r^2 dr`` over-counts a curving
    profile, so the discrete minimum sits above the continuum one. Measured,
    it falls by a factor of four per doubling:

        nodes      41        81       161       321
        excess  6.14e-04  1.54e-04  3.84e-05  9.61e-06

    Second order, which is what the midpoint rule gives, and which says the
    gap is the discretisation rather than a search that stopped early.
    """
    excesses = []
    for nodes in (41, 81, 161):
        result = optimise(DesignProblem(speed=SPEED, nodes=nodes), max_iterations=4000)
        excesses.append(result.excess)
    ratios = [excesses[i] / excesses[i + 1] for i in range(len(excesses) - 1)]
    assert all(3.0 < ratio < 5.5 for ratio in ratios), (excesses, ratios)


def test_the_history_records_every_objective_evaluation(search):
    """What a live view would plot, and it has to go down."""
    assert search.history.size > 10
    assert search.history[0] == pytest.approx(search.initial_energy, rel=1e-12)
    assert float(search.history.min()) == pytest.approx(search.energy, rel=1e-6)
    assert search.history[-1] < search.history[0]


# --- the gradient, and the bug that made it fail -------------------------


def test_the_gradient_is_exact_not_a_finite_difference(problem):
    """Reverse-mode against central differences, on a random point.

    The whole reason a search over a hundred and sixty free nodes is
    affordable: one reverse-mode gradient costs a few objective evaluations
    regardless of the dimension, where differencing costs one per node.
    """
    import jax

    from particlesim.symbolic.codegen import _backend

    _backend(problem.backend)
    generator = np.random.default_rng(7)
    point = generator.uniform(0.1, 0.9, problem.free_count)
    exact = np.asarray(jax.grad(problem.energy)(point), dtype=float)

    step = 1e-6
    for index in (0, problem.free_count // 2, problem.free_count - 1):
        forward, backward = point.copy(), point.copy()
        forward[index] += step
        backward[index] -= step
        difference = (float(problem.energy(forward)) - float(problem.energy(backward))) / (
            2.0 * step
        )
        assert exact[index] == pytest.approx(difference, rel=1e-6)


def test_the_objective_runs_in_double_precision(problem):
    """The ordering hazard that made the first search crash.

    ``_backend`` is what turns on ``jax_enable_x64``, and calling it for the
    first time *inside* a jitted function enables it mid-trace: the
    executable is compiled for the f32 inputs JAX downcast to, and the next
    call passes f64 and fails with "expected parameter 0 of size 636
    (f32[159]) but got 1272 (f64[159])". Here it raised; on
    :func:`particlesim.solvers.nr.bssn._module` the same ordering silently
    produced single-precision initial data instead.
    """
    import jax

    from particlesim.symbolic.codegen import _backend

    _backend(problem.backend)
    assert jax.config.read("jax_enable_x64")
    value = problem.energy(np.linspace(1.0, 0.0, problem.nodes)[1:-1])
    assert value.dtype == np.float64


def test_a_shape_put_back_on_a_grid_reproduces_the_density(problem):
    """The reduced objective is an integral; this is what it integrates.

    A sampled shape is pushed back onto a Cartesian grid and its energy
    density summed, against the same shape's one-dimensional objective. They
    agree to the interpolation error, which is what says the reduction did
    not quietly lose the angular structure.
    """
    shape = problem.analytic_shape()
    count, half = 140, 26.0
    axis = (np.arange(count) + 0.5) * (2 * half / count) - half
    mesh = np.meshgrid(axis, axis, axis, indexing="ij")
    volume = float(np.sum(energy_density(problem, shape, mesh))) * (2 * half / count) ** 3

    radii = problem.radii
    reduced = -(SPEED**2 / 12.0) * float(
        np.trapezoid(np.gradient(shape, radii) ** 2 * radii**2, radii)
    )
    assert volume == pytest.approx(reduced, rel=2e-2)


# --- refusals ------------------------------------------------------------


def test_an_outer_radius_inside_the_passenger_region_is_refused():
    with pytest.raises(ValueError, match="must exceed the passenger radius"):
        DesignProblem(passenger_radius=10.0, outer_radius=5.0)


def test_a_non_positive_passenger_radius_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        DesignProblem(passenger_radius=0.0)


def test_too_few_nodes_is_refused():
    with pytest.raises(ValueError, match="at least three nodes"):
        DesignProblem(nodes=2)


def test_a_mismatched_initial_shape_is_refused(problem):
    with pytest.raises(ValueError, match="interior nodes"):
        optimise(problem, initial=np.zeros(7))
