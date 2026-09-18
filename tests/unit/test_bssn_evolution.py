"""BSSN evolution on a periodic grid: initial data, the integrator, convergence.

The acceptance test of milestone 4's first issue is one line -- the
constraints converge at fourth order on a gauge wave -- and getting there
turned on one thing, which is what most of this file is about.

**Why the conformal Ricci tensor has to be written with ``Gammabar^i``.**
Run the evolution with ``Rbar_ij`` computed straight from the conformal
metric, which is the same tensor and the obvious thing to do when the
symbolic module already has a Ricci formula, and the gauge wave converges
at fourth order from 16 to 32 points, stalls at 64 and blows up at 128.
Measured here before the fix, sampling ``|H|`` every sixteen steps at
128 points:

    3.3e-09  6.3e-08  8.1e-06  1.1e-03  1.4e-01  1.9e+01  2.6e+03

a clean exponential, and halving the time step did not change its rate per
unit time while doubling the resolution did: the growth rate scales like
``1/h``. That is not a Courant violation, it is the continuum system being
only weakly hyperbolic. Carrying ``Gammabar^i`` as an evolved variable and
using it to write the mixed second derivatives leaves a flat wave operator
as the principal part of every component, and the same study then converges
at fourth order at every resolution tried.

The symbolic side of that -- that the two forms are the *same tensor*,
exactly -- is checked in ``test_symbolic_bssn.py``. What is checked here is
the other half: that the form the evolution uses converges. The numbers
above are recorded rather than asserted, because a test that evolved the
broken form to prove it breaks would cost a second twenty-four-equation
derivation to say what the convergence test already says.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.core.grid import derivative
from particlesim.solvers.nr import bssn
from particlesim.symbolic.threeplusone import inverse_metric

SMALL = (16, 8, 8)


@pytest.fixture(scope="module")
def wave():
    """Gauge-wave data and an evolution of it, built once for the module."""
    state, spacing = bssn.gauge_wave(shape=SMALL, amplitude=0.1, extent=1.0)
    evolution = bssn.Evolution.build(spacing, slicing="harmonic", shift_condition="frozen")
    return state, spacing, evolution


# --- initial data -------------------------------------------------------


def test_the_gauge_wave_is_an_exact_solution_of_the_constraints(wave):
    """Flat space in a wavy gauge: both constraints vanish identically.

    Not to truncation error -- identically, because the data is flat
    spacetime. What is left is the round-off of evaluating the closed forms,
    so this is the sharpest test of the initial data there is: a factor
    wrong anywhere in the change of variables shows up at 1e-3, not 1e-13.
    """
    state, _, evolution = wave
    violation = evolution.constraints(state)
    assert violation.hamiltonian < 1e-10
    assert violation.momentum < 1e-12
    assert violation.determinant < 1e-14
    assert violation.trace < 1e-14


def test_the_gauge_wave_connection_is_the_one_the_metric_defines():
    """``Gammabar^i = -d_j gammabar^ij``, and the closed form is that.

    ``Gammabar^i`` is the one evolved variable that is a spatial derivative
    of the others, so initialising it by differencing would seed exactly
    the error a convergence test is trying to measure. It is set from
    ``(2/3) H' H^(-5/3) n^i`` instead, and this checks that against the
    definition -- differenced, so the agreement is to truncation order and
    improves by sixteen when the grid is halved.

    Compared over the interior only. ``core.grid.derivative`` is the
    bounded-domain path and drops to a one-sided second-order stencil
    within the stencil radius of an edge, which on periodic data is simply
    the wrong formula. Including those points measures second-order
    convergence of the edge treatment -- 3.9 and 4.3 per halving, measured
    -- rather than fourth-order agreement of the closed form.
    """
    margin = 3
    ratios = []
    previous = None
    for n in (16, 32, 64):
        state, spacing = bssn.gauge_wave(
            shape=(n, 8, 8), amplitude=0.1, extent=1.0, backend="numpy"
        )
        upper = inverse_metric(
            [[np.asarray(state[f"gt{min(i, j)}{max(i, j)}"]) for j in range(3)] for i in range(3)]
        )
        worst = 0.0
        for i in range(3):
            defined = -sum(derivative(upper[i][j], j, spacing[j], order=4) for j in range(3))
            residual = (defined - np.asarray(state[f"Gt{i}"]))[margin:-margin]
            worst = max(worst, float(np.max(np.abs(residual))))
        if previous is not None:
            ratios.append(previous / worst)
        previous = worst
    assert all(ratio > 12.0 for ratio in ratios), ratios


def test_a_diagonal_gauge_wave_makes_every_metric_component_non_trivial():
    """The searching direction: ``(1,1,1)`` leaves no component switched off.

    The two thresholds differ on purpose. With ``u = (1,1,1)/sqrt(3)`` the
    conformal metric is ``H^(-1/3) (delta_ij + (H-1) u_i u_j)``, and on the
    diagonal the two factors very nearly cancel:

        H^(-1/3) (1 + (H-1)/3) = 1 + (H-1)^2 / 9 + O((H-1)^3)

    so the diagonal components vary at *second* order in the amplitude
    while the off-diagonal ones vary at first. At A = 0.1 that is a factor
    of six between them, measured, and it is worth knowing before reading a
    convergence study on this data: the diagonal components carry much less
    signal than they look like they do.
    """
    state, _ = bssn.gauge_wave(shape=(16, 16, 16), amplitude=0.1, direction=(1.0, 1.0, 1.0))
    for i in range(3):
        for j in range(i, 3):
            deviation = float(np.std(np.asarray(state[f"gt{i}{j}"])))
            assert deviation > (1e-4 if i == j else 1e-3), (i, j, deviation)


def test_the_punctures_miss_the_grid():
    """``psi = 1 + m/(2r)`` is infinite at ``r = 0``, so no sample may sit there.

    The grid runs from zero with ``endpoint=False``, which puts a sample
    exactly at the box centre whenever the shape is even -- and the centre
    is where a single puncture goes. Before the punctures were staggered by
    half a cell this returned ``inf``, every constraint was ``NaN`` from the
    first evaluation, and a forty-step run reported ``NaN`` without ever
    raising. The nearest sample is now ``sqrt(3)/2`` of a cell away.
    """
    shape, extent = (32, 32, 32), 8.0
    state, spacing = bssn.brill_lindquist(
        shape=shape, punctures=((1.0, (0.0, 0.0, 0.0)),), extent=extent
    )
    conformal = np.exp(np.asarray(state["phi"]))
    assert np.all(np.isfinite(conformal))
    # psi_max = 1 + m / (2 r_min) with r_min = sqrt(3)/2 dx.
    closest = np.sqrt(3.0) / 2 * spacing[0]
    assert float(np.max(conformal)) == pytest.approx(1 + 1.0 / (2 * closest), rel=1e-12)


def test_a_puncture_on_a_grid_point_is_refused():
    """Offset it back onto the lattice and the data says so instead of returning inf.

    Half a cell back along *every* axis: the radius only vanishes where all
    three coordinates coincide, so an offset on one axis alone still leaves
    the nearest sample a face diagonal away and is perfectly fine.
    """
    half_cell = 8.0 / 32 / 2
    with pytest.raises(ValueError, match="lands on a grid point"):
        bssn.brill_lindquist(shape=(32, 32, 32), punctures=((1.0, (-half_cell,) * 3),), extent=8.0)


def test_the_gauge_wave_direction_must_be_a_direction():
    with pytest.raises(ValueError, match="non-zero"):
        bssn.gauge_wave(shape=SMALL, direction=(0.0, 0.0, 0.0))


# --- the algebraic constraints -----------------------------------------


def test_the_projection_restores_the_algebraic_constraints(wave):
    """Bend ``det gammabar`` and ``tr Abar`` away from their values and project.

    The two constraints are preserved by the continuum equations and drift
    discretely, so nothing in the right-hand side pulls the state back.
    ``project`` is what does, and it has to do it without moving anything
    else: the trace-free part of ``Abar_ij`` and the conformal class of
    ``gammabar_ij`` are the physical content and must come through
    untouched.
    """
    state, _, evolution = wave
    bent = dict(state)
    for i in range(3):
        bent[f"gt{i}{i}"] = np.asarray(state[f"gt{i}{i}"]) * 1.03
        bent[f"At{i}{i}"] = np.asarray(state[f"At{i}{i}"]) + 0.02
    before = evolution.constraints(bent)
    assert before.determinant > 1e-3
    assert before.trace > 1e-3

    after = evolution.constraints(evolution.project(bent))
    assert after.determinant < 1e-14
    assert after.trace < 1e-14


def test_the_projection_leaves_data_that_already_satisfies_them_alone(wave):
    state, _, evolution = wave
    projected = evolution.project(state)
    for name, value in state.items():
        assert np.allclose(np.asarray(projected[name]), np.asarray(value), atol=1e-13)


# --- the integrator -----------------------------------------------------


def test_flat_space_is_a_fixed_point():
    """Minkowski with harmonic slicing and a frozen shift does not move."""
    state, spacing = bssn.gauge_wave(shape=SMALL, amplitude=0.0, extent=1.0)
    evolution = bssn.Evolution.build(spacing, slicing="harmonic", shift_condition="frozen")
    advanced = evolution.step(state)
    for name, value in state.items():
        assert np.allclose(np.asarray(advanced[name]), np.asarray(value), atol=1e-14), name


def test_the_time_step_follows_the_courant_factor(wave):
    _, spacing, evolution = wave
    assert evolution.time_step == pytest.approx(evolution.courant * min(spacing))
    assert evolution.time_step == pytest.approx(bssn.COURANT / SMALL[0])


def test_dissipation_damps_the_shortest_wavelength_and_spares_the_longest():
    """Kreiss-Oliger, the only thing it is for.

    The operator annihilates polynomials below its degree, so a smooth mode
    is barely touched and the mode the grid cannot represent is damped at
    ``epsilon/h``. Both halves matter: an operator that damps everything is
    a viscosity that would spoil the convergence order.
    """
    # Constructed rather than built: the dissipation operator needs the
    # spacing, the order and the backend, and nothing from the kernel.
    spacing, strength = 0.05, 0.1
    evolution = bssn.Evolution(spacing=(spacing,) * 3, backend="numpy", dissipation=strength)
    operator = evolution._dissipator()
    axis = np.arange(32)
    smooth = np.broadcast_to(np.sin(2 * np.pi * axis / 32)[:, None, None], (32, 32, 32))
    # A checkerboard, so that all three axes contribute rather than one: the
    # operator sums an independent stencil per axis, and a mode constant
    # along y and z is damped at eps/h, not 3 eps/h.
    checker = (-1.0) ** (axis[:, None, None] + axis[None, :, None] + axis[None, None, :])
    damped = operator(np.stack([np.asarray(smooth, dtype=float), checker]))
    assert np.max(np.abs(damped[0])) < 1e-4
    assert np.max(np.abs(damped[1])) == pytest.approx(3 * strength / spacing, rel=1e-12)


def test_the_dissipation_operator_rejects_an_order_it_has_no_weights_for():
    with pytest.raises(ValueError, match="order must be"):
        bssn.dissipation_operator(order=3)


def test_an_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="unknown backend"):
        bssn.gauge_wave(shape=SMALL, backend="cupy")


# --- what the derivation cost, and what it produced ---------------------


def test_the_kernel_reports_the_derivation_it_came_from(wave):
    """The right-hand side is derived, and the numbers say by how much.

    Twenty-four outputs, and the elimination is the difference between a
    kernel that runs and one that does not: the raw operation count is what
    a transcription of the equations would have to be.
    """
    _, _, evolution = wave
    summary = evolution.kernel.summary()
    assert summary["outputs"] == 24
    assert summary["raw_operations"] > 1_000_000
    assert summary["operations"] < 5_000
    assert summary["raw_operations"] / summary["operations"] > 100
    assert summary["stencils"] < 7 * summary["fields"]


# --- the acceptance test ------------------------------------------------


@pytest.mark.slow
@pytest.mark.benchmark
def test_the_constraints_converge_at_fourth_order_on_the_gauge_wave():
    """Issue #47's acceptance criterion, at three resolutions.

    Fourth order means sixteen per halving. All three quantities are
    checked -- the Hamiltonian constraint, the momentum constraint and the
    error against the exact solution -- because a scheme can converge on the
    solution while the constraints do something else, and it is the
    constraints that say the system being solved is still Einstein's.

    The time step is fixed by the Courant factor, so refining the grid
    refines the step too and this measures the scheme rather than the
    spatial stencil alone.
    """
    final = 0.25
    results = []
    for n in (16, 32, 64):
        state, spacing = bssn.gauge_wave(shape=(n, 8, 8), amplitude=0.1, extent=1.0)
        evolution = bssn.Evolution.build(
            spacing, slicing="harmonic", shift_condition="frozen", dissipation=0.1
        )
        steps = max(1, int(round(final / evolution.time_step)))
        history = evolution.run(state, steps, time_step=final / steps)
        exact, _ = bssn.gauge_wave(shape=(n, 8, 8), amplitude=0.1, time=final, extent=1.0)
        results.append(
            (
                history.final.hamiltonian,
                history.final.momentum,
                bssn.solution_error(history.state, exact),
            )
        )
        assert history.final.determinant < 1e-12
        assert history.final.trace < 1e-12

    for index, label in enumerate(("hamiltonian", "momentum", "solution")):
        for coarse, fine in zip(results, results[1:], strict=False):
            ratio = coarse[index] / fine[index]
            assert ratio > 12.0, (label, ratio)
            assert ratio < 20.0, (label, ratio)


@pytest.mark.slow
def test_the_moving_puncture_gauge_runs_bounded():
    """1+log slicing and the Gamma-driver shift on a puncture, forty steps.

    Not a stability claim -- a single puncture on a torus is an infinite
    lattice of them, and stability to t = 1000 M is issue #51 with a proper
    outer boundary. What this checks is that the gauge is wired up and does
    what a moving-puncture gauge does: the lapse starts pre-collapsed at
    ``psi^-2``, the shift starts at zero and the driver pushes it away from
    zero, and nothing runs away.

    The Hamiltonian constraint is large here and that is not a failure: the
    data is exactly conformally flat and time-symmetric, so ``H`` vanishes
    analytically and what is measured is a fourth-order stencil applied to
    ``1/r`` at sqrt(3)/2 of a cell from the singularity. It has to stay bounded,
    not be small.

    Its own kernel, so this pays a derivation the first time it runs.
    """
    state, spacing = bssn.brill_lindquist(
        shape=(32, 32, 32), punctures=((1.0, (0.0, 0.0, 0.0)),), extent=8.0
    )
    evolution = bssn.Evolution.build(
        spacing, slicing="one_plus_log", shift_condition="gamma_driver"
    )
    history = evolution.run(state, 40, sample_every=10)

    violations = [record.hamiltonian for record in history.constraints]
    assert all(np.isfinite(value) for value in violations), violations
    assert max(violations) < 4 * violations[0], violations
    assert history.final.determinant < 1e-12
    assert history.final.trace < 1e-12

    lapse = np.asarray(history.state["alpha"])
    shift = np.asarray(history.state["beta0"])
    assert float(np.min(lapse)) > 0.0
    assert float(np.max(lapse)) <= 1.0 + 1e-12
    assert 0.0 < float(np.max(np.abs(shift))) < 1.0


@pytest.mark.slow
@pytest.mark.benchmark
def test_sixth_order_stencils_converge_at_sixth_order():
    """The same study with ``order=6``, where the target is 64 per halving.

    The time step refines with the grid at a fixed Courant factor, so RK4's
    fourth-order temporal error is mixed in and would cap the rate at 16
    eventually. It has not started to at 64 points -- the measured ratios
    climb, 51 then 60 -- which says the temporal error is still well under
    the spatial one at these resolutions.

    Its own kernel, so this pays a derivation the first time it runs.
    """
    final = 0.25
    results = []
    for n in (16, 32, 64):
        state, spacing = bssn.gauge_wave(shape=(n, 8, 8), amplitude=0.1, extent=1.0)
        evolution = bssn.Evolution.build(
            spacing, order=6, slicing="harmonic", shift_condition="frozen"
        )
        steps = max(1, int(round(final / evolution.time_step)))
        history = evolution.run(state, steps, time_step=final / steps)
        results.append(history.final.hamiltonian)

    ratios = [coarse / fine for coarse, fine in zip(results, results[1:], strict=False)]
    assert all(ratio > 40.0 for ratio in ratios), ratios
    assert ratios[1] > ratios[0], ratios
