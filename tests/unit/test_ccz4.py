"""CCZ4: the Z4 system in the conformal variables, symbolically and on a grid.

Two things have to be true of a constraint-damping formulation, and they
pull in opposite directions. It has to be the *same* physics as BSSN --
otherwise the damping is buying agreement with something that is no longer
Einstein's equations -- and it has to behave *differently* on data that
violates the constraints, or the damping is not doing anything.

The first is checked exactly. With ``Theta = 0`` and ``Z_i = 0`` every CCZ4
right-hand side reduces to one the suite has already verified against exact
solutions, with residual zero rather than small. The second is checked by
measurement: the same violating state, evolved by BSSN and by CCZ4 through
the same integrator, gauge and diagnostics, and the violation goes two
different ways.

Note which checks touch which terms. The Z terms -- ``2 D_i Z^i``,
``2 D_(i Z_j)``, and the damping -- vanish identically wherever ``Theta``
and ``Z_i`` do, so *none* of the exact reduction identities exercise them.
Only the violating-data runs do.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from particlesim.solvers.nr import bssn as bssn_solver
from particlesim.solvers.nr import ccz4 as ccz4_solver
from particlesim.symbolic import bssn, ccz4
from particlesim.symbolic.threeplusone import (
    INDICES,
    adm_rhs,
    hamiltonian_constraint,
    inverse_metric,
    symbolic_slice,
    trace,
)

X, Y, Z = sp.symbols("x y z", real=True)
COORDS = (X, Y, Z)
POINT = {X: sp.Rational(1, 3), Y: sp.Rational(1, 4), Z: sp.Rational(1, 5)}


def _unimodular():
    """``det gamma = 1`` by construction, every component non-zero.

    The same slice the connection-form Ricci identity is checked on, and
    for the same reason: unimodular keeps ``e^(-4 phi) = 1`` so everything
    stays polynomial and SymPy can carry it, while ``det gammabar = 1`` is
    the condition BSSN and CCZ4 both evolve under.
    """
    small = sp.Rational(1, 5) * X + sp.Rational(1, 7) * Y * Z
    middle = sp.Rational(1, 6) * Z + sp.Rational(1, 11) * X * Y
    large = sp.Rational(1, 4) * Y + sp.Rational(1, 9) * X * Z
    metric = [
        [1, small, middle],
        [small, small**2 + 1, small * middle + large],
        [middle, small * middle + large, middle**2 + large**2 + 1],
    ]
    curvature = [
        [X / 3, Y / 5, Z / 7],
        [Y / 5, Y * Z / 4, X / 9],
        [Z / 7, X / 9, Z * X / 6],
    ]
    return 1 + X * Z / 4, [Y / 6, X * Z / 8, (X + Y) / 7], metric, curvature


@pytest.fixture(scope="module")
def variables():
    return bssn.from_adm(symbolic_slice(*_unimodular(), COORDS))


# --- the state ----------------------------------------------------------


def test_the_ccz4_state_is_the_bssn_state_and_theta():
    """One extra equation rather than a renumbering."""
    assert ccz4.STATE_NAMES[:-1] == bssn.STATE_NAMES
    assert ccz4.STATE_NAMES[-1] == "Theta"
    assert len(ccz4.STATE_NAMES) == 25
    state, derivatives, registry = ccz4.abstract_state()
    assert set(state) == set(ccz4.STATE_NAMES)
    # Theta appears differenced once and never twice.
    assert {f"d_Theta_{k}" for k in INDICES} <= set(derivatives)
    assert not any(name.startswith("dd_Theta") for name in derivatives)
    assert len(registry) == len(derivatives)


def test_exact_data_starts_on_the_constraint_surface():
    """``Theta = 0`` for every exact solution, which is why one cannot test the damping."""
    state, _ = ccz4_solver.gauge_wave(shape=(16, 8, 8), amplitude=0.1, extent=1.0)
    assert "Theta" in state
    assert float(np.max(np.abs(np.asarray(state["Theta"])))) == 0.0


# --- the same physics as BSSN -------------------------------------------


@pytest.mark.slow
def test_the_two_forms_of_the_hamiltonian_constraint_agree(variables):
    """``H = R + 2 K^2 / 3 - Abar_ij Abar^ij``, the rewrite ``d_t Theta`` leans on.

    CCZ4 is usually published with that right-hand side spelled out in the
    conformal variables; this module writes ``d_t Theta`` with ``H`` in it
    instead, because that is the fact worth seeing. The two are the same
    quantity and this says so rather than asking a reader to trust it.
    """
    left, right = ccz4.constraint_identity(variables)
    assert sp.simplify((left - right).subs(POINT)) == 0


@pytest.mark.slow
@pytest.mark.benchmark
def test_ccz4_reduces_to_bssn_where_theta_and_z_vanish(variables):
    """Every right-hand side, with residual exactly zero.

    ``from_adm`` sets the connection to the one the metric defines, so
    ``Z_i`` vanishes identically; ``Theta`` is passed as zero. What is left
    has to be the system the suite already trusts.

    The ``d_t K`` comparison is the sharpest of these, because the ADM value
    it is checked against is computed by a different route entirely -- the
    product rule on ``gamma^ij K_ij`` using ``adm_rhs`` -- rather than from
    anything CCZ4 touches. It comes out equal because Z4 does *not*
    substitute the Hamiltonian constraint, where the textbook BSSN equation
    does; the two differ by exactly ``alpha H``.
    """
    slice_ = variables.physical
    inverse = inverse_metric(slice_.metric)
    zero = sp.Integer(0)
    system = ccz4.ccz4_rhs(variables, zero, [zero] * 3, damping=0.0, damping_mix=0.0)
    reference = bssn.bssn_rhs(variables, ricci_form="connection")

    # d_t K by the product rule on gamma^ij K_ij, from the ADM equations.
    dt_metric, dt_curvature = adm_rhs(slice_, inverse=inverse)
    dt_inverse = [
        [
            -sum(inverse[i][a] * inverse[j][b] * dt_metric[a][b] for a in INDICES for b in INDICES)
            for j in INDICES
        ]
        for i in INDICES
    ]
    adm_mean = trace(inverse, dt_curvature) + sum(
        dt_inverse[i][j] * slice_.curvature[i][j] for i in INDICES for j in INDICES
    )

    constraint = hamiltonian_constraint(slice_, inverse=inverse)
    assert _at(system["theta"] - variables.lapse * constraint / 2) == 0.0
    assert _at(system["mean_curvature"] - adm_mean) == 0.0
    assert _at(system["phi"] - reference["phi"]) == 0.0
    for i in INDICES:
        assert _at(system["connection"][i] - reference["connection"][i]) == 0.0
        for j in INDICES:
            mine = system["traceless_curvature"][i][j]
            theirs = reference["traceless_curvature"][i][j]
            assert _at(mine - theirs) == 0.0


def _at(expression) -> float:
    return float(sp.N(sp.sympify(expression).subs(POINT)))


# --- and different where they violate -----------------------------------


def _violating(shape=(32, 8, 8)):
    """Gauge-wave data nudged off the constraint surface.

    A smooth bump on ``K``. It leaves ``det gammabar = 1`` and
    ``gammabar^ij Abar_ij = 0`` alone -- those are statements about the
    conformal variables -- and breaks the Hamiltonian constraint, which is
    the one ``Theta`` is.
    """
    state, spacing = ccz4_solver.gauge_wave(shape=shape, amplitude=0.1, extent=1.0)
    axis = np.linspace(0.0, 1.0, shape[0], endpoint=False)
    bump = np.broadcast_to(0.01 * np.sin(2 * np.pi * axis)[:, None, None], shape)
    return {**state, "trK": state["trK"] + bump}, spacing


@pytest.mark.slow
@pytest.mark.benchmark
def test_z4_keeps_a_constraint_violation_bounded_where_bssn_does_not():
    """The measurement the whole formulation is for.

    One violating state, four crossing times, and the same integrator,
    gauge, dissipation, projection and diagnostics throughout -- only the
    right-hand sides differ, so a difference in the outcome is a difference
    the equations made.

    Sampled once per crossing time. The violation propagates at the
    coordinate light speed on a unit torus, so sampling on the period holds
    the phase fixed and the series is an envelope rather than a sampled
    wave; sampled off the period it oscillates and the ratios mean nothing.

    Measured: BSSN grows roughly linearly to 69 times its initial value,
    both CCZ4 runs stay near 3. The damping is the *smaller* effect here and
    shows up mostly in ``Theta`` -- see the test below.
    """
    state, spacing = _violating()
    final, periods = 4.0, 4
    ratios = {}
    for label, evolution in (
        (
            "bssn",
            bssn_solver.Evolution.build(spacing, slicing="harmonic", shift_condition="frozen"),
        ),
        (
            "ccz4",
            ccz4_solver.build(spacing, slicing="harmonic", shift_condition="frozen", damping=0.1),
        ),
    ):
        start = state if label == "ccz4" else {k: v for k, v in state.items() if k != "Theta"}
        steps = max(1, int(round(final / evolution.time_step)))
        history = evolution.run(
            start, steps, time_step=final / steps, sample_every=max(1, steps // periods)
        )
        series = [record.hamiltonian for record in history.constraints]
        assert all(np.isfinite(value) for value in series), (label, series)
        ratios[label] = series[-1] / series[0]

    assert ratios["bssn"] > 20.0, ratios
    assert ratios["ccz4"] < 10.0, ratios
    assert ratios["bssn"] / ratios["ccz4"] > 5.0, ratios


@pytest.mark.slow
@pytest.mark.benchmark
def test_the_damping_term_damps_theta():
    """``kappa_1`` is what makes ``Z_a = 0`` an attractor rather than merely preserved.

    Undamped Z4 already keeps the violation bounded -- that is the
    formulation, not the damping -- so the damping has to be looked for
    where it acts, which is on ``Theta`` itself. Over four crossing times at
    ``kappa_1 = 0.1`` the peak drops by about a third against the undamped
    run, measured 1.042e-02 to 7.219e-03.

    Two kernels, so this pays two derivations the first time it runs.
    """
    state, spacing = _violating()
    final = 4.0
    peaks = {}
    for damping in (0.0, 0.1):
        evolution = ccz4_solver.build(
            spacing, slicing="harmonic", shift_condition="frozen", damping=damping
        )
        steps = max(1, int(round(final / evolution.time_step)))
        history = evolution.run(state, steps, time_step=final / steps)
        peaks[damping] = float(np.max(np.abs(np.asarray(history.state["Theta"]))))

    assert peaks[0.0] > 0.0
    assert peaks[0.1] < 0.85 * peaks[0.0], peaks


# --- the gauge wave, which cannot tell them apart ------------------------


def test_ccz4_and_bssn_agree_on_exact_initial_data():
    """Same constraints at ``t = 0``: the state is the BSSN one plus a zero."""
    state, spacing = ccz4_solver.gauge_wave(shape=(32, 8, 8), amplitude=0.1, extent=1.0)
    reference, _ = bssn_solver.gauge_wave(shape=(32, 8, 8), amplitude=0.1, extent=1.0)
    violation = bssn_solver.constraints(state, spacing)
    assert violation.hamiltonian == pytest.approx(
        bssn_solver.constraints(reference, spacing).hamiltonian, rel=1e-12
    )
    assert violation.hamiltonian < 1e-10


@pytest.mark.slow
@pytest.mark.benchmark
def test_ccz4_converges_at_fourth_order_on_the_gauge_wave():
    """The BSSN acceptance test, run again on the other system.

    ``Theta`` does not stay at zero here and should not: the *discrete*
    solution violates the constraints at truncation level, ``d_t Theta`` is
    ``alpha H / 2``, so ``Theta`` picks up exactly that and is the size of
    the truncation error. A CCZ4 run that reproduced BSSN bit for bit on
    this data would mean the Z terms were dead code.
    """
    final = 0.25
    violations = []
    for n in (16, 32, 64):
        state, spacing = ccz4_solver.gauge_wave(shape=(n, 8, 8), amplitude=0.1, extent=1.0)
        evolution = ccz4_solver.build(
            spacing, slicing="harmonic", shift_condition="frozen", damping=0.0
        )
        steps = max(1, int(round(final / evolution.time_step)))
        history = evolution.run(state, steps, time_step=final / steps)
        violations.append(history.final.hamiltonian)
        assert history.final.determinant < 1e-12

    ratios = [coarse / fine for coarse, fine in zip(violations, violations[1:], strict=False)]
    assert all(ratio > 12.0 for ratio in ratios), ratios
    assert all(ratio < 20.0 for ratio in ratios), ratios
