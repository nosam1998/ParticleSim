"""Modified CCZ4 (``nr.modified``): the auxiliary-metric gauge, a scalar, and EsGB (issue #52).

Three stages, each checked against something computed another way.

**The modified gauge, in General Relativity.** At ``a = b = 0`` the rates
are CCZ4's, to round-off, on random data with ``Theta`` and ``Z_i`` far from
zero. With ``a`` and ``b`` on, the point of the formulation is a statement
about speeds, and speeds are measurable: on flat space a constraint
violation travels at ``1 / sqrt(1 + b)``, a gauge pulse at
``1 / sqrt(1 + a)``, each independent of the other function. The gauge wave
keeps fourth-order constraint convergence.

**A minimally coupled scalar.** A scalar wave moving through a gauge wave
the other way is an exact solution for a test field, because the gauge wave
is conformally flat in the plane it moves in.

**Einstein-scalar-Gauss-Bonnet.** A static hairy hole from the reduced
action, put on the grid in isotropic coordinates with its own static lapse,
has every time derivative zero; on the grid they are truncation error and
fall at fourth order. The Gauss-Bonnet algebra itself is checked against
four-dimensional geometry in ``test_esgb.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from particlesim.solvers.nr import bssn as bssn_solver
from particlesim.solvers.nr import modified
from particlesim.symbolic import bssn, ccz4
from particlesim.symbolic import modified_ccz4 as mccz4
from particlesim.symbolic.threeplusone import (
    INDICES,
    inverse_metric,
    momentum_constraint,
    symbolic_slice,
)

X, Y, Z = sp.symbols("x y z", real=True)
COORDS = (X, Y, Z)
POINT = {X: sp.Rational(1, 3), Y: sp.Rational(1, 4), Z: sp.Rational(1, 5)}


def _at(expression) -> float:
    return float(sp.N(sp.sympify(expression).subs(POINT)))


def _unimodular():
    """The slice ``test_ccz4.py`` checks the CCZ4 reduction on: unimodular, nothing zero."""
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


# --- structure ------------------------------------------------------------


def test_the_state_is_ccz4s_and_a_scalar():
    assert mccz4.STATE_NAMES[: len(ccz4.STATE_NAMES)] == ccz4.STATE_NAMES
    assert mccz4.STATE_NAMES[-2:] == ("sf", "Ksf")
    state, derivatives, registry = mccz4.abstract_state()
    assert set(state) == set(mccz4.STATE_NAMES)
    assert {f"dd_sf_{i}{j}" for i in INDICES for j in range(i, 3)} <= set(derivatives)
    assert not any(name.startswith("dd_Ksf") for name in derivatives)
    assert len(registry) == len(derivatives)


def test_the_gauge_is_bssns_at_a_equal_to_zero():
    """``modified_gauge`` against :func:`particlesim.symbolic.bssn.gauge_rhs`, term for term."""
    rng = np.random.default_rng(3)
    lapse, mean = 1 + 0.1 * rng.standard_normal(5), rng.standard_normal(5)
    driver = [rng.standard_normal(5) for _ in INDICES]
    rate = [rng.standard_normal(5) for _ in INDICES]
    gradient = [rng.standard_normal(5) for _ in INDICES]
    for slicing in ("harmonic", "one_plus_log"):
        dt_lapse, dt_shift, dt_driver = mccz4.modified_gauge(
            lapse, mean, driver, rate, 0.0, gradient, slicing=slicing, damping=1.5
        )
        assert np.array_equal(dt_lapse, bssn.SLICINGS[slicing](lapse, mean))
        for i in INDICES:
            assert np.allclose(dt_shift[i], 0.75 * driver[i], rtol=1e-15, atol=0)
            assert np.array_equal(dt_driver[i], rate[i] - 1.5 * driver[i])
    # With a on: the slicing slows by 1 + a, and the driver picks up D^i alpha.
    dt_lapse, dt_shift, _ = mccz4.modified_gauge(
        lapse, mean, driver, rate, 0.2, gradient, slicing="harmonic"
    )
    assert np.allclose(dt_lapse, -(lapse**2) * mean / 1.2)
    assert np.allclose(dt_shift[0], (0.75 * driver[0] - 0.2 * lapse * gradient[0]) / 1.2)


def test_the_constraint_terms_vanish_with_b():
    """Every ``b`` term carries ``b / (1 + b)``: none survives ``b = 0``."""
    zero = {"conformal_inverse": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    symbols = sp.symbols("H M0 M1 M2 z0 z1 z2 theta k1 k2 d0 d1 d2", real=True)
    hamiltonian, *rest = symbols
    momentum, upper = rest[0:3], rest[3:6]
    theta, k1, k2 = rest[6:9]
    d_theta = rest[9:12]
    parts = {**zero, "hamiltonian": hamiltonian, "momentum": momentum, "conformal_z": upper}

    class Flat:
        lapse = sp.Symbol("alpha")
        conformal_exponent = 1
        conformal_slice = type("S", (), {"metric": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})()
        physical = type(
            "P", (), {"curvature": [[sp.Symbol(f"K{i}{j}") for j in INDICES] for i in INDICES]}
        )()

    terms = mccz4.constraint_terms(Flat, parts, theta, d_theta, 0, k1, k2)
    assert terms["theta"] == 0 and terms["mean_curvature"] == 0
    assert all(value == 0 for value in terms["connection"])
    terms = mccz4.constraint_terms(Flat, parts, theta, d_theta, sp.Symbol("b"), k1, k2)
    assert sp.simplify(terms["mean_curvature"] - 3 * terms["theta"] / 2) == 0


def test_adr_008_admits_a_modified_ccz4_plugin_and_refuses_an_order_reduced_one():
    from particlesim.theories.base import Coupling, IllPosedRun, Theory
    from particlesim.theories.gauss_bonnet import DILATONIC

    class OrderReduced(Theory):
        id = "test.order_reduced"
        formulation = "order_reduced"
        coupling_function = DILATONIC
        couplings = [Coupling("alpha", 0.0)]

        def gr_limit(self):
            return {"alpha": 0.0}

    class Modified(OrderReduced):
        id = "test.modified"
        formulation = "modified_ccz4"

    with pytest.raises(IllPosedRun):
        modified.admit_theory(OrderReduced(alpha=0.1))
    modified.admit_theory(Modified(alpha=0.1))
    modified.admit_theory(OrderReduced(alpha=0.0))


# --- the symbolic pieces ---------------------------------------------------


@pytest.mark.slow
def test_the_momentum_constraint_is_the_adm_one(variables):
    """The conformal ``M_i`` against the ADM ``momentum_constraint``, lowered."""
    slice_ = variables.physical
    reference = momentum_constraint(slice_)  # M^i, index up
    mine = mccz4.momentum_constraint(variables)
    for i in INDICES:
        lowered = sum(slice_.metric[i][j] * reference[j] for j in INDICES)
        assert _at(mine[i] - lowered) == pytest.approx(0.0, abs=1e-12)
    assert max(abs(_at(value)) for value in mine) > 0.05


@pytest.mark.slow
def test_the_gammahat_ricci_already_holds_the_conformal_z_terms(variables):
    """Building the conformal Ricci with ``Gammahat^i`` adds ``2 Dbar_(i Z_j)``, exactly.

    This is why CCZ4 adds only :func:`~particlesim.symbolic.ccz4.connection_z_terms`
    by hand. Here ``Z_i`` is put in by hand -- a polynomial covector -- and
    the difference of the two tensors is compared with ``2 Dbar_(i Z_j)``.
    """
    conformal = variables.conformal_slice
    inverse = inverse_metric(conformal.metric)
    lower_z = [X * Y / 3 + Z / 5, Y**2 / 7 - X / 4, X * Z / 6 + Y / 9]
    upper_z = [sum(inverse[i][j] * lower_z[j] for j in INDICES) for i in INDICES]
    hatted = [variables.connection[i] + 2 * upper_z[i] for i in INDICES]
    plain = bssn.conformal_connection_ricci(variables)
    moved = bssn.BSSNVariables(**{**variables.__dict__, "connection": hatted})
    d_hatted = [[sp.diff(hatted[i], COORDS[k]) for i in INDICES] for k in INDICES]
    with_z = bssn.conformal_connection_ricci(moved, d_hatted)
    from particlesim.symbolic.threeplusone import christoffel

    gammas = christoffel(conformal, inverse)
    for i in INDICES:
        for j in INDICES:
            gradient = (
                sp.diff(lower_z[j], COORDS[i])
                + sp.diff(lower_z[i], COORDS[j])
                - 2 * sum(gammas[k][i][j] * lower_z[k] for k in INDICES)
            )
            assert _at(with_z[i][j] - plain[i][j] - gradient) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.slow
def test_matter_enters_as_the_map_says(variables):
    """The matter in the symbolic equations, against :func:`matter_terms` outside them.

    The evolution adds matter outside the kernel. That is only right if the
    map is the one the equations apply, ``b`` terms included, which see
    ``rho`` through ``H`` and ``S_i`` through ``M_i``.
    """
    rng = np.random.default_rng(11)
    density = sp.Rational(3, 17)
    momentum = [sp.Rational(int(v), 37) for v in rng.integers(-9, 9, 3)]
    upper = [[sp.Rational(int(v), 41) for v in rng.integers(-9, 9, 3)] for _ in INDICES]
    stress = [[(upper[i][j] + upper[j][i]) / 2 for j in INDICES] for i in INDICES]
    theta = X * Y / 5
    d_theta = [sp.diff(theta, c) for c in COORDS]
    b, k1, k2 = sp.Rational(2, 5), sp.Rational(1, 10), sp.Rational(1, 3)
    vacuum, _ = mccz4.modified_parts(variables, theta, d_theta, b, k1, k2)
    full, _ = mccz4.modified_parts(
        variables, theta, d_theta, b, k1, k2, density=density, stress=stress, momentum=momentum
    )
    terms = mccz4.matter_terms(
        variables.lapse,
        variables.conformal_exponent,
        variables.conformal_metric,
        density,
        momentum,
        stress,
        b,
    )
    checks = [(full["mean_curvature"] - vacuum["mean_curvature"], terms["mean_curvature"])]
    checks.append((full["theta"] - vacuum["theta"], terms["theta"]))
    for i in INDICES:
        checks.append((full["connection"][i] - vacuum["connection"][i], terms["connection"][i]))
        for j in INDICES:
            checks.append(
                (
                    full["traceless_curvature"][i][j] - vacuum["traceless_curvature"][i][j],
                    terms["traceless_curvature"][i][j],
                )
            )
    for got, expected in checks:
        assert _at(got) == pytest.approx(_at(expected), rel=1e-10, abs=1e-12)
    assert max(abs(_at(e)) for _, e in checks) > 0.01


# --- the kernel: a = b = 0 is CCZ4 ---------------------------------------


def _random_state(shape=(16, 16, 16), seed=0, backend="jax"):
    """Smooth periodic data far from any solution: ``Theta``, ``Z_i``, all of it non-zero."""
    rng = np.random.default_rng(seed)
    axes = [np.linspace(0, 1, n, endpoint=False) for n in shape]
    mesh = np.meshgrid(*axes, indexing="ij")

    def smooth(scale):
        total = np.zeros(shape)
        for _ in range(3):
            k = rng.integers(-2, 3, 3)
            phase = rng.uniform(0, 2 * np.pi)
            total += np.sin(2 * np.pi * sum(k[i] * mesh[i] for i in INDICES) + phase)
        return scale * total / 3

    state = {
        "phi": smooth(0.1),
        "trK": smooth(0.1),
        "Theta": smooth(0.05),
        "alpha": 1 + smooth(0.1),
    }
    for i in INDICES:
        state[f"beta{i}"] = smooth(0.1)
        state[f"B{i}"] = smooth(0.1)
        state[f"Gt{i}"] = smooth(0.1)
        for j in range(i, 3):
            state[f"gt{i}{j}"] = (1.0 if i == j else 0.0) + smooth(0.05)
            state[f"At{i}{j}"] = smooth(0.1)
    state["sf"] = np.zeros(shape)
    state["Ksf"] = np.zeros(shape)
    module = bssn_solver._module(backend)
    return {k: module.asarray(v) for k, v in state.items()}, tuple(1 / n for n in shape)


@pytest.mark.slow
@pytest.mark.parametrize("damping", [0.0, 0.1])
def test_at_a_and_b_zero_it_is_ccz4_on_random_data(damping):
    """Every rate, at every point, on data with ``Theta`` and ``Z_i`` of order ten per cent.

    The two kernels are separate derivations with different expression
    trees -- one carries the scalar, the ``b`` terms with ``b`` read as a
    field, the gauge assembled outside -- so agreement is to round-off, not
    to the bit.
    """
    from particlesim.solvers.nr import ccz4 as ccz4_solver

    state, spacing = _random_state()
    reference = ccz4_solver.build(
        spacing, slicing="harmonic", shift_condition="frozen", damping=damping
    ).kernel
    mine = modified.build(
        spacing,
        slicing="harmonic",
        shift_condition="frozen",
        gauge_function=0.0,
        constraint_function=0.0,
        damping=damping,
    ).kernel
    theirs = reference({k: v for k, v in state.items() if k not in ("sf", "Ksf")}, spacing)
    ours = mine(state, spacing)
    for name in ccz4.STATE_NAMES:
        scale = float(np.max(np.abs(np.asarray(theirs[name])))) + 1e-300
        difference = float(np.max(np.abs(np.asarray(ours[name]) - np.asarray(theirs[name]))))
        assert difference <= 1e-12 * max(scale, 1.0), (name, difference, scale)
    # And b does change them, in the three places it should and nowhere else.
    shifted = modified.build(
        spacing,
        slicing="harmonic",
        shift_condition="frozen",
        gauge_function=0.0,
        constraint_function=0.4,
        damping=damping,
    ).kernel(state, spacing)
    changed = {
        name
        for name in ccz4.STATE_NAMES
        if float(np.max(np.abs(np.asarray(shifted[name]) - np.asarray(ours[name])))) > 1e-10
    }
    assert changed == {"trK", "Theta", "Gt0", "Gt1", "Gt2"}


# --- speeds ----------------------------------------------------------------


def _flat(shape=(32, 8, 8)):
    state, spacing = bssn_solver.gauge_wave(shape=shape, amplitude=0.0, extent=1.0)
    return modified.with_scalar(state), spacing


def _frequency(evolution, state, name, final=1.5, every=4):
    """Fit ``cos(omega t)`` to the ``sin(2 pi x)`` content of ``state[name]`` over time."""
    from scipy.optimize import curve_fit

    shape = np.asarray(state[name]).shape
    x = np.linspace(0, 1, shape[0], endpoint=False)[:, None, None] * np.ones(shape)
    mode = np.sin(2 * np.pi * x)
    background = np.asarray(state[name]) - np.mean(np.asarray(state[name]))
    start = 2 * float(np.mean(background * mode))
    step = evolution.time_step
    count = int(round(final / step))
    times, amplitudes = [], []
    current = dict(state)
    for index in range(count + 1):
        if index % every == 0:
            values = np.asarray(current[name])
            times.append(index * step)
            amplitudes.append(2 * float(np.mean((values - np.mean(values)) * mode)) / start)
        current = evolution.step(current, step)
    (omega,), _ = curve_fit(lambda t, w: np.cos(w * t), times, amplitudes, p0=[2 * np.pi * 0.9])
    return omega / (2 * np.pi)


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.parametrize(("a", "b"), [(0.0, 0.0), (0.0, 0.4), (0.2, 0.4)])
def test_a_constraint_violation_travels_at_one_over_root_one_plus_b(a, b):
    """``ghat^cd nabla_c nabla_d Z^a = 0``: ``Theta`` is a wave at ``1/sqrt(1 + b)``.

    A ``1e-6`` bump on ``Theta`` on flat space, nothing else, frozen gauge,
    no damping. The fitted frequency of its fundamental is the speed, and
    it does not depend on ``a``. Before the CCZ4 fix this was ``sqrt(2)``
    at ``b = 0``: the Z terms were counted twice.
    """
    state, spacing = _flat()
    x = np.linspace(0, 1, 32, endpoint=False)[:, None, None] * np.ones((32, 8, 8))
    state["Theta"] = state["Theta"] + 1e-6 * np.sin(2 * np.pi * x)
    evolution = modified.build(
        spacing,
        slicing="frozen",
        shift_condition="frozen",
        gauge_function=a,
        constraint_function=b,
        damping=0.0,
        dissipation=0.0,
    )
    speed = _frequency(evolution, state, "Theta")
    assert speed == pytest.approx(1 / np.sqrt(1 + b), rel=2e-3)


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.parametrize(("a", "b"), [(0.0, 0.0), (0.2, 0.0), (0.2, 0.4)])
def test_the_lapse_travels_at_one_over_root_one_plus_a(a, b):
    """``gtilde``'s cone: a harmonic-slicing gauge pulse at ``1/sqrt(1 + a)``, whatever ``b`` is.

    Flat data with ``alpha = 1 + 1e-6 sin(2 pi x)`` is flat spacetime in a
    different slicing: the constraints stay zero and only the gauge moves.
    """
    state, spacing = _flat()
    x = np.linspace(0, 1, 32, endpoint=False)[:, None, None] * np.ones((32, 8, 8))
    state["alpha"] = state["alpha"] + 1e-6 * np.sin(2 * np.pi * x)
    evolution = modified.build(
        spacing,
        slicing="harmonic",
        shift_condition="frozen",
        gauge_function=a,
        constraint_function=b,
        damping=0.0,
        dissipation=0.0,
    )
    speed = _frequency(evolution, state, "alpha")
    assert speed == pytest.approx(1 / np.sqrt(1 + a), rel=2e-3)


# --- convergence --------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.benchmark
def test_the_gauge_wave_keeps_fourth_order_constraint_convergence_with_a_and_b():
    """The BSSN and CCZ4 acceptance, rerun with ``a = 0.2`` and ``b = 0.4``.

    With ``a`` on, the slicing is no longer harmonic and the gauge wave is
    no longer a solution of the gauge condition, so the solution moves off
    it -- it is still flat spacetime, and the constraints still have to
    converge at the scheme's order.
    """
    final = 0.5
    violations = []
    for n in (16, 32, 64):
        state, spacing = bssn_solver.gauge_wave(shape=(n, 8, 8), amplitude=0.1, extent=1.0)
        state = modified.with_scalar(state)
        evolution = modified.build(
            spacing,
            slicing="harmonic",
            shift_condition="frozen",
            gauge_function=0.2,
            constraint_function=0.4,
            damping=0.0,
        )
        steps = max(1, int(round(final / evolution.time_step)))
        history = evolution.run(state, steps, time_step=final / steps)
        violations.append(history.final.hamiltonian)
        assert history.final.determinant < 1e-12
    ratios = [coarse / fine for coarse, fine in zip(violations, violations[1:], strict=False)]
    assert all(12.0 < ratio < 20.0 for ratio in ratios), (violations, ratios)


@pytest.mark.slow
@pytest.mark.benchmark
def test_a_scalar_through_a_gauge_wave_converges_to_the_closed_form():
    """Stage 2: ``phi = A sin(2 pi (x + t))`` through a gauge wave, fourth order."""
    final = 0.5
    errors = []
    for n in (16, 32, 64):
        state, spacing = modified.scalar_gauge_wave(shape=(n, 8, 8), amplitude=0.1)
        exact, _ = modified.scalar_gauge_wave(shape=(n, 8, 8), amplitude=0.1, time=final)
        evolution = modified.build(
            spacing,
            slicing="harmonic",
            shift_condition="frozen",
            gauge_function=0.0,
            constraint_function=0.4,
            damping=0.0,
        )
        steps = max(1, int(round(final / evolution.time_step)))
        history = evolution.run(state, steps, time_step=final / steps)
        error = bssn_solver.solution_error(history.state, {k: exact[k] for k in ("sf", "Ksf")})
        errors.append(error / 1e-6)
    ratios = [coarse / fine for coarse, fine in zip(errors, errors[1:], strict=False)]
    assert errors[-1] < 1e-4, errors
    assert all(12.0 < ratio < 20.0 for ratio in ratios), (errors, ratios)


# --- Einstein-scalar-Gauss-Bonnet --------------------------------------------

#: Two static holes of the reduced action, with lambda = 1: the scalarized one
#: at M = lambda / 2 that the decoupling limit is compared with, where the
#: coupling is strong, and a dilatonic one of the string's own coupling.
STATIC_HOLES = {
    "scalarized": ("SCALARIZATION", 0.8993, 0.40612252832039697),
    "dilatonic": ("DILATONIC", 6.0, -0.12531528095715458),
}

_RESIDUAL_GROUPS = {
    "phi": ["phi"],
    "gt": [f"gt{i}{j}" for i in INDICES for j in range(i, 3)],
    "trK": ["trK"],
    "At": [f"At{i}{j}" for i in INDICES for j in range(i, 3)],
    "Gt": [f"Gt{i}" for i in INDICES],
    "Theta": ["Theta"],
    "sf": ["sf"],
    "Ksf": ["Ksf"],
}


def _static_residuals(which, coupling_squared=1.0, per=(4, 8, 16), inner=0.8, outer=1.6):
    """Largest ``|d_t|`` of each group over ``inner R_H < R < outer R_H``, at each resolution.

    Each grid is a box around the shell with five cells to spare, so the
    periodic wrap at its faces reaches nothing measured; the hole is half a
    cell off every grid point. The horizon is inside the shell, and so is
    part of the other exterior.
    """
    from particlesim.theories import gauss_bonnet as gb

    name, horizon, scalar = STATIC_HOLES[which]
    hole = modified.IsotropicHole.solve(horizon, getattr(gb, name), horizon_scalar=scalar)
    radius_h = hole.isotropic_horizon
    rows = []
    for count in per:
        h = radius_h / count
        n = 2 * int(np.ceil((outer * radius_h + 5 * h) / h))
        state, spacing, radius = hole.state((n, n, n), n * h)
        evolution = modified.build(
            spacing,
            slicing="frozen",
            shift_condition="frozen",
            gauge_function=0.0,
            constraint_function=0.4,
            coupling=hole.coupling,
            coupling_squared=coupling_squared,
        )
        rates = evolution.kernel(state, spacing)
        mask = (radius >= inner * radius_h) & (radius <= outer * radius_h)
        rows.append(
            {
                group: max(float(np.max(np.abs(np.asarray(rates[m])[mask]))) for m in members)
                for group, members in _RESIDUAL_GROUPS.items()
            }
        )
    return hole, rows
