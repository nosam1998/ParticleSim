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
    # The vacuum solvers still refuse it, and say where it runs.
    with pytest.raises(NotImplementedError, match="nr.modified"):
        bssn_solver.admit_theory(Modified(alpha=0.1))


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
    worst = 0.0
    for name in ccz4.STATE_NAMES:
        scale = float(np.max(np.abs(np.asarray(theirs[name])))) + 1e-300
        difference = float(np.max(np.abs(np.asarray(ours[name]) - np.asarray(theirs[name]))))
        worst = max(worst, difference / max(scale, 1.0))
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
    print(f"kappa1 = {damping}: largest difference from CCZ4, relative, {worst:.2e}")
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
    print(f"a={a} b={b}: Theta speed {speed:.6f}, expected {1 / np.sqrt(1 + b):.6f}")
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
    print(f"a={a} b={b}: lapse speed {speed:.6f}, expected {1 / np.sqrt(1 + a):.6f}")
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
    print("gauge wave, a = 0.2, b = 0.4: |H|", violations, "ratios", ratios)
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
    print("scalar through a gauge wave: relative error", errors, "ratios", ratios)
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
        row = {}
        for group, members in _RESIDUAL_GROUPS.items():
            values = np.stack([np.asarray(rates[m])[mask] for m in members])
            row[group] = float(np.sqrt(np.mean(values**2)))
        rows.append(row)
    return hole, rows


#: The groups whose rate is not zero by construction on time-symmetric data
#: with zero shift: ``phi``, ``gammabar_ij`` and ``Gammahat^i`` are, exactly.
_NONTRIVIAL = ("trK", "At", "Theta", "Ksf")


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.parametrize("which", ["scalarized", "dilatonic"])
def test_a_static_hairy_hole_is_static_to_fourth_order(which):
    """Stage 3: every time derivative of a static hole of the reduced action is truncation error.

    The hole comes from shooting the static equations
    :func:`~particlesim.theories.gauss_bonnet.static_equations` derives from
    the reduced action, in areal coordinates; it is mapped to isotropic ones
    and continued through the horizon by the static slice's inversion
    symmetry, with its own static lapse, which changes sign there. The
    Gauss-Bonnet term, the 7 x 7 solve, the scalar and the ``b`` terms all
    act, and the coupling is not small: the scalarized hole is ``M = lambda/2``.
    L2 norms over ``0.8 R_H < R < 1.6 R_H``, horizon included. Measured for
    the scalarized hole from ``R_H/12`` to ``R_H/16`` to ``R_H/24``: order
    3.98 to 4.00 for each group.
    """
    _, rows = _static_residuals(which, per=(8, 12, 16))
    print(which, rows)
    for group in ("phi", "gt", "Gt"):
        assert all(row[group] < 1e-14 for row in rows), (group, rows)
    pairs = zip(rows[:-1], rows[1:], strict=True)
    for (coarse, fine), ratio in zip(pairs, (12 / 8, 16 / 12), strict=True):
        for group in _NONTRIVIAL:
            order = np.log(coarse[group] / fine[group]) / np.log(ratio)
            assert order > 3.5, (group, order, rows)


@pytest.mark.slow
def test_without_the_coupling_the_hairy_hole_is_not_static():
    """The control: the same data with ``lambda = 0`` is not a solution, and its rates do not fall.

    Measured: ``|d_t K|`` is 7.9e-3 at ``R_H/8`` and at ``R_H/16`` alike,
    against 2.1e-4 and 1.4e-5 with the coupling on.
    """
    _, rows = _static_residuals("scalarized", coupling_squared=0.0, per=(8, 16))
    print("coupling off", rows)
    for group in _NONTRIVIAL:
        assert rows[0][group] / rows[1][group] < 1.2, (group, rows)
        assert rows[1][group] > 1e-3, (group, rows)


@pytest.mark.slow
def test_the_gauss_bonnet_terms_vanish_as_lambda_squared():
    """Stage 3's GR limit: the coupled rates go to the minimally coupled ones like ``lambda^2``.

    At ``lambda = 0`` the Gauss-Bonnet kernel is not called at all, so the
    rates are Stage 2's exactly. Near it, on the scalarized hole's data,
    halving ``lambda^2`` halves the difference, to first order: the 7 x 7
    solve makes it nonlinear, and the deviation from two is of order
    ``lambda^2`` itself.
    """
    from particlesim.theories import gauss_bonnet as gb

    name, horizon, scalar = STATIC_HOLES["scalarized"]
    hole = modified.IsotropicHole.solve(horizon, getattr(gb, name), horizon_scalar=scalar)
    h = hole.isotropic_horizon / 8
    n = 2 * int(np.ceil((1.6 * hole.isotropic_horizon + 5 * h) / h))
    state, spacing, radius = hole.state((n, n, n), n * h)
    mask = (radius > 0.8 * hole.isotropic_horizon) & (radius < 1.6 * hole.isotropic_horizon)

    def rates(coupling_squared):
        evolution = modified.build(
            spacing,
            slicing="frozen",
            shift_condition="frozen",
            constraint_function=0.4,
            coupling=hole.coupling,
            coupling_squared=coupling_squared,
        )
        assert (evolution.kernel.gauss_bonnet is None) == (coupling_squared == 0.0)
        out = evolution.kernel(state, spacing)
        return {name: np.asarray(out[name])[mask] for name in ("trK", "At00", "Ksf", "Theta")}

    base = rates(0.0)
    small, smaller = rates(2e-3), rates(1e-3)
    for name in base:
        first = np.max(np.abs(small[name] - base[name]))
        second = np.max(np.abs(smaller[name] - base[name]))
        print(f"{name}: {first:.3e} at lambda^2 = 2e-3, {second:.3e} at 1e-3")
        assert first > 1e-6, name
        assert first / second == pytest.approx(2.0, rel=1e-2), (name, first, second)


def _shell_value(field, radius, target, spacing):
    """``field`` at isotropic radius ``target``: a quadratic fit over a thin shell of points."""
    mask = np.abs(radius - target) < 1.5 * spacing
    coefficients = np.polyfit(radius[mask] - target, np.asarray(field)[mask], 2)
    return float(coefficients[-1])


@pytest.mark.slow
@pytest.mark.benchmark
def test_hair_grows_from_rest_as_the_decoupling_limit_says():
    """Stage 3: the dilatonic hair on a hole, 3-D and coupled, against ``decoupling.py`` in 1-D.

    Schwarzschild in isotropic coordinates with its static lapse frozen, so
    coordinate time is Killing time and the two evolutions can be compared
    event by event; ``phi = 0`` and ``K_phi = 0`` at ``t = 0``; the full
    coupled system with ``f = e^(-2 phi)`` at ``lambda^2 = 0.05 M^2``,
    where the metric's response is fourth order in ``lambda``. The periodic
    box's faces, where the static data meets its image, are what end the
    comparison: by ``t = 3.25 M`` their disturbance has not reached the
    probes. Measured at ``h = M/4``: within 1.6% at ``r = 2.5 M`` and
    0.8% at ``r = 3 M``, of hair that grows from nothing to ``-3.9e-3``.
    """
    from particlesim.theories.decoupling import DecouplingLimit, tortoise
    from particlesim.theories.gauss_bonnet import DILATONIC

    n, half, final, coupling_squared = 40, 5.0, 3.25, 0.05
    state, spacing, radius = modified.static_schwarzschild((n, n, n), 2 * half)
    evolution = modified.build(
        spacing,
        slicing="frozen",
        shift_condition="frozen",
        gauge_function=0.0,
        constraint_function=0.4,
        coupling=DILATONIC,
        coupling_squared=coupling_squared,
    )
    probes = np.array([2.5, 3.0])
    isotropic = (probes - 1 + np.sqrt(probes**2 - 2 * probes)) / 2
    steps = int(round(final / evolution.time_step))
    step = final / steps
    every = int(round(0.25 / step))
    times, three_d = [], []
    current = dict(state)
    for index in range(1, steps + 1):
        current = evolution.step(current, step)
        if index % every == 0 or index == steps:
            times.append(index * step)
            three_d.append([_shell_value(current["sf"], radius, R, spacing[0]) for R in isotropic])
    three_d = np.array(three_d)
    reference = DecouplingLimit(DILATONIC, coupling_squared, 1.0, points=8001)
    t1, samples, _ = reference.evolve(
        lambda rstar, r: 0.0 * r, final, probes=tortoise(probes), every=0.25
    )
    one_d = samples / probes
    for k, r in enumerate(probes):
        expected = np.interp(times, t1, one_d[:, k])
        difference = np.max(np.abs(three_d[:, k] - expected)) / np.max(np.abs(expected))
        print(f"r = {r}: largest difference {difference:.3e} of {np.max(np.abs(expected)):.3e}")
        assert np.all(np.isfinite(three_d[:, k]))
        assert difference < 0.03, (r, difference)
    # The hair is there to be compared: a coupling ten times smaller would be
    # ten times smaller, and the sign is the dilatonic one.
    assert three_d[-1, 0] < -3e-3


#: Areal radii of the hair probes on the initial slice, and their isotropic radii (M = 1).
HAIR_PROBES = np.array([2.5, 3.0])
HAIR_ISOTROPIC = (HAIR_PROBES - 1 + np.sqrt(HAIR_PROBES**2 - 2 * HAIR_PROBES)) / 2


def octant_hair(cells, coupling_squared, final=3.0, half=5.0, kernel=None):
    """Dilatonic hair from rest on frozen-gauge Schwarzschild, on an octant: ``phi`` at the probes.

    The box is ``2 half`` wide and ``cells`` across, of which the octant
    holds ``cells / 2`` and four mirror cells per axis. Its high faces carry
    the Sommerfeld condition on the departure from the initial
    Schwarzschild data. ``phi`` at a radius is a quadratic fit over a thin
    shell of the octant's own cells, which for a mirror-symmetric field is
    the fit over the whole shell. ``kernel``, if given, maps the evolution's
    kernel to the one run. Returns ``(times, phi)``, every ``M/4``.
    """
    import dataclasses

    from particlesim.solvers.nr import octant as octants
    from particlesim.theories.gauss_bonnet import DILATONIC

    state, spacing, radius, coords = modified.octant_schwarzschild(cells, half)
    evolution = modified.build(
        spacing,
        slicing="frozen",
        shift_condition="frozen",
        gauge_function=0.0,
        constraint_function=0.4,
        coupling=DILATONIC,
        coupling_squared=coupling_squared,
    )
    if kernel is not None:
        evolution = dataclasses.replace(evolution, kernel=kernel(evolution.kernel))
    bounded = octants.OctantBounded(
        modified.OctantModifiedEvolution.from_evolution(evolution),
        octants.OctantRadiative(
            coords=coords,
            spacing=spacing,
            axes=(0, 1, 2),
            width=4,
            background={name: np.asarray(value) for name, value in state.items()},
        ),
    )
    own = (coords[0] > 0) & (coords[1] > 0) & (coords[2] > 0)
    h = spacing[0]
    masks = [own & (np.abs(radius - target) < 1.5 * h) for target in HAIR_ISOTROPIC]

    def sample(field):
        values = np.asarray(field)
        return [
            float(np.polyfit(radius[mask] - target, values[mask], 2)[-1])
            for target, mask in zip(HAIR_ISOTROPIC, masks, strict=True)
        ]

    steps = int(round(final / bounded.time_step))
    step = final / steps
    every = int(round(0.25 / step))
    times, values = [0.0], [sample(state["sf"])]
    current = octants.reflect(state)
    for index in range(1, steps + 1):
        current = bounded.step(current, step)
        if index % every == 0:
            times.append(index * step)
            values.append(sample(current["sf"]))
    return np.array(times), np.array(values)


@pytest.mark.slow
@pytest.mark.benchmark
def test_strong_coupling_hair_carries_the_metric_answer_the_spherical_theory_predicts():
    """Stage 3 at strong coupling: the 3-D hair against the same problem in 1-D, from the action.

    The dilatonic coupling at ``lambda^2 = 0.3 M^2``, hair grown from rest
    on Schwarzschild with its static lapse frozen and zero shift, as in the
    weak-coupling test above, on an octant with a radiative edge. The
    prediction is :class:`~particlesim.theories.esgb_spherical.CoupledSpherical`:
    the same theory, gauge and data in spherical symmetry, with field
    equations varied from the reduced action rather than split 3+1, and
    nothing of this module in it. Held with the metric fixed it is the
    decoupling limit, and the difference is the metric's answer, which
    raises the hair by 9.1% at ``r = 2.5M`` and 3.5% at ``3M`` by
    ``t = 3M``.

    The assertion is that the 3-D hair departs from the 1-D coupled hair by
    less than 15% of that answer, at every sample and both probes. Measured
    at ``h = M/6``: 7.7% at worst, at ``r = 2.5M`` and ``t = 3M``, and
    under 1% at ``3M``; at ``h = M/4``, 19%, and at ``M/8``, 0.8%. The
    control is the same 3-D run with the Gauss-Bonnet stress-energy negated
    in the metric's equations and the scalar's left alone. Its hair is
    *smaller* than the decoupled hair, by 6.8% and 3.2% at ``t = 3M``, as the
    1-D theory flipped the same way says (7.2% and 3.1%). Through this
    assertion at ``h = M/6`` it departs from the true prediction by 175% to
    195% of the answer, and fails.
    """
    from particlesim.theories.esgb_spherical import CoupledSpherical
    from particlesim.theories.gauss_bonnet import DILATONIC

    coupling_squared, final = 0.3, 3.0
    times, three_d = octant_hair(60, coupling_squared, final=final)
    one_d = {}
    for backreaction in (True, False):
        model = CoupledSpherical(DILATONIC, coupling_squared, points=300, backreaction=backreaction)
        t1, samples, _ = model.evolve(model.schwarzschild(), final, radii=HAIR_ISOTROPIC)
        one_d[backreaction] = np.stack([np.interp(times, t1, samples[:, k]) for k in (0, 1)], -1)
    coupled, decoupled = one_d[True], one_d[False]
    answer = coupled - decoupled
    later = times >= 0.25
    departure = np.abs(three_d - coupled)[later] / np.abs(answer[later])
    print("t", times[later])
    print("3-D", three_d[later].tolist())
    print("departure / answer", departure.tolist())
    assert np.all(np.isfinite(three_d))
    assert np.all(departure < 0.15), departure
    # The answer is the size the 1-D theory says, and it raises the hair.
    excess = three_d[-1] / decoupled[-1] - 1
    print("3-D excess over the decoupled hair at t = 3M", excess)
    assert 0.07 < excess[0] < 0.11 and 0.025 < excess[1] < 0.045, excess
