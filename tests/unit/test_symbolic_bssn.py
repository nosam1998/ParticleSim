"""The 3+1 pipeline: ADM algebra, the BSSN change of variables, codegen.

The chain this file checks, link by link:

1. the GR plugin's field equations, projected on the normal, are the ADM
   constraints -- verified on data that satisfies neither, so both sides
   are non-zero and have to agree;
2. the ADM evolution equations reproduce an exact solution's time
   derivatives, symbolically and exactly;
3. the BSSN equations are the same system in different variables --
   verified two independent ways, and the one place they differ is
   identified as exactly ``alpha`` times the Hamiltonian constraint;
4. the emitted finite-difference kernel converges at its stencil's order
   against that same exact solution.

Two conventions hold throughout: Baumgarte and Shapiro's signs, and
``simplify`` avoided on large expressions in favour of evaluating at
several points, because a test that takes a minute to prove an identity
gets deleted.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from particlesim.core.grid import second_derivative
from particlesim.symbolic import bssn, codegen
from particlesim.symbolic.curvature import Eulerian, MetricGeometry
from particlesim.symbolic.threeplusone import (
    DIMENSION,
    INDICES,
    abstract_slice,
    adm_rhs,
    christoffel,
    grid_slice,
    hamiltonian_constraint,
    inverse_metric,
    inverse_metric_derivative,
    momentum_constraint,
    ricci,
    symbolic_slice,
)
from particlesim.theories.gr import GeneralRelativity

X, Y, Z = sp.symbols("x y z", real=True)
COORDS = (X, Y, Z)
TIME = sp.Symbol("t", real=True)
POINTS = (
    {X: 0.4137, Y: -0.9123, Z: 1.2718},
    {X: -1.1031, Y: 0.3319, Z: 0.7712},
)


def _at(expression, point) -> float:
    return float(sp.N(sp.sympify(expression).subs(point)))


def _worst(table, point) -> float:
    return max(abs(_at(table[i][j], point)) for i in INDICES for j in INDICES)


# --- exact solutions ----------------------------------------------------


def _gauge_wave(amplitude=None):
    """The Apples-with-Apples gauge wave: a flat spacetime in a wavy gauge.

    ``ds^2 = H(-dt^2 + dx^2) + dy^2 + dz^2`` with
    ``H = 1 - A sin(2 pi (x - t))``. Both constraints vanish identically
    and every time derivative is known in closed form, which is what makes
    it the standard first test of an evolution scheme.
    """
    amplitude = sp.Rational(1, 10) if amplitude is None else amplitude
    profile = 1 - amplitude * sp.sin(2 * sp.pi * (X - TIME))
    lapse = sp.sqrt(profile)
    metric = [[profile, 0, 0], [0, 1, 0], [0, 0, 1]]
    curvature = [[-sp.diff(profile, TIME) / (2 * lapse), 0, 0], [0, 0, 0], [0, 0, 0]]
    return lapse, [0, 0, 0], metric, curvature


def _schwarzschild():
    """Isotropic Schwarzschild on a static slice: ``K_ij = 0`` and nothing moves."""
    mass = sp.Rational(1, 1)
    radius = sp.sqrt(X**2 + Y**2 + Z**2)
    conformal = 1 + mass / (2 * radius)
    lapse = (1 - mass / (2 * radius)) / (1 + mass / (2 * radius))
    metric = [[conformal**4 if i == j else 0 for j in INDICES] for i in INDICES]
    return lapse, [0, 0, 0], metric, [[0] * DIMENSION for _ in INDICES]


def _generic():
    """Smooth data satisfying nothing, with every component non-zero.

    A metric that is diagonal, or a vanishing shift, or time-symmetric
    data, each switches off whole terms of these equations. This switches
    none of them off, which is why the identities checked against it mean
    something.
    """
    small, smaller = sp.Rational(1, 7), sp.Rational(1, 9)
    metric = [
        [
            1 + small * sp.sin(X) * sp.cos(2 * Y),
            small * smaller * sp.sin(X) * sp.sin(Y),
            -smaller * sp.sin(Z) / 3,
        ],
        [
            small * smaller * sp.sin(X) * sp.sin(Y),
            1 + smaller * sp.cos(Y) * Z / 4,
            small * sp.sin(Z) / 5,
        ],
        [-smaller * sp.sin(Z) / 3, small * sp.sin(Z) / 5, 1 + small * sp.sin(Z) * X / 6],
    ]
    curvature = [
        [sp.cos(X) / 3, small * sp.sin(Y) / 2, sp.sin(Y) / 8],
        [small * sp.sin(Y) / 2, sp.sin(X + Z) / 5, smaller * sp.cos(Z) / 2],
        [sp.sin(Y) / 8, smaller * sp.cos(Z) / 2, sp.cos(Y + Z) / 7],
    ]
    lapse = 1 + sp.sin(X) * sp.cos(Z) / 4
    shift = [sp.sin(Y) / 6, sp.cos(X) * Z / 8, sp.sin(X + Y) / 7]
    return lapse, shift, metric, curvature


@pytest.fixture(scope="module")
def generic():
    lapse, shift, metric, curvature = _generic()
    return symbolic_slice(lapse, shift, metric, curvature, COORDS)


@pytest.fixture(scope="module")
def generic_bssn(generic):
    return bssn.from_adm(generic)


@pytest.fixture(scope="module")
def generic_rhs(generic_bssn):
    return bssn.bssn_rhs(generic_bssn), bssn.bssn_rhs_from_adm(generic_bssn)


# --- the spatial geometry ----------------------------------------------


def test_the_inverse_metric_inverts_the_metric(generic):
    inverse = inverse_metric(generic.metric)
    for point in POINTS:
        for i in INDICES:
            for j in INDICES:
                value = sum(_at(inverse[i][k] * generic.metric[k][j], point) for k in INDICES)
                assert value == pytest.approx(1.0 if i == j else 0.0, abs=1e-13)


def test_the_inverse_metric_derivative_matches_differentiation(generic):
    """``d_k gamma^ij = -gamma^ia gamma^jb d_k gamma_ab``, against ``sp.diff``.

    The identity is used so that the grid path never has to difference the
    inverse metric, which carries a determinant in its denominator. It has
    to be right, so it is checked against actually differentiating.
    """
    inverse = inverse_metric(generic.metric)
    identity = inverse_metric_derivative(inverse, generic.d_metric)
    for point in POINTS:
        for k in INDICES:
            for i in INDICES:
                direct = sp.diff(inverse[i][i], COORDS[k])
                assert _at(identity[k][i][i], point) == pytest.approx(
                    _at(direct, point), rel=1e-10, abs=1e-14
                )


def test_the_connection_is_metric_compatible(generic):
    """``d_k gamma_ij = Gamma^m_ki gamma_mj + Gamma^m_kj gamma_im``.

    The defining property of the Levi-Civita connection, and the one an
    error in the Christoffel formula breaks.
    """
    connection = christoffel(generic)
    for point in POINTS:
        for k in INDICES:
            for i in INDICES:
                for j in INDICES:
                    rebuilt = sum(
                        _at(
                            connection[m][k][i] * generic.metric[m][j]
                            + connection[m][k][j] * generic.metric[i][m],
                            point,
                        )
                        for m in INDICES
                    )
                    assert rebuilt == pytest.approx(
                        _at(generic.d_metric[k][i][j], point), abs=1e-12
                    )


@pytest.mark.benchmark
def test_flat_space_in_spherical_coordinates_has_no_curvature():
    """``diag(1, r^2, r^2 sin^2 theta)`` is flat, and the Ricci tensor says so.

    A Cartesian test cannot catch a Christoffel error because every symbol
    vanishes. This one has non-zero connection coefficients everywhere and
    a Ricci tensor that has to cancel to zero exactly.
    """
    radius, polar, azimuth = sp.symbols("r theta phi", positive=True)
    metric = [[1, 0, 0], [0, radius**2, 0], [0, 0, radius**2 * sp.sin(polar) ** 2]]
    slice_ = symbolic_slice(
        1, [0, 0, 0], metric, [[0] * 3 for _ in INDICES], (radius, polar, azimuth)
    )
    tensor = ricci(slice_, inverse_metric(metric))
    assert sp.simplify(sp.Matrix(tensor)) == sp.zeros(3, 3)


@pytest.mark.benchmark
def test_a_two_sphere_has_the_curvature_of_its_radius():
    """``S^2 x R`` of radius ``a``: ``R_ij = gamma_ij/a^2`` on the sphere.

    The closed-form answer for a genuinely curved space, so the Ricci
    machinery is checked against a number and not only against zero.
    """
    radius, polar, azimuth = sp.symbols("r theta phi", positive=True)
    scale = sp.Symbol("a", positive=True)
    metric = [[1, 0, 0], [0, scale**2, 0], [0, 0, scale**2 * sp.sin(polar) ** 2]]
    slice_ = symbolic_slice(
        1, [0, 0, 0], metric, [[0] * 3 for _ in INDICES], (radius, polar, azimuth)
    )
    tensor = sp.simplify(sp.Matrix(ricci(slice_, inverse_metric(metric))))
    expected = sp.Matrix([[0, 0, 0], [0, 1, 0], [0, 0, sp.sin(polar) ** 2]])
    assert sp.simplify(tensor - expected) == sp.zeros(3, 3)
    from particlesim.symbolic.threeplusone import trace

    scalar = sp.simplify(trace(inverse_metric(metric), ricci(slice_, inverse_metric(metric))))
    assert sp.simplify(scalar - 2 / scale**2) == 0


# --- the plugin round-trip ---------------------------------------------


@pytest.mark.slow
@pytest.mark.benchmark
def test_the_plugin_field_equations_project_onto_the_adm_constraints():
    """The first link: ``G_ab`` from the plugin, projected, *is* the constraint.

        n^a n^b G_ab = (R + K^2 - K_ij K^ij)/2
        -n^a gamma^b_i G_ab = D_j (K^j_i - delta^j_i K)

    Checked on a four-metric that solves nothing, so both sides are
    non-zero: on a solution they would both vanish and the test would pass
    without testing anything. The extrinsic curvature is taken from the
    four-metric's own time dependence rather than chosen, which is what
    makes this a statement about one spacetime rather than about two sets
    of numbers.
    """
    small, smaller = sp.Rational(1, 7), sp.Rational(1, 9)
    first = 1 + small * sp.sin(X + 2 * TIME) * sp.cos(Y)
    second = 1 + smaller * sp.cos(Y - TIME) * sp.sin(Z)
    off = small * smaller * sp.sin(X) * sp.sin(Y) * sp.cos(TIME)
    lapse = 1 + small * sp.cos(X) * sp.sin(TIME) / 2
    metric = sp.Matrix([[first, off, 0], [off, second, 0], [0, 0, 1]])
    four = sp.zeros(4, 4)
    four[0, 0] = -(lapse**2)
    four[1:, 1:] = metric
    curvature = sp.Matrix(3, 3, lambda i, j: -sp.diff(metric[i, j], TIME) / (2 * lapse))

    slice_ = symbolic_slice(lapse, [0, 0, 0], metric.tolist(), curvature.tolist(), COORDS)
    inverse = inverse_metric(metric.tolist())
    hamiltonian = hamiltonian_constraint(slice_, inverse=inverse)
    momentum = momentum_constraint(slice_, inverse=inverse)

    coords = (TIME, *COORDS)
    einstein = MetricGeometry(four, coords).einstein
    stress = GeneralRelativity().effective_stress_energy(einstein, four)
    observer = Eulerian(four, coords)
    density = observer.energy_density(stress)
    flux = observer.momentum_density(stress)
    raised = [sum(inverse[i][j] * flux[j] for j in INDICES) for i in INDICES]

    for point in POINTS:
        full = dict(point)
        full[TIME] = 0.0
        size = abs(_at(hamiltonian, full))
        assert size > 1e-3, "the test data must violate the constraint"
        assert _at(hamiltonian, full) == pytest.approx(_at(16 * sp.pi * density, full), rel=1e-12)
        for i in INDICES:
            assert _at(momentum[i], full) == pytest.approx(
                _at(8 * sp.pi * raised[i], full), rel=1e-10, abs=1e-15
            )


# --- the ADM evolution equations ---------------------------------------


def test_minkowski_does_nothing():
    slice_ = symbolic_slice(1, [0, 0, 0], sp.eye(3).tolist(), sp.zeros(3, 3).tolist(), COORDS)
    dt_metric, dt_curvature = adm_rhs(slice_)
    assert sp.Matrix(dt_metric) == sp.zeros(3, 3)
    assert sp.Matrix(dt_curvature) == sp.zeros(3, 3)
    assert hamiltonian_constraint(slice_) == 0
    assert list(momentum_constraint(slice_)) == [0, 0, 0]


@pytest.mark.benchmark
def test_static_schwarzschild_stays_static():
    """``alpha R_ij = D_i D_j alpha`` on a static slice, to machine precision.

    Nothing moves, so the whole content of the test is that the spatial
    Ricci tensor and the lapse's second covariant derivative cancel -- two
    long expressions computed by different routes, agreeing to 5e-17 on a
    metric with a puncture in it.
    """
    slice_ = symbolic_slice(*_schwarzschild(), COORDS)
    dt_metric, dt_curvature = adm_rhs(slice_)
    for point in POINTS:
        assert _worst(dt_metric, point) < 1e-15
        assert _worst(dt_curvature, point) < 1e-14
        assert abs(_at(hamiltonian_constraint(slice_), point)) < 1e-14
        assert max(abs(_at(v, point)) for v in momentum_constraint(slice_)) < 1e-14


@pytest.mark.benchmark
def test_the_gauge_wave_evolves_the_way_it_is_supposed_to():
    """The ADM right-hand sides equal the exact solution's time derivatives.

    Symbolically zero, not small: the residual simplifies to exactly zero,
    which is the strongest form this check takes.
    """
    lapse, shift, metric, curvature = _gauge_wave()
    slice_ = symbolic_slice(lapse, shift, metric, curvature, COORDS)
    dt_metric, dt_curvature = adm_rhs(slice_)
    expected_metric = sp.Matrix(3, 3, lambda i, j: sp.diff(metric[i][j], TIME))
    expected_curvature = sp.Matrix(3, 3, lambda i, j: sp.diff(curvature[i][j], TIME))
    assert sp.simplify(sp.Matrix(dt_metric) - expected_metric) == sp.zeros(3, 3)
    assert sp.simplify(sp.Matrix(dt_curvature) - expected_curvature) == sp.zeros(3, 3)
    assert sp.simplify(hamiltonian_constraint(slice_)) == 0
    assert [sp.simplify(v) for v in momentum_constraint(slice_)] == [0, 0, 0]


def test_the_matter_terms_enter_where_they_should(generic):
    """A stress tensor proportional to the metric shifts only the trace part."""
    density = 0.05
    stress = [[generic.metric[i][j] * density / 3 for j in INDICES] for i in INDICES]
    _, vacuum = adm_rhs(generic)
    _, sourced = adm_rhs(generic, density=density, stress=stress)
    for point in POINTS:
        for i in INDICES:
            for j in INDICES:
                # S_ij - gamma_ij (S - rho)/2 with S = rho: gamma_ij rho/3 - gamma_ij(rho - rho)/2
                expected = (
                    -8 * np.pi * _at(generic.lapse * generic.metric[i][j] * density / 3, point)
                )
                assert _at(sourced[i][j] - vacuum[i][j], point) == pytest.approx(
                    expected, rel=1e-10
                )


# --- the BSSN change of variables --------------------------------------


def test_the_bssn_variables_satisfy_their_algebraic_constraints(generic_bssn):
    unit, traceless = bssn.algebraic_constraints(generic_bssn)
    for point in POINTS:
        assert abs(_at(unit, point)) < 1e-14
        assert abs(_at(traceless, point)) < 1e-14


def test_the_change_of_variables_is_invertible(generic, generic_bssn):
    metric, curvature = bssn.to_adm(generic_bssn)
    for point in POINTS:
        for i in INDICES:
            for j in INDICES:
                assert _at(metric[i][j], point) == pytest.approx(
                    _at(generic.metric[i][j], point), rel=1e-12
                )
                assert _at(curvature[i][j], point) == pytest.approx(
                    _at(generic.curvature[i][j], point), rel=1e-12, abs=1e-15
                )


@pytest.mark.benchmark
def test_the_conformal_decomposition_reproduces_the_physical_ricci(generic, generic_bssn):
    """``R_ij = Rbar_ij + R^phi_ij``, the identity BSSN rests on."""
    physical = ricci(generic, inverse_metric(generic.metric))
    decomposed = bssn.physical_ricci(generic_bssn)
    for point in POINTS:
        scale = _worst(physical, point)
        assert scale > 1e-3
        for i in INDICES:
            for j in INDICES:
                assert _at(decomposed[i][j] - physical[i][j], point) == pytest.approx(
                    0.0, abs=1e-12 * scale
                )


def _unimodular():
    """A metric with ``det gamma = 1`` exactly, built as ``L L^T``.

    Unit-diagonal triangular ``L`` makes the determinant one by
    construction, so ``e^(-4 phi) = 1`` and the conformal metric *is* the
    metric. Two things follow, and both are wanted here. Everything stays
    polynomial, so SymPy can carry the third derivatives of the metric that
    the connection form needs without the expression swell a cube root
    brings; and ``det gammabar = 1`` is the condition BSSN evolves under, so
    this is the case the identity has to hold in, not a special one.
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
    lapse = 1 + X * Z / 4
    shift = [Y / 6, X * Z / 8, (X + Y) / 7]
    return lapse, shift, metric, curvature


@pytest.mark.slow
def test_the_connection_form_of_the_conformal_ricci_is_the_same_tensor():
    """``Rbar_ij`` written with ``Gammabar^i`` equals ``Rbar_ij`` written without it.

    The rewrite an evolution cannot do without -- carrying ``Gammabar^i``
    as an independent field is what makes BSSN strongly hyperbolic, and
    running the plain formula instead makes the gauge wave blow up at a
    rate proportional to ``1/h`` (measured in
    ``test_bssn_evolution.py``). It is only legitimate because the two are
    algebraically the same tensor whenever ``Gammabar^i = gammabar^jk
    Gammabar^i_jk``, which is what ``from_adm`` sets it to.

    Checked on a unimodular metric with every component non-zero, a
    non-zero shift and non-zero extrinsic curvature, so nothing switches
    off. The difference is not small: it is *exactly zero*.

    Evaluated at **rational** points rather than the module's floats, which
    is what makes that claim checkable. At a float point the two forms
    differ by about 1e-17 -- the rounding of two different orders of
    summation, not a discrepancy -- and an exact comparison there would be
    testing floating-point associativity. Rationals keep SymPy in exact
    arithmetic all the way, so ``== 0`` means what it says.
    """
    points = (
        {X: sp.Rational(1, 3), Y: sp.Rational(1, 4), Z: sp.Rational(1, 5)},
        {X: sp.Rational(-2, 7), Y: sp.Rational(3, 8), Z: sp.Rational(-1, 6)},
    )
    slice_ = symbolic_slice(*_unimodular(), COORDS)
    variables = bssn.from_adm(slice_)
    plain = ricci(variables.conformal_slice)
    connection_form = bssn.conformal_connection_ricci(variables)
    for point in points:
        scale = _worst(plain, point)
        assert scale > 1e-3
        for i in INDICES:
            for j in INDICES:
                difference = sp.simplify((connection_form[i][j] - plain[i][j]).subs(point))
                assert difference == 0, (i, j, difference)


def test_the_conformal_ricci_rejects_a_form_it_does_not_have(generic_bssn):
    with pytest.raises(ValueError, match="unknown Ricci form"):
        bssn.physical_ricci(generic_bssn, form="whatever")


@pytest.mark.slow
@pytest.mark.benchmark
def test_the_bssn_equations_evolve_the_gauge_wave_exactly():
    """All five right-hand sides against the exact solution, including the connection.

    The connection equation is the one that cannot be checked by the
    chain-rule route, because ``Gammabar^i`` is already a derivative of the
    state. Here it is checked directly: ``Gammabar^x`` is a non-zero
    function of ``x - t`` for this solution, and its time derivative has to
    come out of the equation that carries the momentum constraint in it.
    """
    lapse, shift, metric, curvature = _gauge_wave()
    slice_ = symbolic_slice(lapse, shift, metric, curvature, COORDS)
    variables = bssn.from_adm(slice_)
    rhs = bssn.bssn_rhs(variables)

    conformal_exponent = sp.log(sp.Matrix(metric).det()) / 12
    assert sp.simplify(rhs["phi"] - sp.diff(conformal_exponent, TIME)) == 0
    assert sp.simplify(rhs["mean_curvature"] - sp.diff(variables.mean_curvature, TIME)) == 0
    assert sp.simplify(
        sp.Matrix(rhs["conformal_metric"]) - sp.Matrix(variables.conformal_metric).diff(TIME)
    ) == sp.zeros(3, 3)
    assert sp.simplify(
        sp.Matrix(rhs["traceless_curvature"]) - sp.Matrix(variables.traceless_curvature).diff(TIME)
    ) == sp.zeros(3, 3)
    assert sp.simplify(variables.connection[0]) != 0
    for i in INDICES:
        assert sp.simplify(rhs["connection"][i] - sp.diff(variables.connection[i], TIME)) == 0


@pytest.mark.slow
@pytest.mark.benchmark
def test_the_two_bssn_derivations_agree_except_through_the_constraint(
    generic, generic_bssn, generic_rhs
):
    """The round-trip, and the one place the textbook form is *not* the ADM one.

    ``d_t phi``, ``d_t gammabar_ij`` and ``d_t Abar_ij`` agree to machine
    precision with the chain rule applied to the ADM equations. ``d_t K``
    does not, and the difference is exactly ``-alpha H``: the textbook
    equation uses the Hamiltonian constraint to replace the Ricci scalar
    with ``Abar_ij Abar^ij + K^2/3``.

    That is not a discrepancy to be tolerated, it is the most interesting
    fact about the change of variables -- it is why BSSN and ADM behave
    differently on constraint-violating data, which is all data -- so it is
    asserted as an equality rather than hidden in a tolerance.
    """
    textbook, chained = generic_rhs
    hamiltonian = hamiltonian_constraint(generic, inverse=inverse_metric(generic.metric))
    for point in POINTS:
        assert _at(textbook["phi"] - chained["phi"], point) == pytest.approx(0.0, abs=1e-14)
        for key in ("conformal_metric", "traceless_curvature"):
            scale = _worst(textbook[key], point)
            assert scale > 1e-3
            for i in INDICES:
                for j in INDICES:
                    assert _at(textbook[key][i][j] - chained[key][i][j], point) == pytest.approx(
                        0.0, abs=1e-12 * scale
                    )

        difference = _at(textbook["mean_curvature"] - chained["mean_curvature"], point)
        constraint = _at(generic.lapse * hamiltonian, point)
        assert abs(constraint) > 1e-3
        assert difference == pytest.approx(-constraint, rel=1e-10)


def test_the_connection_equation_refuses_what_a_grid_slice_cannot_supply():
    """Two terms an array slice cannot compute, and neither is dropped silently.

    ``Gammabar^i`` is itself a derivative of the state, so its advection term
    needs a *third* derivative of the metric; and the equation needs
    ``d_k d_j beta^i``, a second derivative of the shift. A ``Slice`` built
    from arrays carries one pass of differences and so has neither. A
    symbolic slice differentiates both itself, and a vanishing shift kills
    both terms -- but an array slice with a shift has to refuse, because a
    missing term is worse than an error.

    Both refusals are checked, one after the other: supplying the shift
    Hessian gets past the first and straight into the second.
    """
    shape = (8, 8, 8)
    ones, zeros = np.ones(shape), np.zeros(shape)
    metric = [[ones if i == j else zeros for j in INDICES] for i in INDICES]
    curvature = [[zeros for _ in INDICES] for _ in INDICES]
    moving = grid_slice(ones, [0.1 * ones, zeros, zeros], metric, curvature, 0.1)
    variables = bssn.from_adm(moving)

    with pytest.raises(ValueError, match="second-derivative-of-shift"):
        bssn.bssn_rhs(variables)

    hessian = [[[zeros for _ in INDICES] for _ in INDICES] for _ in INDICES]
    with pytest.raises(ValueError, match="advection term"):
        bssn.bssn_rhs(variables, dd_shift=hessian)

    still = grid_slice(ones, [zeros, zeros, zeros], metric, curvature, 0.1)
    rhs = bssn.bssn_rhs(bssn.from_adm(still))
    assert float(np.max(np.abs(rhs["mean_curvature"]))) == pytest.approx(0.0, abs=1e-12)


# --- stencil substitution, CSE and kernel emission ---------------------


@pytest.fixture(scope="module")
def adm_expressions():
    slice_, _, derivatives = abstract_slice()
    dt_metric, dt_curvature = adm_rhs(slice_)
    expressions = {}
    for i in INDICES:
        for j in range(i, DIMENSION):
            expressions[f"dt_gamma{i}{j}"] = dt_metric[i][j]
            expressions[f"dt_K{i}{j}"] = dt_curvature[i][j]
    return expressions, derivatives


@pytest.fixture(scope="module")
def adm_kernel(adm_expressions):
    expressions, derivatives = adm_expressions
    return codegen.emit(expressions, derivatives, order=4, backend="numpy")


def test_the_abstract_slice_names_every_field_and_derivative():
    slice_, fields, derivatives = abstract_slice()
    assert len(fields) == 1 + 3 + 6 + 6
    assert {"alpha", "beta0", "gamma01", "K22"} <= set(fields)
    # One derivative per field per axis, plus the second derivatives of the
    # lapse and the metric.
    assert len(derivatives) == 16 * 3 + 6 * (1 + 6)
    assert derivatives[sp.Symbol("dd_gamma01_12", real=True)] == ("gamma01", (1, 2))
    assert slice_.coords is None


@pytest.mark.benchmark
def test_common_subexpression_elimination_earns_its_place(adm_kernel):
    """Thirty-two thousand operations become thirteen hundred.

    The twelve outputs share nearly all of their work -- the inverse
    metric, the Christoffels, the Ricci tensor -- so the elimination is run
    over all of them at once rather than one at a time. The measured
    reduction is a factor of twenty-five, which is the difference between a
    kernel that runs and one that does not.
    """
    assert adm_kernel.raw_operations > 30000
    assert adm_kernel.operations < 2000
    assert adm_kernel.reduction < 0.05
    assert adm_kernel.temporaries > 100


def test_only_the_stencils_that_are_needed_are_emitted(adm_kernel):
    """Seventy-five of the ninety possible derivative arrays.

    The ADM right-hand sides never use the second derivative of the shift,
    and a kernel that differenced everything anyway would spend a sixth of
    its time filling arrays nothing reads.
    """
    assert adm_kernel.stencils == 75
    assert "dd_beta" not in adm_kernel.source
    assert adm_kernel.source.count("_d1(") + adm_kernel.source.count("_d2(") > 75
    assert "def _d1(f, axis, h)" in adm_kernel.source
    assert len(adm_kernel.fields) == 16


@pytest.mark.benchmark
@pytest.mark.parametrize(("order", "expected"), [(2, 4.0), (4, 16.0), (6, 64.0)])
def test_the_emitted_kernel_converges_at_its_stencil_order(adm_expressions, order, expected):
    """The whole pipeline against the gauge wave: derivation to convergence.

    The error in ``d_t K_xx`` falls by the stencil's factor for each
    halving of the grid spacing. ``d_t gamma_xx`` is exact at every
    resolution, because with zero shift it is ``-2 alpha K_xx`` and
    involves no derivative at all -- which is a useful thing for the test
    to notice, since it means the convergence being measured is the
    curvature equation's.
    """
    expressions, derivatives = adm_expressions
    kernel = codegen.emit(expressions, derivatives, order=order, backend="numpy")

    amplitude = 0.1
    profile = 1 - amplitude * sp.sin(2 * sp.pi * (X - TIME))
    lapse = sp.sqrt(profile)
    curvature = -sp.diff(profile, TIME) / (2 * lapse)
    closed = {
        "alpha": lapse,
        "gamma00": profile,
        "K00": curvature,
        "dt_gamma00": sp.diff(profile, TIME),
        "dt_K00": sp.diff(curvature, TIME),
    }
    functions = {key: sp.lambdify(X, value.subs(TIME, 0), "numpy") for key, value in closed.items()}

    errors = []
    for count in (32, 64):
        shape = (count, 7, 7)
        grid = codegen.sample({"x": lambda a, b, c: a}, shape, 1.0)
        position = grid["fields"]["x"]
        fields = {name: np.zeros(shape) for name in kernel.fields}
        fields["gamma11"] = np.ones(shape)
        fields["gamma22"] = np.ones(shape)
        for name in ("alpha", "gamma00", "K00"):
            fields[name] = functions[name](position)
        out = kernel(fields, grid["spacing"])
        assert np.max(np.abs(out["dt_gamma00"] - functions["dt_gamma00"](position))) < 1e-14
        for key, value in out.items():
            if key not in ("dt_gamma00", "dt_K00"):
                assert np.max(np.abs(value)) < 1e-9
        errors.append(float(np.max(np.abs(out["dt_K00"] - functions["dt_K00"](position)))))

    assert errors[0] / errors[1] == pytest.approx(expected, rel=0.25)


def test_the_kernel_summary_reports_what_it_did(adm_kernel):
    summary = adm_kernel.summary()
    assert summary["backend"] == "numpy"
    assert summary["order"] == 4
    assert summary["outputs"] == 12
    assert summary["fields"] == 16
    assert 0.0 < summary["reduction"] < 0.05


def test_emission_error_paths(adm_expressions):
    expressions, derivatives = adm_expressions
    with pytest.raises(ValueError, match="order must be one of"):
        codegen.emit(expressions, derivatives, order=3)
    with pytest.raises(ValueError, match="nothing to emit"):
        codegen.emit({}, derivatives)
    with pytest.raises(ValueError, match="unknown backend"):
        codegen.emit({"a": sp.Symbol("alpha")}, derivatives, backend="fortran")


def test_the_sample_grid_is_periodic():
    grid = codegen.sample({"f": lambda x, y, z: np.sin(2 * np.pi * x)}, (8, 4, 4), 1.0)
    assert grid["spacing"] == pytest.approx((0.125, 0.25, 0.25))
    assert grid["fields"]["f"].shape == (8, 4, 4)
    assert grid["mesh"][0].max() == pytest.approx(0.875)
    with pytest.raises(ValueError, match="grid sizes"):
        codegen.sample({"f": lambda x, y, z: x}, (8, 8), (1.0, 1.0, 1.0))


@pytest.mark.slow
@pytest.mark.benchmark
def test_the_jax_kernel_is_the_numpy_kernel(adm_expressions):
    """Same expressions, same stencils, two backends, to 1e-15.

    JAX defaults to single precision, and a right-hand side computed in
    float32 agrees with the double-precision one to six digits -- close
    enough to be dismissed as noise and far too coarse for a constraint
    residual to converge. The emitter turns on ``jax_enable_x64`` for that
    reason, and this test would fail at 1e-7 if it stopped doing so.
    """
    jax = pytest.importorskip("jax")
    expressions, derivatives = adm_expressions
    numpy_kernel = codegen.emit(expressions, derivatives, backend="numpy")
    jax_kernel = codegen.emit(expressions, derivatives, backend="jax")
    assert jax.config.read("jax_enable_x64")

    rng = np.random.default_rng(7)
    shape = (12, 12, 12)
    fields = {name: rng.normal(scale=0.05, size=shape) for name in numpy_kernel.fields}
    for name in ("alpha", "gamma00", "gamma11", "gamma22"):
        fields[name] = fields[name] + 1.0
    spacing = (0.1, 0.1, 0.1)
    expected = numpy_kernel(fields, spacing)
    got = jax_kernel({key: jax.numpy.asarray(value) for key, value in fields.items()}, spacing)
    for key, value in expected.items():
        scale = max(float(np.max(np.abs(value))), 1e-30)
        assert float(np.max(np.abs(np.asarray(got[key]) - value))) < 1e-13 * scale


# --- the second-derivative stencil -------------------------------------


@pytest.mark.parametrize(("order", "expected"), [(2, 4.0), (4, 16.0), (6, 64.0)])
def test_the_second_derivative_stencil_converges(order, expected):
    errors = []
    for count in (48, 96):
        position = np.linspace(0.0, 2 * np.pi, count, endpoint=False)
        spacing = position[1] - position[0]
        values = np.sin(3 * position)
        radius = order // 2
        got = second_derivative(values, 0, spacing, order=order)[radius:-radius]
        errors.append(float(np.max(np.abs(got + 9 * values[radius:-radius]))))
    assert errors[0] / errors[1] == pytest.approx(expected, rel=0.2)


def test_the_second_derivative_stencil_rejects_bad_arguments():
    with pytest.raises(ValueError, match="order must be 2, 4, or 6"):
        second_derivative(np.zeros(10), 0, 0.1, order=3)
    with pytest.raises(ValueError, match="too short"):
        second_derivative(np.zeros(4), 0, 0.1, order=6)
