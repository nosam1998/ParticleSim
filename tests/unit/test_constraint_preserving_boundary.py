"""A constraint-preserving absorbing layer: Einstein's equations in the zone, the radiation damped.

The layer, :class:`~particlesim.solvers.nr.boundary.ConstraintPreserving`,
evolves the interior's own equations in the zone, inside a rim two points
deep that keeps :class:`~particlesim.solvers.nr.boundary.SecondOrder`. Two
terms are added in the layer, for the incoming modes of BSSN linearised about
flat space (harmonic slicing, frozen shift) that are not constraint modes: a
relaxation of the transverse-traceless incoming radiation towards its
Bayliss-Turkel value, and an outgoing radiation condition on the lapse. In
BSSN the constraint modes get nothing, because damping them is unstable
there. On CCZ4 the layer damps them with Z4's own terms, ``kappa_1`` in the
layer only. Each of those statements is checked here against the principal
part of the equations actually evolved.

The fast tests use the linearised system written out by hand, emitted into
a kernel by the same code generator in seconds; the slow ones hold those
kernels to the generated BSSN and CCZ4 kernels' own linearisations, and the
layer's damping to the CCZ4 kernel's ``kappa_1`` terms, which is what makes
the fast ones statements about BSSN and CCZ4 rather than about a model of
them.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from particlesim.solvers.nr import boundary, bssn
from particlesim.symbolic import codegen
from particlesim.symbolic.bssn import STATE_NAMES
from particlesim.symbolic.ccz4 import STATE_NAMES as Z4_NAMES

INDICES = range(3)
EXTENT = 8.0
N = 24
WIDTH = 5
PAIRS = ("00", "01", "02", "11", "12", "22")


def _pair(i, j):
    return f"{min(i, j)}{max(i, j)}"


def linearised_expressions(z4: bool = False):
    """BSSN about flat space, harmonic slicing, frozen shift, to first order.

    ``d_t alpha = -K``, ``d_t phi = -K/6``, ``d_t gammabar_ij = -2 Abar_ij``,
    ``d_t K = -Laplacian alpha``, ``d_t Gammabar^i = -(4/3) d_i K``, and

        d_t Abar_ij = [-d_i d_j alpha - 2 d_i d_j phi - (1/2) Laplacian gammabar_ij
                       + d_(i Gammabar_j)]^TF

    which is the connection-form Ricci tensor of the conformal metric plus
    the conformal factor's, linearised.

    With ``z4``, undamped CCZ4 instead. Linearised, it differs from BSSN in
    three rates: ``d_t K`` gets back ``H + 2 d_k Z_k``, ``d_t Gammahat^i``
    gets ``2 d_i Theta``, and ``d_t Theta = (H + 2 d_k Z_k) / 2``, with
    ``H = -8 Laplacian phi + d_i d_j gammabar_ij`` and
    ``2 Z_i = Gammahat^i - d_j gammabar_ij``. ``d_t Abar_ij`` keeps its form
    with ``Gammahat`` in it, because the Ricci tensor built from the evolved
    connection already carries ``2 d_(i Z_j)``.
    """
    registry = {}

    def d(name, *axes):
        axes = tuple(sorted(axes))
        prefix = "d" * len(axes)
        symbol = sp.Symbol(f"{prefix}_{name}_{''.join(str(a) for a in axes)}", real=True)
        registry[symbol] = (name, axes)
        return symbol

    trace = sp.Symbol("trK", real=True)
    out = {
        "alpha": -trace,
        "phi": -trace / 6,
        "trK": -sum(d("alpha", k, k) for k in INDICES),
    }
    if z4:
        # H + 2 d.Z = -8 Laplacian phi + d.Gammahat: the metric's second
        # derivatives cancel between the two.
        violation = -8 * sum(d("phi", k, k) for k in INDICES) + sum(d(f"Gt{k}", k) for k in INDICES)
        out["trK"] = out["trK"] + violation
        out["Theta"] = violation / 2
    source = {}
    for i in INDICES:
        out[f"Gt{i}"] = -sp.Rational(4, 3) * d("trK", i)
        if z4:
            out[f"Gt{i}"] = out[f"Gt{i}"] + 2 * d("Theta", i)
        out[f"beta{i}"] = sp.Integer(0) * trace
        out[f"B{i}"] = sp.Integer(0) * trace
        for j in INDICES:
            source[i, j] = (
                -d("alpha", i, j)
                - 2 * d("phi", i, j)
                - sum(d(f"gt{_pair(i, j)}", k, k) for k in INDICES) / 2
                + (d(f"Gt{j}", i) + d(f"Gt{i}", j)) / 2
            )
    traced = sum(source[k, k] for k in INDICES)
    for i in INDICES:
        for j in range(i, 3):
            out[f"gt{i}{j}"] = -2 * sp.Symbol(f"At{i}{j}", real=True)
            out[f"At{i}{j}"] = source[i, j] - (traced / 3 if i == j else 0)
    assert set(out) == set(Z4_NAMES if z4 else STATE_NAMES)
    return out, registry


def _layer(dissipation: float, z4: bool = False, **options):
    expressions, registry = linearised_expressions(z4)
    kernel = codegen.emit(expressions, registry, order=4, backend="jax")
    spacing = (EXTENT / N,) * 3
    evolution = bssn.Evolution(
        spacing=spacing,
        kernel=kernel,
        slicing="harmonic",
        shift_condition="frozen",
        enforce=False,
        dissipation=dissipation,
    )
    axis = np.linspace(0.0, EXTENT, N, endpoint=False) - EXTENT / 2 + EXTENT / (2 * N)
    mesh = tuple(np.meshgrid(axis, axis, axis, indexing="ij"))
    radiative = boundary.Radiative(coords=mesh, spacing=spacing, axes=(0, 1, 2), width=WIDTH)
    return boundary.ConstraintPreserving(evolution, radiative, strength=20.0, **options), mesh


@pytest.fixture(scope="module")
def setup():
    """Without dissipation, so that what the layer adds can be read off exactly."""
    return _layer(0.0)


def _flat(names=STATE_NAMES):
    state = {name: np.zeros((N, N, N)) for name in names}
    state["alpha"] = np.ones((N, N, N))
    for i in INDICES:
        state[f"gt{i}{i}"] = np.ones((N, N, N))
    return state


def _rates(layer, state):
    return {k: np.asarray(v) for k, v in layer.right_hand_side(layer.start(state)).items()}


def _einstein(layer, state):
    return {
        k: np.broadcast_to(np.asarray(v), (N, N, N))
        for k, v in layer.evolution.right_hand_side(state).items()
    }


def test_the_angular_laplacian_has_the_spherical_harmonics_eigenvalues():
    """``Delta_S Y_l = -l(l+1) Y_l``, from rotations alone, converging at fourth order."""
    errors = []
    for n in (32, 64):
        h = 12.0 / n
        axis = np.arange(n) * h - 6.0 + h / 2
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        r2 = x**2 + y**2 + z**2
        radial = np.exp(-0.05 * r2)
        shell = (r2 > 4.0) & (r2 < 25.0)
        worst = 0.0
        for ell, harmonic in ((1, z / np.sqrt(r2)), (2, (3 * z**2 - r2) / r2)):
            field = (harmonic * radial)[None]
            result = boundary.angular_laplacian(np, field, (x, y, z), (h,) * 3)[0]
            worst = max(worst, np.max(np.abs(result + ell * (ell + 1) * field[0])[shell]))
        errors.append(worst)
    assert errors[1] < 0.1, errors
    assert np.log2(errors[0] / errors[1]) > 3.5, errors


# --- the layer, on the linearised system ----------------------------------


def test_the_zone_is_a_layer_and_a_rim(setup):
    """The rim is the outer two points of the zone; the layer is the rest of it."""
    layer, _ = setup
    inner, rim = layer.masks()
    zone = layer.boundary.mask()
    assert not np.any(inner & rim)
    assert np.array_equal(inner | rim, zone)
    assert rim[:2].all() and rim[-2:].all() and not rim[2:-2, 2:-2, 2:-2].any()
    assert inner[2:WIDTH, WIDTH:-WIDTH, WIDTH:-WIDTH].all()


def test_the_layer_leaves_minkowski_alone(setup):
    """Every rate, the auxiliaries' included, vanishes on flat space."""
    layer, _ = setup
    rates = _rates(layer, _flat())
    assert set(rates) == set(STATE_NAMES) | set(layer.auxiliary_names(_flat()))
    assert max(float(np.max(np.abs(v))) for v in rates.values()) < 1e-13


def test_outside_the_zone_the_rates_are_the_evolutions_dissipation_and_all():
    """The layer's own dissipation is the evolution's, away from the edge."""
    layer, _ = _layer(0.1)
    rng = np.random.default_rng(5)
    state = _flat()
    for name in STATE_NAMES:
        state[name] = state[name] + 1e-3 * rng.normal(size=(N, N, N))
    mine = _rates(layer, state)
    theirs = _einstein(layer, state)
    outside = ~layer.boundary.mask()
    for name in STATE_NAMES:
        assert np.allclose(mine[name][outside], theirs[name][outside], rtol=0, atol=1e-12), name
    assert np.max(np.abs(theirs["At01"][outside])) > 1e-3


def test_a_pure_constraint_violation_is_left_to_einsteins_equations(setup):
    """Nothing is added to a violation of the connection constraint, in the layer.

    ``Gammabar^i`` disturbed on flat data: ``G^i`` is the disturbance, and
    the incoming radiation and the lapse are untouched by it, so the layer's
    rates are the evolution's to the last bit. Damping it instead is
    unstable; ``docs/benchmarks.md`` has the dispersion relation.
    """
    layer, mesh = setup
    inner, _ = layer.masks()
    r = np.sqrt(sum(m**2 for m in mesh))
    violating = _flat()
    violating["Gt0"] = 1e-4 * np.exp(-((r - 3.5) ** 2))
    violating["Gt2"] = -0.5e-4 * np.exp(-((r - 3.0) ** 2)) * mesh[0] / 4
    mine, theirs = _rates(layer, violating), _einstein(layer, violating)
    assert np.max(np.abs(theirs["At01"][inner])) > 1e-6
    for name in STATE_NAMES:
        assert np.array_equal(mine[name][inner], theirs[name][inner]), name


def _shells(mesh, direction):
    """``c_ij f(t -+ r)/r`` at ``t = 0``, with ``Abar_ij = -d_t gammabar_ij / 2``: out or in."""
    r = np.sqrt(sum(m**2 for m in mesh))
    profile = np.exp(-(((r - 4.0) / 1.2) ** 2))
    slope = -2 * (r - 4.0) / 1.2**2 * profile  # d f / d r
    # f(t - r) has d_t = -d_r; f(t + r) has d_t = +d_r.
    rate = -slope if direction == "out" else slope
    weights = dict(zip(PAIRS, (0.3, 1.0, -0.4, -0.7, 0.6, 0.4), strict=True))
    state = _flat()
    for pair, c in weights.items():
        state[f"gt{pair}"] = state[f"gt{pair}"] + 1e-4 * c * profile / r
        state[f"At{pair}"] = -0.5e-4 * c * rate / r
    return state


def test_an_outgoing_wave_is_left_alone_and_an_incoming_one_is_not(setup):
    """The radiative term acts on the incoming characteristic variable and on nothing else.

    ``w_ij = d_t gammabar_ij + (d_r + 1/r)(gammabar_ij - delta_ij)`` vanishes
    for ``f(t - r)/r`` and is ``2 f'/r`` for ``f(t + r)/r``. The term added
    to ``d_t Abar_ij`` is the transverse-traceless part of it, so it is the
    stencil's error on the first and order one on the second.
    """
    layer, mesh = setup
    inner, _ = layer.masks()
    added = {}
    for direction in ("out", "in"):
        state = _shells(mesh, direction)
        rates, einstein = _rates(layer, state), _einstein(layer, state)
        added[direction] = np.stack(
            [np.where(inner, rates[f"At{p}"] - einstein[f"At{p}"], 0.0) for p in PAIRS]
        )
    assert np.max(np.abs(added["in"])) > 1e-4
    assert np.max(np.abs(added["out"])) < 0.02 * np.max(np.abs(added["in"])), (
        np.max(np.abs(added["out"])),
        np.max(np.abs(added["in"])),
    )

    # Transverse to x^i/r and traceless, at every point.
    term = added["in"]
    full = [[term[PAIRS.index(_pair(i, j))] for j in INDICES] for i in INDICES]
    r = np.sqrt(sum(m**2 for m in mesh))
    n = [m / r for m in mesh]
    scale = np.max(np.abs(term))
    for i in INDICES:
        assert np.max(np.abs(sum(full[i][j] * n[j] for j in INDICES))) < 1e-12 * scale
    assert np.max(np.abs(sum(full[i][i] for i in INDICES))) < 1e-12 * scale


def test_the_lapse_radiates_as_sommerfeld_where_its_auxiliary_is_zero(setup):
    """With the surface auxiliary at zero the lapse's condition *is* Sommerfeld's, to rounding."""
    layer, mesh = setup
    inner, _ = layer.masks()
    r2 = sum(m**2 for m in mesh)
    state = _flat()
    state["alpha"] = 1.0 + 1e-3 * np.exp(-((np.sqrt(r2) - 3.5) ** 2)) * (1 + mesh[2] / 4)
    rates = _rates(layer, state)
    sommerfeld = np.asarray(layer.boundary.rates({"alpha": state["alpha"]})["alpha"])
    # Sommerfeld differences alpha and this differences alpha - 1: the same
    # number to the rounding of a derivative of a field of order one.
    assert np.allclose(rates["alpha"][inner], sommerfeld[inner], rtol=0, atol=1e-13)
    assert np.max(np.abs(sommerfeld[inner])) > 1e-6

    # And the auxiliary moves as Bayliss and Turkel's second factor says:
    # d_t v = c^2 Delta_S u / (2 r^2) - c v / r, with v = 0 here.
    lapse = (state["alpha"] - 1.0)[None]
    coords = tuple(np.asarray(m) for m in mesh)
    expected = 0.5 * boundary.angular_laplacian(np, lapse, coords, layer.boundary.spacing)[0] / r2
    moved = rates[boundary.AUXILIARY + "alpha"]
    assert np.allclose(moved[inner], expected[inner], rtol=1e-10, atol=1e-18)
    assert np.max(np.abs(moved[~inner])) == 0.0


def test_the_rim_is_sommerfeld_where_its_carried_fields_are_zero(setup):
    """The rim's rates are Sommerfeld's plus what it carries: Sommerfeld's, to rounding, at zero.

    Every variable, on a state with something in all of them. This is where
    the condition is :class:`~particlesim.solvers.nr.boundary.SecondOrder`'s
    by construction, and it is checked against Sommerfeld's own code.
    """
    layer, mesh = setup
    _, rim = layer.masks()
    rng = np.random.default_rng(9)
    r = np.sqrt(sum(m**2 for m in mesh))
    state = _flat()
    for name in STATE_NAMES:
        state[name] = state[name] + 1e-4 * rng.normal() * np.exp(-((r - 4.5) ** 2))
    rates = _rates(layer, state)
    sommerfeld = layer.boundary.rates(state)
    for name in STATE_NAMES:
        expected = np.asarray(sommerfeld[name])
        assert np.allclose(rates[name][rim], expected[rim], rtol=0, atol=1e-13), name
    assert max(np.max(np.abs(np.asarray(sommerfeld[n])[rim])) for n in STATE_NAMES) > 1e-6


def test_the_auxiliaries_are_added_and_not_projected(setup):
    layer, _ = setup
    state = layer.start(_flat())
    surface = {boundary.AUXILIARY + n for n in (*boundary.METRIC, "alpha")}
    assert set(state) - set(STATE_NAMES) == surface | {boundary.RIM + n for n in STATE_NAMES}

    class _Projecting:
        def project(self, s):
            return {k: 2 * v for k, v in s.items()}

    shell = object.__new__(boundary.ConstraintPreserving)
    object.__setattr__(shell, "evolution", _Projecting())
    projected = boundary.ConstraintPreserving.project(shell, {**state})
    for name in STATE_NAMES:
        assert np.array_equal(projected[name], 2 * state[name])
    for name in layer.auxiliary_names(state):
        assert np.array_equal(projected[name], state[name])


def test_a_rim_narrower_than_the_stencil_or_a_zone_with_no_layer_is_refused(setup):
    layer, mesh = setup
    with pytest.raises(ValueError, match="stencil radius"):
        boundary.ConstraintPreserving(layer.evolution, layer.boundary, rim=1)
    thin = boundary.Radiative(coords=mesh, spacing=layer.boundary.spacing, axes=(0, 1, 2), width=2)
    with pytest.raises(ValueError, match="no layer"):
        boundary.ConstraintPreserving(layer.evolution, thin)


# --- the Z4 layer: the constraint modes, carried by Theta and Z_i, damped ---

KAPPA, MIX = 4.0, 0.5


@pytest.fixture(scope="module")
def z4_setup():
    """The undamped and the damped layer on linearised CCZ4, without dissipation."""
    plain, mesh = _layer(0.0, z4=True)
    damped, _ = _layer(0.0, z4=True, damping=KAPPA, damping_mix=MIX)
    return plain, damped, mesh


def test_the_layers_damping_is_z4s_and_only_in_the_layer(z4_setup):
    """``Theta`` and ``Gammahat^i`` disturbed: the damping is Z4's three terms, in the layer alone.

    ``d_t Theta -= kappa_1 (2 + kappa_2) Theta``,
    ``d_t K -= 3 kappa_1 (1 + kappa_2) Theta`` and
    ``d_t Gammahat^i -= kappa_1 (Gammahat^i + d_j gammabar^ij)``, with
    ``Gammahat^i + d_j gammabar^ij = 2 gammabar^ij Z_j``. On a flat metric
    the last is ``kappa_1 Gammahat^i`` exactly. Every other rate, and every
    rate outside the layer, is the undamped layer's to the last bit.
    """
    plain, damped, mesh = z4_setup
    inner, _ = plain.masks()
    r = np.sqrt(sum(m**2 for m in mesh))
    state = _flat(Z4_NAMES)
    state["Theta"] = 1e-4 * np.exp(-((r - 3.5) ** 2))
    state["Gt0"] = 1e-4 * np.exp(-((r - 3.2) ** 2)) * mesh[1] / 4
    added = {k: v - _rates(plain, state)[k] for k, v in _rates(damped, state).items()}
    expected = {name: np.zeros((N, N, N)) for name in added}
    expected["Theta"] = -KAPPA * (2 + MIX) * state["Theta"]
    expected["trK"] = -3 * KAPPA * (1 + MIX) * state["Theta"]
    expected["Gt0"] = -KAPPA * state["Gt0"]
    for name in added:
        assert np.allclose(added[name][inner], expected[name][inner], rtol=1e-12, atol=1e-20), name
    # Outside the layer the geometry's rates are untouched. The rim's carried
    # fields are not: they are fed by the layer's rates, damping and all.
    for name in Z4_NAMES:
        assert not np.any(added[name][~inner]), name
    assert np.max(np.abs(expected["Gt0"][inner])) > 1e-5


def test_the_layer_damps_z_with_the_metrics_own_connection(z4_setup):
    """With the metric disturbed, ``Z_i`` is the gap between ``Gammahat^i`` and the metric's own.

    ``gammabar_01`` alone, ``Gammahat^i = 0``: ``d_j gammabar^ij`` is
    ``-d_1 gammabar_01`` for ``i = 0`` and ``-d_0 gammabar_01`` for ``i = 1``
    to first order, and the damping is ``kappa_1`` times it, against the
    closed-form derivative of the Gaussian.
    """
    plain, damped, mesh = z4_setup
    inner, _ = plain.masks()
    r = np.sqrt(sum(m**2 for m in mesh))
    bump = 1e-5 * np.exp(-((r - 3.5) ** 2))
    state = _flat(Z4_NAMES)
    state["gt01"] = bump
    added = {k: v - _rates(plain, state)[k] for k, v in _rates(damped, state).items()}
    slope = -2 * (r - 3.5) * bump / r
    for i, axis in ((0, 1), (1, 0)):
        expected = KAPPA * slope * mesh[axis]  # -kappa (d_j gammabar^ij) = +kappa d gt01
        scale = np.max(np.abs(expected[inner]))
        assert np.max(np.abs(added[f"Gt{i}"] - expected)[inner]) < 2e-2 * scale, i
    assert np.max(np.abs(added["Gt2"][inner])) < 1e-12 * scale


def test_damping_wants_theta_and_z4s_signs(setup, z4_setup):
    """BSSN has no ``Theta`` to damp, and ``kappa_1 < 0`` or ``kappa_2 <= -1`` grow instead."""
    layer, _ = setup
    plain, _, _ = z4_setup
    with pytest.raises(ValueError, match="Z4 system"):
        boundary.ConstraintPreserving(layer.evolution, layer.boundary, damping=1.0)
    for options in ({"damping": -1.0}, {"damping": 1.0, "damping_mix": -1.0}):
        with pytest.raises(ValueError, match="kappa"):
            boundary.ConstraintPreserving(plain.evolution, plain.boundary, **options)


def _violation_left_inside(layer, mesh, final=6.0):
    """A ``Theta`` violation at the centre, evolved; what is left inside the zone at ``final``.

    Linearised Z4 carries ``Theta`` at the speed of light, so a central
    violation reaches the zone at ``t`` of about two and is gone from the
    interior by four, unless the boundary sends it back.
    """
    from particlesim.solvers.nr.stepping import operations

    r2 = sum(m**2 for m in mesh)
    state = layer.start(_flat(Z4_NAMES))
    state["Theta"] = 1e-4 * np.exp(-r2)
    step = layer.time_step
    steps = int(round(final / step))
    advance, finish, _ = operations("jax")
    current = dict(state)
    for _ in range(steps):
        first = layer.right_hand_side(current)
        second = layer.right_hand_side(advance(current, first, step / 2))
        third = layer.right_hand_side(advance(current, second, step / 2))
        fourth = layer.right_hand_side(advance(current, third, step))
        current = finish(current, first, second, third, fourth, step)
    inside = ~layer.boundary.mask()
    return float(np.sqrt(np.mean(np.asarray(current["Theta"])[inside] ** 2))) / 1e-4


@pytest.mark.slow
def test_a_constraint_violation_leaves_through_the_damped_layer(z4_setup):
    """What the rim sends back of an outgoing ``Theta`` wave, the damping absorbs on the way in.

    A Gaussian ``Theta`` at the centre of the box of 8 at 24 points, through
    the linearised layer to ``t = 6``. Measured, the RMS of ``Theta`` left
    inside the zone, over the initial peak, is 0.56 undamped, and 0.12,
    0.10 and 0.20 at ``kappa_1`` = 2, 4 and 10. The best value is where the
    damping rate ``kappa_1 / 2`` of a mode with ``k > kappa_1 / 2`` meets the
    ``k^2 / kappa_1`` of a longer one, near the wave numbers here. A minute
    of stepping a 24^3 grid twice, hence slow.
    """
    plain, damped, mesh = z4_setup
    left_plain = _violation_left_inside(plain, mesh)
    left_damped = _violation_left_inside(damped, mesh)
    assert np.isfinite(left_plain) and np.isfinite(left_damped)
    assert left_damped < 0.3 * left_plain, (left_plain, left_damped)


# --- the linearisation is BSSN's, and CCZ4's -------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("z4", [False, True], ids=["bssn", "ccz4"])
def test_the_linearised_system_is_the_generated_kernels_linearisation(setup, z4_setup, z4):
    """What the fast tests used is the principal part on a flat background.

    The generated kernel at flat space plus ``eps X``, minus flat space,
    over ``eps``, against the hand-linearised kernel on ``X``: they differ
    by ``O(eps)``. A wrong coefficient in the linearisation -- the ``4/3``,
    a missing ``phi`` term, CCZ4's ``H + 2 d_k Z_k`` -- shows at order one.
    CCZ4's kernel is the undamped one, as the layer uses it.
    """
    if z4:
        from particlesim.solvers.nr import ccz4

        layer = z4_setup[0]
        names = Z4_NAMES
        full = ccz4.rhs_kernel(slicing="harmonic", shift_condition="frozen", damping=0.0)
    else:
        layer = setup[0]
        names = STATE_NAMES
        full = bssn.rhs_kernel(slicing="harmonic", shift_condition="frozen")
    rng = np.random.default_rng(11)
    n, h = 16, 0.25
    axis = np.arange(n) * h
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    k = 2 * np.pi / (n * h)
    flat = {name: np.zeros((n, n, n)) for name in names}
    flat["alpha"] = np.ones((n, n, n))
    for i in INDICES:
        flat[f"gt{i}{i}"] = np.ones((n, n, n))
    perturbation = {}
    for name in names:
        if name.startswith(("beta", "B")):
            perturbation[name] = np.zeros((n, n, n))
            continue
        a, b, c = rng.normal(size=3)
        perturbation[name] = np.sin(k * x + a) * np.cos(k * y + b) * np.sin(2 * k * z + c)
    traced = sum(perturbation[f"gt{i}{i}"] for i in INDICES) / 3
    for i in INDICES:
        perturbation[f"gt{i}{i}"] = perturbation[f"gt{i}{i}"] - traced
    traced = sum(perturbation[f"At{i}{i}"] for i in INDICES) / 3
    for i in INDICES:
        perturbation[f"At{i}{i}"] = perturbation[f"At{i}{i}"] - traced

    linear = layer.evolution.kernel(perturbation, (h,) * 3)
    for eps in (1e-5, 1e-6):
        perturbed = {name: flat[name] + eps * perturbation[name] for name in names}
        rates = full(perturbed, (h,) * 3)
        base = full(flat, (h,) * 3)
        for name in names:
            difference = np.broadcast_to(
                (np.asarray(rates[name]) - np.asarray(base[name])) / eps - np.asarray(linear[name]),
                (n, n, n),
            )
            scale = max(1.0, float(np.max(np.abs(np.asarray(linear[name])))))
            assert np.max(np.abs(difference)) < 200 * eps * scale, (name, eps)


@pytest.mark.slow
def test_the_layers_damping_is_the_generated_kernels_kappa_term():
    """The layer's hand-written damping is what CCZ4's kernel does with ``kappa_1``.

    The damping is linear in ``kappa_1``, so the kernel derived at 0.1 less
    the one at zero is 0.1 times it exactly. On random smooth data, every
    variable ten per cent from flat, the layer's damping at 0.1 on the
    undamped kernel matches that difference in the layer to 1e-15, and is
    zero outside it. The ``Gammahat`` part needs the exact inverse metric
    and its derivative at that amplitude, which a linearised check would
    not see.
    """
    from particlesim.solvers.nr import ccz4

    n, h = 20, 0.3
    axis = (np.arange(n) - n / 2 + 0.5) * h
    mesh = tuple(np.meshgrid(axis, axis, axis, indexing="ij"))
    rng = np.random.default_rng(3)
    k = 2 * np.pi / (n * h)
    state = {}
    for name in Z4_NAMES:
        background = 1.0 if name == "alpha" or (name[:2] == "gt" and name[2] == name[3]) else 0.0
        size = 0.0 if name.startswith(("beta", "B")) else 0.05
        a, b, c = rng.normal(size=3)
        state[name] = background + size * np.sin(k * mesh[0] + a) * np.cos(
            k * mesh[1] + b
        ) * np.sin(2 * k * mesh[2] + c)
    undamped = ccz4.build((h,) * 3, slicing="harmonic", shift_condition="frozen", damping=0.0)
    damped = ccz4.build((h,) * 3, slicing="harmonic", shift_condition="frozen", damping=0.1)
    expected = {
        name: np.broadcast_to(
            np.asarray(damped.kernel(state, (h,) * 3)[name])
            - np.asarray(undamped.kernel(state, (h,) * 3)[name]),
            (n, n, n),
        )
        for name in Z4_NAMES
    }
    zone = boundary.Radiative(coords=mesh, spacing=(h,) * 3, axes=(0, 1, 2), width=7)
    plain = boundary.ConstraintPreserving(undamped, zone)
    layer = boundary.ConstraintPreserving(undamped, zone, damping=0.1)
    inner, _ = plain.masks()
    start = plain.start(state)
    p, d = plain.right_hand_side(start), layer.right_hand_side(start)
    for name in Z4_NAMES:
        added = np.asarray(d[name]) - np.asarray(p[name])
        assert np.max(np.abs(added - expected[name])[inner]) < 1e-14, name
        assert not np.any(added[~inner]), name
    assert max(np.max(np.abs(expected[v][inner])) for v in ("Theta", "trK", "Gt0")) > 1e-3


# --- issue #132's acceptance, on the documented setup ---------------------------

TEUKOLSKY_EXTENT = 12.0
TEUKOLSKY_CUBE = 2.5
GHOSTS = 3


class _Octant(boundary.Radiative):
    """The zone on the high faces only: the low faces are planes of reflection symmetry."""

    def mask(self):
        shape = np.asarray(self.coords[0]).shape
        out = np.zeros(shape, dtype=bool)
        for axis in self.axes:
            edge = np.arange(shape[axis]) >= shape[axis] - self.width
            out |= edge.reshape([-1 if k == axis else 1 for k in range(len(shape))])
        return out


def _reflect(state):
    """Ghost planes below each symmetry face, refreshed from the real points with their parities."""
    import jax.numpy as jnp

    from particlesim.solvers.nr import symmetry

    out = {}
    for name, value in state.items():
        a = jnp.asarray(value)
        for axis, sign in enumerate(symmetry.parity(name)):
            moved = jnp.moveaxis(a, axis, 0)
            mirrored = sign * moved[GHOSTS + 1 : 2 * GHOSTS + 1][::-1]
            a = jnp.moveaxis(jnp.concatenate([mirrored, moved[GHOSTS:]], axis=0), 0, axis)
        out[name] = a
    return out


def _octant_teukolsky(
    n: int,
    condition,
    final: float = 10.5,
    every: float = 0.5,
    formulation: str = "bssn",
    amplitude: float = 1e-6,
):
    """The measurement of ``docs/benchmarks.md``'s tables, on one octant.

    Linear Teukolsky wave, ``a = 1e-6``, ``λ = 1``, a box of 12, the zone the
    outer unit, ``|x_i| >= 5``, harmonic slicing and a frozen shift; measured
    in the cube ``|x_i| <= 2.5`` against the closed form. The ``l = 2,
    m = 0`` wave is even under each coordinate reflection with tensor
    parity, so only ``x_i >= 0`` is evolved, with three reflected ghost
    planes refreshed before every right-hand side. The zone is the same set
    of points as on the full grid's positive faces, and the cube norms are
    weighted so they are the full cube's. That is what makes the 72-point
    run affordable; the 96-point one was run the same way.

    ``formulation`` is ``"bssn"`` or ``"ccz4"``; CCZ4 runs the undamped
    kernel, with ``Theta = 0`` at the start, so that any damping is the
    condition's. ``condition(evolution, zone)`` builds the boundary.

    Returns ``(t, largest |h|, RMS error, ||H||, ||M||)`` samples over the
    cube, all divided by the amplitude.
    """
    import jax

    from particlesim.solvers.nr import teukolsky
    from particlesim.solvers.nr.stepping import operations
    from particlesim.symbolic.threeplusone import determinant

    # Through bssn's resolver, so jax_enable_x64 is on before the first array:
    # converting the data first makes it float32, and the error at t = 0 is
    # then 0.05 instead of zero.
    jnp = bssn._module("jax")
    full, spacing = teukolsky.teukolsky_wave(
        shape=(n, n, n), amplitude=amplitude, width=1.0, extent=TEUKOLSKY_EXTENT, backend="numpy"
    )
    low = n // 2 - GHOSTS
    window = (slice(low, None),) * 3
    state = {k: jnp.asarray(np.asarray(v)[window]) for k, v in full.items()}
    del full
    h = spacing[0]
    axis = (np.arange(n - low) - GHOSTS) * h
    mesh = tuple(np.meshgrid(axis, axis, axis, indexing="ij"))
    if formulation == "ccz4":
        from particlesim.solvers.nr import ccz4

        evolution = ccz4.build(spacing, slicing="harmonic", shift_condition="frozen", damping=0.0)
        state["Theta"] = jnp.zeros_like(state["phi"])
    else:
        evolution = bssn.Evolution.build(spacing, slicing="harmonic", shift_condition="frozen")
    zone = _Octant(coords=mesh, spacing=spacing, axes=(0, 1, 2), width=int(round(1.0 / h)))
    bounded = condition(evolution, zone)

    inside = (axis >= -1e-9) & (axis <= TEUKOLSKY_CUBE + 1e-9)
    cube = np.ix_(inside, inside, inside)
    side = np.where(np.abs(axis[inside]) < 1e-9, 1.0, 2.0)
    weights = side[:, None, None] * side[None, :, None] * side[None, None, :]
    weights = weights / weights.sum()
    cube_coords = [m[cube] for m in mesh]
    names = [f"gt{i}{j}" for i in INDICES for j in range(i, 3)]
    constraints = bssn.constraint_kernel()
    reflect = jax.jit(_reflect)

    def sample(time, current):
        current = reflect(current)
        hh = teukolsky.perturbation(time, cube_coords, amplitude, 1.0)
        metric = [[float(i == j) + hh[i][j] for j in INDICES] for i in INDICES]
        conformal = determinant(metric) ** (-1.0 / 3.0)
        largest = error = 0.0
        for name in names:
            i, j = int(name[2]), int(name[3])
            value = np.asarray(current[name])[cube]
            largest = max(largest, float(np.max(np.abs(value - float(i == j)))))
            error += float(np.sum(weights * (value - conformal * metric[i][j]) ** 2))
        geometry = {k: v for k, v in current.items() if not k.startswith(boundary.AUXILIARY)}
        out = constraints(bssn.physical_slice_arrays(geometry), tuple(spacing))
        hamiltonian = np.sum(weights * np.asarray(out["hamiltonian"])[cube] ** 2)
        momentum = sum(
            np.sum(weights * np.asarray(out[f"momentum{i}"])[cube] ** 2) for i in INDICES
        )
        return tuple(
            float(value)
            for value in (
                time,
                largest / amplitude,
                np.sqrt(error) / amplitude,
                np.sqrt(hamiltonian) / amplitude,
                np.sqrt(momentum) / amplitude,
            )
        )

    steps = int(round(final / evolution.time_step))
    step = final / steps
    stride = max(1, int(round(every / step)))
    advance, finish, _ = operations("jax")
    current = dict(bounded.start(state)) if hasattr(bounded, "start") else state
    current = {k: jnp.asarray(v) for k, v in current.items()}

    def rates(s):
        return bounded.right_hand_side(reflect(s))

    samples = [sample(0.0, current)]
    for index in range(1, steps + 1):
        first = rates(current)
        second = rates(advance(current, first, step / 2))
        third = rates(advance(current, second, step / 2))
        fourth = rates(advance(current, third, step))
        current = bounded.project(finish(current, first, second, third, fourth, step))
        if index % stride == 0:
            samples.append(sample(index * step, current))
    return samples


def _acceptance(samples):
    """``(truncation, after, largest left, ||H|| after)``, as the tables in the docs define them."""
    truncation = max(error for t, _, error, _, _ in samples if t <= 3.0 + 1e-9)
    late = [s for s in samples if s[0] >= 8.0 - 1e-9]
    return (
        truncation,
        max(s[2] for s in late),
        max(s[1] for s in late),
        max(s[3] for s in late),
    )


#: Z4's damping in the layer, ``kappa_1``. The interior is undamped CCZ4.
LAYER_DAMPING = 4.0


def _z4_layer(evolution, zone):
    """The acceptance's condition: the layer on CCZ4, its constraint modes damped."""
    return boundary.ConstraintPreserving(evolution, zone, damping=LAYER_DAMPING)


@pytest.mark.slow
@pytest.mark.benchmark
# The 72-point CCZ4 octant is twenty minutes at a load average of fourteen,
# so the project's 600 s default cannot hold the pair.
@pytest.mark.timeout(5400)
@pytest.mark.parametrize(("formulation", "condition"), [("ccz4", _z4_layer)])
def test_a_teukolsky_wave_leaves_through_the_layer_without_a_reflection(formulation, condition):
    """Issue #132's acceptance on the documented setup, at 48 and 72 points.

    Box of 12, zone from ``|x_i| = 5``, cube ``|x_i| <= 2.5``, ``a = 1e-6``,
    as multiples of the amplitude. Measured with the layer on CCZ4, damped
    at ``kappa_1 = 4`` in the layer and undamped inside:

        n     truncation   after      ratio   ||H|| after   left
        48    7.07e-02     3.16e-02   0.45    3.98e-02      0.131
        72    1.48e-02     5.52e-03   0.37    8.06e-03      0.023
        order 3.85         4.30               3.94

    The ``||H||`` bound is the one that changed. The same layer on BSSN,
    the previous condition, gives 3.80e-02 and 1.13e-02, order 3.0: BSSN
    with a frozen shift holds 4/3 of the Teukolsky data's own quadratic
    violation at the centre for good, and cannot damp the constraint
    violation the boundary sends in. Run once with ``("bssn",
    ConstraintPreserving)`` the test fails on that bound alone: every other
    bound holds, with 0.37 of the truncation error after the reflection at
    order 4.4, and the ``||H||`` order is 3.00 against 3.5. See
    ``docs/benchmarks.md``.
    """
    coarse = _acceptance(_octant_teukolsky(48, condition, formulation=formulation))
    fine = _acceptance(_octant_teukolsky(72, condition, formulation=formulation))
    truncation, after, left, hamiltonian = fine
    order = np.log(coarse[1] / fine[1]) / np.log(1.5)
    constraint_order = np.log(coarse[3] / fine[3]) / np.log(1.5)
    assert np.log(coarse[0] / truncation) / np.log(1.5) > 3.7, (coarse, fine)

    # The wave leaves: two orders of magnitude, from 48.
    assert left < 48.0 / 100.0, (coarse, fine)
    # What comes back is below the truncation error and converges as fast.
    assert after < 0.5 * truncation, (coarse, fine)
    assert order > 3.5, (coarse, fine, order)
    # And so does the constraint, at the interior's order.
    assert constraint_order > 3.5, (coarse, fine, constraint_order)
