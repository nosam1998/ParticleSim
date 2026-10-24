"""A constraint-preserving absorbing layer: Einstein's equations in the zone, the radiation damped.

The layer, :class:`~particlesim.solvers.nr.boundary.ConstraintPreserving`,
evolves the interior's own equations in the zone, inside a rim two points
deep that keeps :class:`~particlesim.solvers.nr.boundary.SecondOrder`. Two
terms are added in the layer, for the incoming modes of BSSN linearised about
flat space (harmonic slicing, frozen shift) that are not constraint modes: a
relaxation of the transverse-traceless incoming radiation towards its
Bayliss-Turkel value, and an outgoing radiation condition on the lapse. The
constraint modes get nothing, because damping them is unstable in BSSN. Each
of those statements is checked here against the principal part of the
equations actually evolved.

The fast tests use the linearised system written out by hand, emitted into
a kernel by the same code generator in seconds; the slow one holds that
kernel to the generated BSSN kernel's own linearisation, which is what
makes the fast ones statements about BSSN rather than about a model of it.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from particlesim.solvers.nr import boundary, bssn
from particlesim.symbolic import codegen
from particlesim.symbolic.bssn import STATE_NAMES

INDICES = range(3)
EXTENT = 8.0
N = 24
WIDTH = 5
PAIRS = ("00", "01", "02", "11", "12", "22")


def _pair(i, j):
    return f"{min(i, j)}{max(i, j)}"


def linearised_expressions():
    """BSSN about flat space, harmonic slicing, frozen shift, to first order.

    ``d_t alpha = -K``, ``d_t phi = -K/6``, ``d_t gammabar_ij = -2 Abar_ij``,
    ``d_t K = -Laplacian alpha``, ``d_t Gammabar^i = -(4/3) d_i K``, and

        d_t Abar_ij = [-d_i d_j alpha - 2 d_i d_j phi - (1/2) Laplacian gammabar_ij
                       + d_(i Gammabar_j)]^TF

    which is the connection-form Ricci tensor of the conformal metric plus
    the conformal factor's, linearised.
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
    source = {}
    for i in INDICES:
        out[f"Gt{i}"] = -sp.Rational(4, 3) * d("trK", i)
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
    assert set(out) == set(STATE_NAMES)
    return out, registry


def _layer(dissipation: float):
    expressions, registry = linearised_expressions()
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
    return boundary.ConstraintPreserving(evolution, radiative, strength=20.0), mesh


@pytest.fixture(scope="module")
def setup():
    """Without dissipation, so that what the layer adds can be read off exactly."""
    return _layer(0.0)


def _flat():
    state = {name: np.zeros((N, N, N)) for name in STATE_NAMES}
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


# --- the linearisation is BSSN's -------------------------------------------


@pytest.mark.slow
def test_the_linearised_system_is_the_generated_kernels_linearisation(setup):
    """What the fast tests used is BSSN's principal part on a flat background.

    The generated kernel at flat space plus ``eps X``, minus flat space,
    over ``eps``, against the hand-linearised kernel on ``X``: they differ
    by ``O(eps)``. A wrong coefficient in the linearisation -- the ``4/3``,
    a missing ``phi`` term -- shows at order one.
    """
    layer, _ = setup
    full = bssn.rhs_kernel(slicing="harmonic", shift_condition="frozen")
    rng = np.random.default_rng(11)
    n, h = 16, 0.25
    axis = np.arange(n) * h
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    k = 2 * np.pi / (n * h)
    flat = {name: np.zeros((n, n, n)) for name in STATE_NAMES}
    flat["alpha"] = np.ones((n, n, n))
    for i in INDICES:
        flat[f"gt{i}{i}"] = np.ones((n, n, n))
    perturbation = {}
    for name in STATE_NAMES:
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
        perturbed = {name: flat[name] + eps * perturbation[name] for name in STATE_NAMES}
        rates = full(perturbed, (h,) * 3)
        base = full(flat, (h,) * 3)
        for name in STATE_NAMES:
            difference = np.broadcast_to(
                (np.asarray(rates[name]) - np.asarray(base[name])) / eps - np.asarray(linear[name]),
                (n, n, n),
            )
            scale = max(1.0, float(np.max(np.abs(np.asarray(linear[name])))))
            assert np.max(np.abs(difference)) < 200 * eps * scale, (name, eps)


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


def _octant_teukolsky(n: int, condition, final: float = 10.5, every: float = 0.5):
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
    amplitude = 1e-6
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


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.parametrize("condition", [boundary.ConstraintPreserving])
def test_a_teukolsky_wave_leaves_through_the_layer_without_a_reflection(condition):
    """Issue #132's acceptance on the documented setup, at 48 and 72 points.

    Box of 12, zone from ``|x_i| = 5``, cube ``|x_i| <= 2.5``, as multiples of
    the amplitude. Measured with the layer:

        n     truncation   after      ratio   ||H|| after   left
        48    7.06e-02     3.18e-02   0.45    3.80e-02      0.129
        72    1.48e-02     5.44e-03   0.37    1.13e-02      0.022
        order 3.9          4.4                3.0

    and, from ``docs/benchmarks.md``, with Bayliss and Turkel's second
    condition on its own: after 3.2e-02 and 1.35e-02, ratio 0.45 and 0.91,
    order 2.1; ``||H||`` 3.4e-02 and 1.6e-02, order 1.9. The bounds below sit
    between the two. Run with ``condition = SecondOrder`` on this same
    octant, the test fails; ``docs/benchmarks.md`` records by how much.
    """
    coarse = _acceptance(_octant_teukolsky(48, condition))
    fine = _acceptance(_octant_teukolsky(72, condition))
    truncation, after, left, hamiltonian = fine
    order = np.log(coarse[1] / fine[1]) / np.log(1.5)
    constraint_order = np.log(coarse[3] / fine[3]) / np.log(1.5)
    assert np.log(coarse[0] / truncation) / np.log(1.5) > 3.7, (coarse, fine)

    # The wave leaves: two orders of magnitude, from 48.
    assert left < 48.0 / 100.0, (coarse, fine)
    # What comes back is below the truncation error and converges as fast.
    assert after < 0.5 * truncation, (coarse, fine)
    assert order > 3.5, (coarse, fine, order)
    # And so does the constraint, though more slowly than the interior.
    assert constraint_order > 2.5, (coarse, fine, constraint_order)
