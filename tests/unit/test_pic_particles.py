"""Macro-particles, pushers and charge-conserving deposition (Section 5.4, M2)."""

import numpy as np
import pytest

from particlesim.solvers.pic import (
    Fields,
    Species,
    YeeGrid,
    YeeSolver,
    advance,
    boris,
    common_window,
    deposit_charge,
    esirkepov_current,
    gather,
    gauss_residual,
    lorentz_factor,
    push_momentum,
    shape,
    vay,
    window,
)

ORDERS = (1, 2)


# --- shape functions --------------------------------------------------------


@pytest.mark.parametrize("order", ORDERS)
def test_shapes_are_a_partition_of_unity_with_the_right_centroid(order):
    """Charge is neither created nor displaced by the deposition.

    The weights summing to one says the macro-particle's charge is conserved
    by the scatter. The first moment landing on the particle says it is
    deposited where the particle is, which is what makes the force on a
    particle in a uniform field independent of where in its cell it sits.
    """
    xi = np.linspace(3.01, 4.99, 41)
    indices, weights = window(xi, order)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-15)
    np.testing.assert_allclose((indices * weights).sum(axis=1), xi, atol=1e-14)


@pytest.mark.parametrize("order", ORDERS)
def test_shapes_are_non_negative_and_compactly_supported(order):
    offsets = np.linspace(-4, 4, 401)
    values = shape(offsets, order)
    assert (values >= 0).all()
    support = (order + 1) / 2.0
    assert (values[np.abs(offsets) > support] == 0).all()


def test_unsupported_shape_order_is_refused():
    with pytest.raises(ValueError, match="order must be one of"):
        shape(np.array([0.0]), 3)
    with pytest.raises(ValueError, match="order must be one of"):
        window(np.array([1.0]), 0)


@pytest.mark.parametrize("order", ORDERS)
def test_gather_reproduces_a_linear_field_exactly(order):
    """Any shape of order one or more interpolates linearly exactly, so this
    separates a correct gather from one off by a cell or a half-cell."""
    n, dx = 32, 0.1
    nodes = np.arange(n) * dx
    field = 2.0 * nodes + 1.0
    positions = np.linspace(0.5, 2.5, 9)[:, None]
    got = gather(field, positions, (dx,), (0.0, 0.0, 0.0), order)
    np.testing.assert_allclose(got, 2.0 * positions[:, 0] + 1.0, atol=1e-13)


@pytest.mark.parametrize("order", ORDERS)
def test_gather_respects_the_half_cell_stagger(order):
    """A component stored at ``i + 1/2`` must be read against that lattice.

    Ignoring the offset shifts the force by half a cell. That looks like a
    small phase error and behaves like a systematic drift, so it is worth a
    test that fails loudly rather than slightly.
    """
    n, dx = 32, 0.1
    half_nodes = (np.arange(n) + 0.5) * dx
    field = 3.0 * half_nodes - 2.0
    positions = np.linspace(0.5, 2.5, 9)[:, None]
    exact = 3.0 * positions[:, 0] - 2.0
    np.testing.assert_allclose(
        gather(field, positions, (dx,), (0.5, 0.0, 0.0), order), exact, atol=1e-13
    )
    shifted = gather(field, positions, (dx,), (0.0, 0.0, 0.0), order)
    assert np.abs(shifted - exact).max() == pytest.approx(3.0 * dx / 2, rel=1e-6)


# --- pushers ----------------------------------------------------------------


@pytest.mark.parametrize("pusher", (boris, vay))
def test_a_static_magnetic_field_does_no_work(pusher):
    """The magnetic part is a rotation, so it cannot change the energy.

    Boris makes that exact rather than approximate, which is why it survives
    runs of millions of steps. Checked over twenty thousand.
    """
    u = np.array([[0.5, 0.0, 0.0]])
    E = np.zeros((1, 3))
    B = np.array([[0.0, 0.0, 1.0]])
    start = lorentz_factor(u)[0]
    for _ in range(20000):
        u = pusher(u, E, B, 1.0, 0.05)
    assert abs(lorentz_factor(u)[0] / start - 1.0) < 1e-12


def test_boris_rotates_by_exactly_the_angle_its_construction_implies():
    """The discrete gyrofrequency, not the continuum one.

    Boris rotates by ``theta`` per step with ``tan(theta/2) = q B dt / (2 m
    gamma)``, which tends to ``q B dt / (m gamma)`` as the step shrinks but
    is not equal to it. Choosing ``dt`` so that ``N`` steps make exactly one
    turn therefore has a closed form, and the particle must come back to
    where it started to round-off rather than to the step's accuracy. A
    pusher rotating by the continuum angle instead would miss by the
    difference between the two, which this separates.
    """
    qm, B_z, speed, turns = 1.0, 1.0, 0.01, 1000
    start = np.array([[speed, 0.0, 0.0]])
    gamma = lorentz_factor(start)[0]
    dt = (2.0 * gamma / (qm * B_z)) * np.tan(np.pi / turns)
    # The continuum step for the same angle differs, so this is not circular.
    assert dt != pytest.approx(2.0 * gamma * np.pi / (qm * B_z * turns), rel=1e-9)

    E, B = np.zeros((1, 3)), np.array([[0.0, 0.0, B_z]])
    u = start.copy()
    for _ in range(turns):
        u = boris(u, E, B, qm, dt)
    assert np.abs(u - start).max() < 1e-14


def test_vay_holds_the_crossed_field_equilibrium_and_boris_does_not():
    """The case the two pushers disagree on.

    A particle moving at the ``E x B`` drift satisfies ``E + v x B = 0``
    exactly, so it should never accelerate. Vay's ordering is built to
    reproduce that and does so to round-off at any step size. Boris evaluates
    the rotation's Lorentz factor between its two electric half-kicks, which
    does not, and its error grows with the step.
    """
    E_y, B_z = 0.5, 1.0
    drift = E_y / B_z
    gamma = 1.0 / np.sqrt(1.0 - drift**2)
    start = np.array([[gamma * drift, 0.0, 0.0]])
    E, B = np.array([[0.0, E_y, 0.0]]), np.array([[0.0, 0.0, B_z]])

    boris_errors = []
    for dt in (0.05, 0.2, 1.0):
        u = start.copy()
        for _ in range(2000):
            u = vay(u, E, B, 1.0, dt)
        assert abs(u[0, 0] / lorentz_factor(u)[0] - drift) < 1e-12

        u = start.copy()
        for _ in range(2000):
            u = boris(u, E, B, 1.0, dt)
        boris_errors.append(abs(u[0, 0] / lorentz_factor(u)[0] - drift))

    assert boris_errors[-1] > 1e-3  # plainly wrong at a large step
    assert boris_errors == sorted(boris_errors)  # and worse the larger the step


def test_unknown_pusher_is_refused():
    species = Species.create(-1.0, 1.0, [[0.5]], [[0.1, 0.0, 0.0]])
    with pytest.raises(ValueError, match="unknown pusher"):
        push_momentum(species, np.zeros((1, 3)), np.zeros((1, 3)), 0.01, scheme="rk4")


# --- the species container --------------------------------------------------


def test_species_validates_its_shapes():
    with pytest.raises(ValueError, match="mass must be positive"):
        Species.create(-1.0, 0.0, [[0.5]])
    with pytest.raises(ValueError, match="three momentum components"):
        Species(-1.0, 1.0, np.zeros((2, 1)), np.zeros((2, 2)), np.ones(2))
    with pytest.raises(ValueError, match=r"weight must be \(N,\)"):
        Species(-1.0, 1.0, np.zeros((2, 1)), np.zeros((2, 3)), np.ones(3))


def test_a_one_dimensional_species_still_carries_three_momenta():
    species = Species.create(-1.0, 1.0, [[0.5], [0.7]])
    assert species.ndim == 1
    assert species.momentum.shape == (2, 3)


def test_weight_changes_the_current_but_not_the_trajectory():
    """Weight is how many real particles a macro-particle stands for. It
    multiplies charge and mass together, so it cancels out of the force."""
    E, B = np.array([[0.1, 0.0, 0.0]]), np.array([[0.0, 0.0, 0.5]])
    light = Species.create(-1.0, 1.0, [[0.5]], [[0.2, 0.1, 0.0]], weight=1.0)
    heavy = Species.create(-1.0, 1.0, [[0.5]], [[0.2, 0.1, 0.0]], weight=1e6)
    a = push_momentum(light, E, B, 0.01)
    b = push_momentum(heavy, E, B, 0.01)
    np.testing.assert_array_equal(a.momentum, b.momentum)

    grid = YeeGrid((16,), (0.1,))
    assert np.abs(deposit_charge(grid, heavy)).max() == pytest.approx(
        1e6 * np.abs(deposit_charge(grid, light)).max()
    )


# --- deposition -------------------------------------------------------------


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize(("shape_", "spacing"), [((40,), (0.1,)), ((24, 20), (0.1, 0.08))])
def test_deposited_current_satisfies_discrete_continuity_identically(order, shape_, spacing):
    """The whole point of Esirkepov's scheme.

    ``(rho^{n+1} - rho^n)/dt + div J = 0`` on the grid, as an algebraic
    identity rather than an approximation, for any order and any trajectory
    inside a cell. Without it a particle-in-cell run accumulates spurious
    space charge and needs a Poisson correction; with it, Gauss's law cannot
    drift at all.
    """
    rng = np.random.default_rng(3)
    grid = YeeGrid(shape_, spacing)
    ndim = len(shape_)
    count, dt = 200, 0.02

    before = rng.uniform(0.5, 1.5, size=(count, ndim))
    species = Species.create(
        -1.0,
        1.0,
        before.copy(),
        rng.normal(0, 0.3, size=(count, 3)),
        weight=rng.uniform(0.5, 1.5, count),
    )
    rho_before = deposit_charge(grid, species, order)
    moved = species.with_position(before + dt * species.velocity[:, :ndim])
    rho_after = deposit_charge(grid, moved, order)
    Jx, Jy, _ = esirkepov_current(grid, moved, before, dt, order)

    divergence = (Jx - np.roll(Jx, 1, axis=0)) / spacing[0]
    if ndim == 2:
        divergence = divergence + (Jy - np.roll(Jy, 1, axis=1)) / spacing[1]
    residual = (rho_after - rho_before) / dt + divergence
    scale = max(np.abs((rho_after - rho_before) / dt).max(), np.abs(divergence).max())
    assert np.abs(residual).max() / scale < 1e-12


def test_a_particle_crossing_more_than_a_cell_is_refused():
    """Not a tolerance to widen. Past one cell the continuity identity stops
    holding on the window, and the charge it fails to account for shows up as
    a growing Gauss violation rather than as an error."""
    with pytest.raises(ValueError, match="moved 1.500 cells"):
        common_window(np.array([2.0]), np.array([3.5]), 1)


def test_deposited_charge_totals_the_charge_present():
    grid = YeeGrid((20, 16), (0.1, 0.05))
    rng = np.random.default_rng(5)
    count = 300
    species = Species.create(
        -2.0,
        1.0,
        rng.uniform(0, 1, size=(count, 2)) * np.array(grid.extent),
        weight=rng.uniform(0.5, 2.0, count),
    )
    cell = float(np.prod(grid.spacing))
    total = deposit_charge(grid, species, 2).sum() * cell
    assert total == pytest.approx(-2.0 * species.weight.sum(), rel=1e-12)


# --- the full cycle ---------------------------------------------------------


def _random_run(ndim, shape_, spacing, order, steps, dtype=np.float64, seed=11):
    rng = np.random.default_rng(seed)
    grid = YeeGrid(shape_, spacing)
    solver = YeeSolver(grid, courant=0.5)
    count = 200
    species = Species.create(
        -1.0,
        1.0,
        rng.uniform(0, 1, size=(count, ndim)) * np.array(grid.extent),
        rng.normal(0, 0.2, size=(count, 3)),
        weight=0.01,
        dtype=dtype,
    )
    fields = Fields.zeros(grid)
    reference = gauss_residual(solver, fields, species, order)
    worst = 0.0
    for _ in range(steps):
        fields, species = advance(solver, fields, species, order=order)
        drift = gauss_residual(solver, fields, species, order) - reference
        worst = max(worst, float(np.abs(drift).max()))
    return worst / float(np.abs(reference).max())


@pytest.mark.parametrize("order", ORDERS)
def test_gauss_law_does_not_drift_over_a_short_run(order):
    assert _random_run(1, (64,), (0.05,), order, 200) < 1e-13


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.parametrize("order", ORDERS)
def test_gauss_law_is_preserved_to_round_off_over_ten_thousand_steps(order):
    """The acceptance criterion for issue #31.

    Never solved for and never corrected. ``div curl`` is identically zero on
    the Yee lattice and the deposited current satisfies discrete continuity
    identically, so ``div D - rho`` is an exact invariant of the cycle and
    what is left is the accumulation of floating-point round-off.
    """
    assert _random_run(1, (64,), (0.05,), order, 10000) < 1e-11


@pytest.mark.slow
def test_gauss_law_holds_in_two_dimensions():
    assert _random_run(2, (32, 28), (0.05, 0.06), 1, 1000) < 1e-12


def test_single_precision_storage_keeps_double_precision_accumulators():
    """Single-precision particles halve the memory a large run needs.

    The accumulators stay double whatever the particles are stored in,
    because the conservation identity is a statement about exact sums: a
    single-precision grid loses it to cancellation long before the
    trajectories suffer.
    """
    grid = YeeGrid((32,), (0.05,))
    species = Species.create(
        -1.0,
        1.0,
        [[0.5], [0.9]],
        [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]],
        dtype=np.float32,
    )
    assert species.position.dtype == np.float32
    assert species.momentum.dtype == np.float32

    assert deposit_charge(grid, species).dtype == np.float64
    moved = species.with_position(species.position + 0.001)
    for component in esirkepov_current(grid, moved, np.asarray(species.position, float), 0.01):
        assert component.dtype == np.float64

    # The invariant still holds, at the precision the positions allow, and
    # that precision is the cost: the same run in double precision holds
    # Gauss's law to 1e-13, and in single to about 3e-5. Eight orders of
    # magnitude is the price of halving the particle memory, which is worth
    # paying for a trajectory and not for a conservation law.
    single = _random_run(1, (64,), (0.05,), 1, 200, dtype=np.float32)
    double = _random_run(1, (64,), (0.05,), 1, 200, dtype=np.float64)
    assert single < 1e-3
    assert double < 1e-13
    assert single > 1e3 * double
