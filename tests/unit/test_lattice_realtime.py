"""Classical-statistical lattice fields, held to what the integrator really does.

Issue #65 asks for "energy conserved to round-off in flat space". Taken
literally of ``H`` that is not a property a symplectic integrator has, so the
tests below separate the three claims that *are* exact: ``H`` oscillates at
``O(dt^2)`` without drifting, a modified energy is conserved to round-off for
a free field, and a different modified energy is fourth order for any
potential. Each is asserted on its own terms rather than one tolerance being
applied to all three.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.solvers.lattice import (
    Lattice,
    Potential,
    PowerLaw,
    ScalarField,
    State,
    energy,
    laplacian,
    minkowski,
    mode_energies,
    mode_weights,
    modified_energy,
    quadratic_invariant,
    thermal_ensemble,
)

MASS = 0.7


@pytest.fixture(scope="module")
def lattice() -> Lattice:
    return Lattice(size=4.0, points=16, dimensions=3)


@pytest.fixture(scope="module")
def free(lattice) -> ScalarField:
    return ScalarField(lattice, Potential(mass=MASS))


def random_state(lattice: Lattice, seed: int = 0, scale: float = 0.1) -> State:
    generator = np.random.default_rng(seed)
    return State(
        field=generator.normal(size=lattice.shape) * scale,
        momentum=generator.normal(size=lattice.shape) * scale,
        time=0.0,
    )


def spread(values) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.ptp(array) / abs(array[0]))


# --- the lattice is a theory, not an approximation -----------------------


@pytest.mark.parametrize(("size", "points", "dimensions"), [(0.0, 8, 3), (1.0, 1, 3), (1.0, 8, 4)])
def test_lattice_refuses_a_degenerate_grid(size, points, dimensions):
    with pytest.raises(ValueError):
        Lattice(size=size, points=points, dimensions=dimensions)


def test_laplacian_refuses_the_wrong_shape(lattice):
    with pytest.raises(ValueError, match="must have shape"):
        laplacian(np.zeros((4, 4)), lattice)


def test_the_laplacian_annihilates_a_constant(lattice):
    assert np.abs(laplacian(np.full(lattice.shape, 3.5), lattice)).max() < 1e-12


@pytest.mark.parametrize("modes", [1, 3, 7])
def test_a_plane_wave_is_an_eigenvector_of_the_lattice_laplacian(lattice, modes):
    """Eigenvalue ``-(4/h^2) sin^2(k h/2)``, not ``-k^2``.

    At ``modes = 7`` on a 16-point lattice the lattice ``omega^2`` is 0.51
    of the continuum one, so this is not a small correction to the
    continuum answer -- it is a different operator, and the lattice theory
    is defined by it.
    """
    wavenumber = 2.0 * np.pi * modes / lattice.size
    axis = np.arange(lattice.points) * lattice.spacing
    field = np.cos(wavenumber * axis)[:, None, None] * np.ones(lattice.shape)

    eigenvalue = -4.0 * np.sin(wavenumber * lattice.spacing / 2.0) ** 2 / lattice.spacing**2
    assert np.abs(laplacian(field, lattice) - eigenvalue * field).max() < 1e-11


def test_the_dispersion_is_the_lattice_one_and_differs_from_the_continuum(lattice):
    frequency = lattice.dispersion(MASS**2)
    wavenumber = 2.0 * np.pi * 7 / lattice.size
    lattice_value = MASS**2 + 4.0 * np.sin(wavenumber * lattice.spacing / 2.0) ** 2 / (
        lattice.spacing**2
    )
    assert frequency[7, 0, 0] == pytest.approx(lattice_value, rel=1e-12)
    assert frequency[7, 0, 0] < 0.6 * (MASS**2 + wavenumber**2)


@pytest.mark.parametrize("modes", [1, 5, 7])
def test_a_single_mode_follows_the_exact_discrete_oscillator(lattice, free, modes):
    """``x_n = x_0 cos(n theta)`` with ``cos theta = 1 - omega^2 dt^2 / 2``.

    The sharpest statement available about this solver: it pins the lattice
    dispersion *and* the integrator at once, against a closed form, with no
    frequency fitting in between. Fitting a frequency to the time series
    instead gives agreement at 1e-5, which would hide a 1e-9 error in either.
    """
    wavenumber = 2.0 * np.pi * modes / lattice.size
    axis = np.arange(lattice.points) * lattice.spacing
    initial = np.cos(wavenumber * axis)[:, None, None] * np.ones(lattice.shape)

    step = 0.05
    frequency = lattice.dispersion(MASS**2)[modes, 0, 0]
    angle = np.arccos(1.0 - 0.5 * frequency * step**2)

    state = State(field=initial.copy(), momentum=np.zeros(lattice.shape), time=0.0)
    worst = 0.0
    for index in range(1, 501):
        state = free.step(state, step)
        worst = max(worst, float(np.abs(state.field - np.cos(index * angle) * initial).max()))
    assert worst < 1e-12


# --- the three energies, each on its own terms ---------------------------


@pytest.mark.parametrize("step", [0.05, 0.025])
def test_a_free_field_conserves_the_quadratic_invariant_to_round_off(lattice, free, step):
    """The acceptance of issue #65, met literally rather than to a tolerance.

    Every mode of a free lattice field is an independent harmonic
    oscillator, and this is that oscillator's exact Verlet invariant summed
    over modes -- so the answer is round-off, and it does not improve when
    the step shrinks because there was nothing to improve.
    """
    state = random_state(lattice)
    values = []
    for _ in range(int(20.0 / step)):
        state = free.step(state, step)
        values.append(quadratic_invariant(state, free, step))
    assert spread(values) < 1e-13


def test_the_plain_energy_oscillates_but_does_not_drift(lattice):
    """What symplecticity actually buys: bounded error, not a small one.

    ``H`` swings by 1.0e-2 here while the mean of its first tenth and the
    mean of its last tenth differ by 5.6e-6 -- 0.06% of the oscillation. A
    non-symplectic scheme of the same order would show the two comparable.

    The *ratio* is what is asserted, not the drift itself: the drift
    estimate moves between 7e-7 and 6e-6 depending on which window is used,
    because it is dominated by where the oscillation's phase happens to
    land rather than by any trend. The ratio is stable, and it is also the
    statement worth making.
    """
    solver = ScalarField(lattice, Potential(mass=MASS, coupling=5.0))
    state = random_state(lattice)
    values = []
    for _ in range(8000):
        state = solver.step(state, 0.02)
        values.append(energy(state, solver))

    array = np.array(values)
    window = len(array) // 10
    drift = abs(array[:window].mean() / array[-window:].mean() - 1.0)
    assert spread(array) > 1e-4  # it really does oscillate
    assert drift < 1e-2 * spread(array)  # and really does not drift


def test_the_shadow_energy_is_fourth_order_with_a_quartic_coupling(lattice):
    """``O(dt^4)``, where ``H`` and the quadratic invariant are both ``O(dt^2)``.

    Halving the step divides the shadow energy's excursion by 16 and the
    other two by 4, which is the cleanest way to show the three are
    different objects rather than three tolerances on one.
    """
    solver = ScalarField(lattice, Potential(mass=MASS, coupling=5.0))
    results = {}
    for step in (0.04, 0.02):
        state = random_state(lattice)
        rows = {"plain": [], "quadratic": [], "shadow": []}
        for _ in range(int(20.0 / step)):
            state = solver.step(state, step)
            rows["plain"].append(energy(state, solver))
            rows["quadratic"].append(quadratic_invariant(state, solver, step))
            rows["shadow"].append(modified_energy(state, solver, step))
        results[step] = {key: spread(value) for key, value in rows.items()}

    assert results[0.04]["plain"] / results[0.02]["plain"] == pytest.approx(4.0, rel=0.3)
    assert results[0.04]["quadratic"] / results[0.02]["quadratic"] == pytest.approx(4.0, rel=0.3)
    assert results[0.04]["shadow"] / results[0.02]["shadow"] == pytest.approx(16.0, rel=0.3)


def test_the_step_is_time_reversible_to_round_off(lattice):
    """Run forward, flip the momentum, run back: the same state returns.

    True with the quartic coupling on, because reversibility is a property
    of the splitting rather than of the problem being linear.
    """
    solver = ScalarField(lattice, Potential(mass=MASS, coupling=5.0))
    start = random_state(lattice)

    state = start
    for _ in range(300):
        state = solver.step(state, 0.03)
    state = State(field=state.field, momentum=-state.momentum, time=state.time)
    for _ in range(300):
        state = solver.step(state, 0.03)

    assert np.abs(state.field - start.field).max() < 1e-13
    assert np.abs(state.momentum + start.momentum).max() < 1e-13


# --- the mode decomposition ----------------------------------------------


def test_mode_weights_count_the_lattice_sites(lattice):
    assert mode_weights(lattice).sum() == pytest.approx(lattice.sites, abs=1e-9)


def test_the_mode_energies_sum_to_the_energy(lattice, free):
    """A decomposition, not a proxy -- and only with the half-grid weights."""
    state = thermal_ensemble(lattice, mass=MASS, temperature=1.0, seed=2)
    weights = mode_weights(lattice)
    assert np.sum(weights * mode_energies(state, free)) == pytest.approx(
        energy(state, free), rel=1e-13
    )
    assert np.sum(weights * mode_energies(state, free, 0.02)) == pytest.approx(
        quadratic_invariant(state, free, 0.02), rel=1e-13
    )


def test_free_modes_never_exchange_energy_but_coupled_ones_do(lattice, free):
    """Round-off against a factor of five, from the same initial state.

    The corrected per-mode energy is what makes this a statement about
    coupling: the uncorrected one moves by 4e-2 for the *free* field too,
    which is the step size talking, not the modes.
    """
    step = 0.02
    state = thermal_ensemble(lattice, mass=MASS, temperature=1.0, seed=2)

    start = mode_energies(state, free, step)
    evolved, _ = free.run(state, 40.0, 2000)
    assert np.abs(mode_energies(evolved, free, step) - start).max() / start.mean() < 1e-12
    assert np.abs(mode_energies(evolved, free) - mode_energies(state, free)).max() > 1e-3

    coupled = ScalarField(lattice, Potential(mass=MASS, coupling=8.0))
    begin = mode_energies(state, coupled, step)
    mixed, _ = coupled.run(state, 40.0, 2000)
    assert np.abs(mode_energies(mixed, coupled, step) - begin).max() / begin.mean() > 1.0


@pytest.mark.parametrize("seed", [1, 3, 5])
def test_the_thermal_ensemble_equipartitions(lattice, free, seed):
    """Classical equilibrium: ``T`` per degree of freedom, no zero-point floor.

    The tolerance is the ensemble's own scatter, ``sqrt(2/dof)`` = 2.2% for
    this lattice, rather than a number chosen to pass.
    """
    temperature = 1.5
    state = thermal_ensemble(lattice, mass=MASS, temperature=temperature, seed=seed)
    weights = mode_weights(lattice)
    mean = np.sum(weights * mode_energies(state, free)) / weights.sum()
    assert mean / temperature == pytest.approx(1.0, abs=3.0 * np.sqrt(2.0 / weights.sum()))


def test_thermal_ensemble_refuses_a_non_positive_temperature(lattice):
    with pytest.raises(ValueError, match="temperature must be positive"):
        thermal_ensemble(lattice, mass=MASS, temperature=0.0)


# --- the background -------------------------------------------------------


def test_minkowski_is_the_identity():
    background = minkowski()
    assert background.scale(3.0) == 1.0
    assert background.curvature(3.0) == 0.0


def test_a_radiation_era_is_exactly_flat_for_a_massless_field(lattice):
    """``a`` linear in conformal time means ``a'' = 0``: conformal triviality.

    So the energy is conserved to round-off *in an expanding universe*,
    which is a statement about the conformal invariance of a massless
    scalar rather than about the integrator. It is also the one expanding
    case where nothing is approximated, which makes it the right one to
    pin the background machinery against.
    """
    solver = ScalarField(lattice, Potential(mass=0.0), PowerLaw(index=1.0))
    generator = np.random.default_rng(3)
    state = State(
        field=generator.normal(size=lattice.shape) * 0.1,
        momentum=np.zeros(lattice.shape),
        time=1.0,
    )
    step = 0.02
    values = []
    for _ in range(1000):
        state = solver.step(state, step)
        values.append(quadratic_invariant(state, solver, step))
    assert spread(values) < 1e-13


def test_a_matter_era_does_not_conserve_energy(lattice):
    """And it should not: ``a''/a`` is a time-dependent mass, which is work.

    The companion to the test above. Without it, a background that silently
    did nothing would pass everything. Measured at 1.1e-2 against the
    radiation era's 1e-13 -- eleven orders between a background that does
    work on the field and one that cannot.
    """
    solver = ScalarField(lattice, Potential(mass=0.0), PowerLaw(index=2.0))
    generator = np.random.default_rng(3)
    state = State(
        field=generator.normal(size=lattice.shape) * 0.1,
        momentum=np.zeros(lattice.shape),
        time=1.0,
    )
    first = quadratic_invariant(state, solver, 0.02)
    for _ in range(1000):
        state = solver.step(state, 0.02)
    assert abs(quadratic_invariant(state, solver, 0.02) / first - 1.0) > 1e-3


def test_the_rescaled_mass_is_the_conformal_one(lattice):
    """``a^2 m^2 - a''/a``: both pieces, at a time where neither is one."""
    solver = ScalarField(lattice, Potential(mass=2.0), PowerLaw(index=2.0, reference=1.0))
    assert solver.mass_squared(3.0) == pytest.approx(3.0**4 * 4.0 - 2.0 / 9.0, rel=1e-12)


# --- refusals -------------------------------------------------------------


def test_a_negative_quartic_coupling_is_refused():
    with pytest.raises(ValueError, match="unbounded below"):
        Potential(mass=1.0, coupling=-0.5)


def test_step_refuses_a_non_positive_step(lattice, free):
    with pytest.raises(ValueError, match="must be positive"):
        free.step(random_state(lattice), 0.0)


def test_run_refuses_to_integrate_backwards(lattice, free):
    with pytest.raises(ValueError, match="must exceed the initial"):
        free.run(random_state(lattice), -1.0, 10)
