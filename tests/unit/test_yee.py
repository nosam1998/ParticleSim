"""Yee FDTD Maxwell solver (design doc Section 5.4, Milestone 2)."""

import numpy as np
import pytest

from particlesim.solvers.pic import (
    Conducting,
    Fields,
    PerfectlyMatchedLayer,
    Periodic,
    Vacuum,
    YeeGrid,
    YeeSolver,
    plane_wave,
    yee_frequency,
)

POLARIZATIONS = ("TM", "TE")


def _components(fields: Fields):
    return ("Dx", "Dy", "Dz", "Bx", "By", "Bz")


def _max_difference(a: Fields, b: Fields) -> float:
    return max(float(np.abs(getattr(a, c) - getattr(b, c)).max()) for c in _components(a))


def _amplitude(f: Fields) -> float:
    return max(float(np.abs(getattr(f, c)).max()) for c in _components(f))


# --- the grid ---------------------------------------------------------------


def test_three_dimensions_are_refused_rather_than_half_supported():
    with pytest.raises(ValueError, match="one- and two-dimensional"):
        YeeGrid((8, 8, 8), (0.1, 0.1, 0.1))


def test_grid_validates_its_arguments():
    with pytest.raises(ValueError, match="same length"):
        YeeGrid((8, 8), (0.1,))
    with pytest.raises(ValueError, match="at least two cells"):
        YeeGrid((1,), (0.1,))
    with pytest.raises(ValueError, match="positive"):
        YeeGrid((8,), (-0.1,))


def test_courant_limit_matches_the_cfl_formula():
    grid = YeeGrid((32, 48), (0.02, 0.05))
    expected = 1.0 / np.sqrt(1 / 0.02**2 + 1 / 0.05**2)
    assert grid.courant_limit == pytest.approx(expected)
    # One dimension reduces to the spacing itself, the "magic time step".
    assert YeeGrid((32,), (0.02,)).courant_limit == pytest.approx(0.02)


def test_a_step_above_the_courant_limit_is_refused():
    grid = YeeGrid((32,), (0.02,))
    with pytest.raises(ValueError, match="Courant limit"):
        YeeSolver(grid, dt=grid.courant_limit * 1.01)


# --- the dispersion relation, which is the acceptance criterion -------------


@pytest.mark.parametrize("polarization", POLARIZATIONS)
def test_exact_discrete_plane_wave_in_one_dimension(polarization):
    """The acceptance criterion for issue #22's sibling: a vacuum plane wave
    propagates with the Yee dispersion relation.

    Checked in the strongest available form. ``plane_wave`` builds the
    scheme's *own* eigenmode rather than a sampled continuum wave, so the
    update must reproduce it step for step at round-off. A solver that got
    the staggering or the amplitudes wrong would drift immediately, and one
    that reproduced the continuum relation instead of the lattice's would
    drift at the 0.3% this configuration separates them by.
    """
    nx, dx, steps = 64, 0.05, 200
    grid = YeeGrid((nx,), (dx,))
    dt = 0.5 * grid.courant_limit
    k = (2 * np.pi * 3 / (nx * dx),)
    omega = yee_frequency(k, grid.spacing, dt)

    solver = YeeSolver(grid, dt=dt)
    evolved = solver.run(plane_wave(grid, k, dt, polarization=polarization), steps)
    expected = plane_wave(grid, k, dt, polarization=polarization, phase=-omega * steps * dt)
    assert _max_difference(evolved, expected) < 1e-12
    # Sampling a cosine on a lattice rarely lands on its crest, so the
    # largest sampled value sits just under the amplitude. What matters is
    # that it has not decayed.
    assert _amplitude(evolved) > 0.99

    # The lattice frequency is not the continuum one, so the test above is
    # not satisfied by any wave solver that happens to be stable.
    assert omega / k[0] == pytest.approx(0.9973, abs=1e-3)


@pytest.mark.parametrize("polarization", POLARIZATIONS)
@pytest.mark.parametrize("modes", [(3, 2), (5, 0), (0, 4)])
def test_exact_discrete_plane_wave_in_two_dimensions(polarization, modes):
    shape, spacing, steps = (48, 40), (0.05, 0.06), 150
    grid = YeeGrid(shape, spacing)
    dt = 0.5 * grid.courant_limit
    k = tuple(2 * np.pi * m / (n * d) for m, n, d in zip(modes, shape, spacing, strict=True))
    omega = yee_frequency(k, spacing, dt)

    solver = YeeSolver(grid, dt=dt)
    evolved = solver.run(plane_wave(grid, k, dt, polarization=polarization), steps)
    expected = plane_wave(grid, k, dt, polarization=polarization, phase=-omega * steps * dt)
    assert _max_difference(evolved, expected) < 1e-11


@pytest.mark.benchmark
def test_lattice_frequency_approaches_the_continuum_as_the_grid_refines():
    """The dispersion error is second order in the spacing.

    A first-order result would mean the staggering has been lost somewhere;
    an exactly zero one would mean the continuum relation is being returned
    instead of the lattice's.
    """
    wavelength = 1.0
    k = 2 * np.pi / wavelength
    errors = []
    for cells_per_wavelength in (8, 16, 32, 64):
        dx = wavelength / cells_per_wavelength
        dt = 0.5 * dx
        errors.append(abs(yee_frequency((k,), (dx,), dt) / k - 1.0))
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert all(1.9 < o < 2.1 for o in orders), f"orders were {orders}"


def test_a_mode_too_fine_for_the_lattice_is_refused():
    """Past the lattice's cutoff the scheme is evanescent, not inaccurate,
    so returning a frequency would be returning a fiction.

    Below the Courant limit no mode can reach the cutoff -- that is what the
    limit means -- so this is reachable only by asking for a frequency at a
    step the solver itself would refuse. Answering anyway is the failure
    being guarded against: the number would look ordinary.
    """
    dx = 0.05
    nyquist = np.pi / dx
    assert yee_frequency((nyquist,), (dx,), dx) > 0  # exactly at the limit
    with pytest.raises(ValueError, match="does not propagate"):
        yee_frequency((nyquist,), (dx,), 1.5 * dx)


def test_plane_wave_validates_its_arguments():
    grid = YeeGrid((32,), (0.05,))
    with pytest.raises(ValueError, match="polarization"):
        plane_wave(grid, (1.0,), 0.02, polarization="TEM")
    with pytest.raises(ValueError, match="needs 1 components"):
        plane_wave(grid, (1.0, 1.0), 0.02)


# --- conservation -----------------------------------------------------------


def test_periodic_propagation_conserves_energy_to_round_off():
    grid = YeeGrid((64,), (0.05,))
    dt = 0.5 * grid.courant_limit
    solver = YeeSolver(grid, dt=dt, boundary=Periodic())
    fields = plane_wave(grid, (2 * np.pi * 3 / (64 * 0.05),), dt)
    before = solver.energy(fields)
    after = solver.energy(solver.run(fields, 5000))
    assert abs(after / before - 1.0) < 1e-12


def test_gauss_law_is_preserved_exactly_in_vacuum():
    """``div D`` can only change through the current.

    Both ``div`` and the ``curl`` that drives ``D`` use backward differences,
    and backward differences along different axes commute, so the discrete
    ``div curl = 0`` holds identically rather than approximately. This is the
    property issue #31's charge-conserving deposition will lean on, so it is
    worth pinning before anything leans on it.
    """
    rng = np.random.default_rng(7)
    grid = YeeGrid((32, 24), (0.05, 0.04))
    solver = YeeSolver(grid)
    fields = Fields(*(rng.normal(size=grid.shape) for _ in range(6)))
    before = solver.divergence_D(fields)
    after = solver.divergence_D(solver.run(fields, 50))
    assert np.abs(after - before).max() < 1e-12


# --- boundaries -------------------------------------------------------------


def test_conducting_walls_hold_an_exact_standing_mode():
    """A perfect conductor puts the cavity's own modes at the discrete
    frequencies of the box, and holds them without loss."""
    nx, dx, steps = 201, 0.02, 2000
    grid = YeeGrid((nx,), (dx,))
    dt = 0.5 * grid.courant_limit
    solver = YeeSolver(grid, dt=dt, boundary=Conducting())

    cavity = (nx - 1) * dx  # walls sit on the first and last node
    k = 4 * np.pi / cavity
    omega = yee_frequency((k,), (dx,), dt)
    x = grid.coordinates((0.0,))[0]
    x_half = grid.coordinates((0.5,))[0]
    zero = np.zeros(nx)
    fields = Fields(
        zero.copy(),
        zero.copy(),
        np.sin(k * x),
        zero.copy(),
        -np.cos(k * x_half) * np.sin(omega * dt / 2),
        zero.copy(),
    )

    worst, wall = 0.0, 0.0
    for i in range(1, steps + 1):
        fields = solver.step(fields)
        worst = max(worst, float(np.abs(fields.Dz - np.sin(k * x) * np.cos(omega * i * dt)).max()))
        wall = max(wall, abs(float(fields.Dz[0])), abs(float(fields.Dz[-1])))
    assert worst < 1e-12
    assert wall == 0.0  # the condition is imposed, not approximated


def test_conducting_energy_oscillation_is_bounded_and_halves_with_the_step():
    """Distinguishes the leapfrog's half-step offset from a lossy wall.

    Reading a leapfrog at one instant puts E and B half a step out of phase,
    which makes the energy oscillate by about ``omega dt / 4``. A wall that
    leaked would instead drift in one direction and would not care about the
    step size.
    """
    nx, dx = 201, 0.02
    grid = YeeGrid((nx,), (dx,))
    x, x_half = grid.coordinates((0.0,))[0], grid.coordinates((0.5,))[0]
    k = 4 * np.pi / ((nx - 1) * dx)

    spreads = []
    for courant in (0.5, 0.25):
        dt = courant * grid.courant_limit
        solver = YeeSolver(grid, dt=dt, boundary=Conducting())
        omega = yee_frequency((k,), (dx,), dt)
        zero = np.zeros(nx)
        fields = Fields(
            zero.copy(),
            zero.copy(),
            np.sin(k * x),
            zero.copy(),
            -np.cos(k * x_half) * np.sin(omega * dt / 2),
            zero.copy(),
        )
        reference = solver.energy(fields)
        seen = []
        for _ in range(2000):
            fields = solver.step(fields)
            seen.append(solver.energy(fields) / reference)
        spreads.append(max(seen) - min(seen))

    assert spreads[0] < 0.05  # bounded
    assert spreads[1] == pytest.approx(spreads[0] / 2, rel=0.1)  # first order in dt


@pytest.mark.parametrize("thickness", (8, 12, 20))
def test_the_layer_absorbs_and_absorbs_better_when_thicker(thickness):
    """A pulse sent into the layer must not come back.

    The threshold falls with thickness because a perfectly matched layer is
    backed by a conductor, so the residual is what survives the round trip
    through the graded conductivity. A sponge layer, which tapers the fields
    instead, reflects at its own edge and does not improve this way.
    """
    nx, dx = 400, 0.02
    grid = YeeGrid((nx,), (dx,))
    dt = 0.5 * grid.courant_limit
    solver = YeeSolver(grid, dt=dt, boundary=PerfectlyMatchedLayer(thickness=thickness))

    x = grid.coordinates((0.0,))[0]
    length = nx * dx
    incident = np.exp(-(((x - length / 2) / 0.25) ** 2))
    zero = np.zeros(nx)
    fields = Fields(
        zero.copy(), zero.copy(), incident.copy(), zero.copy(), zero.copy(), zero.copy()
    )

    residual = 0.0
    for _ in range(int(2.5 * length / dt)):
        fields = solver.step(fields)
        if fields.time > 0.75 * length:
            residual = max(residual, float(np.abs(fields.Dz).max()))
    limits = {8: 1e-4, 12: 3e-5, 20: 5e-6}
    assert residual / float(incident.max()) < limits[thickness]


def test_a_layer_thicker_than_its_axis_is_refused():
    grid = YeeGrid((16,), (0.05,))
    with pytest.raises(ValueError, match="too few for two layers"):
        YeeSolver(grid, boundary=PerfectlyMatchedLayer(thickness=10))


def test_layer_validates_its_arguments():
    with pytest.raises(ValueError, match="at least one cell"):
        PerfectlyMatchedLayer(thickness=0)
    with pytest.raises(ValueError, match="strictly between zero and one"):
        PerfectlyMatchedLayer(reflection=1.5)


def test_resetting_a_layer_clears_its_convolution_history():
    grid = YeeGrid((64,), (0.02,))
    layer = PerfectlyMatchedLayer(thickness=8)
    solver = YeeSolver(grid, boundary=layer)
    x = grid.coordinates((0.0,))[0]
    zero = np.zeros(64)
    fields = Fields(
        zero.copy(),
        zero.copy(),
        np.exp(-(((x - 0.6) / 0.1) ** 2)),
        zero.copy(),
        zero.copy(),
        zero.copy(),
    )
    solver.run(fields, 40)
    assert layer._psi, "the layer should have accumulated history"
    layer.reset()
    assert not layer._psi


# --- the constitutive hook --------------------------------------------------


class UniformDielectric:
    """``E = D / permittivity``, ``H = B``: the simplest non-vacuum medium.

    Enough to show the hook is consulted rather than decorative, because it
    changes the wave speed to ``1 / sqrt(permittivity)`` and therefore the
    lattice dispersion relation along with it.
    """

    def __init__(self, permittivity: float):
        self.permittivity = float(permittivity)

    def electric(self, D, B):
        return tuple(d / self.permittivity for d in D)

    def magnetic(self, D, B):
        return B


def test_vacuum_relation_is_the_identity():
    medium = Vacuum()
    D = (np.array([1.0]), np.array([2.0]), np.array([3.0]))
    B = (np.array([4.0]), np.array([5.0]), np.array([6.0]))
    assert medium.electric(D, B) is D
    assert medium.magnetic(D, B) is B


def test_a_dielectric_slows_the_wave_by_the_refractive_index():
    """Exact discrete mode for a uniform permittivity.

    Repeating the amplitude derivation with ``E = D / eps`` gives the same
    mode with ``eps`` dividing the electric amplitude and the dispersion
    relation picking up ``1 / eps`` on its right-hand side. That the solver
    holds it to round-off is what says the hook is inside the update rather
    than beside it.
    """
    nx, dx, steps = 64, 0.05, 200
    permittivity = 4.0
    grid = YeeGrid((nx,), (dx,))
    dt = 0.5 * grid.courant_limit
    k = 2 * np.pi * 3 / (nx * dx)

    # sin(omega dt / 2)^2 / dt^2 = (1 / eps) sin(k dx / 2)^2 / dx^2
    s = dt * np.sin(k * dx / 2) / (dx * np.sqrt(permittivity))
    omega = 2 * np.arcsin(s) / dt
    assert omega / k == pytest.approx(1 / np.sqrt(permittivity), rel=5e-3)

    x, x_half = grid.coordinates((0.0,))[0], grid.coordinates((0.5,))[0]
    zero = np.zeros(nx)
    By0 = -(1.0 / permittivity) * (dt / dx) * np.sin(k * dx / 2) / np.sin(omega * dt / 2)

    def state(t):
        return Fields(
            zero.copy(),
            zero.copy(),
            np.cos(k * x - omega * t),
            zero.copy(),
            By0 * np.cos(k * x_half - omega * (t - dt / 2)),
            zero.copy(),
        )

    solver = YeeSolver(grid, dt=dt, medium=UniformDielectric(permittivity))
    evolved = solver.run(state(0.0), steps)
    assert _max_difference(evolved, state(steps * dt)) < 1e-12
