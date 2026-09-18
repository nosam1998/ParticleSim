"""Particle-mesh gravity against the one case with a closed-form answer.

The Zel'dovich pancake is not an approximate solution that the code should
roughly reproduce -- it is *exact* until shell crossing, so most of what is
below compares against closed forms rather than against tolerances picked to
pass. Where the mesh does have an error, it has a predicted one: the
cloud-in-cell force transfer function is ``sinc^4(k h / 2)``, and that is
asserted to three digits rather than bounded.

The exception is the caustic itself, which a plain particle-mesh gets late.
That is measured rather than excused, because it is the entire argument for
the short-range force that issue #78 also asks for.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.cosmo.nbody import (
    Mesh,
    ParticleMesh,
    State,
    caustic_scale_factor,
    deposit,
    gaussian_field,
    growth_exponents,
    interpolate,
    potential_gradient,
    zeldovich_from_field,
    zeldovich_plane_wave,
)

AMPLITUDE = 0.5
START = 0.1
WAVENUMBER = 2.0 * np.pi


def lattice(mesh: Mesh, transverse: int | None = None) -> np.ndarray:
    """The Lagrangian lattice ``zeldovich_plane_wave`` uses, as ``(3, N)``."""
    transverse = mesh.cells if transverse is None else transverse
    along = (np.arange(mesh.cells) + 0.5) * (mesh.size / mesh.cells)
    across = (np.arange(transverse) + 0.5) * (mesh.size / transverse)
    grid = np.meshgrid(along, across, across, indexing="ij")
    return np.stack([value.ravel() for value in grid])


# --- the background, before any mesh is involved -------------------------


def test_growth_exponents_are_the_einstein_de_sitter_modes():
    """``+1`` and ``-3/2``, from the roots rather than from memory."""
    growing, decaying = growth_exponents()
    assert growing == pytest.approx(1.0, abs=1e-12)
    assert decaying == pytest.approx(-1.5, abs=1e-12)


def test_caustic_scale_factor_is_the_reciprocal_amplitude():
    assert caustic_scale_factor(0.4) == pytest.approx(2.5, abs=1e-14)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_caustic_scale_factor_refuses_a_non_positive_amplitude(bad):
    with pytest.raises(ValueError, match="amplitude must be positive"):
        caustic_scale_factor(bad)


@pytest.mark.parametrize(("size", "cells"), [(0.0, 8), (-1.0, 8), (1.0, 1)])
def test_mesh_refuses_a_degenerate_grid(size, cells):
    with pytest.raises(ValueError):
        Mesh(size=size, cells=cells)


# --- deposition ----------------------------------------------------------


def test_deposit_refuses_the_wrong_shape():
    with pytest.raises(ValueError, match=r"shape \(3, N\)"):
        deposit(np.zeros((2, 10)), Mesh(cells=8))


def test_deposit_conserves_mass():
    """The eight weights are three pairs that each sum to one, so ``<delta> = 0``."""
    mesh = Mesh(cells=8)
    positions = np.random.default_rng(0).uniform(0.0, mesh.size, size=(3, 500))
    assert abs(deposit(positions, mesh).mean()) < 1e-14


def test_one_particle_per_cell_deposits_exactly_uniformly():
    """A lattice of the grid's own spacing leaves no density at all."""
    mesh = Mesh(cells=16)
    assert np.abs(deposit(lattice(mesh), mesh)).max() < 1e-14


def test_a_coarser_transverse_lattice_is_a_grid_of_rods():
    """Why ``transverse`` must be a multiple of ``cells``, stated as a number.

    Eight particles across sixteen cells put every particle on a cell
    corner, so cloud-in-cell gives one cell everything and its neighbour
    nothing. The undisplaced lattice -- which should be featureless -- comes
    out at ``delta = 3``.
    """
    mesh = Mesh(cells=16)
    assert np.abs(deposit(lattice(mesh, transverse=8), mesh)).max() == pytest.approx(3.0, abs=1e-12)


# --- the Poisson solve ---------------------------------------------------


def test_poisson_is_exact_on_a_resolved_mode():
    """``delta = sin(kx)`` has ``grad phi = -cos(kx)/k``, to round-off."""
    mesh = Mesh(size=1.0, cells=32)
    x = np.arange(mesh.cells) * mesh.spacing
    density = np.sin(WAVENUMBER * x)[:, None, None] * np.ones(mesh.shape)

    gradient = potential_gradient(density, mesh)
    want = -np.cos(WAVENUMBER * x)[:, None, None] / WAVENUMBER * np.ones(mesh.shape)
    assert np.abs(gradient[0] - want).max() < 1e-15
    assert np.abs(gradient[1]).max() == 0.0
    assert np.abs(gradient[2]).max() == 0.0


def test_poisson_drops_the_zero_mode():
    """A uniform density exerts no force: the box does not accelerate."""
    mesh = Mesh(cells=8)
    gradient = potential_gradient(np.full(mesh.shape, 2.5), mesh)
    assert max(np.abs(component).max() for component in gradient) < 1e-15


# --- interpolation, and the property that kills the self-force -----------


def test_interpolate_is_exact_at_grid_nodes():
    mesh = Mesh(cells=8)
    field = np.random.default_rng(1).normal(size=mesh.shape)
    axes = [np.arange(mesh.cells) * mesh.spacing] * 3
    nodes = np.stack([v.ravel() for v in np.meshgrid(*axes, indexing="ij")])
    assert np.abs(interpolate(field, nodes, mesh) - field.ravel()).max() == 0.0


def test_interpolate_is_the_adjoint_of_deposit():
    """``sum_p F(x_p) == sum_cells F * rho``, which is why there is no self-force."""
    mesh = Mesh(cells=8)
    rng = np.random.default_rng(2)
    positions = rng.uniform(0.0, mesh.size, size=(3, 400))
    field = rng.normal(size=mesh.shape)

    gathered = interpolate(field, positions, mesh).sum()
    mean = positions.shape[1] / mesh.cells**3
    scattered = ((deposit(positions, mesh) + 1.0) * mean * field).sum()
    assert gathered == pytest.approx(scattered, rel=1e-13)


def test_a_lone_particle_does_not_accelerate_itself():
    """Matched assignment and an odd force kernel: the self-force cancels."""
    mesh = Mesh(cells=16)
    lone = np.array([[0.3137], [0.5211], [0.7419]])
    assert np.abs(ParticleMesh(mesh).acceleration(lone)).max() < 1e-13


def test_the_forces_sum_to_zero():
    """Momentum is conserved because the ``k = 0`` mode was removed."""
    mesh = Mesh(cells=16)
    positions = np.random.default_rng(3).uniform(0.0, mesh.size, size=(3, 500))
    acceleration = ParticleMesh(mesh).acceleration(positions)
    assert np.abs(acceleration.sum(axis=1)).max() < 1e-13 * np.abs(acceleration).max()


# --- what the mesh costs, predicted rather than bounded ------------------


@pytest.mark.parametrize("cells", [16, 32])
def test_the_force_transfer_function_is_sinc_to_the_fourth(cells):
    """Deposit and interpolate each apply ``sinc^2(k h/2)``; the force sees both.

    Measured against the exact plane-parallel force ``D A / k`` by
    projection onto the mode, not by a maximum -- a lattice never samples a
    sine's peak, and taking one hides the agreement behind a few percent of
    sampling error.
    """
    mesh = Mesh(size=1.0, cells=cells)
    state = zeldovich_plane_wave(mesh, AMPLITUDE, scale=START)
    acceleration = ParticleMesh(mesh).acceleration(state.positions)[0]

    q = np.repeat((np.arange(cells) + 0.5) / cells, cells * cells)
    basis = np.sin(WAVENUMBER * q)
    measured = -np.dot(acceleration, basis) / np.dot(basis, basis)

    exact = START * AMPLITUDE / WAVENUMBER
    predicted = exact * np.sinc(1.0 / cells) ** 4
    assert measured == pytest.approx(predicted, rel=5e-3)
    assert measured < exact  # the mesh softens, it never sharpens


# --- initial conditions --------------------------------------------------


@pytest.mark.parametrize("bad", [8, 24, 0])
def test_plane_wave_refuses_a_transverse_count_that_is_not_a_multiple(bad):
    with pytest.raises(ValueError, match="positive multiple"):
        zeldovich_plane_wave(Mesh(cells=16), AMPLITUDE, transverse=bad)


def test_the_plane_wave_map_is_the_zeldovich_displacement():
    """``x = q + D psi(q)`` and ``dx/da = psi``, both exactly."""
    mesh = Mesh(size=1.0, cells=8)
    state = zeldovich_plane_wave(mesh, AMPLITUDE, scale=START)
    q = lattice(mesh)

    psi = -(AMPLITUDE / WAVENUMBER) * np.sin(WAVENUMBER * q[0])
    assert np.abs(state.positions[0] - (q[0] + START * psi)).max() < 1e-15
    assert np.abs(state.positions[1] - q[1]).max() == 0.0
    assert np.abs(state.velocities()[0] - psi).max() < 1e-15


def test_zeldovich_from_a_field_reproduces_the_analytic_plane_wave():
    """The generic path and the closed form agree to round-off, not to a percent.

    The linear density of ``psi = -(A/k) sin(kq)`` is ``delta = +A cos(kq)``,
    and the lattice is the grid itself, so no interpolation enters.
    """
    mesh = Mesh(size=1.0, cells=16)
    x = np.arange(mesh.cells) * mesh.spacing
    density = AMPLITUDE * np.cos(WAVENUMBER * x)[:, None, None] * np.ones(mesh.shape)

    state = zeldovich_from_field(mesh, density, scale=START)
    q = np.stack([v.ravel() for v in np.meshgrid(*[x] * 3, indexing="ij")])
    psi = -(AMPLITUDE / WAVENUMBER) * np.sin(WAVENUMBER * q[0])

    assert np.abs(state.positions[0] - (q[0] + START * psi)).max() < 1e-15
    assert np.abs(state.positions[1] - q[1]).max() == 0.0


def test_gaussian_field_is_real_with_zero_mean():
    mesh = Mesh(size=10.0, cells=16)
    field = gaussian_field(mesh, lambda k: k**-2.0, seed=5)
    assert not np.iscomplexobj(field)
    assert abs(field.mean()) < 1e-14


def test_gaussian_field_recovers_its_power_spectrum():
    """``<|delta_k|^2> = V P(k)`` in shells, within the sample scatter.

    A shell of ``m`` modes has a fractional scatter of ``sqrt(2/m)``, so the
    tolerance is the statistics rather than a guess: 54000 modes give 0.6%.
    """
    mesh = Mesh(size=100.0, cells=64)
    field = gaussian_field(mesh, lambda k: 1e3 * k**-2.0, seed=7)

    modes = np.fft.rfftn(field) * mesh.spacing**3
    _, squared = mesh.wavenumbers()
    magnitude = np.sqrt(squared)
    magnitude[0, 0, 0] = 0.0

    shell = (magnitude > 1.2) & (magnitude < 2.0)
    measured = (np.abs(modes[shell]) ** 2 / mesh.size**3).mean()
    target = (1e3 * magnitude[shell] ** -2.0).mean()
    assert measured == pytest.approx(target, rel=3.0 * np.sqrt(2.0 / shell.sum()))


# --- the integrator ------------------------------------------------------


def test_step_refuses_a_non_positive_step():
    mesh = Mesh(cells=8)
    state = zeldovich_plane_wave(mesh, AMPLITUDE)
    with pytest.raises(ValueError, match="must be positive"):
        ParticleMesh(mesh).step(state, 0.0)


def test_run_refuses_to_integrate_backwards():
    mesh = Mesh(cells=8)
    state = zeldovich_plane_wave(mesh, AMPLITUDE, scale=0.5)
    with pytest.raises(ValueError, match="must exceed the initial"):
        ParticleMesh(mesh).run(state, 0.25, 10)


def growth(cells: int, steps: int, final: float = 1.0):
    """``D(a)`` from projecting the displacement onto the mode it started in."""
    mesh = Mesh(size=1.0, cells=cells)
    state = zeldovich_plane_wave(mesh, AMPLITUDE, scale=START)
    q = np.repeat((np.arange(cells) + 0.5) / cells, cells * cells)
    psi = -(AMPLITUDE / WAVENUMBER) * np.sin(WAVENUMBER * q)

    history: list[tuple[float, float]] = []

    def sample(current: State) -> None:
        offset = (current.positions[0] - q + 0.5) % 1.0 - 0.5
        history.append((current.scale, float(np.dot(offset, psi) / np.dot(psi, psi))))

    ParticleMesh(mesh).run(state, final, steps, sample=sample)
    record = np.array(history)
    return float(np.interp(final, record[:, 0], record[:, 1]))


def test_kick_drift_kick_is_second_order():
    """Halving the step quarters the error, with no exact answer needed.

    The three successive differences are compared to each other, so the
    mesh's own error -- which is identical at every step count -- cancels.
    """
    values = [growth(8, steps) for steps in (50, 100, 200, 400)]
    gaps = np.abs(np.diff(values))
    assert gaps[0] / gaps[1] == pytest.approx(4.0, rel=0.25)
    assert gaps[1] / gaps[2] == pytest.approx(4.0, rel=0.25)


@pytest.mark.slow
@pytest.mark.parametrize("cells", [16, 32])
def test_linear_growth_follows_the_scale_factor(cells):
    """``D = a`` for Einstein-de Sitter, short by the mesh's own softening.

    The shortfall is not free, and predicting it is the test. A force
    weakened by ``eps`` moves the growing exponent to ``1 - 3 eps / 5``,
    from ``n^2 + n/2 - 3(1 - eps)/2 = 0``, so over the ``a/a_i = 10`` run
    the growth falls short by ``1 - 10^(-3 eps / 5)`` -- with ``eps`` the
    ``sinc^4`` transfer function above and nothing fitted.

    What is asserted is the *shortfall*, not ``D`` itself, because a
    fractional tolerance on ``D`` gets vacuously easy as the mesh refines --
    the deficit shrinks with the mesh while the tolerance does not, so
    returning ``a`` unchanged eventually passes. Held this way the same
    11% agreement appears at both resolutions -- the shortfall is 0.0306
    against 0.0345 predicted at ``16^3``, and 0.0079 against 0.0088 at
    ``32^3``. The missing tenth is real and is the harmonics: by ``a = 1``
    the pancake has ``delta ~ 1`` and is no longer one mode, and the mesh
    damps ``2k`` and ``3k`` by more than it damps ``k``. A single-mode
    transfer function cannot account for that, and 15% is how much it
    misses by rather than a tolerance chosen to pass.
    """
    shortfall = 1.0 - np.sinc(1.0 / cells) ** 4
    measured = 1.0 - growth(cells, 200)
    predicted = 1.0 - (1.0 / START) ** (-0.6 * shortfall)
    assert measured == pytest.approx(predicted, rel=0.15)


@pytest.mark.benchmark
def test_the_caustic_is_late_and_that_is_the_case_for_a_short_range_force():
    """A plain particle-mesh forms the pancake late, by a measured amount.

    Issue #78 asks for the caustic time within 2%. This is 13% at ``32^3``
    and 34% at ``16^3`` -- roughly a factor 2.6 per doubling, so a mesh
    alone would need something like ``256^3`` to get there. The reason is
    not the ``sinc^4`` softening asserted above, which is 0.6% at ``32^3``
    and which deconvolving the window does not remove: by the time the
    pancake is thinner than a cell the mesh has no information left about
    it. That is what the short-range half of TreePM supplies, and this test
    exists to keep the number honest rather than to pass.
    """
    cells, mesh = 32, Mesh(size=1.0, cells=32)
    state = zeldovich_plane_wave(mesh, AMPLITUDE, scale=START)
    q = (np.arange(cells) + 0.5) / cells
    history: list[tuple[float, float]] = []

    def sample(current: State) -> None:
        slabs = current.positions[0].reshape(cells, -1)
        offset = ((slabs - q[:, None] + 0.5) % 1.0 - 0.5).mean(axis=1)
        history.append(
            (current.scale, float((1.0 / cells + np.diff(offset, prepend=offset[-1])).min()))
        )

    ParticleMesh(mesh).run(state, 2.4, 300, sample=sample)
    record = np.array(history)

    crossed = np.argmax(record[:, 1] <= 0.0)
    assert crossed > 0, "no shell crossing was reached"
    (late, below), (early, above) = record[crossed], record[crossed - 1]
    caustic = early + (late - early) * above / (above - below)

    exact = caustic_scale_factor(AMPLITUDE)
    assert caustic > exact  # never early
    assert caustic / exact - 1.0 == pytest.approx(0.133, abs=0.02)
