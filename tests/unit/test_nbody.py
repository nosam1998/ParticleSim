"""Particle-mesh gravity against the one case with a closed-form answer.

The Zel'dovich pancake is not an approximate solution that the code should
roughly reproduce -- it is *exact* until shell crossing, so most of what is
below compares against closed forms rather than against tolerances picked to
pass. Where the mesh does have an error, it has a predicted one: the
cloud-in-cell force transfer function is ``sinc(k h)``, and that is
asserted to round-off rather than bounded.

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


def test_the_half_cell_shift_is_exact():
    """``shift=0.5`` evaluates the gradient at the cell centres, not near them.

    It is a phase factor applied to a field that is already in Fourier
    space, so it costs nothing and loses nothing -- unlike reading a
    node-centred field at the centres by cloud-in-cell, which would apply a
    ``cos(k h/2)`` window.
    """
    mesh = Mesh(size=1.0, cells=32)
    x = np.arange(mesh.cells) * mesh.spacing
    density = np.sin(WAVENUMBER * x)[:, None, None] * np.ones(mesh.shape)

    shifted = potential_gradient(density, mesh, shift=0.5)[0]
    centres = (np.arange(mesh.cells) + 0.5) * mesh.spacing
    want = -np.cos(WAVENUMBER * centres)[:, None, None] / WAVENUMBER * np.ones(mesh.shape)
    assert np.abs(shifted - want).max() < 1e-15


@pytest.mark.parametrize(("modes", "expected"), [(1, 0.167), (4, 0.707)])
def test_a_node_aligned_lattice_deposits_spurious_harmonics(modes, expected):
    """Why the Lagrangian lattice is offset half a cell, as a number.

    A particle exactly on a grid point gives it all of its mass, and a
    displacement ``s`` moves ``|s|/h`` to the neighbour *in the direction of
    travel*. The response depends on ``|s|``, not ``s``, so it is rectified
    and a single displaced mode deposits harmonics of itself. At any other
    phase the cloud-in-cell weights are differentiable in ``s`` and the
    harmonics vanish identically -- which is the whole reason
    :func:`zeldovich_from_field` does not put its particles on the grid it
    solved on, convenient though that would be.
    """
    cells, amplitude = 32, 0.02
    mesh = Mesh(size=1.0, cells=cells)
    wavenumber = WAVENUMBER * modes

    def harmonic(phase: float) -> float:
        axis = (np.arange(cells) + phase) * mesh.spacing
        grid = np.meshgrid(axis, axis, axis, indexing="ij")
        lagrangian = np.stack([value.ravel() for value in grid])
        positions = lagrangian.copy()
        positions[0] = (
            lagrangian[0] - (amplitude / wavenumber) * np.sin(wavenumber * lagrangian[0])
        ) % mesh.size
        line = np.fft.rfftn(deposit(positions, mesh))[:, 0, 0]
        return float(abs(line[2 * modes]) / abs(line[modes]))

    assert harmonic(0.0) == pytest.approx(expected, abs=0.01)
    assert harmonic(0.5) < 1e-12


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


@pytest.mark.parametrize("modes", [1, 2, 4, 8, 12])
def test_the_force_transfer_function_is_sinc_of_k_h(modes):
    """The mesh multiplies the force by ``sinc(k h)``, to round-off.

    **Several modes, deliberately.** The textbook ``sinc^4(k h/2)`` --
    ``sinc^2`` for the deposit and ``sinc^2`` for the read-back -- agrees
    with this to fourth order in ``k h``, so on the box's longest mode the
    two are indistinguishable and a one-mode test cannot tell them apart.
    By ``k h = 3 pi / 4`` they are 0.300 against 0.378.

    The textbook value is the deposition window *averaged over sub-cell
    phase*, which a lattice does not sample: every particle sits at the
    same phase. What a lattice gets is the static window plus its
    derivative with respect to phase, because the particles also move
    inside their own cells -- and that sum is ``sinc(k h/2)`` at every
    phase, the phase dependence cancelling. Times ``cos(k h/2)`` for
    reading the field back at the cell centres, that is ``sinc(k h)``.
    """
    cells = 32
    mesh = Mesh(size=1.0, cells=cells)
    wavenumber = WAVENUMBER * modes
    state = zeldovich_plane_wave(mesh, 0.02, scale=1.0, modes=modes)
    acceleration = ParticleMesh(mesh).acceleration(state.positions)[0]

    q = np.repeat((np.arange(cells) + 0.5) / cells, cells * cells)
    basis = np.sin(wavenumber * q)
    measured = -np.dot(acceleration, basis) / np.dot(basis, basis)

    exact = 0.02 / wavenumber
    predicted = exact * np.sinc(wavenumber * mesh.spacing / np.pi)
    assert measured == pytest.approx(predicted, rel=1e-9)
    assert abs(measured) < abs(exact)  # the mesh softens, it never sharpens


def test_sinc_of_k_h_is_not_the_textbook_window():
    """The two candidate laws are far apart where the test above looks.

    Guards the point of that parametrisation: if someone re-derives
    ``sinc^4(k h/2)`` from the usual argument and swaps it in, the
    ``modes = 12`` case has to fail outright rather than pass by a whisker.
    """
    product = WAVENUMBER * 12 / 32
    assert np.sinc(product / np.pi) == pytest.approx(0.300105, abs=1e-6)
    assert np.sinc(product / (2.0 * np.pi)) ** 4 == pytest.approx(0.378213, abs=1e-6)


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

    The linear density of ``psi = -(A/k) sin(kq)`` is ``delta = +A cos(kq)``.
    The lattice is offset half a cell from the grid the density lives on,
    and the displacement is evaluated there by a Fourier phase factor, so
    no interpolation enters and the agreement is round-off rather than
    second order.
    """
    mesh = Mesh(size=1.0, cells=16)
    x = np.arange(mesh.cells) * mesh.spacing
    density = AMPLITUDE * np.cos(WAVENUMBER * x)[:, None, None] * np.ones(mesh.shape)

    state = zeldovich_from_field(mesh, density, scale=START)
    centres = (np.arange(mesh.cells) + 0.5) * mesh.spacing
    q = np.stack([v.ravel() for v in np.meshgrid(*[centres] * 3, indexing="ij")])
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
    ``sinc(k h)`` transfer function above and nothing fitted.

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
    shortfall = 1.0 - np.sinc(2.0 / cells)
    measured = 1.0 - growth(cells, 200)
    predicted = 1.0 - (1.0 / START) ** (-0.6 * shortfall)
    assert measured == pytest.approx(predicted, rel=0.15)


@pytest.mark.benchmark
def test_the_caustic_is_late_at_the_centre_and_a_false_one_forms_beside_it():
    """The pancake forms late, and a naive search for it reports a pass.

    Issue #78 asks for the caustic time within 2%. The caustic is the
    *first* shell crossing, which in the exact solution is at ``q = 0``, so
    that is what is measured here. A plain mesh is 17.4% late at ``32^3``,
    and still 7.7% late at ``128^3`` with the convergence order decaying
    from 0.98 to 0.48 -- it is not on its way to 2%.

    The second assertion is the interesting one. Taking the first crossing
    *anywhere* -- the obvious implementation -- finds a different pair,
    two cells off centre here and three at ``128^3``, that crosses sooner.
    Cloud-in-cell moves force off the density peak and into its wings, so
    the centre is under-pulled while its neighbours are over-pulled. At
    ``128^3`` that spurious crossing lands at ``a = 1.996``, which reads as
    a 0.2% pass and is nothing of the kind. This test pins the gap so the
    detector cannot quietly be replaced by the one that "passes".
    """
    cells, mesh = 32, Mesh(size=1.0, cells=32)
    state = zeldovich_plane_wave(mesh, AMPLITUDE, scale=START)
    q = (np.arange(cells) + 0.5) / cells
    history: list[tuple[float, float, float]] = []

    def sample(current: State) -> None:
        slabs = current.positions[0].reshape(cells, -1)
        offset = ((slabs - q[:, None] + 0.5) % 1.0 - 0.5).mean(axis=1)
        gaps = 1.0 / cells + np.diff(offset, prepend=offset[-1])
        history.append((current.scale, float(gaps[0]), float(gaps.min())))

    ParticleMesh(mesh).run(state, 2.4, 300, sample=sample)
    record = np.array(history)

    def crossing(column: int) -> float:
        index = int(np.argmax(record[:, column] <= 0.0))
        assert index > 0, "no shell crossing was reached"
        before, after = record[index - 1], record[index]
        fraction = before[column] / (before[column] - after[column])
        return float(before[0] + (after[0] - before[0]) * fraction)

    exact = caustic_scale_factor(AMPLITUDE)
    centre = crossing(1)
    anywhere = crossing(2)

    assert centre / exact - 1.0 == pytest.approx(0.174, abs=0.02)
    assert anywhere / exact - 1.0 == pytest.approx(0.133, abs=0.02)
    assert anywhere < centre  # the false caustic always comes first
