"""TreePM: the mesh for the long range, particle pairs for the short (issue #78).

The plain mesh gets the Zel'dovich caustic 17% late at ``32^3``, because once
the pancake is thinner than a cell there is nothing left to represent it
with. What that leaves for the short-range force is the separations a mesh
cannot see, so most of what is below checks it there, against closed forms:
a single pair against Newton plus the background, and the frozen pancake
against the exact force between plane sheets.

**The reference for the pancake is a lattice of sheets, not a continuum.**
Particles laid on a plane lattice are sheets of point masses, and two such
sheets closer than their own pitch do not pull like uniform ones. Aligned
square lattices of pitch ``b`` at separation ``D`` add

    (sigma / 2) sum_{G != 0} exp(-|G| D)

to the uniform sheet's ``sigma / 2``, the sum over the reciprocal lattice.
Uniform sheets obey Gauss's law in one dimension, so each is pulled by
exactly its own displacement, and the lattice sum is the only correction.
It is what makes point masses on a cubic lattice collapse 15% *early* --
and what a finer transverse lattice removes.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.cosmo.nbody import (
    Mesh,
    ParticleMesh,
    TreePM,
    caustic_scale_factor,
    deposit,
    long_range_gradient,
    potential_gradient,
    short_range_acceleration,
    zeldovich_plane_wave,
)

AMPLITUDE = 0.5
START = 0.1
SLABS = 16


def _slab_mean(values, slabs: int = SLABS) -> np.ndarray:
    return np.asarray(values).reshape(slabs, -1).mean(axis=1)


def _slab_positions(state, slabs: int = SLABS) -> tuple[np.ndarray, np.ndarray]:
    """Each slab's ``x`` and its Lagrangian ``q``, unwrapped around ``q``."""
    lagrangian = (np.arange(slabs) + 0.5) / slabs
    offset = _slab_mean((state.positions[0].reshape(slabs, -1) - lagrangian[:, None] + 0.5) % 1.0)
    return lagrangian + offset - 0.5, lagrangian


def _lattice_sheets(position, lagrangian, spacing: float, pitch: float) -> np.ndarray:
    """Exact ``x``-acceleration of aligned square-lattice sheets of point masses.

    ``spacing`` is the sheets' Lagrangian separation, which with unit mean
    density is also their surface density; ``pitch`` the lattice constant
    within each sheet.
    """
    acceleration = (position - lagrangian + 0.5) % 1.0 - 0.5
    index = np.arange(-40, 41)
    reciprocal = 2 * np.pi * np.hypot(index[:, None], index[None, :]).ravel() / pitch
    reciprocal = reciprocal[reciprocal > 0]
    for i in range(position.size):
        for j in range(i + 1, position.size):
            separation = (position[i] - position[j] + 0.5) % 1.0 - 0.5
            extra = 0.5 * spacing * np.exp(-reciprocal * abs(separation)).sum()
            acceleration[i] -= np.sign(separation) * extra
            acceleration[j] += np.sign(separation) * extra
    return acceleration


# --- the split -------------------------------------------------------------


def test_a_pair_feels_newton_and_the_background_at_every_separation():
    """Two particles against ``m/(4 pi r^2) - m r/(3 L^3)``, which is exact to ``O(r^3)``.

    The periodic Green's function with a neutralising background solves
    ``lap G = delta - 1/L^3``, so beside the source it is ``-1/(4 pi r)``
    plus a regular part whose Laplacian is ``-1/L^3``; cubic symmetry makes
    that ``-r^2/(6 L^3)`` up to fourth order. Inside a third of a cell the
    pair force is Newton's to 5e-06 -- where the plain mesh gives a small
    fraction of it -- and to 1% out to four cells, where the mesh takes over.
    Every pair is antisymmetric, so the net force is round-off.
    """
    mesh = Mesh(size=1.0, cells=32)
    tree_pm, plain = TreePM(mesh), ParticleMesh(mesh)
    generator = np.random.default_rng(7)
    mass = 0.5
    for cells_apart, tolerance in ((0.1, 1e-6), (0.3, 1e-5), (1.0, 1e-3), (2.0, 4e-3), (4.0, 1e-2)):
        separation = cells_apart * mesh.spacing
        expected = mass / (4 * np.pi * separation**2) - mass * separation / 3
        for _ in range(6):
            source = generator.uniform(0.0, 1.0, 3)
            direction = generator.normal(size=3)
            direction /= np.linalg.norm(direction)
            positions = np.stack([source, (source + separation * direction) % 1.0], axis=1)
            acceleration = tree_pm.acceleration(positions)
            towards = -acceleration[:, 1] @ direction
            sideways = np.linalg.norm(acceleration[:, 1] + towards * direction)
            assert abs(towards / expected - 1) < tolerance, (cells_apart, towards / expected)
            assert sideways / expected < tolerance, (cells_apart, sideways / expected)
            assert np.abs(acceleration.sum(axis=1)).max() < 1e-12 * expected
            if cells_apart == 0.3:
                mesh_only = -plain.acceleration(positions)[:, 1] @ direction
                assert mesh_only / expected < 0.05


def test_the_long_and_short_halves_add_up_to_the_whole():
    """With the Gaussian at zero width the long range is the whole mesh solve, deconvolved.

    And a short range with nothing inside its cutoff contributes nothing:
    the halves are the two ends of one split, not two approximations.
    """
    mesh = Mesh(size=1.0, cells=16)
    positions = np.random.default_rng(3).uniform(0.0, 1.0, (3, 200))
    density = deposit(positions, mesh)
    narrow = long_range_gradient(density, mesh, split=1e-9)
    grids, _ = mesh.wavenumbers()
    window = np.ones_like(grids[0])
    for component in grids:
        window = window * np.sinc(component * mesh.spacing / (2 * np.pi)) ** 2
    plain = potential_gradient(
        np.fft.irfftn(np.fft.rfftn(density) / window**2, s=mesh.shape, axes=(0, 1, 2)), mesh
    )
    for mine, theirs in zip(narrow, plain, strict=True):
        assert np.allclose(mine, theirs, atol=1e-12 * np.abs(theirs).max())
    far = np.array([[0.1, 0.4], [0.1, 0.1], [0.1, 0.1]])
    assert np.all(short_range_acceleration(far, 1.0, split=0.01, cutoff=0.05) == 0.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"split": 0.0}, "split"),
        ({"softening": -0.1}, "softening"),
        ({"split": 2.0, "cutoff": 4.5}, "half the box"),
    ],
)
def test_treepm_refuses_a_split_it_cannot_honour(kwargs, message):
    """A two-cell split at ``16^3`` reaches 0.56 of the box: minimum image would double count."""
    with pytest.raises(ValueError, match=message):
        TreePM(Mesh(size=1.0, cells=16), **kwargs)


def test_run_hands_the_closing_force_on_without_changing_a_bit():
    """``run`` reuses each step's closing force as the next opening one; ``step`` does not."""
    initial = zeldovich_plane_wave(Mesh(size=1.0, cells=8), AMPLITUDE, scale=START)
    for solver in (ParticleMesh(Mesh(size=1.0, cells=8)), TreePM(Mesh(size=1.0, cells=32))):
        final, _ = solver.run(initial, 0.5, 4)
        current = initial
        for _ in range(4):
            current = solver.step(current, 0.1)
        assert np.array_equal(final.positions, current.positions)
        assert np.array_equal(final.momenta, current.momenta)


# --- the frozen pancake ------------------------------------------------------


@pytest.mark.parametrize(
    "transverse, scale",
    [(1, 1.6), (2, 0.5), (2, 1.9)],
)
def test_the_pancake_force_is_the_lattice_sheets_force(transverse, scale):
    """The exact Zel'dovich state at ``a``, against the exact force on its slabs.

    At ``a = 1.6`` the central gap is a fifth of a slab spacing, and on a
    cubic lattice the facing particles pull more than twice as hard as
    uniform sheets would. TreePM gets that to 1% of the peak force; the
    plain mesh on the particles' own grid is off by 28-100%. Every particle
    in a slab feels the same force, and none feels one across it.
    """
    lattice = Mesh(size=1.0, cells=SLABS)
    state = zeldovich_plane_wave(lattice, AMPLITUDE, scale=scale, transverse=transverse * SLABS)
    position, lagrangian = _slab_positions(state)
    exact = _lattice_sheets(position, lagrangian, lattice.spacing, lattice.spacing / transverse)
    peak = np.abs(exact).max()

    acceleration = TreePM(Mesh(size=1.0, cells=64)).acceleration(state.positions)
    slabs = acceleration[0].reshape(SLABS, -1)
    assert np.abs(_slab_mean(acceleration[0]) - exact).max() < 1e-2 * peak
    assert np.abs(slabs - slabs.mean(axis=1, keepdims=True)).max() < 1e-12 * peak
    assert np.abs(acceleration[1:]).max() < 1e-12 * peak

    mesh_only = ParticleMesh(lattice).acceleration(state.positions)
    assert np.abs(_slab_mean(mesh_only[0]) - exact).max() > 0.05 * peak


def test_a_cubic_lattice_is_not_a_set_of_uniform_sheets_near_collapse():
    """Where the reference above parts from the continuum, and by how much.

    Uniform sheets pull each other by ``sigma/2`` at any separation, so the
    continuum force on a slab is its displacement. At a fifth of a spacing
    the lattice sum triples the central pair's relative pull on a cubic
    lattice. Across, a lattice twice as fine leaves 36% of that, four times
    2.6%, eight times 2e-04: the correction goes as ``exp(-2 pi D / b)``.
    """
    lattice = Mesh(size=1.0, cells=SLABS)
    state = zeldovich_plane_wave(lattice, AMPLITUDE, scale=1.6)
    position, lagrangian = _slab_positions(state)
    continuum = (position - lagrangian + 0.5) % 1.0 - 0.5
    gap = (position[0] - position[-1] + 0.5) % 1.0 - 0.5
    assert gap == pytest.approx(0.2051 * lattice.spacing, rel=1e-3)

    def central_pull(pitch):
        sheets = _lattice_sheets(position, lagrangian, lattice.spacing, pitch)
        return sheets[-1] - sheets[0]

    uniform = continuum[-1] - continuum[0]
    assert central_pull(lattice.spacing) / uniform > 3.0
    assert central_pull(lattice.spacing / 2) / uniform == pytest.approx(1.362, abs=1e-3)
    assert central_pull(lattice.spacing / 4) / uniform == pytest.approx(1.026, abs=1e-3)
    assert central_pull(lattice.spacing / 8) / uniform == pytest.approx(1.0, abs=5e-4)


# --- the caustic ---------------------------------------------------------------


def _caustic(solver, transverse: int, steps: int = 300) -> tuple[float, float]:
    """The central pair's crossing and the first crossing anywhere, as scale factors."""
    lattice = Mesh(size=1.0, cells=SLABS)
    state = zeldovich_plane_wave(lattice, AMPLITUDE, scale=START, transverse=transverse * SLABS)
    lagrangian = (np.arange(SLABS) + 0.5) / SLABS
    history: list[tuple[float, float, float]] = []

    def sample(current) -> None:
        slabs = current.positions[0].reshape(SLABS, -1)
        offset = ((slabs - lagrangian[:, None] + 0.5) % 1.0 - 0.5).mean(axis=1)
        gaps = 1.0 / SLABS + np.diff(offset, prepend=offset[-1])
        history.append((current.scale, float(gaps[0]), float(gaps.min())))

    solver.run(state, 2.4, steps, sample=sample)
    record = np.array(history)

    def crossing(column: int) -> float:
        index = int(np.argmax(record[:, column] <= 0.0))
        assert index > 0, "no shell crossing was reached"
        before, after = record[index - 1], record[index]
        fraction = before[column] / (before[column] - after[column])
        return float(before[0] + (after[0] - before[0]) * fraction)

    return crossing(1), crossing(2)


@pytest.mark.benchmark
def test_point_masses_on_a_cubic_lattice_collapse_early():
    """16% early, where the mesh alone was 34% late -- and neither is the solver's error.

    The sheet model above, integrated on its own with the same steps,
    predicts 15.3% early for sixteen slabs of 16 x 16. TreePM agrees to a
    point: it is solving the discrete problem correctly, and the discrete
    problem is not the continuum one. The first crossing is the central
    pair's, as it should be.
    """
    centre, anywhere = _caustic(TreePM(Mesh(size=1.0, cells=64)), transverse=1)
    exact = caustic_scale_factor(AMPLITUDE)
    assert centre / exact - 1.0 == pytest.approx(-0.153, abs=0.02)
    assert anywhere == centre


@pytest.mark.slow
@pytest.mark.benchmark
def test_the_caustic_is_within_two_percent_with_a_finer_transverse_lattice():
    """Issue #78's acceptance: the Zel'dovich caustic to 2%, at the true caustic.

    Sixteen slabs of 64 x 64 point masses, a quarter of the slab spacing
    apart across, on a ``128^3`` mesh with the split at two cells. The
    central pair crosses at ``a = 1.9806``, 0.97% early; the sheet model
    predicts 1.33%, and 0.06% at 128 x 128. The first crossing anywhere is
    the central pair's -- no false caustic beside it, where the plain mesh
    had one two cells out.
    """
    centre, anywhere = _caustic(TreePM(Mesh(size=1.0, cells=128)), transverse=4)
    exact = caustic_scale_factor(AMPLITUDE)
    assert abs(centre / exact - 1.0) < 0.02
    assert centre / exact - 1.0 == pytest.approx(-0.0097, abs=0.003)
    assert anywhere == centre
