"""Particle-mesh gravity, and the one test that has an exact answer.

Issue #78, Level C3. Collisionless matter in an expanding box: deposit the
particles onto a mesh, solve Poisson with an FFT, interpolate the force
back, and leapfrog.

**The time variable is the scale factor, and the friction is transformed
away.** In comoving coordinates the equation of motion is

    x'' + (3/2a) x' = -(3/2a^2) grad phi,    lap phi = delta

for Einstein-de Sitter, where a prime is ``d/da``. Written that way a
leapfrog has to carry a velocity-dependent damping term, which is not
separable and costs the scheme its symmetry. Substituting ``p = a^(3/2) x'``
removes it exactly:

    dx/da = a^(-3/2) p
    dp/da = -(3/2) a^(-1/2) grad phi

which *is* separable, so kick-drift-kick applies unchanged. The
substitution is not a numerical trick -- it is the canonical momentum for
this time variable, and the damping was the Jacobian of the change of
variables all along.

**Why Einstein-de Sitter and not a general background.** Because it makes
the acceptance test exact rather than approximate; see below. The
generalisation is one factor of ``E(a)`` in each of the two equations and
nothing else structural, and :func:`growth_exponents` is where it would go.

**The Zel'dovich pancake is an exact solution, not an approximation.** For a
plane-parallel perturbation the displacement map ``x = q + D(a) psi(q)``
solves the *nonlinear* system until shell crossing, and the reason is worth
stating because it is what makes the test sharp. Mass conservation gives
``(1 + delta) dx = dq``, so

    d(phi)/dx = int delta dx = int -D psi'(q) dq = -D psi(q)

with no linearisation anywhere: the ``1 + D psi'`` factors cancel between the
density and the Jacobian. Substituting into the equation of motion leaves

    D'' + (3/2a) D' - (3/2a^2) D = 0

whose exponents are ``+1`` and ``-3/2`` -- the growing and decaying modes of
Einstein-de Sitter. So ``D = a`` exactly, and the caustic, where
``1 + D psi' = 0`` first holds, forms at

    a_caustic = -1/min(psi')

For ``psi = -(A/k) sin(kq)`` that is ``a = 1/A``, a closed form with no
tolerance attached.

**And a mesh alone forms it late.** Measured as the first crossing of two
neighbouring Lagrangian slabs, against ``a = 1/A = 2``:

    cells     a_caustic      error     ratio
       16      2.687465     +34.4%
       32      2.265100     +13.3%      2.59
       64      2.081681      +4.1%      3.25

Time-stepping is not the cause: at 64 cells, 300 steps and 1200 steps give
2.081681 and 2.081308, a difference of 0.02%. Nor is it the ``sinc^4``
softening below, which is 0.2% at 64 cells -- and which deconvolving the
window does not remove. It is that once the pancake is thinner than a cell
the mesh has no information about it left to use. That is what the
short-range half of TreePM supplies, and it is why issue #78 asks for
both.

**What the mesh costs, to four digits.** Cloud-in-cell deposition applies
``sinc^2(k h / 2)`` to the density and the interpolation back applies it
again, so a particle feels the true force times ``sinc^4(k h / 2)``. That
is not a scaling argument, it is what comes out, measured on the box's
longest mode against the exact ``D A / k``:

    cells     measured error      sinc^4(k h/2) - 1
       16     -2.550e-02          -2.541e-02
       32     -6.413e-03          -6.407e-03
       64     -1.606e-03          -1.605e-03
      128     -4.034e-04          -4.015e-04

The error has to be read off by *projecting* the force onto the mode. A
maximum over particles will not do it: a lattice of ``cells`` points never
samples a sine's peak, which costs 8% at ``cells = 8`` and 0.5% at
``cells = 32``, and that sampling error is large enough to hide the
agreement above entirely.

**Deconvolving that window is standard, and is deliberately not done.**
Dividing the potential by ``sinc^4`` cancels the suppression on the
fundamental by construction, but it amplifies the aliased power the same
window was holding down, and measured against the exact Zel'dovich state
the force comes out *worse*: on a ``32^3`` mesh the maximum error over
particles goes from 4.1% to 6.4% at ``D = 0.5`` and from 2.9% to 4.2% at
``D = 1``. Near collapse it helps, but only from 37% to 34%, which is not
the kind of help that matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def growth_exponents() -> tuple[float, float]:
    """``(+1, -3/2)``: the Einstein-de Sitter growing and decaying modes.

    Roots of ``n(n-1) + (3/2)n - 3/2 = 0``, which is what
    ``D'' + (3/2a)D' - (3/2a^2)D = 0`` becomes for ``D = a^n``. Returned
    rather than hard-coded so a test can check the equation of motion
    against them instead of against a remembered pair of numbers.
    """
    roots = np.sort(np.roots([1.0, 0.5, -1.5]))
    return float(roots[1]), float(roots[0])


@dataclass(frozen=True)
class Mesh:
    """A periodic cubic box and the grid the forces are computed on.

    ``size`` is the comoving box length and ``cells`` the grid per
    dimension. The particle count is independent of both: a particle-mesh
    code is free to use more or fewer particles than cells, and the usual
    choice of one per cell is a convention rather than a requirement.

    **Grid point ``i`` sits at ``i * spacing``, not at the cell centre.**
    Every routine here agrees on that, and it is worth stating because the
    two conventions differ by half a cell -- which for the box's longest
    mode is a phase error of ``k h / 2 = pi / cells``, a 20% amplitude error
    at ``cells = 16``. A field sampled at cell centres and read back at
    nodes looks plausible and is wrong by exactly that much.
    """

    size: float = 1.0
    cells: int = 32

    def __post_init__(self) -> None:
        if self.size <= 0.0:
            raise ValueError(f"the box size must be positive, got {self.size}")
        if self.cells < 2:
            raise ValueError(f"a mesh needs at least two cells per side, got {self.cells}")

    @property
    def spacing(self) -> float:
        return self.size / self.cells

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.cells, self.cells, self.cells)

    def wavenumbers(self):
        """``(k_x, k_y, k_z)`` for a real FFT, and ``k^2`` with the zero mode safe."""
        full = 2.0 * np.pi * np.fft.fftfreq(self.cells, d=self.spacing)
        half = 2.0 * np.pi * np.fft.rfftfreq(self.cells, d=self.spacing)
        grids = np.meshgrid(full, full, half, indexing="ij")
        squared = sum(value**2 for value in grids)
        squared[0, 0, 0] = 1.0
        return grids, squared


def deposit(positions, mesh: Mesh) -> np.ndarray:
    """Cloud-in-cell density contrast ``delta`` from particle positions.

    ``positions`` is ``(3, N)`` in comoving units. Each particle is spread
    over the eight cells around it with trilinear weights, which is what
    makes the force continuous as a particle crosses a cell boundary --
    nearest-grid-point would make it jump, and the jump is a spurious
    heating term.

    Mass is conserved to round-off by construction: the eight weights are a
    product of three pairs that each sum to one.
    """
    array = np.asarray(positions, dtype=float)
    if array.ndim != 2 or array.shape[0] != 3:
        raise ValueError(f"positions must have shape (3, N), got {array.shape}")

    grid = array / mesh.spacing
    lower = np.floor(grid).astype(np.int64)
    frac = grid - lower

    total = np.zeros(mesh.cells**3, dtype=float)
    for offset in np.ndindex(2, 2, 2):
        weight = np.ones(array.shape[1], dtype=float)
        index = np.zeros(array.shape[1], dtype=np.int64)
        for axis in range(3):
            shift = offset[axis]
            weight = weight * (frac[axis] if shift else 1.0 - frac[axis])
            index = index * mesh.cells + ((lower[axis] + shift) % mesh.cells)
        total += np.bincount(index, weights=weight, minlength=mesh.cells**3)

    density = total.reshape(mesh.shape)
    mean = array.shape[1] / mesh.cells**3
    return density / mean - 1.0


def potential_gradient(density, mesh: Mesh) -> list[np.ndarray]:
    """``grad phi`` on the mesh, from ``lap phi = delta`` by FFT.

    In Fourier space ``-k^2 phi_k = delta_k``, so ``phi_k = -delta_k/k^2``
    and the gradient is ``i k phi_k`` without ever forming ``phi``. The
    ``k = 0`` mode is set to zero, which is the statement that a periodic
    box has no net force -- it is a choice of gauge, not an approximation,
    and dropping it would leave the whole box accelerating.
    """
    field = np.fft.rfftn(np.asarray(density, dtype=float))
    grids, squared = mesh.wavenumbers()
    potential = -field / squared
    potential[0, 0, 0] = 0.0
    return [
        np.fft.irfftn(1j * component * potential, s=mesh.shape, axes=(0, 1, 2))
        for component in grids
    ]


def interpolate(field, positions, mesh: Mesh) -> np.ndarray:
    """Cloud-in-cell interpolation of a mesh field to particle positions.

    The *same* weights as :func:`deposit`, which is not a coincidence and
    not optional: using a different assignment for the force than for the
    density gives each particle a non-zero force from its own mass, and a
    lone particle in an empty box accelerates.
    """
    array = np.asarray(positions, dtype=float)
    grid = array / mesh.spacing
    lower = np.floor(grid).astype(np.int64)
    frac = grid - lower

    out = np.zeros(array.shape[1], dtype=float)
    for offset in np.ndindex(2, 2, 2):
        weight = np.ones(array.shape[1], dtype=float)
        index = []
        for axis in range(3):
            shift = offset[axis]
            weight = weight * (frac[axis] if shift else 1.0 - frac[axis])
            index.append((lower[axis] + shift) % mesh.cells)
        out += weight * np.asarray(field)[tuple(index)]
    return out


@dataclass(frozen=True)
class State:
    """Positions and canonical momenta at one scale factor."""

    positions: np.ndarray
    momenta: np.ndarray
    scale: float

    @property
    def count(self) -> int:
        return self.positions.shape[1]

    def velocities(self) -> np.ndarray:
        """``dx/da = a^(-3/2) p``, which is what the momentum encodes."""
        return self.scale**-1.5 * self.momenta


@dataclass(frozen=True)
class ParticleMesh:
    """The solver: deposit, solve, interpolate, leapfrog.

    ``kick`` and ``drift`` are separate because the system is separable in
    ``(x, p)``, which is the point of the momentum substitution. The step is
    kick-drift-kick, so it is second order and time-symmetric, and a
    convergence test pins that rather than assuming it.
    """

    mesh: Mesh

    def acceleration(self, positions) -> np.ndarray:
        """``-grad phi`` at the particles, as ``(3, N)``."""
        density = deposit(positions, self.mesh)
        gradient = potential_gradient(density, self.mesh)
        return -np.stack([interpolate(component, positions, self.mesh) for component in gradient])

    def step(self, state: State, step: float) -> State:
        """One kick-drift-kick step of size ``step`` in the scale factor."""
        if step <= 0.0:
            raise ValueError(f"the step in the scale factor must be positive, got {step}")
        start = state.scale
        middle = start + 0.5 * step
        end = start + step

        momenta = state.momenta + 0.5 * step * self._force(state.positions, start)
        positions = (state.positions + step * middle**-1.5 * momenta) % self.mesh.size
        momenta = momenta + 0.5 * step * self._force(positions, end)
        return State(positions=positions, momenta=momenta, scale=end)

    def _force(self, positions, scale: float) -> np.ndarray:
        """``dp/da = -(3/2) a^(-1/2) grad phi``."""
        return 1.5 * scale**-0.5 * self.acceleration(positions)

    def run(self, state: State, final: float, steps: int, sample=None):
        """Integrate to ``final``, optionally recording ``sample(state)``."""
        if final <= state.scale:
            raise ValueError(
                f"the final scale factor {final} must exceed the initial {state.scale}"
            )
        step = (final - state.scale) / int(steps)
        current = state
        history = [] if sample is None else [sample(current)]
        for _ in range(int(steps)):
            current = self.step(current, step)
            if sample is not None:
                history.append(sample(current))
        return current, history


# --- initial conditions ---------------------------------------------------


def zeldovich_plane_wave(
    mesh: Mesh,
    amplitude: float,
    scale: float = 0.1,
    modes: int = 1,
    transverse: int | None = None,
) -> State:
    """A Zel'dovich pancake: the test with a closed-form answer.

    ``psi(q) = -(A/k) sin(k q)`` displaced along ``x``, so ``psi' = -A cos(k q)``
    and the caustic forms at ``a = 1/A`` exactly. Particles are laid on a
    regular lattice in ``q`` and displaced by ``D(a) psi = a psi``, with
    momenta set from the exact ``dx/da = psi``.

    ``transverse`` is the particle count along ``y`` and ``z``, and it must be
    a multiple of ``mesh.cells`` -- the default, one particle per cell.
    Fewer is not a cheaper approximation, it is a different problem. A
    lattice whose spacing is an integer number of cells *greater than one*
    puts every particle at a cell corner, where cloud-in-cell gives the
    whole mass to one cell and nothing to its neighbour: eight particles
    across a sixteen-cell mesh leave ``delta = 3`` on an **undisplaced**
    lattice. That is a grid of rods, not a plane wave.

    The failure hides well, which is why the constraint is enforced rather
    than documented. The transverse force still vanishes to round-off, by
    the lattice's own symmetry, so every obvious diagnostic looks right.
    What the spurious rods do instead is corrupt the ``x``-force, because
    the modes they add carry ``k_perp != 0`` and enter ``a_x`` weighted by
    ``k_x^2/k^2``. With eight across sixteen the ``x``-force came out
    *closer* to the continuum answer than the correct lattice gives -- the
    aliases happen to cancel part of the mesh's own suppression -- so the
    error is not even one-signed.
    """
    transverse = mesh.cells if transverse is None else int(transverse)
    if transverse < 1 or transverse % mesh.cells:
        raise ValueError(
            f"the transverse particle count {transverse} must be a positive multiple "
            f"of the mesh size {mesh.cells}; anything else leaves the undisplaced "
            f"lattice non-uniform, which contaminates the force along x"
        )
    wavenumber = 2.0 * np.pi * int(modes) / mesh.size
    along = (np.arange(mesh.cells) + 0.5) * (mesh.size / mesh.cells)
    across = (np.arange(transverse) + 0.5) * (mesh.size / transverse)
    grid = np.meshgrid(along, across, across, indexing="ij")
    lagrangian = np.stack([value.ravel() for value in grid])

    displacement = -(float(amplitude) / wavenumber) * np.sin(wavenumber * lagrangian[0])
    positions = lagrangian.copy()
    positions[0] = (positions[0] + scale * displacement) % mesh.size

    momenta = np.zeros_like(positions)
    momenta[0] = scale**1.5 * displacement
    return State(positions=positions, momenta=momenta, scale=float(scale))


def caustic_scale_factor(amplitude: float) -> float:
    """``a = 1/A``: where ``1 + a psi'`` first vanishes, in closed form."""
    if amplitude <= 0.0:
        raise ValueError(f"the amplitude must be positive, got {amplitude}")
    return 1.0 / float(amplitude)


def gaussian_field(mesh: Mesh, spectrum, seed: int = 0) -> np.ndarray:
    """A Gaussian random ``delta`` with the given ``P(k)``.

    The noise is drawn in **real space** and coloured in Fourier space,
    rather than drawn as independent complex amplitudes. Both give a
    Gaussian field, but only this one gets the symmetry right for free.
    An ``rfftn`` array is not a set of independent modes: the ``k_z = 0``
    and ``k_z = Nyquist`` planes are self-conjugate, so their entries must
    be *real*. Filling them with complex numbers and calling ``irfftn``
    still returns a real field -- ``irfftn`` simply keeps the Hermitian part
    -- but it throws away half the variance on those two planes, which is
    6% of the modes at ``cells = 32`` and silent. Transforming a real field
    forward cannot violate a symmetry it already has.

    The normalisation is fixed by one identity: real white noise of unit
    per-cell variance is a field of constant power ``P = V/N^3``, the cell
    volume. Colouring it therefore means multiplying by
    ``sqrt(P(k) N^3 / V)``, and the convention that comes out the far side
    is ``<|delta_k|^2> = V P(k)`` with ``delta_k = (V/N^3) rfftn(delta)``.
    """
    generator = np.random.default_rng(seed)
    _, squared = mesh.wavenumbers()
    magnitude = np.sqrt(squared)
    magnitude[0, 0, 0] = 0.0

    nonzero = magnitude > 0.0
    power = np.where(nonzero, spectrum(np.where(nonzero, magnitude, 1.0)), 0.0)
    amplitude = np.sqrt(power * mesh.cells**3 / mesh.size**3)

    field = np.fft.rfftn(generator.normal(size=mesh.shape)) * amplitude
    field[0, 0, 0] = 0.0
    return np.fft.irfftn(field, s=mesh.shape, axes=(0, 1, 2))


def zeldovich_from_field(mesh: Mesh, density, scale: float = 0.1) -> State:
    """Zel'dovich displacement from a linear density field.

    ``psi = -grad(lap^-1 delta)``, which is the same solve
    :func:`potential_gradient` does, so the displacement of a plane wave
    comes out as the sine above without that being special-cased anywhere.

    ``density`` is sampled on the grid, meaning at ``i * mesh.spacing`` --
    see :class:`Mesh`. The Lagrangian lattice is that same grid, one
    particle per cell, which is what makes this exact rather than merely
    accurate: the displacement is *read off* the mesh at the points where it
    was computed, with no interpolation to smooth it. A lattice offset to
    the cell centres would need a cloud-in-cell read-back and would carry
    that half-cell phase error into the initial conditions.
    """
    displacement = [-component for component in potential_gradient(density, mesh)]
    axes = [np.arange(mesh.cells) * mesh.spacing for _ in range(3)]
    lagrangian = np.stack([value.ravel() for value in np.meshgrid(*axes, indexing="ij")])

    positions = np.empty_like(lagrangian)
    momenta = np.empty_like(lagrangian)
    for axis in range(3):
        sampled = displacement[axis].ravel()
        positions[axis] = (lagrangian[axis] + scale * sampled) % mesh.size
        momenta[axis] = scale**1.5 * sampled
    return State(positions=positions, momenta=momenta, scale=float(scale))


__all__ = [
    "Mesh",
    "ParticleMesh",
    "State",
    "caustic_scale_factor",
    "deposit",
    "gaussian_field",
    "growth_exponents",
    "interpolate",
    "potential_gradient",
    "zeldovich_from_field",
    "zeldovich_plane_wave",
]
