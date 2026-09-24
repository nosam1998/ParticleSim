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

**And a mesh alone forms it late -- later than the obvious measurement
says.** The caustic is where the *first* shell crossing happens, which in
the exact solution is at ``q = 0``. Looking for it as the first crossing
*anywhere* is the natural thing to do and is a trap, because a mesh
manufactures an earlier one somewhere else:

    cells    at q = 0    error     first anywhere    error    where
       16    2.687637   +34.4%        2.687637      +34.4%    q = 0
       32    2.348872   +17.4%        2.265392      +13.3%    2 cells out
       64    2.214884   +10.7%        2.081681       +4.1%    3 cells out
      128    2.154361    +7.7%        1.996029       -0.2%    3 cells out

Read the right-hand column alone and a ``128^3`` mesh clears issue #78's 2%
with room to spare. It has not. It has found a *different* crossing that
happens to be sweeping past ``a = 2`` at that resolution, and the caustic
the acceptance is about is 7.7% late.

The sign says why. Pair ``j`` sits at ``q = j h`` and crosses, exactly, at
``D = 1/(A cos(k j h))``. Measured against each pair's own exact time, the
central pair is 17.4% late at ``32^3`` and 7.7% late at ``128^3``, while
the pair three cells out goes from 4.6% late at ``32^3`` to 0.4% *early* at
``64^3`` and 1.3% early at ``128^3``. Cloud-in-cell moves force off the
peak and into its wings: the collapsing centre is under-pulled and its
neighbours are over-pulled, and where those two errors cross, a spurious
caustic forms first.

Nothing cheap fixes it. Not the time step -- at 64 cells, 300 and 1200
steps give 2.081681 and 2.081308. Not the slab lattice, whose own caustic
sits at ``(1/A)(kh/2)/sin(kh/2)``, +0.01% at ``128^3``. Not the ``sinc(k h)``
softening above, 0.2% at 64 cells, which deconvolving does not remove.
And not resolution: the central error falls 34.4, 17.4, 10.7, 7.7
as the mesh doubles, an effective order of 0.98, then 0.70, then 0.48 --
decaying, not converging. Once the pancake is thinner than a cell the mesh
has nothing left to represent it with, and that is true at every
resolution. **Issue #78's 2% is not reachable this way**, which is exactly
what the short-range half of TreePM is for, and why the issue asks for
both.

**The caustic tests the pair force and nothing else.** The pancake is
symmetric about ``q = 0``, and in one dimension a uniform sheet pulls with
``sigma/2`` at any distance, so everything outside the central pair pulls
its two members equally and oppositely and cancels. What is left is their
mutual attraction against the background's push, at separations falling to
zero. That is exactly where a mesh has nothing, and where :class:`TreePM`
adds each pair's force back directly.

**It is also where particles stop looking like a sheet.** Two aligned
square lattices of pitch ``b`` at separation ``D`` pull with
``sigma/2 [1 + sum_{G != 0} exp(-|G| D)]``, not ``sigma/2``, and at a fifth
of a slab spacing a cubic lattice triples the central pair's pull. So
point masses on a cubic lattice form the caustic 15% *early*, and that is
the right answer to the discrete problem: a model of sixteen lattice
sheets, integrated with the same steps, predicts it to half a point.
Refining the lattice *across* removes it -- 4.6% early at twice as fine,
0.97% at four times, which is issue #78's 2% at the true caustic. See
``docs/benchmarks.md`` for the tables.

**What the mesh costs is exactly ``sinc(k h)``.** Not to four digits: to
1.5e-13, over every mode of the box at three resolutions. The textbook
answer would be ``sinc^4(k h / 2)`` -- cloud-in-cell applies
``sinc^2(k h/2)`` on the way in and again on the way out -- and it is
wrong here, for a reason worth keeping.

That ``sinc^2`` is the deposition window *averaged over sub-cell phase*,
which is right for particles that sample the box fairly. A lattice does
not: every particle sits at the same phase, so the static window is
``|(1-f) + f e^{-ikh}|``, which is ``cos(k h/2)`` at the cell centres. But
a displaced lattice also moves *within* its cells, and the cloud-in-cell
weights respond to that motion, which contributes the derivative of the
window with respect to phase. Adding the two,

    Omega(f) + i Omega'(f) / (k h)   has magnitude   sinc(k h / 2)

for **any** phase ``f``, the phase dependence cancelling exactly. The force
then carries that once for the deposit and ``cos(k h/2)`` once for reading
the field back at the particle, and

    sinc(k h/2) cos(k h/2) = sin(k h) / (k h) = sinc(k h)

The two laws agree to ``O((kh)^4)``, so on the box's longest mode alone --
where an earlier version of this study stopped -- they are
indistinguishable: -6.413e-03 measured against -6.407e-03 for ``sinc^4``
and -6.413e-03 for ``sinc(kh)``. They part company further up: at
``k h = 3 pi / 4`` the force is suppressed to 0.300105, which is
``sinc(k h)`` exactly and ``sinc^4(k h/2) = 0.378`` not at all. **Testing
one mode per resolution cannot tell two models apart when they differ at
fourth order in that mode.**

The error has to be read off by *projecting* the force onto the mode. A
maximum over particles will not do it: a lattice of ``cells`` points never
samples a sine's peak, which costs 8% at ``cells = 8`` and 0.5% at
``cells = 32``, and that sampling error is large enough to hide the
agreement entirely.

**Deconvolving is standard, and is deliberately not done.** Dividing the
potential by the textbook ``sinc^4`` window -- the usual choice, and the
one a reader would reach for -- cancels a suppression on the fundamental
by construction, but it amplifies the aliased power the same window was
holding down. Measured against the exact Zel'dovich state on a ``32^3``
mesh, the maximum error over particles goes from 4.1% to 6.4% at
``D = 0.5`` and from 2.9% to 4.2% at ``D = 1``. Near collapse it helps,
but only from 37% to 34%, which is not the kind of help that matters --
and there the field is many modes at once, so no single window is the
right one to divide by anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import erfc


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


def potential_gradient(density, mesh: Mesh, shift: float = 0.0) -> list[np.ndarray]:
    """``grad phi`` on the mesh, from ``lap phi = delta`` by FFT.

    In Fourier space ``-k^2 phi_k = delta_k``, so ``phi_k = -delta_k/k^2``
    and the gradient is ``i k phi_k`` without ever forming ``phi``. The
    ``k = 0`` mode is set to zero, which is the statement that a periodic
    box has no net force -- it is a choice of gauge, not an approximation,
    and dropping it would leave the whole box accelerating.

    ``shift`` moves the evaluation points by that fraction of a cell along
    every axis, by the phase factor ``exp(i k h s)``. Since the field is
    already in Fourier space this is **exact**: ``shift = 0.5`` gives the
    gradient at the cell centres with no interpolation and no window, which
    is what :func:`zeldovich_from_field` needs and what reading a
    node-centred field at cell centres by cloud-in-cell would only
    approximate.
    """
    field = np.fft.rfftn(np.asarray(density, dtype=float))
    grids, squared = mesh.wavenumbers()
    potential = -field / squared
    potential[0, 0, 0] = 0.0
    if shift:
        phase = sum(component for component in grids) * (mesh.spacing * float(shift))
        potential = potential * np.exp(1j * phase)
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
        advanced, _ = self._advance(state, step, self._force(state.positions, state.scale))
        return advanced

    def _advance(self, state: State, step: float, force: np.ndarray) -> tuple[State, np.ndarray]:
        """Kick-drift-kick from a force already known at ``state``; returns the force at the end.

        The closing kick's force is the next step's opening one -- same
        positions, same scale factor -- so :meth:`run` hands it on instead
        of computing it twice. The result is bit-for-bit what repeated
        :meth:`step` calls give, at half the force evaluations, which is
        what makes the short-range force affordable.
        """
        if step <= 0.0:
            raise ValueError(f"the step in the scale factor must be positive, got {step}")
        start = state.scale
        middle = start + 0.5 * step
        end = start + step

        momenta = state.momenta + 0.5 * step * force
        positions = (state.positions + step * middle**-1.5 * momenta) % self.mesh.size
        after = self._force(positions, end)
        momenta = momenta + 0.5 * step * after
        return State(positions=positions, momenta=momenta, scale=end), after

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
        force = self._force(current.positions, current.scale)
        for _ in range(int(steps)):
            current, force = self._advance(current, step, force)
            if sample is not None:
                history.append(sample(current))
        return current, history


# --- TreePM: the mesh for the long range, pairs for the short -------------


def long_range_gradient(density, mesh: Mesh, split: float) -> list[np.ndarray]:
    """``grad phi`` of the Gaussian-smoothed long-range potential, on the mesh.

    ``phi_k exp(-k^2 r_s^2)`` with ``r_s = split``, a comoving length -- the
    long-range half of Hernquist and Bode's split, as in GADGET-2. What is
    taken off here is exactly what :func:`short_range_acceleration` puts
    back pair by pair, so the sum is Newton's force wherever both are exact.

    **Here the cloud-in-cell window is divided out, unlike in the plain
    solve.** Deconvolving the whole field amplifies the aliased power the
    window was holding down (see :func:`potential_gradient` and the module
    notes). Behind the Gaussian there is almost none left to amplify: the
    division is at most ``1/sinc^4(pi/2) = 6.1`` per axis at the Nyquist
    frequency, where the filter at ``r_s = 2h`` is ``exp(-4 pi^2) = 7e-18``.
    Measured on the frozen pancake against the exact plane-sheet force,
    dividing takes the worst per-slab error at ``a = 1.9`` from 2.5e-03 to
    5.0e-04 of the peak force on a ``128^3`` mesh.
    """
    if split <= 0.0:
        raise ValueError(f"the split radius must be positive, got {split}")
    field = np.fft.rfftn(np.asarray(density, dtype=float))
    grids, squared = mesh.wavenumbers()
    window = np.ones_like(squared)
    for component in grids:
        # np.sinc(x) is sin(pi x)/(pi x), so this is sinc^2(k h / 2) per axis.
        window = window * np.sinc(component * mesh.spacing / (2.0 * np.pi)) ** 2
    potential = -field / squared * np.exp(-squared * split**2) / window**2
    potential[0, 0, 0] = 0.0
    return [
        np.fft.irfftn(1j * component * potential, s=mesh.shape, axes=(0, 1, 2))
        for component in grids
    ]


def short_range_acceleration(
    positions, size: float, split: float, cutoff: float, softening: float = 0.0
) -> np.ndarray:
    """The pair force the Gaussian filter took off the mesh, summed over neighbours.

    Each particle has mass ``m = size^3 / N`` in the units where
    ``lap phi = delta``, so a point mass pulls with ``m / (4 pi r^2)``, and
    the short-range part of that is

        m / (4 pi r^2) [erfc(r / 2 r_s) + (r / (r_s sqrt(pi))) exp(-r^2 / 4 r_s^2)]

    with ``r_s = split``. It stops at ``cutoff``, a comoving length; at
    ``4.5 r_s`` it has fallen to 1.75% of Newton's there. ``softening`` is a
    Plummer length on the Newtonian factor; zero is point masses.

    Neighbours come from a periodic k-d tree (``scipy.spatial.cKDTree`` with
    ``boxsize``), and within the cutoff the sum is **direct**, not a multipole
    walk. That is the short-range half of P3M behind TreePM's Gaussian split,
    and at the densities measured here -- about 200 neighbours per particle
    -- a walk would have little to group. Separations are minimum-image,
    which is why the cutoff must stay under half the box.
    """
    array = np.asarray(positions, dtype=float)
    if array.ndim != 2 or array.shape[0] != 3:
        raise ValueError(f"positions must have shape (3, N), got {array.shape}")
    if not 0.0 < cutoff < 0.5 * size:
        raise ValueError(
            f"the cutoff {cutoff} must be positive and under half the box {size}: "
            f"separations are minimum-image, so a longer one would count a pair twice"
        )
    count = array.shape[1]
    mass = size**3 / count
    # ``x % size`` can round to exactly ``size`` for a tiny negative x, and
    # the periodic tree refuses anything outside [0, size).
    wrapped = np.mod(array, size)
    wrapped = np.where(wrapped >= size, wrapped - size, wrapped)
    tree = cKDTree(wrapped.T, boxsize=size)
    first, second = tree.query_pairs(cutoff, output_type="ndarray").T

    separation = wrapped[:, second] - wrapped[:, first]
    separation -= size * np.round(separation / size)
    squared = np.sum(separation**2, axis=0)
    distance = np.sqrt(squared)
    newton = mass / (4.0 * np.pi) / (squared + softening**2) ** 1.5
    screen = erfc(distance / (2.0 * split)) + distance / (split * np.sqrt(np.pi)) * np.exp(
        -squared / (4.0 * split**2)
    )
    strength = newton * screen

    out = np.empty_like(array)
    for axis in range(3):
        pull = strength * separation[axis]
        out[axis] = np.bincount(first, weights=pull, minlength=count) - np.bincount(
            second, weights=pull, minlength=count
        )
    return out


@dataclass(frozen=True)
class TreePM(ParticleMesh):
    """Particle-mesh for the long range, particle pairs for the short.

    ``split`` is the Gaussian radius ``r_s`` in mesh cells, ``cutoff`` the
    short-range reach in units of ``r_s``, and ``softening`` a comoving
    Plummer length. Kick-drift-kick and everything else are
    :class:`ParticleMesh`'s; only the force differs.

    **``r_s = 2`` cells rather than GADGET-2's 1.25.** On the frozen pancake
    near collapse, against the exact force between plane lattice sheets,
    1.25 cells leaves 2.7% of the peak force wrong at ``a = 0.5`` on a
    ``16^3`` mesh, and 2 cells on ``64^3`` leaves 0.5%. The central pair's
    relative pull is a small difference, and a 3% error in it moved the
    caustic by four points.

    **The mesh need not match the particles,** and for the caustic it should
    not: what sets the short-range cost is ``r_s`` in comoving units, so a
    finer mesh buys a shorter reach. Sixteen slabs on a ``128^3`` mesh at
    two cells is 200 neighbours a particle.
    """

    split: float = 2.0
    cutoff: float = 4.5
    softening: float = 0.0

    def __post_init__(self) -> None:
        if self.split <= 0.0:
            raise ValueError(f"the split must be a positive number of cells, got {self.split}")
        if self.softening < 0.0:
            raise ValueError(f"the softening must be non-negative, got {self.softening}")
        reach = self.cutoff * self.radius
        if not 0.0 < reach < 0.5 * self.mesh.size:
            raise ValueError(
                f"the short-range reach {self.cutoff} x {self.radius:.4g} = {reach:.4g} "
                f"must be positive and under half the box {self.mesh.size}"
            )

    @property
    def radius(self) -> float:
        """``r_s`` as a comoving length."""
        return self.split * self.mesh.spacing

    def acceleration(self, positions) -> np.ndarray:
        """``-grad phi`` at the particles: the smoothed mesh force plus the pairs."""
        density = deposit(positions, self.mesh)
        gradient = long_range_gradient(density, self.mesh, self.radius)
        long_range = -np.stack(
            [interpolate(component, positions, self.mesh) for component in gradient]
        )
        return long_range + short_range_acceleration(
            positions,
            self.mesh.size,
            self.radius,
            self.cutoff * self.radius,
            self.softening,
        )


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
    see :class:`Mesh`. The Lagrangian lattice is offset from it by **half a
    cell**, and the displacement is evaluated there exactly, by a phase
    factor in the same Fourier space the solve already happened in. So
    there is no interpolation and no window, and the half-cell offset is
    not a compromise to be corrected for later.

    **The offset is not cosmetic: a node-aligned lattice is the one phase
    at which cloud-in-cell degenerates.** A particle sitting exactly on a
    grid point gives that point all of its mass, and a small displacement
    moves a fraction ``|s|/h`` to the neighbour *in the direction of
    travel*. The response is rectified -- it depends on ``|s|``, not ``s``
    -- so a lattice displaced by a single mode deposits spurious harmonics
    at 17% of the fundamental for the box's longest mode and 71% at four
    times that, while the fundamental itself falls below the window. At any
    other phase, cell centres included, the harmonics are *exactly* zero
    and the fundamental is exactly ``sinc(k h / 2)``.
    """
    displacement = [-component for component in potential_gradient(density, mesh, shift=0.5)]
    axes = [(np.arange(mesh.cells) + 0.5) * mesh.spacing for _ in range(3)]
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
    "TreePM",
    "caustic_scale_factor",
    "deposit",
    "gaussian_field",
    "growth_exponents",
    "interpolate",
    "long_range_gradient",
    "potential_gradient",
    "short_range_acceleration",
    "zeldovich_from_field",
    "zeldovich_plane_wave",
]
