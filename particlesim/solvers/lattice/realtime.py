"""Classical-statistical lattice fields, and what a leapfrog actually conserves.

Issue #65, Level C4 groundwork. A scalar field on a periodic lattice, stepped
in real time by velocity Verlet, on Minkowski or a fixed FLRW background.

**"Energy conserved to round-off" is a true statement about the wrong
energy.** A symplectic integrator does not conserve ``H``; it conserves a
*modified* Hamiltonian that differs from ``H`` at ``O(dt^2)``. Reporting
``H`` and finding it constant to ten digits would mean the step was small,
not that the scheme was good, and reporting it oscillating at ``O(dt^2)``
says nothing either -- that is what a symplectic scheme is supposed to do.
Three quantities are therefore offered, each with its own exact claim:

=========================  ================================================
:func:`energy`             ``H`` itself. Oscillates at ``O(dt^2)`` with no
                           secular drift, which is the physical content of
                           symplecticity.
:func:`quadratic_invariant`  ``H - (dt^2/8) |a|^2``. Conserved **exactly**,
                           to round-off, whenever the theory is free --
                           because then every lattice mode is an independent
                           harmonic oscillator and this is that oscillator's
                           exact invariant, summed. Interacting, it is still
                           ``O(dt^2)``, about 2.4 times smaller than ``H``.
:func:`modified_energy`    ``H + (dt^2/24)(2 <pi, U'' pi> - |U'|^2)``, the
                           Verlet shadow Hamiltonian. ``O(dt^4)`` for *any*
                           potential, measured at 16.0x per halving.
=========================  ================================================

The first is the physics, the second is the acceptance, the third is the
diagnostic. Note that they are not orderings of the same thing: the
quadratic invariant is exact for a free field where the shadow energy is
merely fourth order, and fourth order for an interacting one where the
quadratic invariant is merely second.

**The Laplacian is nearest-neighbour, and that is a definition rather than
an approximation.** A classical-statistical lattice theory *is* the lattice
theory, so its dispersion is not an approximation to the continuum one:

    omega^2(k) = m^2 + (4/h^2) sum_i sin^2(k_i h / 2)

exactly, and a mode put on the lattice oscillates at that frequency and no
other. At seven wavelengths across a 16-point lattice that ``omega^2`` is
0.51 of the continuum ``m^2 + k^2`` -- half. Using a spectral Laplacian
would give the continuum value and a different theory, not a
better-resolved one.

**FLRW is carried by a change of variable, not by a friction term.** In
cosmic time a scalar obeys ``phi'' + 3H phi' - lap phi / a^2 + V' = 0``, and
the friction is not separable, so a leapfrog would lose the symmetry that
the whole of the above depends on. In conformal time the rescaled field
``chi = a phi`` obeys

    chi'' - lap chi + (a^2 m^2 - a''/a) chi + lambda chi^3 = 0

with no first derivative anywhere: the expansion has become a time-dependent
mass. (The quartic term is untouched because ``lambda phi^4`` is conformally
invariant in 3+1 dimensions.) This is the same move as the ``p = a^(3/2) x'``
substitution in :mod:`particlesim.cosmo.nbody` and for the same reason -- the
damping was the Jacobian of a change of variables all along.

A radiation era is the sharp case: ``a`` is linear in conformal time, so
``a'' = 0``, and a massless field there is *exactly* a free field in flat
space. The energy is then conserved to round-off in an expanding universe,
which is a statement about conformal invariance rather than about the
integrator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Lattice:
    """A periodic cubic lattice in ``dimensions`` dimensions."""

    size: float = 1.0
    points: int = 32
    dimensions: int = 3

    def __post_init__(self) -> None:
        if self.size <= 0.0:
            raise ValueError(f"the box size must be positive, got {self.size}")
        if self.points < 2:
            raise ValueError(f"a lattice needs at least two points per side, got {self.points}")
        if self.dimensions not in (1, 2, 3):
            raise ValueError(f"only 1, 2 and 3 dimensions are supported, got {self.dimensions}")

    @property
    def spacing(self) -> float:
        return self.size / self.points

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.points,) * self.dimensions

    @property
    def sites(self) -> int:
        return self.points**self.dimensions

    @property
    def volume(self) -> float:
        return self.size**self.dimensions

    def wavenumbers(self) -> list[np.ndarray]:
        """``k_i`` on the real-transform grid, one array per axis."""
        full = 2.0 * np.pi * np.fft.fftfreq(self.points, d=self.spacing)
        half = 2.0 * np.pi * np.fft.rfftfreq(self.points, d=self.spacing)
        axes = [full] * (self.dimensions - 1) + [half]
        return list(np.meshgrid(*axes, indexing="ij"))

    def dispersion(self, mass_squared: float) -> np.ndarray:
        """``omega^2(k) = m^2 + (4/h^2) sum sin^2(k h/2)``: the lattice's own.

        Not the continuum ``m^2 + k^2``. The difference is the whole content
        of putting a field on a lattice, and it is what a mode's measured
        oscillation frequency has to be compared against.
        """
        total = np.zeros_like(self.wavenumbers()[0])
        for component in self.wavenumbers():
            total = total + np.sin(component * self.spacing / 2.0) ** 2
        return mass_squared + 4.0 * total / self.spacing**2


def laplacian(field, lattice: Lattice) -> np.ndarray:
    """Nearest-neighbour periodic Laplacian."""
    array = np.asarray(field, dtype=float)
    if array.shape != lattice.shape:
        raise ValueError(f"the field must have shape {lattice.shape}, got {array.shape}")
    total = -2.0 * lattice.dimensions * array
    for axis in range(lattice.dimensions):
        total = total + np.roll(array, 1, axis=axis) + np.roll(array, -1, axis=axis)
    return total / lattice.spacing**2


# --- the theory -----------------------------------------------------------


@dataclass(frozen=True)
class Potential:
    """``V = m^2 chi^2 / 2 + lambda chi^4 / 4``, with the mass supplied separately.

    The quadratic part is held apart from the quartic because on an
    expanding background the mass is time-dependent while the coupling is
    not, so the two cannot travel together.
    """

    mass: float = 1.0
    coupling: float = 0.0

    def __post_init__(self) -> None:
        if self.coupling < 0.0:
            raise ValueError(f"a negative quartic coupling is unbounded below, got {self.coupling}")

    @property
    def is_free(self) -> bool:
        return self.coupling == 0.0

    def value(self, field, mass_squared: float):
        return 0.5 * mass_squared * field**2 + 0.25 * self.coupling * field**4

    def gradient(self, field, mass_squared: float):
        return mass_squared * field + self.coupling * field**3

    def curvature(self, field, mass_squared: float):
        return mass_squared + 3.0 * self.coupling * field**2


@dataclass(frozen=True)
class Background:
    """A fixed FLRW background in conformal time: ``a(eta)`` and ``a''/a``."""

    def scale(self, time: float) -> float:
        return 1.0

    def curvature(self, time: float) -> float:
        """``a'' / a``, which is what the rescaling leaves behind."""
        return 0.0


def minkowski() -> Background:
    """``a = 1``: flat space, where the energy statements are exact."""
    return Background()


@dataclass(frozen=True)
class PowerLaw(Background):
    """``a = (eta / reference)^index``, so ``a''/a = p(p-1)/eta^2``.

    ``index = 1`` is radiation, and its ``a'' = 0`` makes a massless field
    exactly free -- the one expanding case where nothing is approximated.
    ``index = 2`` is matter.
    """

    index: float = 1.0
    reference: float = 1.0

    def scale(self, time: float) -> float:
        return float((time / self.reference) ** self.index)

    def curvature(self, time: float) -> float:
        return float(self.index * (self.index - 1.0) / time**2)


@dataclass(frozen=True)
class State:
    """The field, its conformal-time momentum, and the time."""

    field: np.ndarray
    momentum: np.ndarray
    time: float


@dataclass(frozen=True)
class ScalarField:
    """The solver: velocity Verlet on ``chi'' = lap chi - U'(chi)``."""

    lattice: Lattice
    potential: Potential
    background: Background = Background()

    def mass_squared(self, time: float) -> float:
        """``a^2 m^2 - a''/a``: what the rescaling turns the mass into."""
        scale = self.background.scale(time)
        return scale**2 * self.potential.mass**2 - self.background.curvature(time)

    def acceleration(self, field, time: float) -> np.ndarray:
        return laplacian(field, self.lattice) - self.potential.gradient(
            np.asarray(field, dtype=float), self.mass_squared(time)
        )

    def step(self, state: State, step: float) -> State:
        """One velocity-Verlet step, which is the whole reason for conformal time."""
        if step <= 0.0:
            raise ValueError(f"the time step must be positive, got {step}")
        half = state.momentum + 0.5 * step * self.acceleration(state.field, state.time)
        field = state.field + step * half
        end = state.time + step
        momentum = half + 0.5 * step * self.acceleration(field, end)
        return State(field=field, momentum=momentum, time=end)

    def run(self, state: State, final: float, steps: int, sample=None):
        """Integrate to ``final``, optionally recording ``sample(state)``."""
        if final <= state.time:
            raise ValueError(f"the final time {final} must exceed the initial {state.time}")
        step = (final - state.time) / int(steps)
        current = state
        history = [] if sample is None else [sample(current)]
        for _ in range(int(steps)):
            current = self.step(current, step)
            if sample is not None:
                history.append(sample(current))
        return current, history


# --- the three energies ---------------------------------------------------


def energy(state: State, solver: ScalarField) -> float:
    """``H``: kinetic, gradient and potential, summed over the lattice.

    The gradient term is written as ``-chi lap chi``, which on a periodic
    lattice equals ``|grad chi|^2`` by summation by parts and uses the same
    stencil the evolution does. Writing it with a separate difference
    operator would make ``H`` disagree with the scheme at ``O(h^2)`` and
    quietly spoil every statement below.
    """
    mass_squared = solver.mass_squared(state.time)
    cell = solver.lattice.spacing**solver.lattice.dimensions
    kinetic = 0.5 * np.sum(state.momentum**2)
    gradient = -0.5 * np.sum(state.field * laplacian(state.field, solver.lattice))
    return float(
        cell * (kinetic + gradient + np.sum(solver.potential.value(state.field, mass_squared)))
    )


def quadratic_invariant(state: State, solver: ScalarField, step: float) -> float:
    """``H - (dt^2/8) |a|^2``, exactly conserved when the theory is free.

    Every mode of a free lattice field is an independent harmonic
    oscillator, and velocity Verlet on ``x'' = -w^2 x`` conserves
    ``v^2/2 + w^2 x^2/2 - (dt^2/8) w^4 x^2`` exactly. Since ``w^2 x`` is
    minus the acceleration, the sum over modes is this, and it needs no
    transform to evaluate.

    With a quartic coupling it is no longer exact -- the modes are not
    independent -- and falls back to ``O(dt^2)``, about 2.4 times smaller
    than ``H``. :func:`modified_energy` is the one to use then.
    """
    acceleration = solver.acceleration(state.field, state.time)
    cell = solver.lattice.spacing**solver.lattice.dimensions
    return energy(state, solver) - cell * (step**2 / 8.0) * float(np.sum(acceleration**2))


def modified_energy(state: State, solver: ScalarField, step: float) -> float:
    """The Verlet shadow Hamiltonian: ``O(dt^4)`` for any potential.

    ``H + (dt^2/24) (2 <pi, U'' pi> - |U'|^2)``, where ``U`` is the whole
    configuration-space part -- gradient term included, so ``U''`` is the
    operator ``-lap + V''`` rather than just ``V''``. Measured at 16.0x per
    halving of the step on a quartic lattice, which is the fourth order the
    Baker-Campbell-Hausdorff expansion promises.
    """
    mass_squared = solver.mass_squared(state.time)
    cell = solver.lattice.spacing**solver.lattice.dimensions
    force = -solver.acceleration(state.field, state.time)

    curved = (
        -laplacian(state.momentum, solver.lattice)
        + solver.potential.curvature(state.field, mass_squared) * state.momentum
    )
    correction = 2.0 * np.sum(state.momentum * curved) - np.sum(force**2)
    return energy(state, solver) + cell * (step**2 / 24.0) * float(correction)


# --- ensembles and diagnostics -------------------------------------------


def thermal_ensemble(lattice: Lattice, mass: float, temperature: float, seed: int = 0) -> State:
    """A classical equilibrium ensemble at ``temperature``.

    Classical, not quantum: every mode carries ``T`` of energy on average,
    by equipartition, with no ``hbar omega / 2`` floor. That is the right
    initial condition for a classical-statistical evolution and the wrong
    one for a vacuum, and the difference matters because the classical
    theory has no ultraviolet fixed point -- the Rayleigh-Jeans divergence
    is physical here, and a finer lattice is a different ensemble.

    ``<|chi_k|^2> = T / omega_k^2`` and ``<|pi_k|^2> = T``, with the
    dispersion the *lattice* one, so equipartition holds mode by mode
    against the frequency the lattice actually has.
    """
    if temperature <= 0.0:
        raise ValueError(f"the temperature must be positive, got {temperature}")
    generator = np.random.default_rng(seed)
    frequency = np.sqrt(lattice.dispersion(mass**2))

    cell = lattice.spacing**lattice.dimensions
    field = np.fft.irfftn(
        np.fft.rfftn(generator.normal(size=lattice.shape))
        * np.sqrt(temperature / (cell * frequency**2)),
        s=lattice.shape,
        axes=tuple(range(lattice.dimensions)),
    )
    momentum = np.fft.irfftn(
        np.fft.rfftn(generator.normal(size=lattice.shape)) * np.sqrt(temperature / cell),
        s=lattice.shape,
        axes=tuple(range(lattice.dimensions)),
    )
    return State(field=field, momentum=momentum, time=0.0)


def mode_weights(lattice: Lattice) -> np.ndarray:
    """``1`` on the two self-conjugate planes of the half-grid, ``2`` elsewhere.

    A real field's transform is stored on half the grid, where every entry
    stands for a conjugate pair except on the ``k = 0`` and ``k = Nyquist``
    planes of the last axis, which are their own conjugates. Weighted this
    way the weights sum to the number of lattice sites -- the number of real
    degrees of freedom -- and ``sum_k w_k E_k`` equals the total energy to
    round-off. Counting the entries equally would get both wrong.
    """
    shape = (lattice.points,) * (lattice.dimensions - 1) + (lattice.points // 2 + 1,)
    weights = np.full(shape, 2.0)
    weights[..., 0] = 1.0
    if lattice.points % 2 == 0:
        weights[..., -1] = 1.0
    return weights


def mode_energies(state: State, solver: ScalarField, step: float | None = None) -> np.ndarray:
    """Energy per lattice mode, ``(|pi_k|^2 + omega_k^2 |chi_k|^2) / 2V``.

    Weighted by :func:`mode_weights` these sum to :func:`energy` exactly, so
    this is a decomposition of the energy rather than a proxy for it.

    Passing ``step`` subtracts each mode's ``(dt^2/8) omega^4 |chi_k|^2``,
    giving the per-mode form of :func:`quadratic_invariant`. For a free
    field *each one* is then conserved to round-off separately, because the
    modes are independent oscillators that never exchange anything. Without
    ``step`` they instead oscillate at ``O(dt^2)`` like ``H`` does, which
    says nothing about whether the modes are coupled -- so a test for
    independence has to use the corrected form or it measures the step size.

    Watching the corrected energies stop being conserved is how a quartic
    coupling makes itself visible, and it is the diagnostic the preheating
    scenario of issue #66 needs.
    """
    lattice = solver.lattice
    axes = tuple(range(lattice.dimensions))
    cell = lattice.spacing**lattice.dimensions
    field = np.fft.rfftn(state.field, axes=axes) * cell
    momentum = np.fft.rfftn(state.momentum, axes=axes) * cell
    frequency = lattice.dispersion(solver.mass_squared(state.time))

    total = np.abs(momentum) ** 2 + frequency * np.abs(field) ** 2
    if step is not None:
        total = total - (step**2 / 4.0) * frequency**2 * np.abs(field) ** 2
    return total / (2.0 * lattice.volume)
