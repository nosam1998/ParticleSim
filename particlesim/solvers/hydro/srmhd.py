"""Special-relativistic magnetohydrodynamics in conservative form.

Issue #59. The same Valencia bookkeeping as :mod:`particlesim.solvers.hydro.srhd`
with the electromagnetic stress folded in, which changes one thing
qualitatively: the primitive recovery is no longer a root-find whose
unknown is the pressure. It is one whose unknown is ``Z = rho h W^2``, and
the velocity is recovered from ``Z`` by a closed form rather than being the
thing solved for.

**Every relation here is checked against the stress tensor it came from.**
``T^(mu nu) = (rho h + b^2) u^mu u^nu + (p + b^2/2) eta^(mu nu) - b^mu b^nu``
is one line; the conserved variables and fluxes are particular components of
it, and the suite forms the tensor from the definition and compares. The
agreement is ``3e-16``, which is a statement about algebra rather than about
a reference, and it is the check that a transcribed flux formula cannot pass
by luck.

**And the whole system reduces to the unmagnetised one exactly.** At ``B =
0`` the conserved variables here are *bitwise* those of
:mod:`~particlesim.solvers.hydro.srhd`, and switching the field on
introduces a difference that scales as ``B^2``: ``2.0e-7`` at ``|B| = 1e-3``
and ``2.0e-13`` at ``1e-6``, a factor of a million for a factor of a
thousand. So the magnetic terms are confirmed against a module that was
tested separately, in a limit where the answer is already known.

**The signal speeds are an upper bound, and are measured as one.** The exact
fast magnetosonic speed is a root of a quartic; what is used here is the
standard estimate that replaces it with an isotropic ``c_ms^2 = c_s^2 +
v_A^2 - c_s^2 v_A^2`` carried into the lab frame by the relativistic
addition law. That is exact when the field is along the sweep or across it
and an over-estimate in between, which costs a little diffusion in HLL and
never costs stability. Over four hundred random states the true eigenvalues
of the numerically differentiated Jacobian never exceed it, and reach
0.99999 of it, so the bound is safe and tight rather than safe and lazy.

**What is not here.** HLLC and HLLD, which need the intermediate states of a
seven-wave fan; multi-dimensional sweeps; and the divergence constraint,
which does not arise in one dimension because ``F^x(B^x)`` is identically
zero and ``B^x`` is therefore a constant of the evolution. The constraint is
:mod:`particlesim.solvers.hydro.transport`'s subject, where it has content.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field

import numpy as np

from particlesim.core.grid import UniformGrid
from particlesim.solvers.hydro.reconstruct import GHOSTS, SCHEMES, reconstruct
from particlesim.solvers.hydro.srhd import ATMOSPHERE, GammaLaw

#: Largest Lorentz factor the recovery will report before refusing.
MAXIMUM_LORENTZ = 1e4

#: Largest velocity a reconstruction may produce.
VELOCITY_CEILING = 1.0 - 1e-12

#: Shu-Osher's third-order strong stability-preserving Runge-Kutta.
SSP_RK3 = ((0.0, 1.0), (0.75, 0.25), (1.0 / 3.0, 2.0 / 3.0))


def _vector(values) -> np.ndarray:
    """A ``(3, N)`` field, from anything shaped like one."""
    array = np.atleast_2d(np.asarray(values, dtype=float))
    if array.shape[0] != 3:
        raise ValueError(f"a three-vector field has leading dimension 3, got {array.shape}")
    return array


def lorentz(velocity) -> np.ndarray:
    velocity = _vector(velocity)
    return 1.0 / np.sqrt(1.0 - np.sum(velocity * velocity, axis=0))


def comoving_field(velocity, field):
    """``(b^0, b^i, b^2)``: the field an observer moving with the fluid measures.

    ``b^2`` is what appears in the stress tensor, and it is *not* ``B^2``:
    the two differ by ``B^2(1 - 1/W^2) + (v.B)^2``, which is the whole of
    the difference between a magnetised and an unmagnetised relativistic
    fluid at fixed lab-frame field.
    """
    velocity = _vector(velocity)
    field = _vector(field)
    factor = lorentz(velocity)
    projection = np.sum(velocity * field, axis=0)
    time_part = factor * projection
    space_part = field / factor + time_part * velocity
    squared = np.sum(field * field, axis=0) / factor**2 + projection**2
    return time_part, space_part, squared


def primitive_to_conserved(density, velocity, pressure, field, eos: GammaLaw):
    """``(D, S^i, tau)`` with the electromagnetic stress included.

    ``S^i = (rho h W^2 + B^2) v^i - (v.B) B^i`` and
    ``tau + D = rho h W^2 - p + B^2/2 + (B^2 v^2 - (v.B)^2)/2``, both of
    which are components of the stress tensor rather than separate
    definitions -- which is how the suite checks them.
    """
    density = np.atleast_1d(np.asarray(density, dtype=float))
    pressure = np.atleast_1d(np.asarray(pressure, dtype=float))
    velocity = _vector(velocity)
    field = _vector(field)

    factor = lorentz(velocity)
    enthalpy = eos.enthalpy(density, pressure)
    inertia = density * enthalpy * factor**2
    squared_field = np.sum(field * field, axis=0)
    projection = np.sum(velocity * field, axis=0)
    speed_squared = np.sum(velocity * velocity, axis=0)

    conserved_density = density * factor
    momentum = (inertia + squared_field) * velocity - projection * field
    energy = (
        inertia
        - pressure
        + 0.5 * squared_field
        + 0.5 * (squared_field * speed_squared - projection**2)
    )
    return conserved_density, momentum, energy - conserved_density


def _speed_squared(inertia, momentum_squared, projection, squared_field):
    """``v^2`` in closed form once ``Z`` is known, which is why ``Z`` is the unknown."""
    return (momentum_squared * inertia**2 + projection**2 * (2.0 * inertia + squared_field)) / (
        inertia**2 * (inertia + squared_field) ** 2
    )


def conserved_to_primitive(
    conserved_density, momentum, energy, field, eos: GammaLaw, tolerance=1e-13
):
    """Recover ``(rho, v^i, p)`` by bisection in ``Z = rho h W^2``.

    Pressure is not a good unknown once there is a field: the velocity no
    longer follows from it in one step, because ``S^i`` mixes ``v^i`` with
    ``B^i``. ``Z`` is, and ``v^2(Z)`` is closed form -- so the root-find
    stays one-dimensional and the rest is substitution.

    The bracket's lower end is where ``v^2`` reaches one, found by walking
    up from the field's own scale rather than by a floor, for the reason
    :mod:`~particlesim.solvers.hydro.srhd` sets out at length: a floor above
    the root does not make bisection fail, it makes it answer.
    """
    conserved_density = np.atleast_1d(np.asarray(conserved_density, dtype=float))
    energy = np.atleast_1d(np.asarray(energy, dtype=float))
    momentum = _vector(momentum)
    field = _vector(field)

    total = energy + conserved_density
    squared_field = np.sum(field * field, axis=0)
    projection = np.sum(momentum * field, axis=0)
    momentum_squared = np.sum(momentum * momentum, axis=0)

    # Below the critical inertia the implied speed exceeds one, and the
    # residual there is deliberately minus infinity so that bisection walks
    # up out of it. Evaluating the expressions on the way overflows, which is
    # arithmetic doing what it was asked to, not a warning worth raising.
    def residual(inertia):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            return _residual(inertia)

    def _residual(inertia):
        speed_squared = _speed_squared(inertia, momentum_squared, projection, squared_field)
        unphysical = (speed_squared >= 1.0) | ~np.isfinite(speed_squared)
        speed_squared = np.where(unphysical, 0.0, speed_squared)
        factor = 1.0 / np.sqrt(1.0 - speed_squared)
        aligned = projection / inertia
        from_energy = (
            inertia
            - total
            + 0.5 * squared_field
            + 0.5 * (squared_field * speed_squared - aligned**2)
        )
        from_state = (
            (eos.gamma - 1.0) * (inertia - conserved_density * factor) / (eos.gamma * factor**2)
        )
        return np.where(unphysical, -np.inf, from_energy - from_state)

    # ``|S|`` is not a lower bound on ``Z`` once there is a field: ``S`` picks
    # up ``B^2 v``, and it sits above the true inertia in 28% of random states.
    # ``|S| - B^2`` does bound it, and the walk below then stops at the only
    # hard limit there is, where ``v^2`` reaches one. No atmosphere floor: a
    # floor above the root does not make bisection fail, it makes it answer.
    low = np.maximum(np.sqrt(momentum_squared) - squared_field, np.finfo(float).tiny)
    high = np.maximum(2.0 * (total + squared_field), 10.0 * low)

    for _ in range(200):
        climbing = residual(high) < 0.0
        if not climbing.any():
            break
        high = np.where(climbing, 2.0 * high, high)

    unbracketed = residual(low) > 0.0
    if np.any(unbracketed):
        index = int(np.argmax(unbracketed))
        raise ValueError(
            f"no inertia between {low[index]:.6g} and {high[index]:.6g} satisfies the recovery "
            f"for (D, tau) = ({conserved_density[index]:.6g}, {energy[index]:.6g}) with "
            f"B^2 = {squared_field[index]:.6g}; these conserved variables do not correspond "
            "to any state of this equation of state"
        )

    lower, upper = np.log(low), np.log(high)
    for _ in range(400):
        middle = 0.5 * (lower + upper)
        above = residual(np.exp(middle)) > 0.0
        upper = np.where(above, middle, upper)
        lower = np.where(above, lower, middle)
        if np.all(upper - lower < tolerance):
            break

    inertia = np.exp(0.5 * (lower + upper))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        speed_squared = np.clip(
            _speed_squared(inertia, momentum_squared, projection, squared_field), 0.0, None
        )
    factor = 1.0 / np.sqrt(1.0 - speed_squared)
    if np.any(factor > MAXIMUM_LORENTZ):
        raise ValueError(
            f"recovery reached a Lorentz factor of {float(np.max(factor)):.3g}, past the "
            f"{MAXIMUM_LORENTZ:.0g} this solver reports"
        )
    aligned = projection / inertia
    velocity = (momentum + aligned * field) / (inertia + squared_field)
    density = conserved_density / factor
    pressure = (eos.gamma - 1.0) * (inertia - conserved_density * factor) / (eos.gamma * factor**2)

    # Bisection always returns something. Whether that something is a root
    # is a separate question, and for conserved variables no fluid produces
    # it is not: unguarded, ``tau = -0.5`` came back with ``rho = 1`` and
    # ``p = -0.333`` -- finite, plausible, and not a state of anything. At a
    # genuine root the residual is below 5e-16 of the energy scale, so there
    # are eight orders of headroom between the two cases.
    failed = (
        ~np.isfinite(pressure)
        | (pressure <= 0.0)
        | ~np.isfinite(density)
        | (density <= 0.0)
        | (np.abs(residual(inertia)) > 1e-8 * (total + squared_field))
    )
    if np.any(failed):
        index = int(np.argmax(failed))
        raise ValueError(
            f"the recovery did not reach a physical state for (D, tau) = "
            f"({conserved_density[index]:.6g}, {energy[index]:.6g}) with "
            f"B^2 = {squared_field[index]:.6g}: it ended at density {density[index]:.6g} "
            f"and pressure {pressure[index]:.6g}, so these conserved variables do not "
            "correspond to any state of this equation of state"
        )
    return density, velocity, pressure


def flux(density, velocity, pressure, field, eos: GammaLaw, axis: int = 0):
    """``(F(D), F(S^j), F(tau), F(B^j))`` along ``axis``, as stress-tensor components.

    ``F^x(S^j) = T^(xj)``, ``F^x(tau) = T^(0x) - D v^x`` and
    ``F^x(B^j) = v^x B^j - v^j B^x``. The last is identically zero for
    ``j = x``, which is why ``B^x`` is a constant of a one-dimensional
    evolution and why the divergence constraint has no content here.
    """
    density = np.atleast_1d(np.asarray(density, dtype=float))
    pressure = np.atleast_1d(np.asarray(pressure, dtype=float))
    velocity = _vector(velocity)
    field = _vector(field)

    factor = lorentz(velocity)
    _, space_part, squared = comoving_field(velocity, field)
    conserved_density, momentum, _ = primitive_to_conserved(density, velocity, pressure, field, eos)

    sweep = velocity[axis]
    flux_density = conserved_density * sweep
    flux_momentum = momentum * sweep - field[axis] * space_part / factor
    flux_momentum[axis] += pressure + 0.5 * squared
    flux_energy = momentum[axis] - conserved_density * sweep
    flux_field = sweep * field - velocity * field[axis]
    return flux_density, flux_momentum, flux_energy, flux_field


def fast_speeds(density, velocity, pressure, field, eos: GammaLaw, axis: int = 0):
    """An upper bound on the outermost characteristics, and a measured one.

    The exact fast magnetosonic speed solves a quartic. This is the standard
    isotropic estimate -- ``c_ms^2 = c_s^2 + v_A^2 - c_s^2 v_A^2`` with
    ``v_A^2 = b^2/(rho h + b^2)`` -- carried into the lab frame by the
    relativistic addition law. Exact when the field is along the sweep or
    across it, an over-estimate in between; never an under-estimate, which
    is what HLL needs, and the suite checks that against the Jacobian's own
    eigenvalues rather than taking it on faith.
    """
    density = np.atleast_1d(np.asarray(density, dtype=float))
    pressure = np.atleast_1d(np.asarray(pressure, dtype=float))
    velocity = _vector(velocity)

    enthalpy = eos.enthalpy(density, pressure)
    _, _, squared = comoving_field(velocity, field)
    sound = eos.sound_speed_squared(density, pressure)
    alfven = squared / (density * enthalpy + squared)
    combined = sound + alfven - sound * alfven

    speed_squared = np.sum(velocity * velocity, axis=0)
    sweep = velocity[axis]
    denominator = 1.0 - speed_squared * combined
    discriminant = combined * (1.0 - speed_squared) * (denominator - sweep**2 * (1.0 - combined))
    spread = np.sqrt(np.maximum(discriminant, 0.0))
    centre = (1.0 - combined) * sweep
    return (centre - spread) / denominator, (centre + spread) / denominator


def _pack(conserved_density, momentum, energy, field) -> np.ndarray:
    """``(D, S^x, S^y, S^z, tau, B^y, B^z)``: what a one-dimensional sweep evolves."""
    return np.stack(
        [conserved_density, momentum[0], momentum[1], momentum[2], energy, field[1], field[2]]
    )


def hll(left, right, normal_field, eos: GammaLaw):
    """One averaged state across the whole fan. Positive, diffusive, and enough.

    HLLD resolves the Alfven and slow waves that this folds into the
    average, and needs the intermediate states of a seven-wave fan to do
    it. What is here is the two-state solver, and the cost is stated rather
    than hidden: a rotational discontinuity comes out smeared.
    """
    fluxes = []
    states = []
    for density, velocity, pressure, field in (left, right):
        conserved_density, momentum, energy = primitive_to_conserved(
            density, velocity, pressure, field, eos
        )
        parts = flux(density, velocity, pressure, field, eos)
        states.append(_pack(conserved_density, momentum, energy, field))
        fluxes.append(_pack(parts[0], parts[1], parts[2], parts[3]))

    left_low, left_high = fast_speeds(*left, eos)
    right_low, right_high = fast_speeds(*right, eos)
    low = np.minimum(np.minimum(left_low, right_low), 0.0)
    high = np.maximum(np.maximum(left_high, right_high), 0.0)
    del normal_field
    return (high * fluxes[0] - low * fluxes[1] + high * low * (states[1] - states[0])) / (
        high - low
    )


@dataclass
class MagnetisedTube:
    """A one-dimensional relativistic MHD sweep. ``B^x`` is a constant of it."""

    grid: UniformGrid
    normal_field: float
    eos: GammaLaw = dataclass_field(default_factory=GammaLaw)
    reconstruction: str = "minmod"
    boundary: str = "outflow"
    courant: float = 0.3
    recovery_tolerance: float = 1e-13

    def __post_init__(self) -> None:
        if self.grid.ndim != 1:
            raise ValueError(f"this solver is one-dimensional, got {self.grid.ndim} axes")
        if self.reconstruction not in SCHEMES:
            raise ValueError(
                f"unknown reconstruction {self.reconstruction!r}; expected one of {SCHEMES}"
            )
        if self.boundary not in ("periodic", "outflow"):
            raise ValueError(
                f"unknown boundary {self.boundary!r}; expected 'periodic' or 'outflow'"
            )
        if not 0.0 < self.courant <= 1.0:
            raise ValueError(f"the Courant number must lie in (0, 1], got {self.courant}")

    @property
    def spacing(self) -> float:
        return float(self.grid.spacing[0])

    @property
    def centres(self) -> np.ndarray:
        return self.grid.axis(0)

    def _field(self, transverse) -> np.ndarray:
        return np.stack([np.full(transverse.shape[1], self.normal_field), *transverse])

    def conserved(self, density, velocity, pressure, field) -> np.ndarray:
        conserved_density, momentum, energy = primitive_to_conserved(
            density, velocity, pressure, field, self.eos
        )
        return _pack(conserved_density, momentum, energy, _vector(field))

    def primitives(self, state):
        field = self._field(np.asarray(state)[5:7])
        density, velocity, pressure = conserved_to_primitive(
            state[0], state[1:4], state[4], field, self.eos, self.recovery_tolerance
        )
        return density, velocity, pressure, field

    def _pad(self, values: np.ndarray) -> np.ndarray:
        if self.boundary == "periodic":
            return np.concatenate([values[-GHOSTS:], values, values[:GHOSTS]])
        return np.concatenate([np.full(GHOSTS, values[0]), values, np.full(GHOSTS, values[-1])])

    def _faces(self, density, velocity, pressure, field):
        count = self.grid.shape[0]
        window = slice(GHOSTS - 1, count + GHOSTS)
        stack = [density, *velocity, pressure, field[1], field[2]]
        sides = ([], [])
        for values in stack:
            low, high = reconstruct(self._pad(values), self.reconstruction)
            sides[0].append(low[window])
            sides[1].append(high[window])

        floors = (ATMOSPHERE * float(np.max(density)), ATMOSPHERE * float(np.max(pressure)))
        built = []
        for side in sides:
            speed = np.clip(np.stack(side[1:4]), -VELOCITY_CEILING, VELOCITY_CEILING)
            magnitude = np.sqrt(np.sum(speed * speed, axis=0))
            excess = magnitude > VELOCITY_CEILING
            speed = np.where(
                excess, speed * VELOCITY_CEILING / np.maximum(magnitude, 1e-300), speed
            )
            built.append(
                (
                    np.maximum(side[0], floors[0]),
                    speed,
                    np.maximum(side[4], floors[1]),
                    np.stack([np.full(side[5].shape, self.normal_field), side[5], side[6]]),
                )
            )
        return built[0], built[1]

    def rhs(self, state: np.ndarray) -> np.ndarray:
        left, right = self._faces(*self.primitives(state))
        fluxes = hll(left, right, self.normal_field, self.eos)
        return -(fluxes[:, 1:] - fluxes[:, :-1]) / self.spacing

    def time_step(self, state: np.ndarray) -> float:
        density, velocity, pressure, field = self.primitives(state)
        low, high = fast_speeds(density, velocity, pressure, field, self.eos)
        fastest = float(max(np.max(np.abs(low)), np.max(np.abs(high))))
        return self.courant * self.spacing / fastest

    def step(self, state: np.ndarray, step_size: float) -> np.ndarray:
        updated = state
        for old, new in SSP_RK3:
            updated = old * state + new * (updated + step_size * self.rhs(updated))
        return updated

    def run(self, state: np.ndarray, duration: float, steps: int | None = None) -> np.ndarray:
        if duration < 0.0:
            raise ValueError(f"the duration must not be negative, got {duration}")
        if steps is not None:
            if steps < 1:
                raise ValueError(f"a fixed-step run needs at least one step, got {steps}")
            size = duration / steps
            for _ in range(steps):
                state = self.step(state, size)
            return state
        elapsed = 0.0
        while elapsed < duration:
            size = min(self.time_step(state), duration - elapsed)
            state = self.step(state, size)
            elapsed += size
        return state

    def totals(self, state: np.ndarray) -> np.ndarray:
        return np.sum(np.asarray(state), axis=1) * self.spacing


__all__ = [
    "MAXIMUM_LORENTZ",
    "SSP_RK3",
    "VELOCITY_CEILING",
    "MagnetisedTube",
    "comoving_field",
    "conserved_to_primitive",
    "fast_speeds",
    "flux",
    "hll",
    "lorentz",
    "primitive_to_conserved",
]
