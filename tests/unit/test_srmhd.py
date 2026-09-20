"""Relativistic MHD against the tensor it comes from, and the limit it reduces to.

Issue #59's acceptance is that the divergence of ``B`` is preserved to
round-off. It is, and the more interesting half of the test is that the
*other* divergence stencil applied to the same field is ``6e-4`` and always
was -- so the acceptance is only meaningful once it says which operator it
is talking about.

Everything else here is held to one of two things that need no reference: a
component of ``T^(mu nu)`` formed from its own definition, or the
unmagnetised solver that was tested separately, which this one has to
reproduce exactly at ``B = 0``.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.core.grid import UniformGrid
from particlesim.solvers.hydro import srhd
from particlesim.solvers.hydro.srhd import GammaLaw
from particlesim.solvers.hydro.srmhd import (
    MagnetisedTube,
    _speed_squared,
    comoving_field,
    conserved_to_primitive,
    fast_speeds,
    flux,
    lorentz,
    primitive_to_conserved,
)
from particlesim.solvers.hydro.transport import (
    StaggeredField,
    advect,
    apply_emf,
    from_vector_potential,
    transport_stage,
    unstaggered_advect,
)

GAMMA = 5.0 / 3.0


@pytest.fixture(scope="module")
def eos() -> GammaLaw:
    return GammaLaw(GAMMA)


def _one(value) -> float:
    return float(np.ravel(value)[0])


def _states(count, seed, speed=0.97):
    rng = np.random.default_rng(seed)
    density = 10.0 ** rng.uniform(-2.0, 1.0, count)
    pressure = 10.0 ** rng.uniform(-3.0, 1.0, count)
    direction = rng.normal(size=(3, count))
    direction /= np.linalg.norm(direction, axis=0)
    velocity = direction * rng.uniform(0.0, speed, count)
    field = rng.normal(size=(3, count)) * 10.0 ** rng.uniform(-2.0, 1.0, count)
    return density, velocity, pressure, field


# --- the variables, against the tensor they are components of -------------


def test_the_conserved_variables_and_fluxes_are_stress_tensor_components(eos):
    """``T = (rho h + b^2) u u + (p + b^2/2) eta - b b``, formed here from scratch.

    ``S^j`` is ``T^(0j)`` and ``F^x(S^j)`` is ``T^(xj)``; there is nothing
    else they could be. Building the tensor from the definition and
    comparing is the check a transcribed flux formula cannot pass by luck,
    and it agrees to ``5e-16``.
    """
    metric = np.diag([-1.0, 1.0, 1.0, 1.0])
    sample = _states(200, 29, speed=0.9)
    worst = 0.0
    for index in range(200):
        density = sample[0][index : index + 1]
        pressure = sample[2][index : index + 1]
        velocity = sample[1][:, index : index + 1]
        field = sample[3][:, index : index + 1]

        factor = _one(lorentz(velocity))
        time_part, space_part, squared = comoving_field(velocity, field)
        four_velocity = np.concatenate([[factor], factor * velocity.ravel()])
        four_field = np.concatenate([[_one(time_part)], space_part.ravel()])
        enthalpy = _one(eos.enthalpy(density, pressure))
        tensor = (
            (_one(density) * enthalpy + _one(squared)) * np.outer(four_velocity, four_velocity)
            + (_one(pressure) + 0.5 * _one(squared)) * metric
            - np.outer(four_field, four_field)
        )

        _, momentum, _ = primitive_to_conserved(density, velocity, pressure, field, eos)
        parts = flux(density, velocity, pressure, field, eos)
        scale = max(float(np.max(np.abs(tensor))), 1.0)
        worst = max(
            worst,
            float(np.max(np.abs(momentum.ravel() - tensor[0, 1:]))) / scale,
            float(np.max(np.abs(parts[1].ravel() - tensor[1, 1:]))) / scale,
        )
    assert worst < 5e-15


def test_the_normal_flux_of_the_normal_field_is_identically_zero(eos):
    """Which is why ``B^x`` is a constant of a one-dimensional sweep."""
    density, velocity, pressure, field = _states(200, 31)
    assert np.max(np.abs(flux(density, velocity, pressure, field, eos)[3][0])) == 0.0


def test_the_comoving_field_is_not_the_lab_field(eos):
    """``b^2 = B^2/W^2 + (v.B)^2``, which is the whole magnetic correction."""
    velocity = np.array([[0.8], [0.0], [0.0]])
    field = np.array([[0.0], [2.0], [0.0]])
    _, _, squared = comoving_field(velocity, field)
    assert _one(squared) == pytest.approx(4.0 * (1.0 - 0.64), rel=1e-12)
    still = comoving_field(np.zeros((3, 1)), field)[2]
    assert _one(still) == pytest.approx(4.0, rel=1e-12)


# --- the recovery ---------------------------------------------------------


def test_the_conserved_variables_round_trip(eos):
    """Up to a Lorentz factor of four, over four decades of density and pressure."""
    density, velocity, pressure, field = _states(5000, 5)
    conserved = primitive_to_conserved(density, velocity, pressure, field, eos)
    recovered = conserved_to_primitive(*conserved, field, eos)
    assert np.max(np.abs(recovered[0] / density - 1.0)) < 1e-9
    assert np.max(np.abs(recovered[1] - velocity)) < 1e-9
    assert np.max(np.abs(recovered[2] / pressure - 1.0)) < 1e-7


def test_the_momentum_is_not_a_lower_bound_on_the_inertia(eos):
    """Which is why the bracket subtracts ``B^2`` before using it.

    At ``B = 0`` the momentum is ``Z v`` and so below ``Z`` always. With a
    field it is ``(Z + B^2) v - (v.B) B``, and it sits *above* the true
    inertia in more than a quarter of random states -- so bracketing the
    root with ``|S|`` would put it outside, and bisection answers when that
    happens rather than failing. This is the same trap the unmagnetised
    recovery documents, in a place where it is easier to fall into.
    """
    density, velocity, pressure, field = _states(3000, 5)
    _, momentum, _ = primitive_to_conserved(density, velocity, pressure, field, eos)
    factor = lorentz(velocity)
    inertia = density * eos.enthalpy(density, pressure) * factor**2
    magnitude = np.sqrt(np.sum(momentum * momentum, axis=0))
    squared_field = np.sum(field * field, axis=0)

    assert np.mean(magnitude > inertia) > 0.2
    assert np.all(magnitude - squared_field <= inertia)


def test_the_recovery_bracket_contains_the_root(eos):
    """Checked by its sign, not by the accuracy of the answer it produced.

    A bracket that excludes the root does not make bisection fail; it makes
    it converge to the bracket end. The only way that shows is to evaluate
    the residual at the lower end and look at the sign.
    """
    density, velocity, pressure, field = _states(4000, 7)
    conserved_density, momentum, energy = primitive_to_conserved(
        density, velocity, pressure, field, eos
    )
    total = energy + conserved_density
    squared_field = np.sum(field * field, axis=0)
    projection = np.sum(momentum * field, axis=0)
    momentum_squared = np.sum(momentum * momentum, axis=0)
    low = np.maximum(np.sqrt(momentum_squared) - squared_field, np.finfo(float).tiny)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        speed_squared = _speed_squared(low, momentum_squared, projection, squared_field)
        reachable = (speed_squared < 1.0) & np.isfinite(speed_squared)
        factor = 1.0 / np.sqrt(1.0 - np.where(reachable, speed_squared, 0.0))
        aligned = projection / low
        residual = (
            low
            - total
            + 0.5 * squared_field
            + 0.5 * (squared_field * speed_squared - aligned**2)
            - (eos.gamma - 1.0) * (low - conserved_density * factor) / (eos.gamma * factor**2)
        )
    assert not np.any(reachable & (residual > 0.0))


@pytest.mark.parametrize("energy", [-0.9, -0.5, -0.2])
def test_recovery_refuses_conserved_variables_no_fluid_produces(eos, energy):
    """Bisection always returns something; the question is whether it is a root.

    At ``tau = -0.5`` the unguarded recovery came back with ``rho = 1`` and
    ``p = -0.333`` -- finite, plausible-looking, and not a state of any
    equation of state. The guard is the residual at the answer, which is
    below ``5e-16`` of the energy scale at a genuine root and nowhere near
    it here.
    """
    with pytest.raises(ValueError, match="do not correspond to any state"):
        conserved_to_primitive(
            np.array([1.0]),
            np.zeros((3, 1)),
            np.array([energy]),
            np.zeros((3, 1)),
            eos,
        )


# --- the limit it has to reduce to ----------------------------------------


@pytest.mark.parametrize("strength", [1e-12, 0.0])
def test_a_vanishing_field_reproduces_the_unmagnetised_module_bitwise(eos, strength):
    """Not approximately. The same numbers, to the last bit.

    The magnetic terms are additive corrections, so at zero field they must
    add exactly nothing -- and a check that allows a tolerance would pass a
    module whose magnetic terms were wrong by something small.
    """
    velocity = np.array([[0.6], [0.0], [0.0]])
    field = np.array([[1.0], [0.5], [-0.3]]) * strength
    magnetised = primitive_to_conserved(np.array([2.0]), velocity, np.array([0.7]), field, eos)
    plain = srhd.primitive_to_conserved(2.0, 0.6, 0.7, eos)
    assert _one(magnetised[0]) == _one(plain[0])
    assert _one(magnetised[1]) == _one(plain[1])
    assert _one(magnetised[2]) == _one(plain[2])

    magnetic_flux = flux(np.array([2.0]), velocity, np.array([0.7]), field, eos)
    plain_flux = srhd.flux(2.0, 0.6, 0.7, eos)
    assert _one(magnetic_flux[0]) == _one(plain_flux[0])
    assert _one(magnetic_flux[1]) == _one(plain_flux[1])
    assert _one(magnetic_flux[2]) == _one(plain_flux[2])


@pytest.mark.parametrize(("strength", "expected"), [(1e-3, 2.04e-7), (1e-6, 2.04e-13)])
def test_the_magnetic_correction_is_second_order_in_the_field(eos, strength, expected):
    """A thousandth of the field gives a millionth of the correction.

    Which is the statement that the correction is ``B^2`` and not ``B``, and
    it is what makes the bitwise agreement above a check on the magnetic
    terms rather than on their absence.
    """
    velocity = np.array([[0.6], [0.0], [0.0]])
    field = np.array([[1.0], [0.5], [-0.3]]) * strength
    magnetised = primitive_to_conserved(np.array([2.0]), velocity, np.array([0.7]), field, eos)
    plain = srhd.primitive_to_conserved(2.0, 0.6, 0.7, eos)
    assert abs(_one(magnetised[1]) - _one(plain[1])) == pytest.approx(expected, rel=0.05)


def test_the_fast_speeds_reduce_to_the_sound_speeds(eos):
    velocity = np.array([[0.6], [0.0], [0.0]])
    low, high = fast_speeds(np.array([2.0]), velocity, np.array([0.7]), np.zeros((3, 1)), eos)
    plain = srhd.characteristic_speeds(2.0, 0.6, 0.7, eos)
    assert _one(low) == pytest.approx(_one(plain[0]), abs=1e-15)
    assert _one(high) == pytest.approx(_one(plain[2]), abs=1e-15)


@pytest.mark.benchmark
def test_the_fast_speed_estimate_is_never_exceeded_by_the_jacobian(eos):
    """An upper bound, measured as one rather than asserted as one.

    The exact fast magnetosonic speed solves a quartic; this is the standard
    isotropic estimate. What HLL needs is that it is never smaller than the
    true outermost characteristic, so the true ones are obtained by
    differentiating the flux numerically and diagonalising. Over the states
    sampled here the ratio reaches 0.99999 and never passes one -- safe, and
    tight rather than lazy.
    """
    rng = np.random.default_rng(37)
    worst = 0.0
    for _ in range(25):
        density = np.array([float(10.0 ** rng.uniform(-1.0, 0.5))])
        pressure = np.array([float(10.0 ** rng.uniform(-2.0, 0.5))])
        direction = rng.normal(size=3)
        velocity = (direction / np.linalg.norm(direction) * rng.uniform(0.0, 0.8)).reshape(3, 1)
        field = (rng.normal(size=3) * float(10.0 ** rng.uniform(-1.0, 0.5))).reshape(3, 1)

        conserved_density, momentum, energy = primitive_to_conserved(
            density, velocity, pressure, field, eos
        )
        packed = np.array(
            [
                _one(conserved_density),
                momentum[0, 0],
                momentum[1, 0],
                momentum[2, 0],
                _one(energy),
                field[1, 0],
                field[2, 0],
            ]
        )

        def evaluate(vector, normal=field[0, 0]):
            whole = np.array([[normal], [vector[5]], [vector[6]]])
            recovered = conserved_to_primitive(
                np.array([vector[0]]),
                vector[1:4].reshape(3, 1),
                np.array([vector[4]]),
                whole,
                eos,
            )
            parts = flux(recovered[0], recovered[1], recovered[2], whole, eos)
            return np.array(
                [
                    _one(parts[0]),
                    parts[1][0, 0],
                    parts[1][1, 0],
                    parts[1][2, 0],
                    _one(parts[2]),
                    parts[3][1, 0],
                    parts[3][2, 0],
                ]
            )

        jacobian = np.empty((7, 7))
        for column in range(7):
            shift = np.zeros(7)
            shift[column] = 1e-6 * max(abs(packed[column]), 1.0)
            jacobian[:, column] = (evaluate(packed + shift) - evaluate(packed - shift)) / (
                2.0 * shift[column]
            )
        measured = np.linalg.eigvals(jacobian).real
        low, high = fast_speeds(density, velocity, pressure, field, eos)
        worst = max(worst, measured.max() / _one(high), measured.min() / _one(low))
        assert np.max(np.abs(measured)) < 1.0
    assert worst <= 1.0 + 1e-7


# --- constrained transport: the acceptance --------------------------------


def _seed(points):
    spacing = 1.0 / points
    faces = np.arange(points) * spacing
    potential = (
        np.sin(2.0 * np.pi * 2.0 * faces)[:, None] * np.cos(2.0 * np.pi * 2.0 * faces)[None, :]
        + 0.3 * np.cos(2.0 * np.pi * 3.0 * faces)[:, None] * np.sin(2.0 * np.pi * faces)[None, :]
    )
    centres = (np.arange(points) + 0.5) * spacing
    sweep_x = 0.4 + 0.2 * np.sin(2.0 * np.pi * centres)[:, None] * np.ones(points)[None, :]
    sweep_y = 0.3 + 0.2 * np.cos(2.0 * np.pi * centres)[None, :] * np.ones(points)[:, None]
    return from_vector_potential(potential, (spacing, spacing)), (sweep_x, sweep_y), spacing


def test_a_vector_potential_seeds_a_field_that_is_divergence_free_to_begin_with():
    """``B = curl(A)`` on the staggered grid: a discrete identity, not a sampling.

    Seeding by sampling an analytic ``B`` instead starts with a truncation
    error that constrained transport then preserves forever, which is the
    other way to fail this acceptance.
    """
    field, _, _ = _seed(64)
    assert field.relative_divergence() < 1e-15


@pytest.mark.benchmark
def test_the_staggered_divergence_stays_at_round_off(eos):
    """Issue #59's acceptance, and the number is relative to ``|B|/dx``.

    Two thousand steps of a sheared flow: ``3e-16`` after the first and
    ``2e-14`` at the end, which is accumulation of round-off rather than
    anything converging or diverging.
    """
    del eos
    field, velocity, spacing = _seed(64)
    step = 0.2 * spacing / max(np.max(np.abs(part)) for part in velocity)
    first = None
    for count in range(2000):
        field = transport_stage(field, velocity, step)
        if count == 0:
            first = field.relative_divergence()
    assert first < 1e-14
    assert field.relative_divergence() < 1e-12


def test_the_constraint_survives_an_electromotive_force_that_is_pure_noise():
    """Because the cancellation is in the stencil, not in the physics.

    No velocity, no fluxes, no equation of state -- random numbers on the
    corners, which grow the field by a factor of sixty. The divergence does
    not move off round-off, and that is the only way to demonstrate that the
    preservation has nothing to do with the scheme being any good.
    """
    field, _, _ = _seed(48)
    rng = np.random.default_rng(0)
    before = max(float(np.max(np.abs(part))) for part in field.centred())
    for _ in range(50):
        field = apply_emf(field, rng.normal(size=field.shape) * 20.0, 0.01)
    after = max(float(np.max(np.abs(part))) for part in field.centred())
    assert after > 10.0 * before
    assert field.relative_divergence() < 1e-14


@pytest.mark.benchmark
def test_the_unstaggered_update_loses_the_constraint():
    """Same fluxes, no corner. Thirteen orders of magnitude apart.

    Not a criticism of the unstaggered update: it is an ordinary
    conservative scheme for an ordinary variable, and it has no reason to
    keep a constraint nothing put into its stencil.
    """
    field, velocity, spacing = _seed(64)
    duration = 0.4
    carried = advect(field, velocity, duration)
    plain = unstaggered_advect(field.centred(), velocity, (spacing, spacing), duration)
    loose = StaggeredField(plain[0], plain[1], (spacing, spacing))

    assert carried.relative_divergence() < 1e-13
    assert loose.relative_centred_divergence() > 1e-2
    assert loose.relative_centred_divergence() / carried.relative_divergence() > 1e10


@pytest.mark.benchmark
def test_the_centred_divergence_is_a_truncation_error_and_converges():
    """The half of the acceptance that says which operator it is about.

    The same field, at the same instant, measured two ways: one is flat at
    machine epsilon because it is an identity, the other falls with the grid
    because it is an approximation. Quoting the second as "the divergence
    error" makes a correct scheme look broken.
    """
    staggered = []
    centred = []
    for points in (32, 64, 128):
        field, velocity, _ = _seed(points)
        field = advect(field, velocity, 0.1)
        staggered.append(field.relative_divergence())
        centred.append(field.relative_centred_divergence())

    assert max(staggered) < 1e-14
    assert max(staggered) / min(staggered) < 10.0
    assert centred[0] / centred[1] > 2.0
    assert centred[1] / centred[2] > 2.0
    assert centred[-1] > 100.0 * staggered[-1]


def test_the_staggered_field_refuses_shapes_it_cannot_use():
    with pytest.raises(ValueError, match="same shape"):
        StaggeredField(np.zeros((4, 4)), np.zeros((4, 5)))
    with pytest.raises(ValueError, match="two-dimensional"):
        StaggeredField(np.zeros(4), np.zeros(4))
    with pytest.raises(ValueError, match="two-dimensional"):
        from_vector_potential(np.zeros(4))
    field, _, _ = _seed(8)
    with pytest.raises(ValueError, match="shape of the grid"):
        apply_emf(field, np.zeros((4, 4)), 0.1)


def test_advect_refuses_a_negative_duration_or_a_bad_courant_number():
    field, velocity, _ = _seed(16)
    with pytest.raises(ValueError, match="must not be negative"):
        advect(field, velocity, -1.0)
    with pytest.raises(ValueError, match="Courant number"):
        advect(field, velocity, 0.1, courant=2.0)


# --- the magnetised sweep -------------------------------------------------


def _tube(points, eos, scheme="minmod"):
    solver = MagnetisedTube(
        UniformGrid([(0.0, 1.0)], (points,)),
        normal_field=0.5,
        eos=eos,
        reconstruction=scheme,
        boundary="outflow",
        courant=0.2,
    )
    left = solver.centres < 0.5
    density = np.where(left, 1.0, 0.125)
    pressure = np.where(left, 1.0, 0.1)
    field = np.stack([np.full(points, 0.5), np.where(left, 1.0, -1.0), np.zeros(points)])
    return solver, solver.conserved(density, np.zeros((3, points)), pressure, field)


@pytest.mark.benchmark
def test_the_normal_field_and_the_rest_mass_both_survive_a_magnetised_shock(eos):
    """``B^x`` exactly, rest mass to round-off -- for different reasons.

    ``B^x`` is constant because its flux is identically zero, so nothing is
    ever added to it. The rest mass is conserved because the update is a
    difference of fluxes. One is exact by construction and the other by
    telescoping, and a shock tube is where anything softer would show.

    The second claim has a precondition, which the next test supplies
    without one: an outflow boundary lets mass leave, so "to round-off"
    holds only while nothing has reached the edge. At 400 cells and
    ``t = 0.4`` the edges are still untouched and the drift is ``2e-14``;
    at 100 cells the diffused precursor of the fast wave has arrived and
    the drift is ``2e-8``, which is the boundary doing its job rather than
    the scheme failing at it.
    """
    solver, state = _tube(400, eos)
    start = solver.totals(state)
    evolved = solver.run(state, 0.4)
    density, _, pressure, field = solver.primitives(evolved)

    assert np.ptp(field[0]) == 0.0
    assert abs(solver.totals(evolved)[0] / start[0] - 1.0) < 1e-12
    assert density.min() > 0.0
    assert pressure.min() > 0.0


@pytest.mark.benchmark
def test_a_periodic_magnetised_run_conserves_everything_to_round_off(eos):
    """No boundary, no qualification: the flux differences telescope exactly.

    Every conserved variable, not only the rest mass, and through a run long
    enough for the perturbation to steepen. This is the unconditional form
    of the statement the shock tube can only make while its edges are
    untouched.
    """
    points = 200
    solver = MagnetisedTube(
        UniformGrid([(0.0, 1.0)], (points,)),
        normal_field=0.6,
        eos=eos,
        reconstruction="mc",
        boundary="periodic",
        courant=0.2,
    )
    phase = 2.0 * np.pi * solver.centres
    density = 1.0 + 0.4 * np.sin(phase)
    velocity = np.stack([0.3 * np.cos(phase), np.zeros(points), np.zeros(points)])
    field = np.stack([np.full(points, 0.6), 0.5 + 0.2 * np.cos(phase), 0.2 * np.sin(phase)])
    state = solver.conserved(density, velocity, np.ones(points), field)

    start = solver.totals(state)
    finish = solver.totals(solver.run(state, 1.0))
    for index, total in enumerate(start):
        if abs(total) > 1e-8:
            assert abs(finish[index] / total - 1.0) < 1e-12, index


@pytest.mark.benchmark
def test_the_magnetised_tube_converges_under_refinement(eos):
    """Self-convergence, because there is no exact solution to compare with.

    The relativistic MHD Riemann problem has seven waves and no closed form
    to sample, unlike its unmagnetised counterpart -- so this is the weaker
    statement that refinement changes the answer less and less, and it is
    labelled as the weaker statement rather than dressed up as an accuracy.
    """
    profiles = {}
    for points in (200, 400, 800):
        solver, state = _tube(points, eos)
        profiles[points] = (solver.centres, solver.primitives(solver.run(state, 0.4))[0])

    coarse = float(np.mean(np.abs(np.interp(profiles[200][0], *profiles[400]) - profiles[200][1])))
    fine = float(np.mean(np.abs(np.interp(profiles[400][0], *profiles[800]) - profiles[400][1])))
    assert fine < coarse


@pytest.mark.benchmark
@pytest.mark.parametrize("normal", [0.5, 1.0])
def test_a_standing_alfven_wave_oscillates_at_the_alfven_speed(eos, normal):
    """``cos(2 pi v_A t)`` with ``v_A = B^x / sqrt(rho h + B^2)``, to ``1e-4``.

    A transverse field perturbation with the fluid at rest is the sum of the
    two Alfven waves, so it stands rather than travels and its amplitude
    follows a cosine. Sampling that at several times measures the
    relativistic Alfven speed through a sign change -- a closed form for the
    magnetic part of the solver, where the shock tube has none.
    """
    points = 256
    amplitude = 1e-4
    solver = MagnetisedTube(
        UniformGrid([(0.0, 1.0)], (points,)),
        normal_field=normal,
        eos=eos,
        reconstruction="mc",
        boundary="periodic",
        courant=0.2,
    )
    field = np.stack(
        [
            np.full(points, normal),
            amplitude * np.sin(2.0 * np.pi * solver.centres),
            np.zeros(points),
        ]
    )
    state = solver.conserved(np.ones(points), np.zeros((3, points)), np.ones(points), field)
    enthalpy = eos.enthalpy(1.0, 1.0)
    alfven = normal / np.sqrt(enthalpy + normal**2 + amplitude**2)

    initial = np.fft.rfft(field[1])[1]
    for elapsed in (0.2, 0.5, 1.0):
        carried = solver.primitives(solver.run(state, elapsed))[3]
        ratio = float(np.real(np.fft.rfft(carried[1])[1] / initial))
        assert ratio == pytest.approx(np.cos(2.0 * np.pi * alfven * elapsed), abs=2e-3)


def test_a_field_free_tube_reproduces_the_unmagnetised_solver(eos):
    """The whole machinery, switched off, has to give the solver it reduces to.

    Against HLLE rather than HLLC, because the magnetised sweep carries the
    two-state solver only -- so this compares the same Riemann solver on the
    same data, which is what makes a ``1e-10`` agreement meaningful rather
    than a coincidence between two diffusive schemes.
    """
    from particlesim.solvers.hydro.evolve import RelativisticHydro, riemann_initial_data

    points = 100
    grid = UniformGrid([(0.0, 1.0)], (points,))
    plain = RelativisticHydro(
        grid, eos, reconstruction="minmod", solver="hlle", boundary="outflow", courant=0.2
    )
    magnetised = MagnetisedTube(
        grid, normal_field=0.0, eos=eos, reconstruction="minmod", boundary="outflow", courant=0.2
    )
    left = (1.0, 0.0, 1.0)
    right = (0.125, 0.0, 0.1)
    reference = plain.run(riemann_initial_data(plain, left, right), 0.2)

    inside = magnetised.centres < 0.5
    state = magnetised.conserved(
        np.where(inside, left[0], right[0]),
        np.zeros((3, points)),
        np.where(inside, left[2], right[2]),
        np.zeros((3, points)),
    )
    evolved = magnetised.run(state, 0.2)
    assert np.max(np.abs(evolved[0] - reference[0])) < 1e-10
    assert np.max(np.abs(evolved[4] - reference[2])) < 1e-10


def test_the_tube_refuses_a_configuration_it_cannot_run(eos):
    grid = UniformGrid([(0.0, 1.0)], (16,))
    with pytest.raises(ValueError, match="unknown reconstruction"):
        MagnetisedTube(grid, 0.5, eos, reconstruction="spline")
    with pytest.raises(ValueError, match="unknown boundary"):
        MagnetisedTube(grid, 0.5, eos, boundary="reflecting")
    with pytest.raises(ValueError, match="Courant number"):
        MagnetisedTube(grid, 0.5, eos, courant=0.0)
    with pytest.raises(ValueError, match="one-dimensional"):
        MagnetisedTube(UniformGrid([(0.0, 1.0)] * 2, (4, 4)), 0.5, eos)
