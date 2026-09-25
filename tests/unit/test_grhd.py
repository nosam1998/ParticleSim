"""General-relativistic hydrodynamics in 3-D, and its coupling to BSSN (issue #57)."""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.solvers.hydro.evolve import RelativisticHydro, grid_for
from particlesim.solvers.hydro.grhd import (
    FLUID,
    CoupledEvolution,
    Primitives,
    Slice,
    Valencia,
    characteristic_speeds,
    homogeneous_state,
    stress_energy,
    to_conserved,
    to_primitives,
)
from particlesim.solvers.hydro.srhd import GammaLaw

EOS = GammaLaw(5.0 / 3.0)


def _along(values: np.ndarray, axis: int, width: int = 3) -> np.ndarray:
    """A 1-D profile laid along ``axis`` of a 3-D grid, constant across it."""
    shape = [width, width, width]
    shape[axis] = len(values)
    view = [1, 1, 1]
    view[axis] = len(values)
    return np.broadcast_to(values.reshape(view), shape).copy()


def _curved_slice(shape, seed=0) -> Slice:
    """A smooth, non-trivial slice: lapse, shift, metric and curvature all varying."""
    x, y, z = np.meshgrid(*(np.arange(n) / n * 2 * np.pi for n in shape), indexing="ij")
    wave = np.sin(x + 2 * y) * np.cos(z)
    alpha = 1.0 + 0.1 * wave
    beta = (0.05 * np.cos(x), 0.03 * np.sin(y + z), 0.02 * wave)
    off = 0.05 * np.sin(x - z)
    gamma = (
        (1.1 + 0.1 * wave, off, 0.02 * np.cos(y)),
        (off, 1.0 + 0.05 * np.cos(x), 0.03 * wave),
        (0.02 * np.cos(y), 0.03 * wave, 0.9 + 0.1 * np.sin(z)),
    )
    curvature = tuple(
        tuple(0.1 * gamma[i][j] * np.cos(x + i + j) for j in range(3)) for i in range(3)
    )
    return Slice(alpha, beta, gamma, curvature)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_on_flat_space_the_3d_fluid_is_the_1d_special_relativistic_one(axis):
    """Same reconstruction, same HLLE, same speeds: the right-hand sides agree to round-off."""
    n = 48
    x = (np.arange(n) + 0.5) / n
    rho = 1 + 0.3 * np.sin(2 * np.pi * x)
    v = 0.4 * np.cos(2 * np.pi * x)
    p = 1 + 0.2 * np.sin(4 * np.pi * x)
    reference = RelativisticHydro(
        grid=grid_for(1.0, n), eos=EOS, reconstruction="mc", solver="hlle"
    )
    expected = reference.rhs(reference.conserved(rho, v, p))

    shape = _along(rho, axis).shape
    zero = np.zeros(shape)
    velocity = [zero, zero, zero]
    velocity[axis] = _along(v, axis)
    slice_ = Slice.static(np.ones(shape))
    primitives = Primitives(_along(rho, axis), tuple(velocity), _along(p, axis))
    rates = Valencia((1 / n,) * 3, EOS, "mc").rhs(to_conserved(primitives, slice_, EOS), slice_)
    line = [slice(0, 1)] * 3
    line[axis] = slice(None)
    for index, name in enumerate(("D", f"S{axis}", "tau")):
        got = rates[name][tuple(line)].ravel()
        assert np.abs(got - expected[index]).max() < 1e-12 * np.abs(expected[index]).max()


def test_primitives_survive_a_round_trip_on_a_curved_slice():
    slice_ = _curved_slice((8, 8, 8))
    rng = np.random.default_rng(3)
    shape = (8, 8, 8)
    velocity = tuple(0.3 * rng.uniform(-1, 1, shape) for _ in range(3))
    primitives = Primitives(rng.uniform(0.5, 2, shape), velocity, rng.uniform(0.1, 3, shape))
    back = to_primitives(to_conserved(primitives, slice_, EOS), slice_, EOS)
    assert np.allclose(back.density, primitives.density, rtol=1e-11)
    assert np.allclose(back.pressure, primitives.pressure, rtol=1e-10)
    for got, expected in zip(back.velocity, primitives.velocity, strict=True):
        assert np.allclose(got, expected, atol=1e-12)


def test_rest_mass_is_conserved_to_round_off_on_a_curved_slice():
    """``D`` has no source and the fluxes telescope on a torus, whatever the slice."""
    shape = (12, 10, 8)
    slice_ = _curved_slice(shape)
    x, y, z = np.meshgrid(*(np.arange(n) / n * 2 * np.pi for n in shape), indexing="ij")
    primitives = Primitives(
        1 + 0.2 * np.sin(x) * np.cos(y),
        (0.2 * np.cos(z), 0.1 * np.sin(x + y), 0.05 * np.cos(x)),
        1 + 0.3 * np.cos(y - z),
    )
    fluid = to_conserved(primitives, slice_, EOS)
    rates = Valencia(tuple(1 / n for n in shape), EOS).rhs(fluid, slice_)
    assert abs(rates["D"].sum()) < 1e-11 * np.abs(rates["D"]).max()


def _hydrostatic(n: int, reconstruction: str) -> float:
    """The largest momentum rate of an isentropic star-like profile in a static lapse.

    At rest, the momentum equation is ``p' = -(e + p) alpha' / alpha``, whose
    isentropic solution is ``h alpha = constant``. The flux ``d_x (alpha p)``
    and the source ``-e d_x alpha`` cancel only if both are right.
    """
    x = (np.arange(n) + 0.5) / n
    lapse = 1 - 0.1 * np.cos(2 * np.pi * x)
    enthalpy = 1.3 / lapse
    kappa = 1.0
    rho = ((enthalpy - 1) * (EOS.gamma - 1) / (EOS.gamma * kappa)) ** (1 / (EOS.gamma - 1))
    slice_ = Slice.static(_along(lapse, 0, 2))
    shape = slice_.alpha.shape
    zero = np.zeros(shape)
    primitives = Primitives(
        _along(rho, 0, 2), (zero, zero, zero), _along(kappa * rho**EOS.gamma, 0, 2)
    )
    rates = Valencia((1 / n,) * 3, EOS, reconstruction).rhs(
        to_conserved(primitives, slice_, EOS), slice_
    )
    return float(np.abs(rates["S0"]).max())


def test_hydrostatic_balance_in_a_static_lapse_converges_at_second_order():
    """Unlimited slopes: order 1.9 to 2.0.

    The MC limiter clips at the extrema, and costs an order there.
    """
    linear = [_hydrostatic(n, "linear") for n in (32, 64, 128)]
    orders = np.log2(np.array(linear[:-1]) / np.array(linear[1:]))
    assert np.all(orders > 1.85)
    limited = [_hydrostatic(n, "mc") for n in (32, 64, 128)]
    assert np.log2(limited[-2] / limited[-1]) == pytest.approx(1.0, abs=0.1)


def test_characteristic_speeds_are_special_relativity_shifted_and_scaled():
    """Special relativity's speeds, scaled by the lapse and moved by ``-beta``."""
    shape = (2, 2, 2)
    one, zero = np.ones(shape), np.zeros(shape)
    primitives = Primitives(one, (0.5 * one, zero, zero), one)
    sound = np.sqrt(EOS.sound_speed_squared(1.0, 1.0))
    low, high = characteristic_speeds(primitives, Slice.static(one), EOS, 0)
    assert np.allclose(high, (0.5 + sound) / (1 + 0.5 * sound))
    assert np.allclose(low, (0.5 - sound) / (1 - 0.5 * sound))
    shifted = Slice(
        2 * one, (0.1 * one, zero, zero), Slice.static(one).gamma, Slice.static(one).curvature
    )
    low2, high2 = characteristic_speeds(primitives, shifted, EOS, 0)
    assert np.allclose(high2, 2 * high - 0.1)
    assert np.allclose(low2, 2 * low - 0.1)


def test_the_matter_terms_of_bssn_for_a_fluid_at_rest():
    """``rho_ADM = rho (1 + eps)``, ``S_i = 0`` and ``S_ij = p gamma_ij``.

    So ``d_t K`` gains ``4 pi (e + 3p)``, and nothing else changes.
    """
    state = homogeneous_state(1e-3, 0.5, EOS, (4, 4, 4))
    slice_ = Slice.from_bssn(state)
    fluid = {name: state[name] for name in FLUID}
    primitives = to_primitives(fluid, slice_, EOS)
    energy, momentum, stress = stress_energy(primitives, slice_, EOS)
    pressure = EOS.pressure(1e-3, 0.5)
    assert np.allclose(energy, 1e-3 * 1.5)
    assert all(np.allclose(m, 0.0) for m in momentum)
    assert np.allclose(stress[0][0], pressure) and np.allclose(stress[0][1], 0.0)

    class Geometry:
        shift_condition = "gamma_driver"

    rates = CoupledEvolution(Geometry(), Valencia((0.25,) * 3, EOS)).matter_rates(
        state, primitives, slice_
    )
    assert np.allclose(rates["trK"], 4 * np.pi * (1.5e-3 + 3 * pressure))
    assert all(np.allclose(rates[f"At{i}{j}"], 0.0) for i in range(3) for j in range(i, 3))
    assert all(np.allclose(rates[f"B{i}"], rates[f"Gt{i}"]) for i in range(3))


@pytest.mark.slow
@pytest.mark.benchmark
def test_a_homogeneous_universe_expands_as_friedmann_says():
    """BSSN and the fluid together, against the Friedmann equation integrated separately.

    An ideal gas with ``Gamma = 4/3`` and ``eps = 0.5``, so both the rest mass
    and the pressure matter. With a frozen lapse and shift, coordinate time
    is proper time. Nothing in the coupled code knows about Friedmann: the
    expansion comes from BSSN's ``d_t K`` with the matter term, and the
    cooling from the fluid's ``S_tau``.

    Measured: the scale factor's error falls 6.6e-7, 4.0e-8, 2.4e-9 as the
    step halves, fourth order, and ``K``, ``rho`` and ``p`` follow. The
    Hamiltonian constraint, matter included, stays at 1e-8 of ``16 pi e``.
    First use derives the frozen-gauge BSSN kernel, about five minutes.
    """
    from scipy.integrate import solve_ivp

    from particlesim.solvers.nr.bssn import Evolution

    eos = GammaLaw(4.0 / 3.0)
    rho0, eps0, n = 1e-3, 0.5, 6
    spacing = (1.0 / n,) * 3
    geometry = Evolution.build(
        spacing, backend="numpy", slicing="frozen", shift_condition="frozen", dissipation=0.0
    )
    coupled = CoupledEvolution(geometry, Valencia(spacing, eos))
    state = homogeneous_state(rho0, eps0, eos, (n, n, n))
    assert np.abs(coupled.constraints(state)["hamiltonian"]).max() < 1e-12

    gamma = eos.gamma

    def energy(a):
        return rho0 * a**-3 * (1 + eps0 * a ** (-3 * (gamma - 1)))

    hubble0 = np.sqrt(8 * np.pi * energy(1.0) / 3)
    duration = 1.0 / hubble0
    reference = solve_ivp(
        lambda t, y: [y[0] * np.sqrt(8 * np.pi * energy(y[0]) / 3)],
        (0, duration),
        [1.0],
        rtol=1e-13,
        atol=1e-15,
    )
    a_exact = reference.y[0, -1]
    errors = []
    for steps in (20, 40, 80):
        current = dict(state)
        for _ in range(steps):
            current = coupled.step(current, duration / steps)
        a = np.exp(2 * np.mean(current["phi"]))
        errors.append(abs(a / a_exact - 1))
        slice_ = Slice.from_bssn(current)
        primitives = to_primitives({k: current[k] for k in FLUID}, slice_, eos)
        assert np.mean(primitives.density) == pytest.approx(rho0 / a_exact**3, rel=1e-5)
        assert np.mean(current["trK"]) == pytest.approx(
            -3 * np.sqrt(8 * np.pi * energy(a_exact) / 3), rel=1e-5
        )
        constraint = np.abs(coupled.constraints(current)["hamiltonian"]).max()
        assert constraint < 1e-5 * 16 * np.pi * energy(a_exact)
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(orders > 3.8)
    assert errors[-1] < 1e-8
