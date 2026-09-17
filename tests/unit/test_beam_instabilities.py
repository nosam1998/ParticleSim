"""Current filamentation of counter-streaming beams (Section 3.3, M2)."""

import numpy as np
import pytest

from particlesim.scenarios.beams import (
    counter_streaming_beams,
    filamentation_growth_rate,
    filamentation_maximum_rate,
    magnetic_mode_amplitude,
)
from particlesim.scenarios.plasma import growth_rate
from particlesim.solvers.pic import YeeGrid, YeeSolver, advance

# --- linear theory ----------------------------------------------------------


def test_the_rate_rises_with_wavenumber_and_saturates_at_the_textbook_maximum():
    """``beta0 wp / sqrt(gamma)``.

    The rate never turns over, so the fastest mode is set by whatever cuts
    the spectrum off rather than by the dispersion relation. Reproducing that
    limit is what says the derivation put the Lorentz factors where they
    belong: the published forms differ by powers of gamma depending on
    whether the perturbing field lies along the drift or across it.
    """
    for drift in (0.3, 0.5, 0.8):
        ceiling = filamentation_maximum_rate(drift)
        gamma = 1.0 / np.sqrt(1.0 - drift**2)
        assert ceiling == pytest.approx(drift / np.sqrt(gamma))

        rates = [filamentation_growth_rate(k, drift) for k in (0.5, 1, 2, 5, 20, 200)]
        assert rates == sorted(rates)
        assert all(r < ceiling for r in rates)
        assert rates[-1] == pytest.approx(ceiling, rel=1e-4)


def test_a_uniform_perturbation_does_not_grow():
    assert filamentation_growth_rate(0.0, 0.5) == 0.0


def test_the_drift_must_be_a_speed():
    for bad in (0.0, 1.0, 1.5, -0.2):
        with pytest.raises(ValueError, match="strictly between zero and one"):
            filamentation_growth_rate(1.0, bad)


def test_the_rate_scales_with_the_plasma_frequency():
    """Everything in the dispersion relation carries ``wp``, so doubling the
    density's square root doubles the rate at the matched wavenumber."""
    base = filamentation_growth_rate(2.0, 0.5, plasma_frequency=1.0)
    scaled = filamentation_growth_rate(4.0, 0.5, plasma_frequency=2.0)
    assert scaled == pytest.approx(2.0 * base, rel=1e-12)


# --- the setup --------------------------------------------------------------


def test_the_beams_are_equal_opposite_and_neutral():
    grid = YeeGrid((32,), (0.1,))
    setup = counter_streaming_beams(grid, density=2.0, drift=0.4, per_cell=8)
    assert setup.plasma_frequency == pytest.approx(np.sqrt(2.0))
    assert setup.species.momentum[:, 1].sum() == pytest.approx(0.0, abs=1e-12)
    speeds = np.abs(setup.species.velocity[:, 1])
    np.testing.assert_allclose(speeds, 0.4, rtol=1e-12)
    total = setup.species.charge * setup.species.weight.sum()
    assert total == pytest.approx(-2.0 * grid.extent[0], rel=1e-12)
    # Neutral cell by cell and currents cancelling, so no initial field.
    assert np.abs(setup.fields.Bz).max() == 0.0


def test_the_seed_displaces_the_two_beams_oppositely():
    """What makes a current filament rather than a density ripple.

    Displacing both beams the same way perturbs the density with no net
    current, and nothing magnetic grows from it. The sign of the seed is
    therefore the whole difference between measuring this instability and
    measuring noise.
    """
    grid = YeeGrid((32,), (0.1,))
    setup = counter_streaming_beams(grid, drift=0.4, per_cell=8, amplitude=1e-2)
    positions = setup.species.position[:, 0]
    half = len(positions) // 2
    length = grid.extent[0]
    unperturbed = ((np.arange(half) + 0.5) / half) * length
    forward = np.mod(positions[:half] - unperturbed + length / 2, length) - length / 2
    backward = np.mod(positions[half:] - unperturbed + length / 2, length) - length / 2
    np.testing.assert_allclose(forward, -backward, atol=1e-12)
    assert np.abs(forward).max() > 1e-4


def test_the_magnetic_mode_is_what_is_measured():
    """``B_z`` and not ``E_x``. A run carrying both modes reports whichever
    grew faster if the wrong field is watched."""
    grid = YeeGrid((16,), (0.1,))
    setup = counter_streaming_beams(grid, drift=0.4, per_cell=4)
    assert magnetic_mode_amplitude(setup.fields, 1) == 0.0


# --- the benchmark ----------------------------------------------------------


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.parametrize(("drift", "wavenumber"), [(0.3, 2.0), (0.5, 2.0), (0.5, 5.0), (0.8, 2.0)])
def test_filamentation_growth_matches_linear_theory(drift, wavenumber):
    """Acceptance for issue #36: 5%.

    Four configurations spanning a factor of two in growth rate and a factor
    of two and a half in wavenumber, each within one per cent. The theory is
    the dispersion relation derived in the module, whose short-wavelength
    limit is the published maximum.
    """
    nx = 64
    grid = YeeGrid((nx,), (2.0 * np.pi / wavenumber / nx,))
    solver = YeeSolver(grid, courant=0.5)
    setup = counter_streaming_beams(grid, density=1.0, drift=drift, per_cell=64, amplitude=1e-4)
    assert setup.wavenumber == pytest.approx(wavenumber, rel=1e-12)

    fields, species = setup.fields, setup.species
    times, amplitudes = [], []
    for _ in range(int(round(40.0 / solver.dt))):
        fields, species = advance(solver, fields, species, order=1)
        times.append(fields.time)
        amplitudes.append(magnetic_mode_amplitude(fields, 1))

    assert growth_rate(times, amplitudes) == pytest.approx(setup.growth_rate, rel=0.05)
