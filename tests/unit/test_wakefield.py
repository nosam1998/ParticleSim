"""One-dimensional laser wakefield (design doc Section 10, Milestone 2)."""

from dataclasses import replace

import numpy as np
import pytest

from particlesim.scenarios.wakefield import (
    LaserWakefield,
    gaussian_wake,
    linear_wake_amplitude,
    measure_wake,
    nonlinear_wake,
    resonant_benchmark,
    resonant_length,
    uniform_plasma,
)
from particlesim.solvers.pic import (
    LaserPulse,
    PerfectlyMatchedLayer,
    PlaneWaveSource,
    YeeGrid,
    YeeSolver,
    advance,
)

WAVELENGTH = 1.0
DENSITY_RATIO = 0.1  # omega_p / omega_0


def _plasma_scales(ratio: float = DENSITY_RATIO):
    omega0 = 2 * np.pi / WAVELENGTH
    plasma_frequency = ratio * omega0
    return plasma_frequency, plasma_frequency, plasma_frequency**2


# --- the one-dimensional theory ---------------------------------------------


def test_the_nonlinear_wake_reduces_to_the_linear_formula():
    """Where the two must agree, they do, which is what licenses using the
    nonlinear solver where only it applies."""
    _, k_p, _ = _plasma_scales()
    sigma = resonant_length(k_p)
    behind = 4 * sigma + 0.5 * 2 * np.pi / k_p
    for a0 in (0.01, 0.05, 0.1):
        measured = gaussian_wake(a0, sigma, k_p).amplitude(behind)
        assert measured == pytest.approx(linear_wake_amplitude(a0, sigma, k_p), rel=0.01)


def test_the_wake_falls_below_the_linear_formula_as_the_drive_grows():
    """Not a correction anyone can ignore.

    A strong pulse lengthens the plasma wave, so a pulse chosen resonant for
    the linear response stops being resonant. At ``a0 = 0.8`` the linear
    formula is eleven per cent high, which is twice this benchmark's
    tolerance.
    """
    _, k_p, _ = _plasma_scales()
    sigma = resonant_length(k_p)
    behind = 4 * sigma + 0.5 * 2 * np.pi / k_p
    ratios = [
        gaussian_wake(a0, sigma, k_p).amplitude(behind) / linear_wake_amplitude(a0, sigma, k_p)
        for a0 in (0.1, 0.5, 0.8, 1.0)
    ]
    assert ratios == sorted(ratios, reverse=True)
    assert ratios[0] == pytest.approx(1.0, abs=0.01)
    assert ratios[2] < 0.90


def test_resonance_is_where_the_linear_wake_peaks():
    """``k_p sigma = sqrt(2)``, giving ``0.3801 a0^2``. Making the pulse
    longer does not help: the exponential in the pulse length takes back what
    the length gives."""
    _, k_p, _ = _plasma_scales()
    widths = np.linspace(0.2, 12.0, 4001)
    amplitudes = [linear_wake_amplitude(1.0, s, k_p) for s in widths]
    best = widths[int(np.argmax(amplitudes))]
    assert best == pytest.approx(resonant_length(k_p), rel=1e-3)
    assert max(amplitudes) == pytest.approx(0.3801, rel=1e-3)


def test_stepping_past_the_singular_point_is_refused_and_refinement_fixes_it():
    """Tells a step too coarse from a drive too strong.

    The source carries ``1/(1 + phi)^2`` and stiffens as the wake steepens.
    A coarse step walks past ``1 + phi = 0`` into a region where the answer
    is finite, smooth and meaningless, so the integration stops there. The
    same drive integrates cleanly once ``s`` is refined, which is what says
    the refusal was about resolution and not about physics.
    """
    _, k_p, _ = _plasma_scales()
    sigma = resonant_length(k_p)
    with pytest.raises(ValueError, match="Refine s first"):
        gaussian_wake(30.0, sigma, k_p, points=2001)
    solution = gaussian_wake(30.0, sigma, k_p, points=20001)
    assert (1.0 + solution.potential).min() > 0.0


def test_wake_solver_validates_its_inputs():
    s = np.linspace(0.0, 1.0, 11)
    with pytest.raises(ValueError, match="same shape"):
        nonlinear_wake(s, np.zeros(5), 1.0)
    with pytest.raises(ValueError, match="at least three"):
        nonlinear_wake(s[:2], np.zeros(2), 1.0)
    with pytest.raises(ValueError, match="does not extend past"):
        nonlinear_wake(s, np.zeros(11), 1.0).amplitude(behind=10.0)


def test_an_undriven_plasma_stays_flat():
    s = np.linspace(0.0, 40.0, 4001)
    solution = nonlinear_wake(s, np.zeros_like(s), 1.0)
    assert np.abs(solution.field).max() < 1e-12


# --- the plasma setup -------------------------------------------------------


def test_a_partly_filled_box_starts_with_no_field_at_all():
    """The mistake this guards against costs six wave-breaking fields.

    Spreading the ion background uniformly over the box is right when the
    plasma fills it and wrong when it does not: it leaves net negative charge
    in the plasma and net positive in the vacuum. Because the cycle conserves
    ``div D - rho`` exactly, that initial error never goes away.
    """
    grid = YeeGrid((400,), (0.05,))
    fields, species, ions = uniform_plasma(grid, 0.4, per_cell=8, start=5.0, ramp=1.0)
    assert np.abs(fields.Dx).max() == 0.0
    assert species.position[:, 0].min() >= 5.0
    # The ions exactly mirror the electrons, so the box is neutral cell by cell.
    from particlesim.solvers.pic import deposit_charge

    np.testing.assert_allclose(ions, -deposit_charge(grid, species, 1), atol=1e-15)


def test_the_density_ramp_is_smooth_and_reaches_full_density():
    grid = YeeGrid((400,), (0.05,))
    _, species, ions = uniform_plasma(grid, 0.4, per_cell=8, start=4.0, ramp=2.0)
    x = grid.coordinates((0.0,))[0]
    # Away from x = 0, where the periodic deposition wraps the plasma's far
    # edge back around.
    assert ions[(x > 0.5) & (x < 4.0)].max() < 1e-12
    plateau = ions[(x > 8.0) & (x < 15.0)]
    assert plateau.mean() == pytest.approx(0.4, rel=0.02)
    # Monotone through the ramp, so it introduces no step of its own.
    ramp = ions[(x >= 4.0) & (x <= 6.0)]
    assert np.all(np.diff(ramp) > -1e-12)


def test_measure_wake_refuses_before_the_pulse_has_travelled():
    grid = YeeGrid((200,), (0.05,))
    fields, _, _ = uniform_plasma(grid, 0.4, per_cell=4, start=0.0)
    with pytest.raises(ValueError, match="not travelled far enough"):
        measure_wake(grid, fields, 0.63, pulse_width=3.0, plasma_start=0.0)


# --- the benchmark ----------------------------------------------------------


def _run_wakefield(a0: float, cells_per_wavelength: int = 16, per_cell: int = 16):
    _, k_p, _ = _plasma_scales()
    sigma = resonant_length(k_p)
    run = resonant_benchmark(a0, cells_per_wavelength, per_cell, DENSITY_RATIO)
    fields, _ = run.run()
    behind = 4 * sigma + 0.5 * 2 * np.pi / k_p
    return run.wake(fields), gaussian_wake(a0, sigma, k_p).amplitude(behind), sigma, k_p


def test_the_run_description_is_the_loop_it_replaces():
    """``LaserWakefield.run`` against the loop the benchmark used to spell
    out, bit for bit, over a short stretch at low resolution."""
    run = replace(resonant_benchmark(0.3, cells_per_wavelength=8, per_cell=4), steps=200)
    fields, species = run.run()

    grid = YeeGrid((run.cells,), (run.spacing,))
    solver = YeeSolver(grid, courant=0.5, boundary=PerfectlyMatchedLayer(thickness=24))
    source = PlaneWaveSource(run.pulse, index=120)
    source.attach(solver)
    by_hand, electrons, _ = uniform_plasma(
        grid, run.density, per_cell=4, start=120 * run.spacing + 22.0, ramp=3.0
    )
    for _ in range(200):
        by_hand, electrons = advance(solver, by_hand, electrons, order=1, source=source)
    for name in ("Dx", "Dy", "Dz", "Bx", "By", "Bz"):
        assert np.array_equal(getattr(fields, name), getattr(by_hand, name))
    assert np.array_equal(species.momentum, electrons.momentum)
    assert np.abs(fields.Dz).max() > 0.1  # the pulse is in


def test_the_run_description_refuses_what_it_cannot_run():
    pulse = LaserPulse()
    for bad, message in (
        ({"cells": 1}, "two cells"),
        ({"source_index": 0}, "inside the box"),
        ({"boundary": "open"}, "boundary"),
        ({"order": 3}, "order"),
        ({"pusher": "higuera"}, "pusher"),
        ({"ramp": -1.0}, "non-negative"),
    ):
        settings = {"pulse": pulse, "density": 0.1, "cells": 64, "spacing": 0.1, "steps": 1}
        with pytest.raises(ValueError, match=message):
            LaserWakefield(**(settings | bad))


@pytest.mark.slow
@pytest.mark.benchmark
def test_wakefield_amplitude_matches_one_dimensional_theory():
    """Acceptance for issue #34: 5%.

    A full particle-in-cell run, carrier and all, with no envelope
    approximation anywhere: the ponderomotive force that drives the wake
    emerges from electrons quivering in a resolved laser field.
    """
    measured, theory, _, _ = _run_wakefield(0.3)
    assert measured == pytest.approx(theory, rel=0.05)


@pytest.mark.slow
@pytest.mark.benchmark
def test_the_wakefield_benchmark_measures_the_nonlinear_response():
    """The same run at a drive where linear theory is not good enough.

    At ``a0 = 0.8`` the linear formula is eleven per cent above the
    nonlinear one, twice the tolerance. The simulation follows the nonlinear
    result, so this benchmark is not satisfied by any code that happens to
    reproduce the linear scaling.
    """
    measured, theory, sigma, k_p = _run_wakefield(0.8)
    linear = linear_wake_amplitude(0.8, sigma, k_p)
    assert linear / theory > 1.10
    assert measured == pytest.approx(theory, rel=0.05)
    assert not measured == pytest.approx(linear, rel=0.05)


def test_the_wake_integration_is_fourth_order_as_its_runge_kutta_claims():
    """The wake is a fourth-order Runge-Kutta over a sampled drive envelope,
    and its half steps need that envelope halfway between the points it has.
    Averaging the two neighbours for it is second-order accurate, and no
    number of Runge-Kutta stages repairs an integrand that is already wrong:
    the error enters each step with an ``O(ds)`` weight and there are ``1/ds``
    steps, so the whole solve came out second order. It measured 2.00 at
    every rung of the ladder below.

    The probe is the potential at the last grid point rather than its peak.
    ``linspace(0, L, 2500 k + 1)`` grids share their endpoints as they are
    refined, so that point is the same place on every grid. The peak is not:
    near a smooth maximum the discrete one is low by ``O(ds^2)`` from
    sampling alone, which is its own second-order error and would have
    reported the fixed solve as second order too.
    """
    reference = gaussian_wake(a0=0.6, sigma=1.0, plasma_wavenumber=1.0, points=640001)
    target = float(reference.potential[-1])

    errors = []
    for points in (2501, 5001, 10001, 20001):
        wake = gaussian_wake(a0=0.6, sigma=1.0, plasma_wavenumber=1.0, points=points)
        errors.append(abs(float(wake.potential[-1]) - target))

    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert all(3.7 < o < 4.3 for o in orders), f"orders were {orders}, errors {errors}"
    # Second order reaches 1.3e-8 at the default resolution; fourth reaches 7.1e-13.
    assert errors[-1] < 1e-11, f"error at the default resolution was {errors[-1]:.3e}"
