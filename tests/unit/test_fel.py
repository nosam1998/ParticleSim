"""One-dimensional free-electron laser gain (design doc Section 3.3, M2)."""

import numpy as np
import pytest

from particlesim.scenarios.fel import (
    FIELD_GROWTH_RATE,
    POWER_GROWTH_RATE,
    gain_curve,
    gain_length,
    linear_growth_rate,
    resonant_wavelength,
    run_fel,
    saturation_power_fraction,
    undulator_parameter,
)

# --- the machine ------------------------------------------------------------


def test_the_undulator_parameter_matches_the_practical_formula():
    """``K = 0.934 B[T] lambda_u[cm]``, the form every undulator is quoted in."""
    assert undulator_parameter(1.0, 0.03) == pytest.approx(0.934 * 1.0 * 3.0, rel=1e-3)
    assert undulator_parameter(0.0, 0.03) == 0.0
    with pytest.raises(ValueError, match="non-negative"):
        undulator_parameter(-1.0, 0.03)


def test_a_helical_undulator_resonates_at_a_different_wavelength():
    """``1 + K^2`` against ``1 + K^2/2``.

    The half is the mean square of a sinusoid: a planar undulator's
    transverse velocity oscillates and a helical one's is constant. Using the
    planar formula for a helical machine misses the resonance by many times
    the gain bandwidth, which is a machine that does not lase.
    """
    period, gamma, strength = 0.03, 1000.0, 2.0
    planar = resonant_wavelength(period, gamma, strength)
    helical = resonant_wavelength(period, gamma, strength, helical=True)
    assert planar == pytest.approx(period * (1 + 2.0) / (2 * gamma**2))
    assert helical == pytest.approx(period * (1 + 4.0) / (2 * gamma**2))
    assert helical / planar == pytest.approx(5.0 / 3.0)


def test_resonance_validates_its_arguments():
    with pytest.raises(ValueError, match="gamma > 1"):
        resonant_wavelength(0.03, 0.5, 1.0)
    with pytest.raises(ValueError, match="positive period"):
        resonant_wavelength(0.0, 100.0, 1.0)


def test_the_gain_length_is_the_scaling_definition():
    """``L_g = lambda_u / (4 pi sqrt(3) rho)``, the acceptance criterion's form."""
    assert gain_length(1e-3, 0.03) == pytest.approx(0.03 / (4 * np.pi * np.sqrt(3) * 1e-3))
    # Halving rho doubles the gain length, which is why rho is the figure of merit.
    assert gain_length(5e-4, 0.03) == pytest.approx(2 * gain_length(1e-3, 0.03))
    assert saturation_power_fraction(1e-3) == 1e-3
    for bad in ((0.0, 0.03), (1e-3, 0.0), (-1e-3, 0.03)):
        with pytest.raises(ValueError, match="must be positive"):
            gain_length(*bad)


# --- linear theory ----------------------------------------------------------


def test_the_resonant_growth_rate_is_the_cube_root_of_i():
    """The universal number. Linearizing leaves a cubic whose growing root at
    resonance is ``exp(i pi / 6)``, so the field grows at ``sqrt(3)/2`` and
    the power at ``sqrt(3)``, which is exactly the stated gain length."""
    assert linear_growth_rate(0.0) == pytest.approx(FIELD_GROWTH_RATE, abs=1e-12)
    assert FIELD_GROWTH_RATE == pytest.approx(np.sqrt(3) / 2)
    assert POWER_GROWTH_RATE == pytest.approx(2 * FIELD_GROWTH_RATE)


def test_the_gain_curve_peaks_at_resonance_and_is_asymmetric():
    """Growth survives far below the resonance and cuts off sharply above it.
    That is the cubic's doing, not any machine's."""
    detunings = np.linspace(-6, 3, 601)
    rates = np.array([linear_growth_rate(d) for d in detunings])
    assert detunings[int(rates.argmax())] == pytest.approx(0.0, abs=0.02)
    assert linear_growth_rate(-3.0) > 0.5  # still growing well below
    assert linear_growth_rate(2.0) == 0.0  # stable just above
    assert linear_growth_rate(-1.0) > linear_growth_rate(1.0)


# --- the integration --------------------------------------------------------


def test_field_energy_comes_out_of_the_beam_exactly():
    """``|A|^2 + <p>`` is conserved by these equations identically.

    ``d|A|^2/dz = 2 Re(A* b)`` and ``d<p>/dz = -2 Re(A* b)``, so the sum
    cannot move. At peak field the run has ``|A|^2 = 1.3720`` and
    ``<p> = -1.3720``: every unit of radiation came from the electrons, and
    a violation here would mean the model was making energy.
    """
    run = run_fel()
    invariant = np.abs(run.amplitude) ** 2 + run.mean_energy
    assert np.abs(invariant - invariant[0]).max() < 1e-9

    peak = int(np.abs(run.amplitude).argmax())
    assert run.mean_energy[peak] == pytest.approx(-(np.abs(run.amplitude[peak]) ** 2), abs=1e-8)


def test_the_invariant_converges_at_fourth_order():
    """Which is what says the integrator is the one claimed."""
    drifts = []
    for step in (0.02, 0.01, 0.005):
        run = run_fel(step=step)
        invariant = np.abs(run.amplitude) ** 2 + run.mean_energy
        drifts.append(float(np.abs(invariant - invariant[0]).max()))
    orders = [np.log2(drifts[i] / drifts[i + 1]) for i in range(len(drifts) - 1)]
    assert all(3.7 < o < 4.3 for o in orders), f"orders were {orders}"


def test_the_run_saturates_rather_than_growing_for_ever():
    run = run_fel()
    assert np.abs(run.amplitude).max() == pytest.approx(1.17, rel=0.1)
    assert 0.0 < run.bunching.max() < 1.0  # complete bunching is never reached
    assert run.mean_energy.min() < -1.0  # the beam pays for it


def test_a_quiet_start_is_deterministic():
    """Evenly spaced phases, not sampled ones. Random phases carry shot noise
    of order ``1/sqrt(N)`` in the bunching, which for a seeded run competes
    with the intended seed and makes start-up depend on the draw."""
    assert run_fel().growth_rate() == run_fel().growth_rate()
    spread = run_fel(energy_spread=0.1, rng_seed=1)
    assert spread.growth_rate() != run_fel().growth_rate()


def test_run_validates_its_arguments():
    with pytest.raises(ValueError, match="at least eight particles"):
        run_fel(particles=4)
    for kwargs in ({"step": 0.0}, {"length": -1.0}):
        with pytest.raises(ValueError, match="must be positive"):
            run_fel(**kwargs)


def test_an_unseeded_quiet_start_does_not_lase():
    """Evenly spaced phases have no bunching, so there is nothing to amplify.

    A real machine starts from shot noise, which a quiet start deliberately
    has none of. The field stays at round-off, which is the check: not that
    it is exactly zero, since the average of 512 evenly spaced phasors is
    1e-17 rather than 0, but that nothing grows out of it.
    """
    quiet = run_fel(seed_field=0j, length=16.0)
    assert np.abs(quiet.amplitude).max() < 1e-12
    assert quiet.bunching.max() < 1e-12


def test_a_field_that_never_grew_is_reported():
    from particlesim.scenarios.fel import FELRun

    flat = FELRun(
        np.linspace(0, 1, 10),
        np.zeros(10, dtype=complex),
        np.zeros(10),
        np.zeros(10),
        0.0,
    )
    with pytest.raises(ValueError, match="never grew"):
        flat.growth_rate()


def test_a_run_that_never_left_start_up_is_reported():
    from particlesim.scenarios.fel import FELRun

    short = FELRun(np.linspace(0, 1, 5), np.full(5, 1.0 + 0j), np.zeros(5), np.zeros(5), 0.0)
    with pytest.raises(ValueError, match="too few points"):
        short.growth_rate()


# --- the benchmark ----------------------------------------------------------


@pytest.mark.benchmark
def test_the_gain_length_matches_theory_at_resonance():
    """Acceptance for issue #35: 5%.

    ``L_g = lambda_u / (4 pi sqrt(3) rho)`` is the scaling's definition, so
    what a simulation can be wrong about is the ``sqrt(3)``. Measured here
    through the field's growth rate, which is half of it.
    """
    measured = run_fel().growth_rate()
    assert measured == pytest.approx(FIELD_GROWTH_RATE, rel=0.05)
    # And therefore the gain length itself, to the same relative accuracy.
    period, rho = 0.03, 1e-3
    implied = period / (4 * np.pi * rho * 2 * measured)
    assert implied == pytest.approx(gain_length(rho, period), rel=0.05)


@pytest.mark.benchmark
@pytest.mark.parametrize("detuning", (-3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 1.5))
def test_the_whole_gain_curve_matches_the_cubic(detuning):
    """Not only the peak. Seven detunings spanning the unstable range, each
    against the cubic's largest root rather than against a fitted shape."""
    measured = run_fel(detuning=detuning).growth_rate()
    assert measured == pytest.approx(linear_growth_rate(detuning), rel=0.05)


def test_the_gain_curve_helper_reports_zero_where_nothing_grows():
    detunings, rates = gain_curve(np.array([0.0, 2.5]), length=8.0)
    assert rates[0] > 0.5
    assert rates[1] == 0.0
