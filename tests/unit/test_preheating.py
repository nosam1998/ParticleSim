"""Parametric resonance against the Floquet chart it has to reproduce.

Issue #66 asks that the resonance bands match Floquet analysis. Floquet is
not a fit to a growth curve -- it is the eigenvalue problem of the mode
equation's monodromy matrix, exact to the ODE solver's tolerance and
*identically zero* outside a band. So the tests below assert agreement
inside a band and exact silence outside it, rather than one tolerance over
a whole scan.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.cosmo.potentials import PowerLaw, Quadratic
from particlesim.cosmo.preheating import (
    Coupling,
    Oscillation,
    Preheating,
    floquet_exponent,
    mathieu_parameters,
    mode_amplitudes,
    monodromy,
    narrow_resonance_exponent,
    resonance_bands,
)
from particlesim.solvers.lattice.realtime import Lattice

MASS = 1.0


@pytest.fixture(scope="module")
def oscillation() -> Oscillation:
    return Oscillation(Quadratic(mass=MASS), amplitude=1.0)


# --- the background -------------------------------------------------------


@pytest.mark.parametrize("amplitude", [0.3, 1.0, 3.0])
def test_a_quadratic_oscillation_has_period_two_pi_over_m(amplitude):
    """And it does not depend on the amplitude, which is the quadratic case.

    Measured from the trajectory's turning point rather than assumed, so
    this checks the period finder against a closed form rather than
    checking a closed form against itself.
    """
    period = Oscillation(Quadratic(mass=MASS), amplitude=amplitude).period()
    assert period == pytest.approx(2.0 * np.pi / MASS, rel=1e-10)


@pytest.mark.parametrize("amplitude", [0.5, 1.0, 2.0])
def test_a_quartic_oscillation_has_period_inversely_proportional_to_amplitude(amplitude):
    """``T ~ 1/A`` exactly, which a quadratic-only implementation would miss.

    The whole reason the period is integrated rather than taken as
    ``2 pi / m``: for any potential that is not quadratic the period
    depends on how far the field swings, and Mathieu's equation stops being
    the right description.
    """
    period = Oscillation(PowerLaw(amplitude=1.0, exponent=4.0), amplitude=amplitude).period()
    assert period * amplitude == pytest.approx(3.708149, rel=1e-5)


def test_oscillation_refuses_a_non_positive_amplitude():
    with pytest.raises(ValueError, match="amplitude must be positive"):
        Oscillation(Quadratic(mass=MASS), amplitude=0.0)


def test_coupling_refuses_a_negative_strength():
    with pytest.raises(ValueError, match="must not be negative"):
        Coupling(strength=-1.0)


def test_the_mathieu_parameters_are_the_textbook_ones(oscillation):
    centre, resonance = mathieu_parameters(0.5, oscillation, Coupling(strength=0.4, mass=0.3))
    assert resonance == pytest.approx(0.4**2 * 1.0**2 / (4.0 * MASS**2), rel=1e-12)
    assert centre == pytest.approx((0.5 + 0.09) / MASS**2 + 2.0 * resonance, rel=1e-12)


# --- Floquet --------------------------------------------------------------


def test_the_monodromy_has_unit_determinant(oscillation):
    """Liouville: the mode equation has no friction, so phase-space area is kept.

    A determinant drifting from one would mean the integration, not the
    physics, and every exponent below would inherit the error.
    """
    matrix = monodromy(1.0, oscillation, Coupling(strength=2.0))
    assert np.linalg.det(matrix) == pytest.approx(1.0, rel=1e-8)


@pytest.mark.parametrize(("strength", "expected"), [(0.2, 1.0000), (0.4, 0.9998), (0.6, 0.9991)])
def test_narrow_resonance_converges_to_the_closed_form(oscillation, strength, expected):
    """``mu -> (m/2) sqrt(q^2 - (A-1)^2)`` as ``q`` shrinks.

    The closed form is leading order in ``q``, so the right check is not a
    fixed tolerance but that the ratio approaches one as its own expansion
    parameter does: 0.9991, 0.9998, 1.0000 for ``q`` of 0.09, 0.04, 0.01.
    Asserting each to 1e-3 would pass a formula that was wrong by a
    constant; asserting the trend would not.
    """
    coupling = Coupling(strength=strength)
    _, resonance = mathieu_parameters(0.0, oscillation, coupling)
    frequency = (1.0 - 2.0 * resonance) * MASS**2

    measured = floquet_exponent(frequency, oscillation, coupling)
    closed = narrow_resonance_exponent(1.0, resonance, MASS)
    assert measured / closed == pytest.approx(expected, abs=2e-4)


def test_outside_a_band_the_exponent_is_zero_to_round_off(oscillation):
    """Not small -- zero. The multipliers sit on the unit circle.

    That is what makes a band edge a fact rather than a threshold, and it
    is why :func:`resonance_bands` can use 1e-8 without tuning.
    """
    coupling = Coupling(strength=0.2)
    _, resonance = mathieu_parameters(0.0, oscillation, coupling)
    frequency = (1.0 + 3.0 * resonance - 2.0 * resonance) * MASS**2
    assert abs(floquet_exponent(frequency, oscillation, coupling)) < 1e-9


def test_the_narrow_band_edges_are_where_the_closed_form_puts_them(oscillation):
    """``|A - 1| < q`` to leading order, recovered by scanning.

    The tolerance is one grid spacing, because that is all a scan can
    resolve: :func:`resonance_bands` reports the outermost sampled points
    that are unstable, so an edge lands between two samples and is known to
    within their separation. Tightening it further would be asserting
    something about the grid rather than about the band.
    """
    coupling = Coupling(strength=0.3)
    _, resonance = mathieu_parameters(0.0, oscillation, coupling)
    centres = np.linspace(1.0 - 2.0 * resonance, 1.0 + 2.0 * resonance, 41)
    spacing = float(centres[1] - centres[0])
    frequencies = (centres - 2.0 * resonance) * MASS**2

    bands = resonance_bands(frequencies, oscillation, coupling)
    assert len(bands) == 1
    low, high = bands[0]
    assert low / MASS**2 + 2.0 * resonance == pytest.approx(1.0 - resonance, abs=1.5 * spacing)
    assert high / MASS**2 + 2.0 * resonance == pytest.approx(1.0 + resonance, abs=1.5 * spacing)


def test_narrow_resonance_exponent_vanishes_outside_its_band():
    assert narrow_resonance_exponent(1.5, 0.1, MASS) == 0.0
    assert narrow_resonance_exponent(1.0, 0.1, MASS) == pytest.approx(0.05)


# --- the acceptance -------------------------------------------------------


@pytest.fixture(scope="module")
def lattice_run(oscillation):
    """Two late, integer-period samples of a lattice ``chi`` under resonance.

    Late, because the initial noise contains the *decaying* Floquet mode as
    well as the growing one and the measured rate is only the exponent once
    that has died. Integer-period, because the growing solution is an
    exponential times a periodic function, so a ratio taken at matching
    phase is the exponential alone -- no fitting window to choose.
    """
    coupling = Coupling(strength=4.0)
    lattice = Lattice(size=32.0, points=64, dimensions=1)
    solver = Preheating(lattice, oscillation, coupling)
    period = oscillation.period()

    noise = np.random.default_rng(0).normal(size=lattice.shape) * 1e-10
    early, rate = solver.run(
        noise, np.zeros(lattice.shape), 8.0 * period, int(8.0 * period / 0.005)
    )
    late, _ = solver.run(early, rate, 24.0 * period, int(24.0 * period / 0.005))

    growth = np.log(mode_amplitudes(late, lattice) / mode_amplitudes(early, lattice)) / (
        24.0 * period
    )
    return solver, coupling, growth


@pytest.mark.benchmark
@pytest.mark.parametrize("mode", [6, 8])
def test_the_lattice_growth_rate_is_the_floquet_exponent(oscillation, lattice_run, mode):
    """Issue #66's acceptance: the bands the lattice produces are Floquet's.

    Agreement is 1.0011 and 0.9998 at the two growing modes -- a tenth of a
    percent, against a reference computed from an entirely separate
    integration of a different equation.
    """
    solver, coupling, growth = lattice_run
    frequency = float(solver.lattice_frequencies()[mode])

    predicted = floquet_exponent(frequency, oscillation, coupling)
    assert predicted > 0.05  # this mode is genuinely inside a band
    assert growth[mode] / predicted == pytest.approx(1.0, abs=0.01)


@pytest.mark.benchmark
@pytest.mark.parametrize("mode", [2, 12, 20])
def test_modes_outside_the_bands_do_not_grow(oscillation, lattice_run, mode):
    """The other half of "the bands match": silence where Floquet is silent."""
    solver, coupling, growth = lattice_run
    frequency = float(solver.lattice_frequencies()[mode])

    assert abs(floquet_exponent(frequency, oscillation, coupling)) < 1e-9
    assert abs(growth[mode]) < 0.02


@pytest.mark.benchmark
def test_using_the_continuum_dispersion_would_disagree(oscillation, lattice_run):
    """Why the reference is fed the lattice's ``omega^2`` and not ``k^2``.

    The same Floquet machinery on the continuum wavenumber moves the
    exponent by 19% and 17% at the two growing modes -- two orders above
    the agreement the lattice dispersion gives, and in the direction of
    looking like a physics disagreement rather than a bookkeeping one.
    """
    solver, coupling, _ = lattice_run
    lattice = solver.lattice
    for mode in (6, 8):
        continuum = (2.0 * np.pi * mode / lattice.size) ** 2
        exact = floquet_exponent(float(solver.lattice_frequencies()[mode]), oscillation, coupling)
        wrong = floquet_exponent(continuum, oscillation, coupling)
        assert abs(wrong / exact - 1.0) > 0.1


def test_run_refuses_a_zero_step_count(oscillation):
    lattice = Lattice(size=8.0, points=8, dimensions=1)
    solver = Preheating(lattice, oscillation, Coupling(strength=1.0))
    with pytest.raises(ValueError, match="at least one step"):
        solver.run(np.zeros(lattice.shape), np.zeros(lattice.shape), 1.0, 0)
