"""Preheating from a registry potential, and the floor under its spectrum.

Issue #66's acceptance is that the resonance bands match Floquet analysis.
That was already shown for a quadratic inflaton; what is new here is that it
holds for a potential Mathieu's equation says nothing about, and that the
spectrum now says which of its modes are still worth reading.

The second half is the substance. A resonant band grows by orders of
magnitude, and once the loudest mode is about ``1/eps`` times a quiet one,
double-precision round-off in the field exceeds the quiet mode's amplitude.
From then on every silent mode tracks the loudest at a fixed ratio and
reports a steady, wrong, plausible growth rate.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.cosmo.preheating import mathieu_parameters
from particlesim.scenarios.cosmo.lattice import (
    RESOLUTION_FLOOR,
    PreheatingScenario,
    defect_count,
    defect_sites,
    measure_growth,
    phase_winding,
    total_winding,
    wrapped_difference,
)

QUADRATIC = PreheatingScenario(
    potential_id="inflation.quadratic", potential_parameters=(("mass", 1.0),)
)
STAROBINSKY = PreheatingScenario(
    potential_id="inflation.starobinsky", potential_parameters=(("amplitude", 1.0),)
)


@pytest.fixture(scope="module")
def starobinsky_short():
    return measure_growth(STAROBINSKY, warmup_periods=2.0, measure_periods=6.0)


@pytest.fixture(scope="module")
def starobinsky_long():
    return measure_growth(STAROBINSKY, warmup_periods=8.0, measure_periods=24.0)


# --- the acceptance, for a potential Mathieu cannot describe ---------------


@pytest.mark.benchmark
def test_floquet_matches_the_lattice_for_a_registry_potential(starobinsky_short):
    """Six bands under a Starobinsky inflaton, every one within a percent.

    The acceptance says the bands match Floquet analysis. Here they do for a
    potential supplied by the registry rather than hard-coded, and one whose
    oscillation is anharmonic -- the period is measured from the trajectory,
    not taken from a mass.
    """
    report = starobinsky_short
    assert np.count_nonzero(report.banded) >= 4
    for ratio in report.agreement():
        assert ratio == pytest.approx(1.0, abs=0.01)


@pytest.mark.benchmark
def test_the_same_holds_for_the_quadratic_inflaton():
    """The known case, through the same registry path."""
    report = measure_growth(QUADRATIC, warmup_periods=8.0, measure_periods=8.0)
    assert np.count_nonzero(report.banded) >= 2
    for ratio in report.agreement():
        assert ratio == pytest.approx(1.0, abs=0.02)


def test_the_mathieu_closed_form_refuses_a_potential_it_does_not_describe():
    """Floquet is potential-agnostic; the Mathieu parameters are not.

    ``A`` and ``q`` are defined through the inflaton's mass, which a
    Starobinsky potential does not have. Raising is the right answer, and it
    is asserted so that a future change cannot quietly invent one.
    """
    with pytest.raises(AttributeError):
        mathieu_parameters(1.0, STAROBINSKY.oscillation, STAROBINSKY.coupling)
    centre, resonance = mathieu_parameters(1.0, QUADRATIC.oscillation, QUADRATIC.coupling)
    assert resonance == pytest.approx(4.0)
    assert centre == pytest.approx(9.0)


# --- the floor under the spectrum ------------------------------------------


@pytest.mark.benchmark
def test_a_long_run_pushes_the_quiet_modes_below_round_off(starobinsky_long):
    """Past a dynamic range of ``1e16``, silent modes report the loud one.

    Every unresolvable mode here is one Floquet puts outside a band, and
    every one of them reports a steady positive rate of a few hundredths --
    neither zero nor the dominant 0.20, but a fixed fraction of it, because
    what is being measured is round-off in the field rather than the mode.
    """
    report = starobinsky_long
    assert report.dynamic_range > 1e15
    unresolvable = ~report.resolvable
    assert np.count_nonzero(unresolvable) > 10

    # none of them is in a band, and all of them look like they are growing
    assert not np.any(report.banded & unresolvable)
    spurious = report.measured[unresolvable]
    assert np.median(spurious) > 0.02
    assert np.all(report.predicted[unresolvable] < 1e-8)


@pytest.mark.benchmark
def test_the_bands_survive_the_floor_that_the_silence_does_not(starobinsky_long):
    """The loud modes are unharmed: agreement to ``1e-3`` at the same time.

    Which is what makes the failure dangerous rather than obvious. The part
    of the run being quoted is still right; only the part being read as
    "no resonance here" has quietly become noise.
    """
    report = starobinsky_long
    assert np.all(report.resolvable[report.banded])
    for ratio in report.agreement():
        assert ratio == pytest.approx(1.0, abs=1e-3)


@pytest.mark.benchmark
def test_a_floored_mode_is_told_from_a_quiet_one_by_its_sign(starobinsky_short, starobinsky_long):
    """Quiet modes scatter both ways; floored ones are uniformly positive.

    A stable mode's amplitude oscillates, and the two samples sit at an
    arbitrary relative phase of that oscillation -- so a single ratio is a
    poor estimator of "no growth" and an unlucky mode near a node can read
    as large as 0.13. What it cannot do is pick a side: across the 27
    out-of-band modes only 41% come out positive, with a median magnitude
    of 0.009.

    Once round-off takes over, every one of those 27 is positive, with a
    median seven times larger, because they are all tracking the same
    growing mode. That is the signature, and it is sharper than any
    threshold on the magnitude.
    """
    quiet = starobinsky_short.measured[~starobinsky_short.banded]
    floored = starobinsky_long.measured[~starobinsky_long.banded]

    assert starobinsky_short.dynamic_range < 1e12
    assert np.all(starobinsky_short.resolvable)
    assert 0.2 < np.count_nonzero(quiet > 0) / quiet.size < 0.8
    assert np.median(np.abs(quiet)) < 0.02

    assert np.all(floored > 0.0)
    assert np.median(np.abs(floored)) > 3.0 * np.median(np.abs(quiet))


@pytest.mark.benchmark
def test_the_dynamic_range_is_what_separates_the_two(starobinsky_short, starobinsky_long):
    assert starobinsky_long.dynamic_range / starobinsky_short.dynamic_range > 1e8
    assert np.count_nonzero(~starobinsky_short.resolvable) == 0
    assert np.count_nonzero(~starobinsky_long.resolvable) > 10
    assert RESOLUTION_FLOOR > np.finfo(float).eps


def test_the_report_summarises_without_hiding_the_unresolvable_count():
    """A row that omitted the floor would be the misleading part."""
    report = measure_growth(QUADRATIC, warmup_periods=1.0, measure_periods=2.0)
    row = report.as_row()
    assert set(row) == {
        "dynamic_range",
        "banded_modes",
        "resolvable_modes",
        "unresolvable_modes",
        "agreement",
    }
    assert row["resolvable_modes"] + row["unresolvable_modes"] == report.measured.size


# --- the scenario --------------------------------------------------------


def test_an_unknown_potential_is_refused_with_the_known_ones_listed():
    with pytest.raises(KeyError, match="unknown inflaton potential"):
        PreheatingScenario(potential_id="inflation.not_a_potential")


def test_the_scenario_builds_its_pieces_from_the_registry():
    from particlesim.cosmo.potentials import Starobinsky

    assert isinstance(STAROBINSKY.potential, Starobinsky)
    assert STAROBINSKY.coupling.strength == pytest.approx(4.0)
    assert STAROBINSKY.oscillation.amplitude == pytest.approx(1.0)
    assert STAROBINSKY.period() == pytest.approx(6.55728, rel=1e-4)


def test_a_measurement_needs_a_positive_window():
    with pytest.raises(ValueError, match="positive window"):
        measure_growth(QUADRATIC, measure_periods=0.0)
    with pytest.raises(ValueError, match="non-negative warm-up"):
        measure_growth(QUADRATIC, warmup_periods=-1.0)


# --- defects --------------------------------------------------------------


def grid(size: int):
    return np.meshgrid(np.arange(size), np.arange(size), indexing="ij")


def test_the_total_winding_of_a_periodic_lattice_is_exactly_zero():
    """An integer identity, not a tolerance.

    Each link enters two plaquettes with opposite sign, so the sum cancels
    term by term whatever the field. That is why this asserts equality with
    ``0`` rather than closeness to it, and why it holds for random phases as
    firmly as for a smooth one.
    """
    rng = np.random.default_rng(0)
    for phases in (
        rng.uniform(0.0, 2.0 * np.pi, size=(32, 32)),
        np.zeros((16, 16)),
        rng.normal(size=(24, 24)) * 10.0,
    ):
        assert total_winding(phases) == 0


def test_a_vortex_pair_is_found_with_opposite_charges():
    """A torus cannot hold a net charge, so defects come in pairs."""
    first, second = grid(32)
    phases = np.arctan2(first - 8, second - 8) - np.arctan2(first - 24, second - 24)
    winding = phase_winding(phases)
    assert sorted(winding[winding != 0].tolist()) == [-1, 1]
    assert defect_count(phases) == 2
    assert defect_sites(phases).shape == (2, 2)
    assert total_winding(phases) == 0


def test_a_smooth_field_has_no_defects_at_all():
    first, second = grid(32)
    phases = 0.5 * np.sin(2.0 * np.pi * second / 32.0) + 0.3 * np.cos(2.0 * np.pi * first / 32.0)
    assert defect_count(phases) == 0
    assert not phase_winding(phases).any()


def test_random_phases_make_many_defects_and_still_balance():
    rng = np.random.default_rng(0)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(32, 32))
    winding = phase_winding(phases)
    assert defect_count(phases) > 100
    assert np.max(np.abs(winding)) == 1
    assert total_winding(phases) == 0


def test_a_single_vortex_cannot_be_planted_on_a_torus():
    """The attempt shows up as spurious charges on the seam.

    Which is the identity making itself felt: a phase that winds once around
    one point is not periodic, and the lattice reports the discontinuity
    rather than a net charge it cannot carry.
    """
    first, second = grid(32)
    phases = np.arctan2(first - 16, second - 16)
    assert total_winding(phases) == 0
    assert defect_count(phases) > 1


def test_the_wrapped_difference_stays_inside_one_turn():
    rng = np.random.default_rng(1)
    phases = rng.uniform(-20.0, 20.0, size=(8, 8))
    for axis in (0, 1):
        folded = wrapped_difference(phases, axis)
        assert np.all(folded > -np.pi - 1e-12)
        assert np.all(folded <= np.pi + 1e-12)


def test_winding_is_defined_for_two_dimensional_fields_only():
    with pytest.raises(ValueError, match="two-dimensional"):
        phase_winding(np.zeros((4, 4, 4)))
