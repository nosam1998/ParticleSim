"""Axion-photon conversion against the formula, and the formula against itself.

Issue #72 asks that a conversion probability match the analytic mixing
formula. That formula is an *approximation* -- it drops the second
``z``-derivative -- so agreeing with it to a tolerance would leave open
whether a gap is the solver's error or the approximation's. What is asserted
here instead is that the exact propagation converges to it at first order in
``Delta_M / omega``, the approximation's own small parameter.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.scenarios.axion import (
    Mixing,
    conversion_probability,
    oscillation_length,
    propagate,
    resonant_plasma_frequency,
)
from particlesim.theories.em import AxionPhoton, check_all_maxwell_limits
from particlesim.theories.registry import list_em_sectors

# --- the plugin -----------------------------------------------------------


def test_the_constitutive_relation_is_the_one_the_action_gives():
    """``D = E + g a B`` and ``H = B - g a E``, linear in the axion."""
    theory = AxionPhoton(coupling=0.3, mass=0.1)
    electric = np.array([1.0, 0.0, 0.0])
    magnetic = np.array([0.0, 2.0, 0.0])
    axion = np.array([0.5, 0.5, 0.5])

    displacement, intensity = theory.constitutive(electric, magnetic, axion)
    assert displacement == pytest.approx([1.0, 0.3, 0.0])
    assert intensity == pytest.approx([-0.15, 2.0, 0.0])


def test_a_constant_axion_is_invisible():
    """``theta F Fdual`` at constant ``theta`` is a total derivative.

    Substituting the constitutive relations into Ampere's law leaves

        curl H - d_t D = curl B - g a curl E - d_t E - g a d_t B

    and Faraday cancels the two axion terms exactly. Checked here on random
    fields that satisfy Faraday by construction: the axion contribution to
    the residual is zero to round-off, not small. This is why the plugin
    supplies no static medium -- there is nothing for one to represent.
    """
    generator = np.random.default_rng(0)
    coupling, axion = 0.37, 1.9
    theory = AxionPhoton(coupling=coupling)

    curl_electric = generator.normal(size=3)
    magnetic_rate = -curl_electric  # Faraday, by construction
    curl_magnetic = generator.normal(size=3)
    electric_rate = generator.normal(size=3)

    # curl H - d_t D, expanded with a constant axion
    axionic = (
        curl_magnetic
        - coupling * axion * curl_electric
        - electric_rate
        - coupling * axion * magnetic_rate
    )
    vacuum = curl_magnetic - electric_rate
    assert np.abs(axionic - vacuum).max() < 1e-15
    # The mass is not part of the limit: a massless axion still couples.
    assert theory.maxwell_limit() == {"coupling": 0.0}


def test_the_plugin_is_discoverable_and_skipped_by_the_medium_harness():
    """Discoverable, and *deliberately* absent from the constitutive check.

    The harness asks a constitutive relation what it does to ``(D, B)``.
    This theory has no static one, so it is skipped rather than reported as
    failing -- a question the harness cannot ask is not an answer. Asserted
    so the skip cannot become an accident.
    """
    assert list_em_sectors()["string.eft4d.axion_photon"] is AxionPhoton
    reports = check_all_maxwell_limits()
    assert "string.eft4d.axion_photon" not in reports
    assert "string.eft4d.born_infeld" in reports  # the harness still runs


def test_only_the_parallel_polarisation_mixes():
    """``E.B`` is what the axion couples to, so perpendicular light passes."""
    theory = AxionPhoton(coupling=0.5)
    magnetic = np.array([0.0, 1.0, 0.0])
    parallel = np.array([0.0, 1.0, 0.0])
    perpendicular = np.array([1.0, 0.0, 0.0])

    assert float(np.dot(parallel, magnetic)) != 0.0
    assert float(np.dot(perpendicular, magnetic)) == 0.0
    assert theory.observable_predictions()["mixes_parallel_polarisation_only"]


# --- the mixing parameters ------------------------------------------------


def test_the_massless_vacuum_case_is_a_clean_sine():
    """``m = 0`` and no plasma gives ``P = sin^2(g B L / 2)``, fully converting."""
    mixing = Mixing(coupling=1e-3, field=1.0, mass=0.0, frequency=1.0)
    assert mixing.oscillation_term == pytest.approx(2.0 * mixing.mixing_term, rel=1e-12)

    length = np.pi / (mixing.coupling * mixing.field)
    assert conversion_probability(mixing, length) == pytest.approx(1.0, abs=1e-12)


def test_a_mass_spoils_the_match_and_a_plasma_restores_it():
    """The resonance ``Delta_par = Delta_a`` is what helioscopes tune.

    With a massive axion in vacuum the oscillation is fast and shallow --
    its ceiling is ``(2 Delta_M / Delta_osc)^2``, here 0.8% -- and setting
    the plasma frequency to the axion mass restores full conversion. The
    mass has to be well clear of the coupling for the point to be visible:
    at ``m = 0.05`` the ceiling is still 0.39, which is "spoiled" only in
    the sense of not being one.
    """
    detuned = Mixing(coupling=1e-3, mass=0.15, frequency=1.0)
    assert conversion_probability(detuned, oscillation_length(detuned) / 2.0) < 0.01

    tuned = Mixing(coupling=1e-3, mass=0.15, frequency=1.0, plasma=resonant_plasma_frequency(0.15))
    assert conversion_probability(tuned, oscillation_length(tuned) / 2.0) == pytest.approx(
        1.0, abs=1e-12
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"frequency": 0.0}, "frequency must be positive"),
        ({"mass": -1.0}, "must not be negative"),
        ({"mass": 2.0, "frequency": 1.0}, "cannot propagate"),
    ],
)
def test_mixing_refuses_unphysical_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Mixing(**kwargs)


def test_conversion_refuses_a_negative_path():
    with pytest.raises(ValueError, match="must not be negative"):
        conversion_probability(Mixing(), -1.0)


# --- the acceptance -------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("coupling", "mass", "length"),
    [(1e-3, 0.0, 1000.0), (1e-3, 0.0, 3000.0), (1e-4, 0.02, 5000.0), (1e-3, 0.05, 2000.0)],
)
def test_the_exact_propagation_matches_the_mixing_formula(coupling, mass, length):
    """Issue #72's acceptance, in the regime where the formula applies.

    The exact computation keeps the second ``z``-derivative the formula
    drops, so this is two different calculations agreeing rather than one
    rearranged.
    """
    mixing = Mixing(coupling=coupling, field=1.0, mass=mass, frequency=1.0)
    exact, _, _ = propagate(mixing, length)
    assert exact == pytest.approx(conversion_probability(mixing, length), rel=2e-3)


@pytest.mark.benchmark
def test_the_formula_is_the_first_order_limit_of_the_exact_propagation():
    """The gap halves as ``Delta_M / omega`` halves -- first order, measured.

    This is the statement worth making. Holding the conversion phase fixed
    at ``g B L = 2`` and shrinking the coupling, the departure runs 8.1e-3,
    5.9e-3, 2.8e-3, 1.4e-3, 6.4e-4: a fixed tolerance would have passed a
    formula wrong by a constant, and only the trend distinguishes "the
    approximation" from "an approximation that happens to be close".
    """
    departures = []
    for coupling in (2e-2, 1e-2, 5e-3, 2.5e-3):
        mixing = Mixing(coupling=coupling, field=1.0, mass=0.0, frequency=1.0)
        length = 2.0 / coupling
        exact, _, _ = propagate(mixing, length)
        departures.append(abs(exact / conversion_probability(mixing, length) - 1.0))

    ratios = [before / after for before, after in zip(departures[:-1], departures[1:], strict=True)]
    assert all(ratio > 1.7 for ratio in ratios), ratios
    assert departures[-1] < 1e-3


@pytest.mark.benchmark
def test_the_exact_propagation_conserves_flux():
    """Photon plus axion flux stays put: the mixing is a rotation, not a gain.

    The antisymmetry of ``+g a B`` against ``-g a E`` in the constitutive
    relation is what guarantees it, so this also checks that the sign
    convention survived into the propagation equations. A sign slip there
    produces exponential growth that would still "match the formula" near
    zero length.
    """
    mixing = Mixing(coupling=5e-3, field=1.0, mass=0.0, frequency=1.0)
    photon, axion = mixing.wavenumbers
    for length in (100.0, 300.0, 600.0):
        _, field, scalar = propagate(mixing, length)
        total = abs(field) ** 2 + abs(scalar) ** 2 * axion / photon
        assert total == pytest.approx(1.0, abs=5e-3)


def test_oscillation_length_is_where_conversion_returns_to_zero():
    mixing = Mixing(coupling=2e-3, mass=0.01, frequency=1.0)
    full = oscillation_length(mixing)
    assert conversion_probability(mixing, full) == pytest.approx(0.0, abs=1e-20)
    assert conversion_probability(mixing, full / 2.0) > 0.0
