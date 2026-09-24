"""Tier B quantum-corrected plugins and the harness that checks their limits."""

import numpy as np
import pytest
import sympy as sp

from particlesim.scenarios.singularity import FLRWBackground
from particlesim.scenarios.singularity.flrw import W_MATTER
from particlesim.theories import get_theory, list_theories
from particlesim.theories.asafety.rg_improved import RGImprovedSchwarzschild
from particlesim.theories.limits import check_all, check_gr_limit
from particlesim.theories.lqg.lqc import RHO_CRITICAL_PLANCK, EffectiveLQC


def test_the_quantum_corrected_plugins_are_registered_as_tier_b():
    registry = list_theories()
    for tid in ("lqg.lqc", "lqg.polymer_bh", "asafety.rg_improved"):
        assert tid in registry
        assert registry[tid].tier == "B"


# --- Effective loop quantum cosmology -------------------------------------


def test_lqc_is_parameterised_so_its_gr_limit_is_reachable():
    """A plugin whose GR limit sits at infinite coupling can never have that
    limit checked; using the inverse puts it at zero, where it is a value."""
    theory = EffectiveLQC()
    assert theory.gr_limit() == {"inverse_rho_c": 0.0}
    assert np.isfinite(theory.rho_critical)
    at_limit = EffectiveLQC(inverse_rho_c=0.0)
    assert at_limit.rho_critical == float("inf")

    report = check_gr_limit(EffectiveLQC)
    assert report.passed, report.reasons
    assert report.reduced_equations_match is True


def test_lqc_default_critical_density_is_the_published_value():
    assert RHO_CRITICAL_PLANCK == pytest.approx(0.41)
    assert EffectiveLQC().rho_critical == pytest.approx(0.41, rel=1e-12)


def test_lqc_friedmann_equation_bends_over_at_the_critical_density():
    eq = EffectiveLQC().reduced_equations("flrw")
    rho_c = RHO_CRITICAL_PLANCK
    # Far below rho_c it is indistinguishable from GR.
    assert eq(1e-6) == pytest.approx(8 * np.pi / 3 * 1e-6, rel=1e-5)
    # It vanishes exactly at rho_c: that is the bounce.
    assert eq(rho_c) == pytest.approx(0.0, abs=1e-15)
    # And is maximal halfway.
    assert eq(rho_c / 2) > eq(rho_c * 0.9)


def test_lqc_refuses_sectors_it_does_not_cover():
    with pytest.raises(NotImplementedError, match="FLRW"):
        EffectiveLQC().reduced_equations("spherical")


@pytest.mark.benchmark
def test_lqc_bounce_occurs_at_the_critical_density():
    """The acceptance criterion: a collapsing universe bounces at rho_c."""
    theory = EffectiveLQC()
    bg = FLRWBackground(
        components={W_MATTER: 1e-3},
        density_correction=theory.density_correction(),
    )
    sol = bg.evolve(a0=1.0, t_max=5000.0, expanding=False)
    assert sol.bounced
    assert sol.density.max() == pytest.approx(theory.rho_critical, rel=1e-2)

    # The same data is singular without the correction.
    plain = FLRWBackground(components={W_MATTER: 1e-3})
    assert plain.evolve(a0=1.0, t_max=5000.0, expanding=False).outcome == "singularity"


def test_lqc_regime_of_validity_and_predictions():
    theory = EffectiveLQC()
    assert theory.regime_of_validity(0.1)
    assert not theory.regime_of_validity(10.0)
    pred = theory.observable_predictions()
    assert pred["bounce"] and pred["singularity_resolved"]
    assert pred["max_density"] == pytest.approx(RHO_CRITICAL_PLANCK)
    # At the GR limit it predicts no bounce at all.
    assert not EffectiveLQC(inverse_rho_c=0.0).observable_predictions()["bounce"]


# --- RG-improved Schwarzschild --------------------------------------------


def test_rg_improved_reduces_to_schwarzschild_symbolically():
    report = check_gr_limit(RGImprovedSchwarzschild)
    assert report.passed, report.reasons
    assert report.metric_family_matches is True

    at_limit = RGImprovedSchwarzschild(omega=0.0)
    r = sp.Symbol("r", positive=True)
    diff = sp.simplify(
        at_limit.metric_family({"mass": 2.0}) - get_theory("gr").metric_family({"mass": 2.0})
    )
    assert diff.is_zero_matrix
    assert r  # symbol used only for readability of the comparison above


def test_running_coupling_vanishes_at_the_origin_and_recovers_far_away():
    """The whole point of the construction: gravity weakens in the deep
    interior, which is what removes the singularity."""
    theory = RGImprovedSchwarzschild()
    assert theory.running_g(1e-6, 1.0) < 1e-10
    assert theory.running_g(1e6, 1.0) == pytest.approx(1.0, rel=1e-5)
    assert np.all(np.diff(theory.running_g(np.linspace(0.1, 50, 200), 1.0)) > 0)


def test_large_black_hole_has_two_horizons_and_matches_schwarzschild_outside():
    theory = RGImprovedSchwarzschild()
    roots = theory.horizon_radii(mass=100.0)
    assert len(roots) == 2
    inner, outer = min(roots), max(roots)
    assert inner < outer
    # For a large mass the outer horizon is very close to 2M.
    assert outer == pytest.approx(200.0, rel=1e-2)


@pytest.mark.benchmark
def test_horizon_disappears_below_a_critical_mass():
    """The plugin's sharpest falsifiable claim, and the reason horizon_radii
    returns a list rather than a number."""
    theory = RGImprovedSchwarzschild()
    m_crit = theory.critical_mass()
    assert 0.1 < m_crit < 100.0
    assert theory.horizon_radii(m_crit * 1.05) != []
    assert theory.horizon_radii(m_crit * 0.95) == []
    # General relativity has a horizon at every mass, so this is a genuine
    # difference rather than a numerical artefact.
    assert (RGImprovedSchwarzschild(omega=0.0).lapse_function(2.0 * 0.01, 0.01)) == pytest.approx(
        0.0
    )


def test_check_all_covers_every_registered_plugin_including_tier_b():
    reports = check_all(check_action=False)
    assert {"gr", "gr.lambda", "lqg.lqc", "lqg.polymer_bh", "asafety.rg_improved"} <= set(reports)
    for tid, report in reports.items():
        assert report.checked, f"{tid} was not checked at all: {report.reasons}"
        assert report.passed, f"{tid} fails its declared GR limit: {report.reasons}"


def test_a_tier_b_plugin_with_a_wrong_limit_is_caught():
    """The Tier B path must be able to fail, like the Tier A one."""

    class BrokenLQC(EffectiveLQC):
        id = "test.broken_lqc"

        def gr_limit(self) -> dict[str, float]:
            return {"inverse_rho_c": 5.0}  # wrong: still corrected

    report = check_gr_limit(BrokenLQC)
    assert report.checked and not report.passed
    assert report.reduced_equations_match is False
    assert any("reduced FLRW dynamics differ" in r for r in report.reasons)
