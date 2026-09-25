"""The quintic compactification's emitted plugin (issue #87).

The acceptance is that emitted couplings agree with a published numerical
result. The Yukawa coupling the plugin carries is built from instanton
numbers that ``test_special_geometry`` holds to Candelas et al. (1991). This
file checks that the plugin carries what the modules compute, and that it
is gated like every other plugin.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.theories.compactify import (
    DEFAULT_QUINTIC,
    QuinticCompactification,
    QuinticVacuum,
    emit_quintic_plugin,
)
from particlesim.theories.limits import check_gr_limit
from particlesim.theories.registry import list_theories
from particlesim.theories.special_geometry import QuinticModuli


def test_the_quintic_plugin_is_registered_and_passes_the_gr_limit():
    assert list_theories()["string.compactify.quintic"] is QuinticCompactification
    report = check_gr_limit(QuinticCompactification, check_action=False)
    assert report.checked
    assert report.passed, report.reasons


def test_listing_the_plugin_does_not_compute_the_kaluza_klein_scale():
    """The Kaluza-Klein scale needs a metric and a Laplacian, so it waits until asked for."""
    plugin = emit_quintic_plugin(QuinticVacuum())
    assert "kaluza_klein_scale" in plugin.derived
    assert plugin.derived._values == {}


def test_the_emitted_yukawa_coupling_is_special_geometrys():
    derived = QuinticCompactification.derived
    t = DEFAULT_QUINTIC.kahler_modulus
    assert derived["yukawa_coupling"] == QuinticModuli().normalized_yukawa(t)
    # At t = 2i the instantons raise kappa by 0.2%. The zeta(3) term raises
    # e^-K by 3.6% and lowers G by 14%, which puts the coupling 21% above its
    # large-volume value, 2/sqrt(3).
    assert derived["yukawa_coupling"] == pytest.approx(1.3933, abs=1e-4)
    assert derived["gauge_coupling"] == pytest.approx(0.1)


def test_the_field_content_is_the_quintics():
    """101 complex-structure moduli, one Kahler modulus, the dilaton; 100 generations."""
    moduli = DEFAULT_QUINTIC.moduli()
    kinds = [m.kind for m in moduli]
    assert kinds.count("kahler") == 1
    assert kinds.count("complex_structure") == 101
    assert kinds.count("axio_dilaton") == 1
    assert QuinticCompactification.derived["generations"] == 100
    assert QuinticCompactification.derived["modulus_count"] == 103


def test_the_kaluza_klein_scale_comes_from_the_numerical_metric():
    """``sqrt(lambda_1 / Im t)``, with ``lambda_1`` the Laplacian's first level.

    A small balanced metric keeps this fast. The first level is 20-fold
    degenerate, and scaling the Kahler class by ``Im t`` divides every
    eigenvalue by ``Im t``.
    """
    vacuum = QuinticVacuum(kahler_modulus=3.0j, metric_degree=2, points=20000)
    spectrum = vacuum.spectrum()
    levels = spectrum.levels()
    assert levels[1][1] == 20
    first = spectrum.eigenvalues[1:21].mean()
    scale = vacuum.kaluza_klein_scale()
    assert scale**2 * 3.0 == pytest.approx(first, rel=1e-12)
    assert first * spectrum.volume ** (1 / 3) == pytest.approx(42.0, rel=0.03)
    plugin = emit_quintic_plugin(vacuum)
    assert plugin.derived["kaluza_klein_scale"] == scale
    assert np.isfinite(scale) and scale > 0


def test_a_vacuum_near_the_conifold_is_refused():
    with pytest.raises(ValueError, match="too small"):
        QuinticVacuum(kahler_modulus=1.0j)
    with pytest.raises(ValueError, match="string coupling"):
        QuinticVacuum(string_coupling=1.5)
