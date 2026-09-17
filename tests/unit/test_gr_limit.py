"""The GR-limit harness, including negative controls (design doc Section 4.2)."""

import sympy as sp

from particlesim.symbolic.curvature import MetricGeometry
from particlesim.theories.base import Coupling, FieldSpec, Theory
from particlesim.theories.gr import GeneralRelativity
from particlesim.theories.limits import check_all, check_gr_limit


class _GRPlusLambda(Theory):
    """GR with a cosmological constant; recovers GR at Lambda = 0."""

    id = "test.gr_lambda"
    tier = "A"
    fields = [FieldSpec("g", "metric", rank=2)]
    couplings = [Coupling("Lambda", 0.0, "inverse_length_squared", (-10.0, 10.0))]
    provenance = "test fixture"

    def lagrangian(self, metric, coords):
        geom = MetricGeometry(metric, coords)
        lam = self.values["Lambda"]
        return sp.sqrt(-metric.det()) * (geom.ricci_scalar - 2 * lam) / (16 * sp.pi)

    def effective_stress_energy(self, einstein, metric):
        return einstein / (8 * sp.pi) + self.values["Lambda"] * metric / (8 * sp.pi)

    def gr_limit(self):
        return {"Lambda": 0.0}


class _WrongLimit(_GRPlusLambda):
    """Negative control: claims a GR limit that does not recover GR."""

    id = "test.wrong_limit"

    def gr_limit(self):
        return {"Lambda": 1.0}


class _RSquaredWrongLimit(Theory):
    """Negative control whose extra term is curvature-dependent.

    R^2 vanishes on any Ricci-flat or flat metric, so this plugin is
    indistinguishable from GR there however wrong its declared limit is.
    It exists to show why the default test metric must be curved.
    """

    id = "test.r_squared"
    tier = "A"
    fields = [FieldSpec("g", "metric", rank=2)]
    couplings = [Coupling("alpha", 0.0, "length_squared", (-10.0, 10.0))]
    provenance = "test fixture"

    def lagrangian(self, metric, coords):
        geom = MetricGeometry(metric, coords)
        r = geom.ricci_scalar
        return sp.sqrt(-metric.det()) * (r + self.values["alpha"] * r**2) / (16 * sp.pi)

    def effective_stress_energy(self, einstein, metric):
        return einstein / (8 * sp.pi)

    def gr_limit(self):
        return {"alpha": 1.0}  # wrong: alpha must be 0 to recover GR


class _LimitAtInfinity(_GRPlusLambda):
    id = "test.limit_at_infinity"

    def gr_limit(self):
        return {"Lambda": float("inf")}


def test_gr_itself_passes():
    report = check_gr_limit(GeneralRelativity)
    assert report.passed and report.checked
    assert report.stress_energy_matches and report.lagrangian_matches


def test_correct_limit_passes():
    report = check_gr_limit(_GRPlusLambda)
    assert report.passed
    assert report.stress_energy_matches and report.lagrangian_matches


def test_wrong_limit_is_caught_by_both_checks():
    """A harness that cannot fail proves nothing."""
    report = check_gr_limit(_WrongLimit)
    assert report.checked and not report.passed
    assert report.stress_energy_matches is False
    assert report.lagrangian_matches is False
    assert any("stress-energy differs" in r for r in report.reasons)


def test_default_test_metric_must_be_curved_to_discriminate():
    """A curvature-dependent term is invisible on a flat metric.

    Minkowski clears the R^2 plugin despite its wrong declared limit; the
    conformally flat default catches it. A cosmological constant, by
    contrast, shows up even in flat space, so it is the curvature-coupled
    case that sets the requirement on the default metric.
    """
    flat = sp.diag(-1, 1, 1, 1)
    assert check_gr_limit(_RSquaredWrongLimit, metric=flat).lagrangian_matches is True
    caught = check_gr_limit(_RSquaredWrongLimit)
    assert caught.lagrangian_matches is False
    assert not caught.passed


def test_limit_at_infinity_is_reported_not_silently_passed():
    report = check_gr_limit(_LimitAtInfinity)
    assert not report.passed and not report.checked
    assert any("infinity" in r for r in report.reasons)


def test_check_all_covers_registered_plugins():
    reports = check_all(check_action=False)
    assert "gr" in reports
    assert reports["gr"].passed
    for tid, report in reports.items():
        assert report.passed, f"{tid} fails its declared GR limit: {report.reasons}"
