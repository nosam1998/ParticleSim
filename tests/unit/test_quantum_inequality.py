import numpy as np
import pytest

from particlesim.analysis.quantum_inequality import (
    check_ford_roman,
    ford_roman_bound,
    lorentzian_weight,
    scan_sampling_times,
)


def test_bound_formula_and_scaling():
    assert ford_roman_bound(1.0) == pytest.approx(-3.0 / (32 * np.pi**2))
    # The bound weakens as tau0^-4.
    assert ford_roman_bound(2.0) == pytest.approx(ford_roman_bound(1.0) / 16)
    with pytest.raises(ValueError):
        ford_roman_bound(0.0)


def test_lorentzian_weight_is_normalised_on_a_wide_window():
    tau = np.linspace(-4000.0, 4000.0, 400001)
    assert np.trapezoid(lorentzian_weight(tau, 1.0), tau) == pytest.approx(1.0, abs=1e-3)


def test_vacuum_and_positive_energy_satisfy_the_bound():
    tau = np.linspace(-50.0, 50.0, 4001)
    for rho in (np.zeros_like(tau), np.ones_like(tau)):
        res = check_ford_roman(tau, rho, 1.0)
        assert res.satisfied
        assert res.sampled_energy >= res.bound


def test_sustained_negative_energy_violates_at_long_sampling_times():
    """Constant negative density -c averages to -c, which beats the bound once
    3/(32 pi^2 tau0^4) < c."""
    c = 1e-3
    tau = np.linspace(-500.0, 500.0, 200001)
    rho = np.full_like(tau, -c)
    crossover = (3.0 / (32 * np.pi**2 * c)) ** 0.25
    short = check_ford_roman(tau, rho, crossover / 3, outside="extend")
    long_ = check_ford_roman(tau, rho, crossover * 3, outside="extend")
    assert short.satisfied
    assert not long_.satisfied
    assert long_.sampled_energy == pytest.approx(-c, rel=1e-2)
    assert long_.violation_factor > 1


def test_brief_negative_pulse_within_the_bound_is_allowed():
    """A pulse short and weak enough passes: this is the regime quantum
    inequalities permit."""
    tau = np.linspace(-20.0, 20.0, 20001)
    width, tau0 = 0.05, 1.0
    depth = abs(ford_roman_bound(tau0)) * 0.05
    rho = -depth * np.exp(-((tau / width) ** 2))
    res = check_ford_roman(tau, rho, tau0)
    assert res.satisfied
    assert res.sampled_energy < 0


def test_curvature_flag_and_scan():
    tau = np.linspace(-20.0, 20.0, 2001)
    rho = np.full_like(tau, -1e-2)
    # Kretschmann 1e-4 gives a curvature radius of 10, so tau0 = 1 is fine and 50 is not.
    assert check_ford_roman(tau, rho, 1.0, kretschmann_max=1e-4).curvature_ok
    assert not check_ford_roman(tau, rho, 50.0, kretschmann_max=1e-4).curvature_ok
    assert check_ford_roman(tau, rho, 1.0).curvature_ok is None
    results = scan_sampling_times(tau, rho, np.array([0.5, 1.0, 5.0]), outside="extend")
    assert [r.tau0 for r in results] == [0.5, 1.0, 5.0]
    # Violation grows monotonically with sampling time for a constant density.
    assert results[0].violation_factor < results[-1].violation_factor


def test_outside_zero_does_not_renormalize_a_localized_pulse():
    """A pulse confined to the window must give the same answer however wide
    the window is; renormalizing by captured weight would inflate it."""
    width, tau0, depth = 0.2, 2.0, 1.0
    answers = []
    for half in (8.0, 40.0, 200.0):
        tau = np.linspace(-half, half, 40001)
        rho = -depth * np.exp(-((tau / width) ** 2))
        res = check_ford_roman(tau, rho, tau0)
        answers.append(res.sampled_energy)
        assert res.weight_captured <= 1.0 + 1e-9
    assert max(answers) - min(answers) < 1e-6 * abs(answers[0])
    # "extend" on the narrow window inflates the magnitude, which is the trap.
    tau = np.linspace(-8.0, 8.0, 40001)
    rho = -depth * np.exp(-((tau / width) ** 2))
    ext = check_ford_roman(tau, rho, tau0, outside="extend")
    assert abs(ext.sampled_energy) > abs(answers[0])


def test_input_validation():
    tau = np.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError):
        check_ford_roman(tau, tau[:-1], 1.0)
    with pytest.raises(ValueError):
        check_ford_roman(tau[::-1], tau, 1.0)
    with pytest.raises(ValueError):
        check_ford_roman(tau, tau, 1.0, outside="nonsense")
