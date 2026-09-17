"""Alternative early-universe backgrounds against their defining solutions.

Each of the three scenarios is defined by a scaling solution, and each one
here is checked against the closed form derived in its module: the
ekpyrotic ``a ~ (-t)^(2/c^2)``, the dilaton-driven ``a ~ |t|^(-1/sqrt(d))``,
and the string gas's radiation era after its winding modes annihilate.

Two of the checks are exact symmetries rather than fitted exponents, and
those are the sharpest ones: scale-factor duality in the pre-big-bang
system and T-duality in the string gas both hold to machine precision or
not at all.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from particlesim.cosmo.dynamics import evolve
from particlesim.cosmo.early import (
    CRITICAL_STEEPNESS_SQUARED,
    MAXIMUM_LARGE_DIMENSIONS,
    DilatonVacuum,
    EkpyroticPotential,
    StringGas,
    bounce_report,
    contraction_report,
    dual,
    evolve_contraction,
    evolve_dilaton,
    hagedorn_gas,
    scaling_exponent,
    scaling_solution,
    winding_modes_annihilate,
)
from particlesim.theories.registry import get_theory

STEEPNESS = [4.0, 10.0, 30.0]
DIMENSIONS = [2, 3, 4, 9]


# --- ekpyrotic slow contraction ----------------------------------------


@pytest.mark.benchmark
@pytest.mark.parametrize("steepness", STEEPNESS)
def test_ekpyrotic_scaling_solution_is_reproduced(steepness):
    """``a ~ (-t)^p`` with ``p = 2/c^2``, from the integrated equations.

    The exponent is what defines the model, and it comes out to a part in
    ten million from a run spanning six decades of cosmic time. The
    constraint is monitored rather than imposed, so its drift is an
    independent statement about the integration.
    """
    potential = EkpyroticPotential(steepness=steepness)
    run = evolve_contraction(potential)
    exponent = scaling_exponent(run.time, run.scale_factor)
    assert exponent == pytest.approx(potential.scaling_exponent, rel=1e-7)
    assert float(run.epsilon[-1]) == pytest.approx(potential.epsilon, rel=1e-9)
    assert run.constraint_drift < 1e-9
    assert abs(run.singular_time) < 1e-5
    assert not run.reached_ceiling


@pytest.mark.parametrize("steepness", STEEPNESS)
def test_ekpyrotic_trajectory_matches_the_closed_form(steepness):
    potential = EkpyroticPotential(steepness=steepness)
    run = evolve_contraction(potential)
    scale, field, velocity, hubble = scaling_solution(potential, run.time)
    normalised = (run.scale_factor / run.scale_factor[0]) / (scale / scale[0])
    assert np.allclose(normalised, 1.0, rtol=1e-6)
    assert np.allclose(run.field, field, atol=1e-6)
    assert np.allclose(run.velocity, velocity, rtol=1e-6)
    assert np.allclose(run.hubble, hubble, rtol=1e-6)


@pytest.mark.parametrize("steepness", STEEPNESS)
def test_the_ekpyrotic_equation_of_state_is_stiff(steepness):
    potential = EkpyroticPotential(steepness=steepness)
    run = evolve_contraction(potential)
    assert float(run.equation_of_state[-1]) == pytest.approx(potential.equation_of_state, rel=1e-9)
    assert potential.equation_of_state > 1.0


def test_a_shallow_potential_has_no_ekpyrotic_scaling_solution():
    """``c^2 <= 6`` is ``epsilon <= 3``, where the required amplitude flips sign.

    The field equation demands ``A = (2 - 6p)/c^2 > 0`` for the potential
    to be negative, and that is the same inequality as diluting anisotropy.
    A shallower potential has no such solution at all, rather than a
    marginal one, and the constructor says so.
    """
    assert CRITICAL_STEEPNESS_SQUARED == 6.0
    with pytest.raises(ValueError, match="no ekpyrotic scaling solution"):
        EkpyroticPotential(steepness=math.sqrt(6.0))
    with pytest.raises(ValueError, match="no ekpyrotic scaling solution"):
        EkpyroticPotential(steepness=2.0)
    with pytest.raises(ValueError, match="scale must be positive"):
        EkpyroticPotential(scale=0.0, steepness=10.0)


@pytest.mark.parametrize(("factor", "tolerance"), [(1.01, 1e-3), (1.1, 3e-3), (2.0, 1e-2)])
def test_the_ekpyrotic_scaling_solution_is_an_attractor(factor, tolerance):
    """Extra kinetic energy is absorbed by moving the crunch, not the exponent.

    ``epsilon`` returns to ``c^2/2`` from a start well off the attractor,
    and what the perturbation leaves behind is a shifted singular time --
    the family's zero mode. The run then reaches the curvature ceiling
    before the requested stop, which is the crunch arriving early rather
    than a failure.
    """
    potential = EkpyroticPotential(steepness=10.0)
    exact = scaling_solution(potential, -1e6)
    run = evolve_contraction(potential, velocity=float(exact[2]) * factor)
    assert float(run.epsilon[-1]) == pytest.approx(potential.epsilon, rel=tolerance)
    assert run.reached_ceiling
    assert run.singular_time < 0.0
    assert run.constraint_drift < 1e-9


def test_the_ekpyrotic_contraction_is_slow():
    """Many Hubble times, almost no contraction. That is the point of it.

    Over six decades of cosmic time the curvature grows by six decades
    while the scale factor falls by a third. A dust or radiation
    contraction covering the same range of curvature would have collapsed
    by a factor of ten thousand or more.
    """
    run = evolve_contraction(EkpyroticPotential(steepness=10.0))
    assert run.contraction_factor == pytest.approx(1.32, rel=0.02)
    assert abs(run.hubble[-1] / run.hubble[0]) == pytest.approx(1e6, rel=1e-3)


def test_the_ekpyrotic_contraction_dilutes_anisotropy():
    run = evolve_contraction(EkpyroticPotential(steepness=10.0))
    report = contraction_report(float(run.epsilon[-1]))
    assert report.dilutes_anisotropy
    assert report.ekpyrotic
    assert report.anisotropy_ratio(1000.0) < 1e-100


def test_ekpyrotic_error_paths():
    potential = EkpyroticPotential(steepness=10.0)
    with pytest.raises(ValueError, match="need start < stop < 0"):
        evolve_contraction(potential, start=-1.0, stop=-10.0)
    with pytest.raises(ValueError, match="need start < stop < 0"):
        evolve_contraction(potential, start=1.0, stop=10.0)
    with pytest.raises(ValueError, match="t < 0"):
        scaling_solution(potential, 1.0)
    with pytest.raises(ValueError, match="does not exceed the depth"):
        evolve_contraction(potential, velocity=0.0)


# --- pre-big-bang dilaton vacuum ---------------------------------------


@pytest.mark.benchmark
@pytest.mark.parametrize("dimension", DIMENSIONS)
@pytest.mark.parametrize("branch", [1, -1])
def test_dilaton_vacuum_exponents_are_reproduced(dimension, branch):
    """``a ~ |t|^(-1/(branch sqrt(d)))`` and ``e^phi ~ |t|^q``, both fitted.

    The unreduced system is integrated and the independent constraint is
    monitored, so agreement here is a statement about the equations rather
    than about a substitution.
    """
    background = DilatonVacuum(dimension=dimension, branch=branch)
    run = evolve_dilaton(background)
    exponent = scaling_exponent(run.time, run.scale_factor)
    coupling = float(np.polyfit(np.log(-run.time), run.dilaton, 1)[0])
    assert exponent == pytest.approx(background.scale_exponent, rel=1e-9)
    assert coupling == pytest.approx(background.coupling_exponent, rel=1e-9)
    assert run.constraint_drift < 1e-12


@pytest.mark.benchmark
def test_the_three_dimensional_exponents_are_the_published_ones():
    """``a ~ |t|^(-0.5774)`` and ``e^phi ~ |t|^(-2.7321)`` in three dimensions."""
    background = DilatonVacuum(dimension=3, branch=1)
    assert background.scale_exponent == pytest.approx(-1.0 / math.sqrt(3.0), rel=1e-15)
    assert background.coupling_exponent == pytest.approx(-(1.0 + math.sqrt(3.0)), rel=1e-15)
    assert background.superinflating
    assert not DilatonVacuum(dimension=3, branch=-1).superinflating


def test_the_dilaton_branch_runs_into_strong_coupling_and_high_curvature():
    """Both diverge together, which is why the scenario needs a graceful exit.

    Over four decades of cosmic time the curvature grows by four and the
    coupling by eleven. Neither this module nor the tree-level action it
    integrates has anything to stop that: the ``alpha'`` corrections that
    would are not here, and the docstring says so rather than the run
    quietly ending.
    """
    run = evolve_dilaton(DilatonVacuum())
    summary = run.summary()
    assert summary["curvature_growth"] == pytest.approx(1e4, rel=1e-3)
    assert summary["coupling_growth"] > 1e10
    assert summary["expansion"] > 100.0
    assert np.all(np.diff(run.coupling) > 0.0)


@pytest.mark.benchmark
def test_scale_factor_duality_is_exact():
    """``a -> 1/a``, ``phi -> phi - 2 d ln a`` leaves the constraint term by term.

    The dual's ``H`` and ``phi_dot`` are different numbers from the
    original's, so satisfying the same constraint to machine precision is a
    statement about the map and not an identity of the arrays. The shifted
    dilaton is invariant, which is the compact way to say the same thing.
    """
    run = evolve_dilaton(DilatonVacuum())
    mirror = dual(run)
    residual = mirror.background.constraint(mirror.hubble, mirror.velocity)
    scale = float(np.abs(mirror.velocity**2).max())
    assert float(np.abs(residual).max()) / scale < 1e-12
    assert np.allclose(mirror.shifted_dilaton, run.shifted_dilaton, atol=1e-12)
    assert np.allclose(mirror.hubble, -run.hubble)
    assert mirror.background.branch == -run.background.branch
    assert mirror.summary()["expansion"] == pytest.approx(
        1.0 / run.summary()["expansion"], rel=1e-9
    )
    assert mirror.summary()["curvature_growth"] == pytest.approx(run.summary()["curvature_growth"])


@pytest.mark.benchmark
def test_the_dual_is_itself_a_solution_of_the_evolution():
    """Integrate the dual branch from the dual's own initial data.

    This is the dynamical version of the previous test: rather than check
    that the mapped trajectory satisfies the constraint, integrate the
    other branch forwards and see whether it lands on the mapped
    trajectory. It does, to a part in ten million over four decades, which
    is the accuracy of the two independent integrations.
    """
    run = evolve_dilaton(DilatonVacuum())
    mirror = dual(run)
    independent = evolve_dilaton(
        DilatonVacuum(dimension=3, branch=-1), hubble=float(mirror.hubble[0])
    )
    ratio = (independent.scale_factor / independent.scale_factor[0]) / (
        mirror.scale_factor / mirror.scale_factor[0]
    )
    assert np.allclose(ratio, 1.0, rtol=1e-7)
    assert np.allclose(independent.hubble, mirror.hubble, rtol=1e-7)


def test_the_dilaton_velocity_solves_the_constraint_on_both_branches():
    for branch in (1, -1):
        background = DilatonVacuum(dimension=3, branch=branch)
        velocity = float(background.velocity_from_hubble(0.3))
        assert float(background.constraint(0.3, velocity)) == pytest.approx(0.0, abs=1e-14)


def test_dilaton_error_paths():
    with pytest.raises(ValueError, match="dimension must be at least 1"):
        DilatonVacuum(dimension=0)
    with pytest.raises(ValueError, match=r"branch must be \+1 or -1"):
        DilatonVacuum(branch=2)
    with pytest.raises(ValueError, match="need start < stop < 0"):
        evolve_dilaton(start=-1.0, stop=-10.0)
    with pytest.raises(ValueError, match="t < 0"):
        DilatonVacuum().scaling_solution(0.0)


# --- string gas --------------------------------------------------------


@pytest.mark.parametrize("dimension", [2, 3, 6])
def test_the_string_gas_equation_of_state_interpolates(dimension):
    """Pure momentum is radiation, pure winding is its negative, equal is dust-free.

    ``w = (1/d)(E_momentum - E_winding)/(E_momentum + E_winding)``, and the
    pressure is checked against ``-dE/dV`` independently so that the
    closed form is not merely restated.
    """
    momentum = StringGas(momentum_scale=1.0, winding_scale=0.0, dimension=dimension)
    winding = StringGas(momentum_scale=0.0, winding_scale=1.0, dimension=dimension)
    balanced = StringGas(dimension=dimension)
    assert momentum.equation_of_state == pytest.approx(1.0 / dimension, rel=1e-14)
    assert winding.equation_of_state == pytest.approx(-1.0 / dimension, rel=1e-14)
    assert balanced.equation_of_state == pytest.approx(0.0, abs=1e-15)
    for gas in (momentum, winding, balanced):
        assert gas.pressure / gas.density == pytest.approx(gas.equation_of_state, abs=1e-14)


@pytest.mark.parametrize("radius", [0.25, 0.5, 1.0, 2.0, 7.0])
def test_t_duality_leaves_the_string_gas_energy_invariant(radius):
    gas = StringGas(momentum_scale=1.3, winding_scale=0.7, radius=radius)
    assert gas.t_dual().energy == pytest.approx(gas.energy, rel=1e-15)
    assert gas.t_dual().t_dual() == gas


@pytest.mark.benchmark
def test_the_radion_sits_at_the_self_dual_radius():
    """The energy's minimum, found by scanning, is where the closed form says."""
    gas = StringGas(momentum_scale=2.0, winding_scale=0.5)
    assert gas.self_dual_radius == pytest.approx(2.0, rel=1e-15)
    radii = np.linspace(0.5, 6.0, 20001)
    curve = gas.energy_curve(radii)
    assert radii[int(np.argmin(curve))] == pytest.approx(gas.self_dual_radius, abs=1e-3)
    at_minimum = gas.at_radius(gas.self_dual_radius)
    assert at_minimum.radion_force == pytest.approx(0.0, abs=1e-15)
    assert at_minimum.stabilised
    assert gas.at_radius(1.0).radion_force > 0.0
    assert gas.at_radius(4.0).radion_force < 0.0


@pytest.mark.parametrize("radius", [0.5, 1.0, 3.0])
def test_a_pressureless_string_gas_is_a_stabilised_radion(radius):
    """One condition, not two: ``E_n/R = E_w R`` *is* ``R = sqrt(E_n/E_w)``."""
    gas = hagedorn_gas(radius=radius)
    assert gas.hagedorn
    assert gas.equation_of_state == pytest.approx(0.0, abs=1e-15)
    assert gas.self_dual_radius == pytest.approx(radius, rel=1e-14)
    assert gas.radion_force == pytest.approx(0.0, abs=1e-14)
    assert gas.stabilised


@pytest.mark.benchmark
def test_radiation_scaling_after_winding_annihilation():
    """``a ~ t^(1/2)`` once only momentum modes are left.

    The equation of state is the only thing the string gas supplies; the
    background comes from the Friedmann integrator in
    :mod:`particlesim.cosmo.dynamics`, and the exponent is fitted against
    the singular time the initial Hubble rate implies.
    """
    radiation = StringGas().after_winding_annihilation()
    assert radiation.equation_of_state == pytest.approx(1.0 / 3.0, rel=1e-15)
    assert radiation.expansion_exponent == pytest.approx(0.5, rel=1e-15)
    run = evolve(equation_of_state=radiation.equation_of_state, density=1e-6, contracting=False)
    singular = -radiation.expansion_exponent / float(run.hubble[0])
    fitted = scaling_exponent(run.time, run.scale_factor, singular)
    assert fitted == pytest.approx(radiation.expansion_exponent, rel=1e-8)


@pytest.mark.benchmark
def test_a_pressureless_gas_expands_as_two_thirds():
    gas = hagedorn_gas()
    assert gas.expansion_exponent == pytest.approx(2.0 / 3.0, rel=1e-15)
    run = evolve(equation_of_state=gas.equation_of_state, density=1e-6, contracting=False)
    singular = -gas.expansion_exponent / float(run.hubble[0])
    fitted = scaling_exponent(run.time, run.scale_factor, singular)
    assert fitted == pytest.approx(gas.expansion_exponent, rel=1e-8)


def test_the_brandenberger_vafa_dimension_count():
    """Two two-dimensional worldsheets meet generically only if ``2 + 2 >= D``."""
    assert MAXIMUM_LARGE_DIMENSIONS == 3
    assert [winding_modes_annihilate(d) for d in (1, 2, 3)] == [True, True, True]
    assert [winding_modes_annihilate(d) for d in (4, 5, 9)] == [False, False, False]
    with pytest.raises(ValueError, match="at least 1"):
        winding_modes_annihilate(0)


def test_string_gas_error_paths():
    with pytest.raises(ValueError, match="radius must be positive"):
        StringGas(radius=0.0)
    with pytest.raises(ValueError, match="dimension must be at least 1"):
        StringGas(dimension=0)
    with pytest.raises(ValueError, match="non-negative"):
        StringGas(momentum_scale=-1.0)
    with pytest.raises(ValueError, match="has no energy"):
        StringGas(momentum_scale=0.0, winding_scale=0.0)
    with pytest.raises(ValueError, match="not stabilised"):
        _ = StringGas(winding_scale=0.0).self_dual_radius
    with pytest.raises(ValueError, match="radii must be positive"):
        StringGas().energy_curve([1.0, 0.0])


# --- diagnostics -------------------------------------------------------


@pytest.mark.parametrize(
    ("epsilon", "dilutes"), [(1.0, False), (2.9, False), (3.0, False), (3.1, True), (50.0, True)]
)
def test_the_anisotropy_threshold_is_epsilon_three(epsilon, dilutes):
    """``rho_shear ~ a^-6`` against ``a^(-2 epsilon)``: the line is ``epsilon = 3``."""
    report = contraction_report(epsilon)
    assert report.dilutes_anisotropy is dilutes
    ratio = report.anisotropy_ratio(100.0)
    if dilutes:
        assert ratio < 1.0
    else:
        assert ratio >= 1.0
    assert contraction_report(3.0).anisotropy_ratio(100.0) == pytest.approx(1.0, rel=1e-12)
    assert report.equation_of_state == pytest.approx(2.0 * epsilon / 3.0 - 1.0, rel=1e-14)
    assert report.scaling_exponent == pytest.approx(1.0 / epsilon, rel=1e-14)


def test_contraction_report_refuses_a_non_positive_epsilon():
    with pytest.raises(ValueError, match="epsilon must be positive"):
        contraction_report(0.0)
    with pytest.raises(ValueError, match="contraction factor must be positive"):
        contraction_report(5.0).anisotropy_ratio(0.0)


@pytest.mark.benchmark
@pytest.mark.parametrize("equation_of_state", [0.0, 1.0 / 3.0])
def test_a_loop_quantum_bounce_does_not_violate_the_null_energy_condition(equation_of_state):
    """The bounce came from the theory, not from exotic matter.

    In general relativity ``H_dot = -(1/2)(rho + p)``, so a bounce needs
    ``rho + p < 0``. The loop-quantum-cosmology plugin bounces with
    ordinary matter -- ``rho + p > 0`` at the turning point, measured
    rather than assumed -- which is only possible because the Friedmann
    equation being integrated is not the general-relativistic one. That is
    the whole content of the Tier B hook, and this is where it shows up as
    a number.
    """
    run = evolve(theory=get_theory("lqg.lqc"), equation_of_state=equation_of_state, density=1e-6)
    report = bounce_report(
        run.time,
        run.scale_factor,
        run.hubble,
        run.density,
        equation_of_state * run.density,
        run.bounce_time,
    )
    assert report.bounced
    assert report.null_energy_violated is False
    assert report.null_energy_sum > 0.0
    assert report.hubble_slope > 0.0
    assert report.bounce_time == pytest.approx(run.bounce_time, rel=1e-12)


def test_general_relativity_crunches_instead_of_bouncing():
    run = evolve(theory=None, equation_of_state=0.0, density=1e-6)
    report = bounce_report(run.time, run.scale_factor, run.hubble, run.density, 0.0 * run.density)
    assert not report.bounced
    assert report.bounce_time is None
    assert report.null_energy_violated is None
    assert run.reached_singularity
    assert set(report.summary()) == {
        "bounced",
        "bounce_time",
        "minimum_scale_factor",
        "maximum_density",
        "hubble_slope",
        "null_energy_violated",
    }


def test_bounce_report_interpolates_its_own_bounce_time():
    time = np.array([0.0, 1.0, 2.0, 3.0])
    hubble = np.array([-2.0, -1.0, 1.0, 2.0])
    report = bounce_report(time, np.array([2.0, 1.5, 1.5, 2.0]), hubble)
    assert report.bounce_time == pytest.approx(1.5)
    assert report.hubble_slope == pytest.approx(2.0)
    assert math.isnan(report.maximum_density)


def test_bounce_report_checks_its_shapes():
    with pytest.raises(ValueError, match="same shape"):
        bounce_report([0.0, 1.0], [1.0, 1.0], [1.0])


def test_scaling_exponent_refuses_a_run_through_the_singular_time():
    with pytest.raises(ValueError, match="trim the run"):
        scaling_exponent([-2.0, -1.0, 0.0], [1.0, 1.0, 1.0])
