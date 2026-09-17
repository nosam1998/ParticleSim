"""Theory-driven background evolution and the LQC bounce (Sections 3.1, 4.3)."""

import numpy as np
import pytest

from particlesim.cosmo import bounce_density, evolve
from particlesim.cosmo.dynamics import FRIEDMANN, general_relativity_hubble_squared
from particlesim.theories.gr import GeneralRelativity
from particlesim.theories.lqg.lqc import RHO_CRITICAL_PLANCK, EffectiveLQC

EQUATIONS_OF_STATE = (0.0, 1.0 / 3.0, 1.0)


# --- the bounce density -----------------------------------------------------


def test_the_bounce_density_is_the_critical_density():
    """Acceptance for issue #40: 1%. Achieved at 6e-11.

    Found by root-finding on the plugin's own ``H^2(rho)``, so the answer
    does not depend on a step size or on where an evolution happened to be
    sampled.
    """
    found = bounce_density(EffectiveLQC())
    assert found == pytest.approx(RHO_CRITICAL_PLANCK, rel=0.01)
    assert found == pytest.approx(RHO_CRITICAL_PLANCK, rel=1e-9)


def test_a_theory_without_a_bounce_reports_none():
    """Not a large number that would look like a bounce nobody can reach."""
    assert bounce_density(GeneralRelativity()) is None
    assert bounce_density(None) is None
    assert bounce_density(EffectiveLQC(inverse_rho_c=0.0)) is None


def test_the_bounce_moves_with_the_critical_density():
    for critical in (0.1, 0.41, 2.0, 50.0):
        found = bounce_density(EffectiveLQC(inverse_rho_c=1.0 / critical))
        assert found == pytest.approx(critical, rel=1e-9)


def test_a_bracket_that_starts_past_the_bounce_is_refused():
    with pytest.raises(ValueError, match="no expanding branch"):
        bounce_density(EffectiveLQC(), lower=10.0)


def test_the_general_relativistic_friedmann_equation_is_the_default():
    rho = np.array([0.1, 1.0, 10.0])
    np.testing.assert_allclose(general_relativity_hubble_squared(rho), FRIEDMANN * rho)
    assert FRIEDMANN == pytest.approx(8 * np.pi / 3)


# --- the evolution ----------------------------------------------------------


@pytest.mark.parametrize("equation_of_state", EQUATIONS_OF_STATE)
def test_a_contracting_universe_bounces_at_the_critical_density(equation_of_state):
    """The dynamical half of the acceptance: the bounce is reached, not just
    predicted. Read at the turning point the integrator located, where
    ``H = 0`` exactly."""
    run = evolve(
        theory=EffectiveLQC(), equation_of_state=equation_of_state, density=1e-6, contracting=True
    )
    assert run.bounced
    assert not run.reached_singularity
    assert run.bounce_density == pytest.approx(RHO_CRITICAL_PLANCK, rel=1e-8)
    assert 0.0 < run.bounce_time < run.time[-1]
    assert run.minimum_scale_factor < 0.2
    assert run.constraint_drift < 1e-8


@pytest.mark.parametrize("starting_density", (1e-3, 1e-6, 1e-9))
def test_the_bounce_does_not_depend_on_where_the_collapse_started(starting_density):
    run = evolve(theory=EffectiveLQC(), equation_of_state=0.0, density=starting_density)
    assert run.bounce_density == pytest.approx(RHO_CRITICAL_PLANCK, rel=1e-8)


def test_general_relativity_crunches_instead_of_bouncing():
    """The contrast that makes the bounce a prediction rather than a setting.

    The crunch is a terminal event, not an integrator failure: run into
    ``a = 0`` without one, the solver stops with a step-size message and the
    physics is reported as an exception.
    """
    run = evolve(theory=None, equation_of_state=0.0, density=1e-6, contracting=True)
    assert not run.bounced
    assert run.bounce_time is None
    assert run.bounce_density is None
    assert run.reached_singularity
    assert run.constraint_drift < 1e-8


def test_the_plugin_at_its_general_relativistic_limit_also_crunches():
    """The declared limit, checked dynamically rather than only in the
    algebra: at ``1/rho_c = 0`` the plugin's universe ends in a singularity
    like general relativity's."""
    run = evolve(theory=EffectiveLQC(inverse_rho_c=0.0), equation_of_state=0.0, density=1e-6)
    assert not run.bounced
    assert run.reached_singularity


def test_an_expanding_universe_never_reaches_the_bounce():
    run = evolve(theory=EffectiveLQC(), equation_of_state=0.0, density=1e-6, contracting=False)
    assert not run.bounced
    assert run.maximum_density <= 1e-6 * (1 + 1e-9)
    assert run.scale_factor[-1] > run.scale_factor[0]


def test_starting_past_the_bounce_is_refused():
    with pytest.raises(ValueError, match="past this theory's bounce"):
        evolve(theory=EffectiveLQC(), density=2.0 * RHO_CRITICAL_PLANCK)


def test_the_duration_is_taken_from_the_physics_not_picked():
    """Regression: a fixed duration is the trap here.

    At a starting density of 1e-6 the Hubble time is 345, so a
    plausible-looking duration of 1 moves the scale factor by three parts in
    a thousand and finds no bounce at all. The default is ``2.5/|H|``, which
    clears the collapse time for every equation of state.
    """
    default = evolve(theory=EffectiveLQC(), equation_of_state=0.0, density=1e-6)
    assert default.bounced
    assert default.time[-1] > 100.0

    too_short = evolve(theory=EffectiveLQC(), equation_of_state=0.0, density=1e-6, duration=1.0)
    assert not too_short.bounced
    assert too_short.minimum_scale_factor > 0.99


def test_the_sampled_maximum_understates_a_sharp_peak():
    """Why ``bounce_density`` exists alongside ``maximum_density``.

    The density peaks sharply at the turning point, and an output grid coarse
    enough to span the whole run misses it by tens of per cent. Quoting the
    sampled maximum as the bounce density would be a resolution artefact
    reported as physics.
    """
    coarse = evolve(theory=EffectiveLQC(), equation_of_state=1.0, density=1e-6, points=201)
    assert coarse.bounced
    assert coarse.bounce_density == pytest.approx(RHO_CRITICAL_PLANCK, rel=1e-8)
    assert coarse.maximum_density < 0.9 * RHO_CRITICAL_PLANCK

    fine = evolve(theory=EffectiveLQC(), equation_of_state=1.0, density=1e-6, points=200_001)
    assert fine.maximum_density == pytest.approx(RHO_CRITICAL_PLANCK, rel=1e-3)


def test_the_summary_reports_the_outcome():
    summary = evolve(theory=EffectiveLQC(), equation_of_state=0.0, density=1e-6).summary()
    assert summary["bounced"] is True
    assert summary["reached_singularity"] is False
    assert summary["bounce_density"] == pytest.approx(RHO_CRITICAL_PLANCK, rel=1e-8)
    assert summary["constraint_drift"] < 1e-8


@pytest.mark.parametrize("equation_of_state", [0.0, 0.2, 1.0 / 3.0, 0.5, 1.0])
def test_a_general_relativistic_collapse_ends_cleanly_at_every_equation_of_state(
    equation_of_state,
):
    """A crunch is an outcome the run carries, not an integrator failure.

    The scale-factor floor cannot do this alone, and the reason is
    arithmetic. For radiation ``rho ~ a^-4``, so ``a = 1e-8`` means
    ``|H| ~ 1e13`` and a dynamical time of 1e-13; at a cosmic time of order
    a hundred that step is below the spacing between neighbouring doubles,
    and the integrator fails with a step-size message before the event can
    fire. Only ``w = 0`` survived it. The ceiling on ``|H|`` is reached
    first and stops the run where the classical description has run out
    anyway.
    """
    run = evolve(theory=None, equation_of_state=equation_of_state, density=1e-6)
    assert run.reached_singularity
    assert not run.bounced
    assert run.constraint_drift < 1e-8


@pytest.mark.parametrize("equation_of_state", [0.0, 1.0 / 3.0, 1.0])
def test_the_curvature_ceiling_cannot_cut_a_bounce_short(equation_of_state):
    """``|H|`` peaks at 0.93 on the loop-quantum branch, three orders below the ceiling.

    ``H^2 = (8 pi/3) rho (1 - rho/rho_c)`` is largest at ``rho = rho_c/2``,
    which is 0.93 at ``rho_c = 0.41``. Asserting that here is what keeps the
    ceiling honest: if someone lowers it to a value a bounce can reach, the
    bounce would be reported as a crunch and this test says so.
    """
    run = evolve(theory=EffectiveLQC(), equation_of_state=equation_of_state, density=1e-6)
    assert run.bounced
    assert not run.reached_singularity
    assert float(np.abs(run.hubble).max()) < 1.0
    assert run.bounce_density == pytest.approx(RHO_CRITICAL_PLANCK, rel=1e-6)
