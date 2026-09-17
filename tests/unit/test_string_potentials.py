"""String-inspired potentials against the predictions their sources quote.

Three kinds of check, and the third is the one that matters:

* the derivatives, against central differences, as for every potential;
* the closed forms this module derives -- the plateau family's universal
  tilt, fibre inflation's ``eps = (3/2) eta^2``, the D-brane model's
  ``eta = -5/(6N)`` -- which are algebra and hold to machine precision or
  to a stated order;
* the numbers the source papers quote, through the full Mukhanov-Sasaki
  pipeline. That is what the milestone asks for, and it is the only one of
  the three that could fail because of something outside this file.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from particlesim.cosmo.inflation import (
    InflationNeverEnds,
    efolds,
    end_of_inflation,
    field_at_efolds,
    run_to_end,
    slow_roll,
    slow_roll_prediction,
)
from particlesim.cosmo.perturbations import power_spectrum
from particlesim.cosmo.potentials import Potential, PowerLaw, Starobinsky
from particlesim.cosmo.string_potentials import (
    FIBRE_EXPONENT,
    AxionMonodromy,
    DBraneInflation,
    FibreInflation,
    KahlerModuli,
    PlateauInflation,
)
from particlesim.theories.registry import get_inflaton_potential, list_inflaton_potentials

STRING_POTENTIALS = [
    (PlateauInflation(coefficient=2.0, decay_constant=1.2247), 6.0),
    (KahlerModuli(), 3.5),
    (FibreInflation(), 6.0),
    (AxionMonodromy(exponent=1.0), 15.0),
    (AxionMonodromy(exponent=2.0 / 3.0), 12.0),
    (AxionMonodromy(exponent=1.0, modulation=1e-14, decay_constant=0.1), 15.0),
    (DBraneInflation(scale=1.0), 4.0),
    (DBraneInflation(scale=0.1), 0.6),
]


def _derivative(f, x, step=1e-5):
    return (f(x + step) - f(x - step)) / (2.0 * step)


def _compare(analytic, numerical, tolerance=1e-6):
    if abs(numerical) < 1e-30:
        assert abs(analytic) < 1e-30
    else:
        assert analytic == pytest.approx(numerical, rel=tolerance)


@pytest.mark.parametrize(("potential", "phi"), STRING_POTENTIALS)
def test_derivatives_match_differences(potential, phi):
    _compare(float(potential.gradient(phi)), float(_derivative(potential.value, phi)))
    _compare(float(potential.curvature(phi)), float(_derivative(potential.gradient, phi)))
    _compare(
        float(potential.third_derivative(phi)),
        float(_derivative(potential.curvature, phi, step=1e-4)),
        tolerance=1e-5,
    )


@pytest.mark.parametrize(("potential", "phi"), STRING_POTENTIALS)
def test_every_string_potential_declares_where_it_comes_from(potential, phi):
    """Provenance and the truncation used, which the milestone asks for by name."""
    assert len(potential.provenance) > 40
    assert len(potential.truncation) > 40
    assert potential.id.startswith("string.cosmo.")


def test_the_registry_finds_every_potential_under_a_unique_id():
    found = list_inflaton_potentials()
    assert len(found) == len(set(found))
    assert {
        "inflation.quadratic",
        "inflation.starobinsky",
        "string.cosmo.fibre",
        "string.cosmo.kahler",
        "string.cosmo.monodromy",
        "string.cosmo.dbrane",
        "string.cosmo.plateau",
    } <= set(found)
    for identifier, cls in found.items():
        assert issubclass(cls, Potential)
        assert cls.id == identifier
        assert cls.provenance


def test_the_registry_instantiates_with_parameters():
    potential = get_inflaton_potential("string.cosmo.fibre", amplitude=2e-10)
    assert isinstance(potential, FibreInflation)
    assert potential.amplitude == 2e-10
    with pytest.raises(KeyError, match="unknown inflaton potential"):
        get_inflaton_potential("string.cosmo.nonexistent")


# --- the plateau family -------------------------------------------------


@pytest.mark.parametrize("coefficient", [1.0, 2.0, 4.0])
@pytest.mark.parametrize("decay_constant", [0.8, 1.2247, 1.7321])
@pytest.mark.parametrize("exponent", [1.0, 4.0 / 3.0, 2.0])
def test_the_plateau_tilt_is_universal(coefficient, decay_constant, exponent):
    """``n_s = 1 - 2/N`` for every ``(C, f, p)``, and the gap shrinks with ``N``.

    Derived in the module docstring: ``eta N = -1`` follows from the
    exponential shape alone, with the parameters cancelling. This is the
    "robust, model-independent" tilt Kaehler moduli inflation is quoted
    for, and the point of sweeping twenty-seven parameter sets is that it
    is a property of the family rather than of a tuning.
    """
    potential = PlateauInflation(
        coefficient=coefficient, decay_constant=decay_constant, exponent=exponent
    )
    near = slow_roll_prediction(potential, 55.0).spectral_index - (1.0 - 2.0 / 55.0)
    far = slow_roll_prediction(potential, 200.0).spectral_index - (1.0 - 2.0 / 200.0)
    assert abs(near) < 6e-3
    assert abs(far) < 1.5e-3


@pytest.mark.parametrize("decay_constant", [0.8, 1.2247, 1.7321])
def test_the_plateau_tilt_gap_shrinks_with_the_e_fold_count(decay_constant):
    """For ``p = 1`` the subleading term falls by an order of magnitude over 55 to 200.

    Asserted separately from the sweep above because it does *not* hold
    for every exponent: at ``p = 4/3`` the gap carries a ``u^(2/p - 2)``
    factor that is itself ``N``-dependent, and for some parameter sets it
    passes through zero near fifty-five e-folds and grows again. The
    universality claim is that the gap is small, not that it is monotonic,
    and conflating the two would have made this file assert something
    false.
    """
    potential = PlateauInflation(coefficient=1.0, decay_constant=decay_constant, exponent=1.0)
    near = slow_roll_prediction(potential, 55.0).spectral_index - (1.0 - 2.0 / 55.0)
    far = slow_roll_prediction(potential, 200.0).spectral_index - (1.0 - 2.0 / 200.0)
    assert near > 0.0 and far > 0.0
    assert 5.0 < near / far < 15.0


@pytest.mark.parametrize("decay_constant", [0.5, 1.2247, 1.7321, 5.0])
@pytest.mark.parametrize("exponent_value", [5.0, 10.0, 20.0])
def test_the_plateau_pins_epsilon_to_eta_squared(decay_constant, exponent_value):
    """``eps = (f^2/2) eta^2`` exactly for ``p = 1``, at any field value.

    Not a slow-roll limit: for a single exponential both parameters carry
    the same ``C e^-u/(1 - C e^-u)``, so the ratio is pure ``f``. At
    ``f^2 = 3`` this is fibre inflation's published ``eps = (3/2) eta^2``
    and at ``f^2 = 3/2`` it is Starobinsky's ``3/4``.
    """
    potential = PlateauInflation(coefficient=1.0, decay_constant=decay_constant, exponent=1.0)
    parameters = slow_roll(potential, decay_constant * exponent_value)
    assert parameters.epsilon / parameters.eta**2 == pytest.approx(
        0.5 * decay_constant**2, rel=1e-12
    )


def test_the_plateau_coefficient_is_a_field_shift_at_unit_exponent():
    """``C`` is pure gauge for ``p = 1``: ``phi -> phi + f ln C`` absorbs it.

    So the observables must be *identical*, not merely close, and a
    prediction that moved with ``C`` would mean the e-fold integral had
    picked up the shift somewhere it should have cancelled.
    """
    reference = slow_roll_prediction(PlateauInflation(coefficient=1.0, decay_constant=1.5), 55.0)
    for coefficient in (0.3, 2.0, 17.0):
        shifted = PlateauInflation(coefficient=coefficient, decay_constant=1.5)
        if coefficient < 1.0:
            # No minimum exists, so there is no end of inflation to count from.
            with pytest.raises(InflationNeverEnds):
                slow_roll_prediction(shifted, 55.0)
            continue
        other = slow_roll_prediction(shifted, 55.0)
        assert other.spectral_index == pytest.approx(reference.spectral_index, rel=1e-10)
        assert other.tensor_to_scalar == pytest.approx(reference.tensor_to_scalar, rel=1e-10)
        assert other.field_value - reference.field_value == pytest.approx(
            1.5 * math.log(coefficient), rel=1e-8
        )


def test_a_plateau_without_a_minimum_has_no_end_of_inflation():
    """``C < 1`` leaves a monotonic ramp from ``A(1-C)``, not a potential well.

    A real model has further terms that produce the minimum; the
    truncation here does not, and it says so instead of returning the edge
    of a bracket.
    """
    with pytest.raises(InflationNeverEnds, match="no end of inflation"):
        end_of_inflation(PlateauInflation(coefficient=0.5))


def test_the_plateau_reproduces_starobinsky_at_its_own_parameters():
    """Starobinsky is the ``C = 2``, ``f = sqrt(3/2)``, ``p = 1`` member.

    Its potential is ``(1 - e^(-phi/f))^2 = 1 - 2 e^(-phi/f) + e^(-2phi/f)``,
    so the plateau family is its leading behaviour and the difference is
    the dropped square -- a few parts in a thousand of ``n_s`` at
    fifty-five e-folds, and seven per cent of ``r``. Both are asserted, and
    from below: the family is an approximation to Starobinsky and should
    not agree with it exactly.
    """
    plateau = slow_roll_prediction(
        PlateauInflation(coefficient=2.0, decay_constant=math.sqrt(1.5)), 55.0
    )
    exact = slow_roll_prediction(Starobinsky(), 55.0)
    assert 1e-4 < abs(plateau.spectral_index - exact.spectral_index) < 3e-3
    assert 0.02 < abs(plateau.tensor_to_scalar / exact.tensor_to_scalar - 1.0) < 0.15


@pytest.mark.benchmark
@pytest.mark.parametrize("number", [55.0, 60.0])
def test_kahler_moduli_inflation_lands_in_the_published_tilt_range(number):
    """Conlon and Quevedo quote ``n_s = 1 - 2/N_e = 0.960`` to 0.967 at ``N_e = 50-60``.

    The subleading term here is 2e-3 *below* the leading-order value, where
    Starobinsky's is above it, so this is not the same test as the
    Starobinsky gate wearing a different name.
    """
    prediction = slow_roll_prediction(KahlerModuli(), number)
    assert prediction.spectral_index == pytest.approx(1.0 - 2.0 / number, abs=3e-3)
    assert 0.957 < prediction.spectral_index < 0.967


@pytest.mark.benchmark
def test_kahler_moduli_inflation_has_an_unobservable_tensor_ratio():
    """The signature of a blow-up modulus: ``p = 4/3`` suppresses ``r`` by ``1/sqrt(u)``.

    Three orders below fibre inflation at the same tilt, which is the
    whole reason the tensor ratio distinguishes plateau models that the
    tilt cannot.
    """
    blow_up = slow_roll_prediction(KahlerModuli(), 55.0)
    fibre = slow_roll_prediction(FibreInflation(), 55.0)
    assert blow_up.tensor_to_scalar < 1e-3
    assert fibre.tensor_to_scalar / blow_up.tensor_to_scalar > 5.0


# --- fibre inflation ----------------------------------------------------


@pytest.mark.benchmark
def test_fibre_inflation_pins_epsilon_to_three_halves_eta_squared():
    """The source paper's ``eps ~= (3/2) eta^2``, reproduced to 2e-4.

    The quartic term is what makes this approximate rather than exact; it
    is 2e-6 against 3 at the pivot, and the residual here is its size.
    """
    parameters = slow_roll_prediction(FibreInflation(), 55.0)
    assert parameters.epsilon / parameters.eta**2 == pytest.approx(1.5, rel=1e-3)


@pytest.mark.benchmark
@pytest.mark.parametrize("number", [50.0, 55.0, 57.0])
def test_fibre_inflation_tensor_ratio_is_in_the_published_range(number):
    """``r ~= 0.005`` to 0.01, the range the source paper quotes for the model."""
    prediction = slow_roll_prediction(FibreInflation(), number)
    assert 0.005 <= prediction.tensor_to_scalar <= 0.01


@pytest.mark.benchmark
def test_fibre_inflation_relates_the_tensor_ratio_to_the_tilt():
    """``r ~= 6 (n_s - 1)^2``, to the 13 per cent the relation itself costs.

    Exactly, the plateau gives ``r = 24 eta^2``; the published form
    replaces ``2 eta`` with ``n_s - 1``, which drops the ``-6 eps`` term.
    That is a 13 per cent effect at this tilt and it is asserted as such
    rather than hidden in a loose tolerance -- the ratio has to be *below*
    one, and by about the right amount.
    """
    prediction = slow_roll_prediction(FibreInflation(), 55.0)
    quoted = 6.0 * (prediction.spectral_index - 1.0) ** 2
    assert 0.85 < prediction.tensor_to_scalar / quoted < 0.90
    assert prediction.tensor_to_scalar == pytest.approx(24.0 * prediction.eta**2, rel=1e-3)


def test_fibre_inflation_has_a_quadratic_minimum_at_the_origin():
    """``V -> 2 A phi^2``, which is where inflation ends and why it can."""
    potential = FibreInflation(amplitude=1.0)
    assert float(potential.value(0.0)) == pytest.approx(0.0, abs=1e-15)
    for phi in (1e-3, 1e-2):
        assert float(potential.value(phi)) == pytest.approx(2.0 * phi**2, rel=1e-2)
    assert float(potential.value(50.0)) == pytest.approx(3.0, rel=1e-10)
    assert end_of_inflation(potential) == pytest.approx(0.9176, abs=1e-3)


@pytest.mark.benchmark
def test_fibre_inflation_mode_spectrum_matches_its_slow_roll_prediction():
    run = run_to_end(FibreInflation(), 55.0, margin=8.0)
    spectrum = power_spectrum(run, efolds_remaining=55.0)
    prediction = slow_roll_prediction(FibreInflation(), 55.0)
    assert spectrum.spectral_index == pytest.approx(prediction.spectral_index, abs=1e-3)
    assert 0.005 <= spectrum.tensor_to_scalar <= 0.01
    assert spectrum.drift < 1e-4


# --- axion monodromy ----------------------------------------------------


@pytest.mark.parametrize("exponent", [2.0 / 3.0, 1.0, 3.0 / 2.0])
def test_unmodulated_monodromy_is_the_power_law_family(exponent):
    monodromy = AxionMonodromy(amplitude=1e-10, exponent=exponent)
    monomial = PowerLaw(amplitude=1e-10, exponent=exponent)
    for phi in (5.0, 12.0):
        assert float(monodromy.value(phi)) == pytest.approx(float(monomial.value(phi)), rel=1e-14)
        assert float(monodromy.gradient(phi)) == pytest.approx(
            float(monomial.gradient(phi)), rel=1e-14
        )
    prediction = slow_roll_prediction(monodromy, 60.0)
    denominator = 240.0 + exponent
    assert prediction.spectral_index == pytest.approx(
        1.0 - (2.0 * exponent + 4.0) / denominator, rel=1e-10
    )
    assert prediction.tensor_to_scalar == pytest.approx(16.0 * exponent / denominator, rel=1e-10)


@pytest.mark.benchmark
def test_linear_monodromy_reproduces_the_published_tensor_ratio():
    """McAllister, Silverstein and Westphal quote ``r ~= 0.07`` for the linear model.

    Through the mode solver at sixty e-folds this is 0.0664, which is the
    quoted value to the one significant figure it is quoted in. The tilt
    that comes with it, 0.9749, is also reported: the model is predictive
    in both and in tension with the measured tilt, which is a fact about
    the model and not about this pipeline.
    """
    run = run_to_end(AxionMonodromy(exponent=1.0), 60.0, margin=8.0)
    spectrum = power_spectrum(run, efolds_remaining=60.0)
    assert spectrum.tensor_to_scalar == pytest.approx(0.07, rel=0.06)
    assert spectrum.tensor_to_scalar == pytest.approx(0.0664, rel=5e-3)
    assert spectrum.spectral_index == pytest.approx(0.9749, abs=1e-3)


def test_the_instanton_modulation_makes_the_tilt_oscillate():
    """The one thing that distinguishes monodromy from a monomial.

    A monomial's tilt is a smooth function of the pivot. With the
    instanton term the tilt oscillates with the field, period ``2 pi f``,
    because ``V''`` picks up a harmonic that is comparable to the monomial's
    curvature. The amplitude scales with the modulation and vanishes with
    it.
    """
    smooth = AxionMonodromy(exponent=1.0)
    pivots = np.linspace(55.0, 58.0, 25)
    plain = np.array([slow_roll_prediction(smooth, n).spectral_index for n in pivots])
    assert np.all(np.diff(plain) > 0.0)

    previous = 0.0
    for modulation in (1e-16, 1e-15, 1e-14):
        wiggly = AxionMonodromy(exponent=1.0, modulation=modulation, decay_constant=0.1)
        tilts = np.array([slow_roll_prediction(wiggly, n).spectral_index for n in pivots])
        swing = float(np.ptp(tilts - plain))
        assert swing > 3.0 * previous
        previous = swing
    assert previous > 1e-3


def test_monodromy_rejects_bad_parameters():
    with pytest.raises(ValueError, match="exponent must be positive"):
        AxionMonodromy(exponent=0.0)
    with pytest.raises(ValueError, match="decay_constant must be positive"):
        AxionMonodromy(decay_constant=0.0)


# --- D-brane inflation --------------------------------------------------


@pytest.mark.parametrize("scale", [1.0, 0.1, 0.01])
def test_d_brane_inflation_e_fold_count_matches_the_closed_form(scale):
    """``N = phi^6/(24 mu^4)``, approached as ``mu`` shrinks.

    The correction is the end-of-inflation term ``phi_end^6``, which is
    ``O(mu^6)`` relative and so vanishes with the scale; at ``mu = 0.01``
    the closed form is right to a part in ten thousand.
    """
    potential = DBraneInflation(scale=scale)
    phi = field_at_efolds(potential, 55.0)
    assert efolds(potential, phi) == pytest.approx(55.0, rel=1e-10)
    assert phi**6 / (24.0 * scale**4) == pytest.approx(55.0, rel=0.03 * scale**0.5 + 1e-4)


@pytest.mark.benchmark
@pytest.mark.parametrize("scale", [1.0, 0.1, 0.01])
def test_d_brane_inflation_tilt_is_robust(scale):
    """``n_s = 1 - 5/(3N)``, independent of the brane separation scale.

    ``eta = -20 mu^4/phi^6 = -5/(6N)`` once ``N = phi^6/(24 mu^4)`` is used,
    and the ``mu`` cancels. Three scales spanning four orders of magnitude
    in ``r`` give the same tilt, which is the content of the KKLMMT
    prediction.
    """
    prediction = slow_roll_prediction(DBraneInflation(scale=scale), 60.0)
    assert prediction.eta == pytest.approx(-5.0 / 360.0, rel=0.02)
    assert prediction.spectral_index == pytest.approx(1.0 - 5.0 / 180.0, abs=3e-4)


@pytest.mark.benchmark
def test_d_brane_inflation_tensor_ratio_is_not_a_prediction():
    """``r`` carries ``mu^(4/3)`` and spans decades where the tilt does not.

    This is the honest content of the model and the reason it sits next to
    fibre inflation, where the opposite is true: there ``r`` is pinned to
    the tilt, here it is a free parameter.
    """
    ratios = [slow_roll_prediction(DBraneInflation(scale=s), 60.0) for s in (1.0, 0.1, 0.01)]
    tilts = [r.spectral_index for r in ratios]
    tensors = [r.tensor_to_scalar for r in ratios]
    assert max(tilts) - min(tilts) < 3e-4
    assert tensors[0] / tensors[2] > 100.0
    # mu^(4/3) scaling: two decades of mu is 8/3 decades of r.
    assert math.log10(tensors[0] / tensors[1]) == pytest.approx(4.0 / 3.0, rel=0.02)


@pytest.mark.benchmark
def test_d_brane_inflation_mode_spectrum_matches_the_closed_form_tilt():
    run = run_to_end(DBraneInflation(scale=0.1), 60.0, margin=8.0)
    spectrum = power_spectrum(run, efolds_remaining=60.0)
    assert spectrum.spectral_index == pytest.approx(1.0 - 5.0 / 180.0, abs=3e-4)
    assert spectrum.tensor_to_scalar < 1e-4


def test_d_brane_potential_is_negative_below_the_separation_scale():
    potential = DBraneInflation(amplitude=1.0, scale=1.0)
    assert potential.domain == (1.0, math.inf)
    assert not bool(potential.contains(0.5))
    assert float(potential.value(0.5)) < 0.0
    assert float(potential.value(1e3)) == pytest.approx(1.0, rel=1e-10)
    with pytest.raises(ValueError, match="scale must be positive"):
        DBraneInflation(scale=0.0)


def test_the_fibre_exponent_is_one_over_root_three():
    assert FIBRE_EXPONENT == pytest.approx(1.0 / math.sqrt(3.0), rel=1e-15)
