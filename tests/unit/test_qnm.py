"""Quasinormal modes: the frequencies, and reading one off a time series.

Issue #50's acceptance criterion is the Schwarzschild fundamental to 1%.
That is met by a wide margin -- every published digit agrees -- so the tests
here are mostly about the things that would let a wrong answer look right:
agreement with a table is agreement with whoever typed the table, so the
load-bearing check is the eikonal limit, which is analytic and derived from
the light ring rather than looked up.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.analysis import qnm

#: Published values, in ``M omega``. Leaver (1985) for the gravitational and
#: scalar modes; the electromagnetic ones are the same calculation at s=1.
#: Quoted to six decimals, which is what limits the comparison below.
PUBLISHED = {
    (2, 2, 0): 0.373672 - 0.088962j,
    (2, 2, 1): 0.346711 - 0.273915j,
    (2, 2, 3): 0.251505 - 0.705148j,
    (3, 2, 0): 0.599443 - 0.092703j,
    (3, 2, 1): 0.582644 - 0.281298j,
    (4, 2, 0): 0.809178 - 0.094163j,
    (0, 0, 0): 0.110455 - 0.104896j,
    (1, 0, 0): 0.292936 - 0.097660j,
    (2, 0, 0): 0.483644 - 0.096759j,
    (1, 1, 0): 0.248263 - 0.092488j,
    (2, 1, 0): 0.457596 - 0.095004j,
}


# --- the acceptance criterion -------------------------------------------


@pytest.mark.benchmark
def test_the_schwarzschild_fundamental_to_one_percent():
    """Issue #50's acceptance criterion, met to six digits rather than two.

    ``M omega = 0.373672 - 0.088962i``. What comes out is
    ``0.373671684418 - 0.088962315689i``, so the 1% asked for is cleared by
    four orders of magnitude and the residual difference is the rounding in
    the published value.
    """
    found = qnm.quasinormal_frequency(l=2, overtone=0, spin=2)
    expected = PUBLISHED[(2, 2, 0)]
    assert abs(found - expected) / abs(expected) < 0.01
    # and, for the record, far better than that
    assert abs(found - expected) / abs(expected) < 5e-6
    assert found.imag < 0.0, "a quasinormal mode decays in the e^(-i omega t) convention"


@pytest.mark.parametrize(("l", "spin", "overtone"), sorted(PUBLISHED))
def test_every_published_digit_agrees(l, spin, overtone):
    """Across three spins and four multipoles, from the eikonal guess alone.

    Overtones above the first are reached by continuation, because the
    eikonal guess for ``n = 2`` lands in the ``n = 1`` basin -- which is the
    reason :func:`quasinormal_spectrum` exists.
    """
    expected = PUBLISHED[(l, spin, overtone)]
    if overtone == 0:
        found = qnm.quasinormal_frequency(l=l, overtone=0, spin=spin)
    else:
        found = qnm.quasinormal_spectrum(l=l, overtones=overtone + 1, spin=spin)[overtone]
    assert abs(found - expected) / abs(expected) < 3e-6, (found, expected)


# --- the check that does not come from a table ---------------------------


@pytest.mark.benchmark
def test_the_eikonal_limit_is_approached_as_one_over_l_squared():
    """The independent check: the light ring, not a published number.

    ``M omega -> ((l + 1/2) - i(n + 1/2))/(3 sqrt 3)`` is the orbital
    frequency and Lyapunov exponent of the photon sphere at ``r = 3M``, both
    of which come straight out of the null geodesic equation. Measured, the
    relative gap closes at second order:

        l          2       4       8      16      32
        gap     2.2e-1  6.5e-2  1.8e-2  4.8e-3  1.2e-3
        gap*l^2   0.88    1.04    1.16    1.22    1.26

    Second order rather than first, and the coefficient settles near 1.3.
    Nothing in this comparison was looked up.
    """
    gaps = {}
    for l in (4, 8, 16, 32):
        found = qnm.quasinormal_frequency(l=l, overtone=0, spin=2)
        limit = qnm.eikonal_frequency(l, 0)
        gaps[l] = abs(found - limit) / abs(limit)

    scaled = [gaps[l] * l**2 for l in sorted(gaps)]
    # Bounded above and below: a first-order approach would have these
    # growing like l, and a third-order one shrinking like 1/l.
    assert all(0.8 < value < 1.6 for value in scaled), scaled
    assert scaled[-1] > scaled[0], scaled
    assert gaps[32] < gaps[4] / 30.0, gaps


@pytest.mark.benchmark
def test_the_damping_rate_reaches_the_lyapunov_exponent_exactly():
    """``Im(M omega) -> -1/(6 sqrt 3) = -0.09622504486`` at large ``l``.

    Half the light ring's Lyapunov exponent, and a cleaner target than the
    real part because it has no ``l`` in it at all: every fundamental mode
    should approach the same number.
    """
    target = -0.5 * qnm.LIGHT_RING
    assert target == pytest.approx(-0.0962250448649, abs=1e-12)
    for l, tolerance in ((8, 1e-3), (32, 1e-4), (128, 1e-5)):
        found = qnm.quasinormal_frequency(l=l, overtone=0, spin=2)
        assert found.imag == pytest.approx(target, abs=tolerance), (l, found)


def test_the_continued_fraction_vanishes_at_a_published_frequency():
    """Straight evaluation, no root-finding: is the condition actually zero there?

    In units ``2M = 1``, so the published ``M omega`` doubles. The residual
    is around 1e-5, which is what six published decimals buys -- not a
    limitation of the recursion.
    """
    for (l, spin, overtone), value in PUBLISHED.items():
        if overtone != 0:
            continue
        residual = abs(qnm.continued_fraction(2 * value, l=l, spin=spin, depth=600))
        assert residual < 1e-4, (l, spin, residual)


def test_the_continued_fraction_converges_in_depth_and_then_stops_changing():
    """Ten digits at depth 50, and bit-identical from 400 upwards.

    The minimal solution's terms fall off geometrically, so the fundamental
    is cheap: depth 50 already agrees with the converged value to 5e-11,
    depth 100 to 6e-15, and 400 and 1600 give the same floating-point
    number. High overtones are what need the depth, which is why
    :func:`quasinormal_spectrum` asks for more of it.
    """
    values = {
        depth: qnm.quasinormal_frequency(l=2, overtone=0, spin=2, depth=depth)
        for depth in (50, 100, 400, 1600)
    }
    assert values[400] == values[1600]
    assert values[100] == pytest.approx(values[1600], abs=1e-13)
    assert values[50] == pytest.approx(values[1600], abs=1e-9)
    assert values[50] != values[1600], "depth 50 is converged but not exactly"


# --- the spectrum and its refusals ---------------------------------------


def test_the_overtone_ladder_is_monotonically_more_damped():
    """Each overtone decays faster than the last, which is how a slip is caught.

    Continuation can land back on a mode already found; the spectrum
    function checks for that rather than returning a duplicate.
    """
    spectrum = qnm.quasinormal_spectrum(l=2, overtones=5, spin=2)
    damping = [value.imag for value in spectrum]
    assert damping == sorted(damping, reverse=True)
    assert all(value.real > 0 for value in spectrum)
    # and the spacing is roughly the eikonal one, which is what makes the
    # continuation guess work at all
    steps = np.diff(damping)
    assert all(-0.30 < step < -0.12 for step in steps), steps


def test_a_multipole_below_the_spin_weight_is_refused():
    """There is no ``l = 1`` gravitational perturbation to have a mode."""
    with pytest.raises(ValueError, match="below the spin weight"):
        qnm.quasinormal_frequency(l=1, spin=2)
    with pytest.raises(ValueError, match="below the spin weight"):
        qnm.quasinormal_frequency(l=0, spin=1)


def test_a_depth_inside_the_inversion_is_refused():
    """Truncating at the term whose root is sought is not an approximation."""
    with pytest.raises(ValueError, match="no tail beyond inversion"):
        qnm.continued_fraction(0.7 - 0.17j, depth=5, inversion=5)


def test_a_wild_guess_still_finds_the_fundamental():
    """Worth recording: the basin is enormous.

    ``500 + 500i`` is a thousand times the answer and Newton on the
    continued fraction still walks back to ``l=2, n=0``. That is why the
    eikonal guess never needed to be good, and why the residual guard below
    has to be tested by other means -- a bad guess does not trigger it.
    """
    found = qnm.quasinormal_frequency(l=2, overtone=0, spin=2, guess=500.0 + 500.0j)
    assert found == pytest.approx(PUBLISHED[(2, 2, 0)], rel=3e-6)


def test_a_converged_non_root_is_refused(monkeypatch):
    """The guard against returning something that is not a root.

    Newton reports convergence on its own step size, which can stall
    somewhere the function is not zero. Rather than trust it, the frequency
    is checked against the continued fraction before being returned. Forcing
    the root-finder to hand back a non-root exercises that check.
    """
    monkeypatch.setattr(qnm, "newton", lambda *args, **kwargs: 3.0 + 3.0j)
    with pytest.raises(RuntimeError, match="is not a root"):
        qnm.quasinormal_frequency(l=2, overtone=0, spin=2)


def test_the_quality_factor_is_a_couple_of_cycles():
    """2.10 for the fundamental, which is why a second overtone is hard to see."""
    assert qnm.quality_factor(qnm.quasinormal_frequency()) == pytest.approx(2.100, abs=0.01)


# --- reading a frequency off a time series -------------------------------


def _ringdown(times, *modes):
    """``sum a_j exp(-i omega_j t)`` for given (amplitude, frequency) pairs."""
    return sum(amplitude * np.exp(-1j * frequency * times) for amplitude, frequency in modes)


def test_the_fit_recovers_the_frequencies_it_was_built_from():
    """The loop closing: synthesise from Leaver, extract, compare.

    Two modes, clean, complex. Recovered to 4e-14 relative, and the
    amplitudes come back as the 1.0 and 0.4i they were built with -- which
    is the check that the least-squares stage is solving for the right
    design matrix and not just fitting something.
    """
    first, second = qnm.quasinormal_spectrum(l=2, overtones=2, spin=2)
    times = np.linspace(0.0, 80.0, 1200)
    signal = _ringdown(times, (1.0, first), (0.4j, second))

    fit = qnm.ringdown_fit(times, signal, modes=2)
    assert fit.residual < 1e-12
    assert fit.frequencies[0] == pytest.approx(first, rel=1e-9)
    assert fit.frequencies[1] == pytest.approx(second, rel=1e-9)
    assert fit.amplitudes[0] == pytest.approx(1.0, abs=1e-9)
    assert fit.amplitudes[1] == pytest.approx(0.4j, abs=1e-9)


def test_a_real_signal_needs_twice_the_modes():
    """One real damped sinusoid is *two* complex exponentials.

    ``omega`` and ``-conj(omega)``: same damping, opposite sense of
    rotation. So one physical mode in real data wants ``modes=2``, which
    fits it to 9e-15, and ``modes=1`` cannot and does not -- it has to
    average the conjugate pair and lands near the real axis.

    The conjugate pair is also why the bug this method was written with hid
    for as long as it did. The right singular vectors span the Hankel row
    space as the rows of ``Vh``; conjugating them spans the Vandermonde
    vectors of ``conj(z)`` and returns ``-conj(omega)``. That is *already*
    in a real signal's spectrum, so the real case looked perfect while the
    complex case above came back with a flipped real part.
    """
    frequency = qnm.quasinormal_frequency()
    times = np.linspace(0.0, 60.0, 900)
    signal = np.real(_ringdown(times, (1.0, frequency)))

    poor = qnm.ringdown_fit(times, signal, modes=1)
    assert poor.residual > 1e-2

    exact = qnm.ringdown_fit(times, signal, modes=2)
    assert exact.residual < 1e-12
    recovered = qnm.dominant_frequency(times, signal, modes=2)
    assert recovered == pytest.approx(frequency, rel=1e-9)
    # the conjugate image is there too, with the same damping
    mirrored = [value for value in exact.frequencies if value.real < 0]
    assert len(mirrored) == 1
    assert mirrored[0] == pytest.approx(-np.conj(frequency), rel=1e-9)
    # and the amplitude splits evenly between the two
    assert exact.amplitudes[0] == pytest.approx(0.5, abs=1e-9)


def test_the_fit_degrades_gracefully_with_noise():
    """Roughly linearly in the noise level, and the fundamental beats the overtone.

    Measured relative error on the fundamental: 3e-9, 5e-7, 1.4e-6, 3e-4,
    1.4e-3 for noise from 1e-8 to 1e-2. The overtone is consistently an
    order worse, because it has decayed away over most of the window.
    """
    first, second = qnm.quasinormal_spectrum(l=2, overtones=2, spin=2)
    times = np.linspace(0.0, 80.0, 1200)
    clean = _ringdown(times, (1.0, first), (0.4j, second))
    generator = np.random.default_rng(20260918)

    previous = 0.0
    for level in (1e-6, 1e-4, 1e-2):
        noise = generator.standard_normal(times.size) + 1j * generator.standard_normal(times.size)
        fit = qnm.ringdown_fit(times, clean + level * noise, modes=2)
        error = abs(fit.frequencies[0] - first) / abs(first)
        assert error < 200 * level, (level, error)
        assert error > previous / 10.0
        previous = error


def test_one_mode_fitted_late_converges_on_the_fundamental():
    """The physical statement: the overtone dies, so a late window is cleaner.

    Fitting a single mode to a two-mode signal is wrong by however much the
    overtone still contributes. Measured relative error 4.7e-2, 7.5e-3,
    1.2e-3, 3.2e-5 for windows starting at ``t = 0, 10, 20, 40``. That is
    the ringdown-fitting problem in one line, and the reason a real
    extraction has to choose a start time.
    """
    first, second = qnm.quasinormal_spectrum(l=2, overtones=2, spin=2)
    times = np.linspace(0.0, 80.0, 1200)
    signal = _ringdown(times, (1.0, first), (0.4j, second))

    errors = []
    for start in (0.0, 20.0, 40.0):
        window = times >= start
        fit = qnm.ringdown_fit(times[window], signal[window], modes=1)
        errors.append(abs(fit.frequencies[0] - first) / abs(first))
    assert errors == sorted(errors, reverse=True), errors
    assert errors[-1] < 1e-4, errors
    assert errors[0] > 1e-2, errors


def test_the_damping_time_is_the_reciprocal_of_the_imaginary_part():
    frequency = qnm.quasinormal_frequency()
    times = np.linspace(0.0, 60.0, 600)
    fit = qnm.ringdown_fit(times, _ringdown(times, (1.0, frequency)), modes=1)
    assert fit.damping_times[0] == pytest.approx(-1.0 / frequency.imag, rel=1e-8)
    assert fit.evaluate(times) == pytest.approx(_ringdown(times, (1.0, frequency)), abs=1e-10)


def test_non_uniform_samples_are_refused():
    """The matrix pencil's whole basis is a constant ratio between samples."""
    times = np.array([0.0, 0.1, 0.25, 0.4])
    with pytest.raises(ValueError, match="not uniformly spaced"):
        qnm.ringdown_fit(times, np.ones(4), modes=1)


def test_too_few_samples_is_refused_rather_than_fitted():
    with pytest.raises(ValueError, match="too few"):
        qnm.ringdown_fit(np.linspace(0, 1, 4), np.ones(4), modes=3)


def test_a_pure_decay_has_no_ringdown_frequency():
    """``dominant_frequency`` refuses rather than returning a damping rate."""
    times = np.linspace(0.0, 10.0, 200)
    with pytest.raises(ValueError, match="decay rather than a ringdown"):
        qnm.dominant_frequency(times, np.exp(-0.3 * times), modes=1)
