"""The head-on benchmark's analysis: extraction plans, radius extrapolation, Richardson, tails."""

import numpy as np
import pytest

from examples.headon_benchmark import (
    FUNDAMENTAL,
    configuration,
    energy_table,
    extraction_plan,
    extrapolate,
    radiated_energy,
    richardson,
)
from particlesim.solvers.nr import puncture

#: The medium run's plan (``runs/headon-z03``: ``n = 48``, buffer 6, margin 5).
MEDIUM_PLAN = [(6.0, 2), (8.0, 1), (10.0, 1), (12.0, 1), (16.0, 0), (20.0, 0), (24.0, 0)]
RADII = (6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0)
#: The medium run's areal radii.
AREAL = np.array([6.81, 8.88, 10.93, 12.96, 17.00, 21.01, 25.01])


def _setup(**overrides):
    config = {**configuration(), **overrides}
    setup, _ = puncture.NestedPuncture.build(**config, backend="numpy")
    return setup


def test_extrapolate_recovers_the_energy_at_infinity_and_the_wrong_model_does_not():
    """``E_inf (1 + b/R^2)`` and ``E_inf (1 + a/R + b/R^2)``, at the medium run's radii.

    Each model recovers its own data's ``E_inf`` and coefficients exactly.
    The control: fitting ``b/R^2`` alone to data with ``a = 3`` misses
    ``E_inf`` by 9%, and the radii below 10.5 are left out of every fit.
    """
    pure = 1.3 * (1 + 43 / AREAL**2)
    fit = extrapolate(AREAL, pure, "b/R2")
    assert fit["E"] == pytest.approx(1.3, rel=1e-12)
    assert fit["b"] == pytest.approx(43, rel=1e-10)

    mixed = 0.9 * (1 + 3 / AREAL + 20 / AREAL**2)
    fit = extrapolate(AREAL, mixed, "a/R+b/R2")
    assert fit["E"] == pytest.approx(0.9, rel=1e-10)
    assert fit["a"] == pytest.approx(3, rel=1e-8)
    assert fit["b"] == pytest.approx(20, rel=1e-8)
    assert extrapolate(AREAL, 0.8 * (1 + 7 / AREAL), "a/R")["E"] == pytest.approx(0.8)

    wrong = extrapolate(AREAL, mixed, "b/R2")
    assert abs(wrong["E"] / 0.9 - 1) > 0.05
    spoiled = pure.copy()
    spoiled[:2] *= 10  # r = 6 and 8, below the cut: must not matter
    assert extrapolate(AREAL, spoiled, "b/R2")["E"] == pytest.approx(1.3, rel=1e-12)
    with pytest.raises(ValueError, match="radii"):
        extrapolate(AREAL[:4], pure[:4], "a/R+b/R2")


def test_richardson_recovers_a_known_power_of_the_spacing():
    """``Q(h) = 2 + 5 h^p`` at ``h`` and ``1.5 h``: the limit and ``Q(h)``'s own error, exactly."""
    for order in (2, 4):
        h = 0.1
        fine, coarse = 2 + 5 * h**order, 2 + 5 * (1.5 * h) ** order
        limit, error = richardson(fine, coarse, 1.5, order)
        assert limit == pytest.approx(2.0, rel=1e-12)
        assert error == pytest.approx(-5 * h**order, rel=1e-10)
    wrong, _ = richardson(2 + 5 * 0.1**4, 2 + 5 * 0.15**4, 1.5, 2)
    assert abs(wrong - 2) > 1e-4


def test_a_four_cell_margin_at_n32_reproduces_the_medium_runs_extraction_plan():
    """``n = 32``, box 16, buffer 4: the medium run's levels need a margin of four cells.

    The plan is held equal at ``6 + 4 (3/4) = 9`` and ``12 + 4 (3/2) = 18``,
    the evolved interiors' edges. The control, the default margin of five,
    moves ``r = 6`` and ``r = 12`` a level coarser. The defaults still give
    the medium run's own plan.
    """
    small = _setup(n=32, box=16, buffer=4)
    assert extraction_plan(small, RADII, margin=4) == MEDIUM_PLAN
    control = extraction_plan(small, RADII)
    assert control != MEDIUM_PLAN
    assert dict(control)[6.0] == 1 and dict(control)[12.0] == 0
    assert extraction_plan(_setup(), RADII) == MEDIUM_PLAN


def test_a_radius_no_level_holds_is_refused():
    """The medium grid's coarsest interior ends at 42 M (a 3-cell zone of 2 M), spheres 10 M short.

    So 32 M is the largest even radius it holds, and 34 M is refused.
    """
    setup = _setup()
    assert extraction_plan(setup, (32.0,)) == [(32.0, 0)]
    with pytest.raises(ValueError, match="no level holds a sphere of radius 34"):
        extraction_plan(setup, (34.0,))


def test_the_tail_bound_holds_the_ringdown_left_after_the_record():
    """A news ringing as ``e^(-gamma t) cos(omega t + phase)``, recorded to 40 M of 400.

    It is switched on smoothly from zero, as the news from ``t = 0`` is.
    ``energy_table``'s tail is at least what the longer record adds and at
    most 1.6 times it, for any phase, to within the 2% that Simpson's news
    carries at this sampling (at ``dt = M/4`` the lower end reads 1.002).
    The control: a record that stops at 5 M, before the ringing, misses 97%
    of the energy, and the "bound" says 0.5%. It holds only for a record
    that ends in the ringdown, which the table's retarded ``end`` shows.
    """
    times = np.arange(0.0, 400.0, 0.5)
    omega, gamma = FUNDAMENTAL.real, -FUNDAMENTAL.imag
    for phase in np.linspace(0, np.pi, 7):
        switch = (1 - np.exp(-times / 2)) ** 4
        ringing = np.exp(-gamma * times) * np.cos(omega * times + phase)
        rate = -np.exp(-gamma * times) * (
            gamma * np.cos(omega * times + phase) + omega * np.sin(omega * times + phase)
        )
        on = 2 * (1 - np.exp(-times / 2)) ** 3 * np.exp(-times / 2)
        psi4 = switch * rate + on * ringing
        whole = radiated_energy(times, psi4, 10.0)
        short = times <= 40.0
        report = dict(
            waves=[
                dict(
                    time=t, spheres=[dict(radius=10.0, level=0, areal=10.0, modes={"2,0": [v, 0]})]
                )
                for t, v in zip(times[short], psi4[short], strict=True)
            ]
        )
        (row,) = energy_table(report, whole)
        missing = 1 - row["ratio"]
        assert 0.97 * missing <= row["tail"] <= 1.65 * missing, (phase, missing, row["tail"])
    early = dict(waves=report["waves"][:11])
    (row,) = energy_table(early, whole)
    assert row["tail"] < 0.05 * (1 - row["ratio"])
