"""A puncture on two levels with a radiative boundary (issue #48).

The long run lives in ``docs/benchmarks.md``; a thousand ``M`` is hours, not
a test. What is checked here is the setup a long run depends on and a short
stretch of the evolution through the part that used to fail: the first ten
``M``, where the lapse collapses onto a trumpet and a centred-advection run at
a fine spacing of ``M/4`` went to ``NaN``.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.solvers.nr import puncture


def test_the_data_is_brill_lindquist():
    """``psi = 1 + m/2r``, the lapse pre-collapsed to ``psi^-2``, the rest flat."""
    axis = np.linspace(-2.0, 2.0, 9) + 0.125
    grid = np.meshgrid(axis, axis, axis, indexing="ij")
    state = puncture.puncture_state(grid, (0.0, 0.0, 0.0), mass=1.0, backend="numpy")
    radius = np.sqrt(sum(g**2 for g in grid))
    conformal = 1.0 + 1.0 / (2 * radius)
    assert np.allclose(state["phi"], np.log(conformal))
    assert np.allclose(state["alpha"], conformal**-2)
    assert np.all(state["gt00"] == 1.0) and np.all(state["gt01"] == 0.0)
    assert np.all(state["trK"] == 0.0) and np.all(state["beta0"] == 0.0)


def test_a_sample_on_the_puncture_is_refused():
    axis = np.linspace(-1.0, 1.0, 5)
    grid = np.meshgrid(axis, axis, axis, indexing="ij")
    with pytest.raises(ValueError, match="lands on the puncture"):
        puncture.puncture_state(grid, (0.0, 0.0, 0.0), backend="numpy")


def test_a_perturbed_puncture_is_the_puncture_plus_a_wave_linear_in_its_amplitude():
    """At zero amplitude the data are Brill-Lindquist; the change is linear in the wave.

    Except in ``phi`` and ``K``: the wave and its rate are traceless, so the
    determinant and the trace move only at second order.
    """
    axis = np.linspace(-6.0, 6.0, 25) + 0.125
    grid = np.meshgrid(axis, axis, axis, indexing="ij")
    plain = puncture.puncture_state(grid, (0.0, 0.0, 0.0), backend="numpy")
    still = puncture.perturbed_puncture_state(grid, (0.0, 0.0, 0.0), 0.0, backend="numpy")
    for name, value in plain.items():
        assert np.allclose(still[name], value, atol=1e-14), name
    small = puncture.perturbed_puncture_state(grid, (0.0, 0.0, 0.0), 1e-6, backend="numpy")
    double = puncture.perturbed_puncture_state(grid, (0.0, 0.0, 0.0), 2e-6, backend="numpy")

    def change(state, name):
        return np.asarray(state[name]) - np.asarray(still[name])

    for name in ("gt00", "gt02", "At11", "Gt2"):
        once, twice = change(small, name), change(double, name)
        assert np.max(np.abs(once)) > 1e-7, name
        assert np.max(np.abs(twice - 2 * once)) < 1e-4 * np.max(np.abs(once)), name
    for name in ("phi", "trK"):
        assert np.max(np.abs(change(small, name))) < 1e-10, name


@pytest.mark.slow
def test_two_levels_carry_a_puncture_through_the_collapse_of_the_lapse():
    """Fifteen ``M`` on two levels with a radiative edge, upwinded.

    Coarse spacing ``M/1.5``, fine ``M/3``, the outer boundary at ``8 M``. The
    lapse starts at 0.09 at the nearest sample and has to collapse, not
    recover -- a trumpet forming -- while the constraint outside ``2 M``
    stays bounded and nothing goes non-finite.

    Also checks the geometry the whole thing rests on: the puncture sits half
    a fine cell off the fine grid, which is a quarter of a coarse cell off the
    coarse one, so neither level samples it.
    """
    setup, coarse, fine = puncture.TwoLevelPuncture.build(n=24, extent=16.0, box=16, zone=2.0)
    coarse_step = setup.coarse_axis[1] - setup.coarse_axis[0]
    fine_step = setup.fine_axis[1] - setup.fine_axis[0]
    offset = setup.position[0] - setup.coarse_axis[12]
    assert offset == pytest.approx(coarse_step / 4)
    nearest_fine = np.min(np.abs(setup.fine_axis - setup.position[0]))
    assert nearest_fine == pytest.approx(fine_step / 2)

    start = setup.diagnostics(coarse, fine)
    per_m = int(round(1.0 / setup.time_step))
    rows = []
    for k in range(1, 15 * per_m + 1):
        coarse, fine = setup.step(coarse, fine, 1.0 / per_m)
        if k % (5 * per_m) == 0:
            rows.append(setup.diagnostics(coarse, fine))
    assert all(row["finite"] for row in rows), rows
    assert rows[-1]["lapse_min"] < 0.5 * start["lapse_min"], (start, rows)
    assert max(row["hamiltonian_fine"] for row in rows) < 0.1, rows
    assert 0.0 < rows[-1]["shift_max"] < 1.0, rows
    # And the hole is still there: a run that dissolves it stays finite.
    assert rows[-1]["phi_max"] > 0.5, rows


def test_the_trumpet_radius_is_the_integral_it_closes():
    """``ln r = ln R - int_R^inf (1/f - 1) dR'/R'``, and ``R(r)`` inverts it."""
    from scipy.integrate import quad

    c = 3 * np.sqrt(3) / 4

    def f(R):
        return np.sqrt(1 - 2 / R + c**2 / R**4)

    for areal in (1.5001, 1.6, 2.0, 3.0, 10.0, 100.0):
        integral, _ = quad(lambda x: (1 / f(x) - 1) / x, areal, np.inf, limit=400)
        expected = areal * np.exp(-integral)
        assert puncture.trumpet_isotropic_radius(areal) == pytest.approx(expected, rel=1e-9)
    r = np.geomspace(1e-3, 50.0, 40)
    assert np.allclose(puncture.trumpet_isotropic_radius(puncture.trumpet_areal_radius(r)), r)
    assert puncture.trumpet_areal_radius(np.array([1e-12]))[0] == pytest.approx(1.5, abs=1e-6)


@pytest.mark.slow
def test_the_trumpet_is_stationary_under_the_unadvected_gauge():
    """The right-hand side on the trumpet is truncation error, fourth order.

    Measured at ``2 <= r <= 4 M``: 1.4e-03, 1.0e-04 and 7.1e-06 at spacings
    ``M/2``, ``M/4`` and ``M/8``. Brill-Lindquist data under the same gauge
    has rates of 1.2e-02 there at both ``M/2`` and ``M/4``: not truncation
    error, but a lapse and shift that still have to settle.
    """
    from particlesim.solvers.nr import bssn

    worst = []
    for n in (32, 64):
        spacing = 16.0 / n
        axis = np.arange(n) * spacing
        position = (n // 2 * spacing + spacing / 4,) * 3
        grid = np.meshgrid(axis, axis, axis, indexing="ij")
        state = puncture.trumpet_state(grid, position)
        evolution = bssn.Evolution.build((spacing,) * 3, dissipation=0.0, advect=False)
        rates = evolution.right_hand_side(state)
        r = np.sqrt(sum((g - p) ** 2 for g, p in zip(grid, position, strict=True)))
        shell = (r > 2.0) & (r < 4.0)
        worst.append(max(float(np.max(np.abs(np.asarray(v)[shell]))) for v in rates.values()))
    assert worst[0] < 2e-3
    assert worst[0] / worst[1] > 10


# --- more than two levels (issue #136) -------------------------------------


@pytest.mark.parametrize(("levels", "multiple"), [(1, 1), (2, 1), (3, 3), (4, 5), (5, 11)])
def test_the_puncture_is_put_where_no_level_samples_it(levels, multiple):
    """An odd number of the finest half-spacings, chosen so every level stays clear.

    Every level's points lie on the finest lattice, which makes any even
    multiple a grid point somewhere. Among the odd ones the choice is the one
    whose worst level is furthest from a sample, and that is at least a
    quarter of a cell on every level for every depth tested.
    """
    assert puncture.staggered_offset(levels) == multiple
    for level in range(levels):
        cells = multiple * 2**level / 2**levels
        distance = min(cells % 1.0, 1.0 - cells % 1.0)
        assert distance >= 0.25 - 1e-12, (level, distance)


def test_nested_levels_share_a_centre_and_halve_the_spacing():
    """Three levels of 32 points over 16 M: ``M/2``, ``M/4``, ``M/8``, each centred in its parent.

    Thirty-two is the fewest that fit: each box starts eight points in, and
    fourth-order interpolation reaching two past it leaves exactly the six
    points of buffer its parent needs.
    """
    setup, states = puncture.NestedPuncture.build(n=32, extent=16.0, levels=3)
    assert setup.levels == len(states) == 3
    assert [setup.spacing(level) for level in range(3)] == pytest.approx([0.5, 0.25, 0.125])
    for level, state in enumerate(states):
        assert np.shape(state["alpha"]) == (32, 32, 32), level
        assert setup.axes[level][16] == pytest.approx(8.0), level
        cells = (setup.position[0] - setup.axes[level][0]) / setup.spacing(level)
        assert abs(cells - round(cells)) >= 0.25 - 1e-12, level
    # The second-order edge's auxiliary fields are on the coarsest level only.
    assert any(name.startswith("aux:") for name in states[0])
    assert not any(name.startswith("aux:") for state in states[1:] for name in state)
    # Each level is the closed form on its own grid, not an interpolant:
    # the lapse at the sample nearest the puncture is psi^-2 there. (Levels
    # one and two both sit a sixteenth of M off it along each axis.)
    for level, state in enumerate(states):
        offsets = np.abs(setup.axes[level] - setup.position[0])
        nearest = np.sqrt(3.0) * float(np.min(offsets))
        expected = (1.0 + 1.0 / (2.0 * nearest)) ** -2
        assert float(np.min(np.asarray(state["alpha"]))) == pytest.approx(expected), level
    with pytest.raises(ValueError, match="interpolated rather than evolved"):
        puncture.NestedPuncture.build(n=24, extent=12.0, levels=3)


def test_a_nested_puncture_is_refused_what_it_cannot_centre():
    with pytest.raises(ValueError, match="at least two levels"):
        puncture.NestedPuncture.build(n=16, extent=8.0, levels=1)
    with pytest.raises(ValueError, match="no centre point"):
        puncture.NestedPuncture.build(n=16, extent=8.0, levels=2, box=7)


def test_a_wave_goes_onto_every_level():
    """``wave`` swaps the data for the perturbed puncture, on each level's own grid."""
    wave = {"amplitude": 1e-3, "width": 1.0, "time": -3.0}
    _, plain = puncture.NestedPuncture.build(n=16, extent=8.0, levels=2, zone=1.0)
    _, struck = puncture.NestedPuncture.build(n=16, extent=8.0, levels=2, zone=1.0, wave=wave)
    for level in range(2):
        change = np.asarray(struck[level]["gt00"]) - np.asarray(plain[level]["gt00"])
        assert 1e-5 < float(np.max(np.abs(change))) < 1e-2, level
