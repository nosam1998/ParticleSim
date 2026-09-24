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
