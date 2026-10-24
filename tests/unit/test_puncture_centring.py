"""A puncture at every level's reflection centre stays there; one a half fine cell off does not.

Issue #48's long two-level runs lost the hole when its coordinate position
drifted out of the finest box. On one uniform level the push was traced to
the stencils' error on either side of an off-centre puncture. With
cell-centred levels the puncture is at a reflection centre of every level,
so the push cancels. This holds a short full-grid run, without any symmetry
imposed, to that.
"""

import numpy as np
import pytest

from particlesim.solvers.nr import puncture

SETTINGS = dict(
    n=24,
    extent=6,
    box=16,
    levels=2,
    data="trumpet",
    conformal="W",
    boundary_background="trumpet",
    zone=0.75,
    advect=False,
    buffer=6,
)


def _centre(setup, state):
    """The ``1/2 - alpha``-weighted centre of the region where the lapse is below one half."""
    axis = setup.full_axis(setup.levels - 1)
    lapse = np.asarray(state["alpha"])
    weights = np.maximum(0.5 - lapse, 0)
    grids = np.meshgrid(*(axis,) * 3, indexing="ij")
    return np.array(
        [
            np.sum((g - p) * weights) / np.sum(weights)
            for g, p in zip(grids, setup.position, strict=True)
        ]
    )


def _drift_and_mass(extra, time=2.5):
    from particlesim.analysis.horizon import Slice, find_apparent_horizon
    from particlesim.solvers.nr import bssn

    setup, states = puncture.NestedPuncture.build(**SETTINGS, **extra)
    start = _centre(setup, states[-1])
    for _ in range(int(round(time / setup.time_step))):
        states = setup.step(states)
    drift = _centre(setup, states[-1]) - start
    axis = setup.full_axis(setup.levels - 1)
    geometry = Slice(
        bssn.physical_slice_arrays(states[-1], backend="numpy"),
        (setup.spacing(setup.levels - 1),) * 3,
        centre=tuple(p - axis[0] for p in setup.position),
    )
    horizon = find_apparent_horizon(
        geometry, float(puncture.trumpet_isotropic_radius(2)), 16, 32, 0.004, 200
    )
    return drift, horizon.irreducible_mass


@pytest.mark.slow
def test_a_cell_centred_puncture_stays_at_its_centre():
    """Measured at 2.5 M: 8e-17 M per axis, mass 0.99993.

    The control, run once, is the vertex-centred hierarchy with the puncture
    half a fine cell, a quarter coarse cell, off the boxes' centre along
    (1,1,1) (``centering="vertex", offset=1``). Over the same 2.5 M its
    centre moves by -6.6e-05 M per axis, and the drift tolerance fails it
    by six orders of magnitude. Its mass, 0.99990, passes: at this length
    the mass bound guards the cell-centred run's health, not the centring.
    """
    drift, mass = _drift_and_mass(dict(centering="cell"))
    assert np.max(np.abs(drift)) < 1e-10
    assert mass == pytest.approx(1.0, abs=1e-3)
