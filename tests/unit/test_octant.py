"""An octant must be the reflection-symmetric full grid, at an eighth of the cells."""

import numpy as np
import pytest

from particlesim.solvers.nr import octant, symmetry

NAMES = ("W", "alpha", "beta0", "beta2", "Gt1", "B2", "gt00", "gt01", "gt12", "At02", "aux:beta1")


def _symmetric_state(size=12, seed=5):
    random = np.random.default_rng(seed)
    state = {name: random.normal(size=(size,) * 3) for name in NAMES}
    return symmetry.project(state)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_fold_unfold_and_ghosts_reproduce_the_symmetric_grid(backend):
    full = _symmetric_state()
    folded = octant.fold(full)
    for name, value in folded.items():
        assert value.shape == (6 + octant.GHOSTS,) * 3
        # The octant's ghosts are already the mirror images, so refilling them changes nothing.
        refilled = octant.reflect(folded, backend=backend)[name]
        assert np.asarray(refilled) == pytest.approx(value, abs=1e-15)
    unfolded = octant.unfold(folded)
    for name in full:
        assert np.array_equal(unfolded[name], full[name]), name


def test_ghosts_take_each_components_parity_and_overwrite_garbage():
    full = _symmetric_state()
    folded = octant.fold(full)
    spoiled = {name: np.array(value) for name, value in folded.items()}
    for value in spoiled.values():
        value[: octant.GHOSTS] = np.nan
        value[:, : octant.GHOSTS] = 1e9
    filled = octant.reflect(spoiled, backend="numpy")
    for name in full:
        assert np.array_equal(filled[name], folded[name]), name
    # A wrong parity is visible: beta0 is odd in x, so mirroring it evenly is not the field.
    wrong = octant.fill_ghosts({"alpha": folded["beta0"]})["alpha"]
    assert np.max(abs(wrong - folded["beta0"])) > 0.1


def test_odd_grids_and_small_arrays_are_refused():
    with pytest.raises(ValueError, match="even"):
        octant.fold({"W": np.zeros((9, 8, 8))})
    with pytest.raises(ValueError, match="small"):
        octant.fill_ghosts({"W": np.zeros((6, 8, 8))})


def test_an_octant_requires_symmetric_cell_centred_data():
    from particlesim.solvers.nr import puncture

    settings = dict(n=24, extent=6, box=16, levels=2, centering="cell", octant=True)
    for wrong in ({"centering": "vertex"}, {"offset": 1}, {"reflection_symmetry": True}):
        with pytest.raises(ValueError, match="octant"):
            puncture.NestedPuncture.build(**{**settings, **wrong})
    with pytest.raises(ValueError, match="symmetric"):
        puncture.NestedPuncture.build(**settings, punctures=((0.5, (0.0, 0.0, 0.3)),))


@pytest.mark.slow
def test_octant_steps_match_the_symmetric_full_grid_and_a_wrong_parity_does_not():
    """One subcycled two-level step, octant against full grid, to rounding.

    Measured: 4e-16 on the coarse level and 2e-16 on the fine after one
    step, 9e-16 and 1e-15 after two. The test's own control mirrors every
    component as a scalar: then the octant's vectors and tensors are wrong
    beside each plane and the step differs at order one.
    """
    from particlesim.solvers.nr import puncture

    settings = dict(
        n=24,
        extent=6,
        box=16,
        levels=2,
        data="trumpet",
        conformal="W",
        centering="cell",
        boundary_background="trumpet",
    )
    full, full_states = puncture.NestedPuncture.build(**settings, reflection_symmetry=True)
    part, part_states = puncture.NestedPuncture.build(**settings, octant=True)
    assert part_states[1]["W"].shape == (16 + octant.GHOSTS,) * 3
    full_states = full.step(full_states)
    stepped = part.step(part_states)
    for level, unfolded in enumerate(part.full_states(stepped)):
        for name, value in unfolded.items():
            assert value == pytest.approx(np.asarray(full_states[level][name]), abs=1e-12), name
    assert np.max(abs(np.asarray(stepped[1]["W"]) - np.asarray(part_states[1]["W"]))) > 1e-6

    original = octant.parity
    try:
        octant.parity = lambda name: (1, 1, 1)
        octant._filler.cache_clear()
        broken = part.step(part_states)
    finally:
        octant.parity = original
        octant._filler.cache_clear()
    worst = max(
        float(np.max(abs(np.asarray(broken[1][name]) - np.asarray(stepped[1][name]))))
        for name in ("beta0", "Gt1", "gt01", "At02")
    )
    assert worst > 1e-6
