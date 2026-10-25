"""A symmetry restriction must transform vectors and tensors, not freeze fields."""

import numpy as np
import pytest

from particlesim.solvers.nr import symmetry


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_projection_obeys_tensor_parities_and_is_idempotent(backend):
    random = np.random.default_rng(73)
    names = ("W", "trK", "Theta", "beta0", "Gt2", "B1", "gt00", "gt01", "At12", "aux:gt02")
    state = {name: random.normal(size=(8, 10, 6)) for name in names}
    result = symmetry.project(state, backend=backend)
    again = symmetry.project(result, backend=backend)
    for name, field in result.items():
        for axis, sign in enumerate(symmetry.parity(name)):
            assert np.array_equal(field, sign * np.flip(field, axis=axis))
        assert np.array_equal(field, again[name])
        assert not np.array_equal(field, state[name])
    numpy = symmetry.project(state)
    for name in state:
        assert np.asarray(result[name]) == pytest.approx(numpy[name], abs=1e-15)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_reflection_keeps_arbitrary_symmetric_physical_fields(backend):
    axis = np.arange(8) - 3.5
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    r2 = x * x + y * y + z * z
    state = {
        "alpha": np.exp(-r2),
        "beta0": x / (1 + r2),
        "gt01": x * y / (1 + r2),
        "At22": 7 + z * z,
        "aux:beta2": z * np.exp(-r2),
    }
    result = symmetry.project(state, backend=backend)
    for name in state:
        assert np.asarray(result[name]) == pytest.approx(state[name], abs=1e-15)


def test_unsupported_fields_and_grids_are_rejected():
    with pytest.raises(ValueError, match="unknown"):
        symmetry.project({"density": np.zeros((8, 8, 8))})
    with pytest.raises(ValueError, match="even"):
        symmetry.project({"W": np.zeros((9, 8, 8))})
    with pytest.raises(ValueError, match="backend"):
        symmetry.project({}, backend="invalid")


@pytest.mark.slow
def test_symmetric_puncture_evolves_and_rejects_incompatible_layouts():
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
        reflection_symmetry=True,
    )
    setup, states = puncture.NestedPuncture.build(**settings)
    advanced = setup.step(states)
    # The numerical solution evolves under truncation error; no initial
    # analytic field is substituted to create apparent stability.
    assert np.max(abs(np.asarray(advanced[-1]["W"] - states[-1]["W"]))) > 1e-8
    for state in advanced:
        for name, value in state.items():
            for axis, sign in enumerate(symmetry.parity(name)):
                assert np.array_equal(value, sign * np.flip(value, axis=axis)), name
    for incompatible in ({"offset": 1}, {"centering": "vertex"}, {"wave": {"amplitude": 0.01}}):
        with pytest.raises(ValueError, match="requires"):
            puncture.NestedPuncture.build(**{**settings, **incompatible})
