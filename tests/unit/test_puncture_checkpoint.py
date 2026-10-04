"""A long numerical experiment must resume its actual saved grid and fields."""

import numpy as np
import pytest

from examples.puncture_benchmark import checkpoint, restore


def test_checkpoint_restores_exact_fields_without_pickle_and_checks_configuration(tmp_path):
    config = {"n": 4, "levels": 2}
    states = [{"W": np.arange(64, dtype=float).reshape((4,) * 3)}, {"alpha": np.ones((8,) * 3)}]
    path = tmp_path / "state.npz"
    checkpoint(path, states, 2.5, config)
    restored, clock = restore(path, states, config, "numpy")
    assert clock == 2.5
    for source, got in zip(states, restored, strict=True):
        for name in source:
            assert np.array_equal(source[name], got[name])
    assert not path.with_suffix(".tmp.npz").exists()
    with pytest.raises(ValueError, match="configuration"):
        restore(path, states, {**config, "n": 8}, "numpy")
    with pytest.raises(ValueError, match="field"):
        restore(path, [{"W": np.ones((2,) * 3)}], config, "numpy")


def test_checkpoint_rejects_invalid_time_or_fields(tmp_path):
    config = {"n": 4}
    states = [{"W": np.ones((4,) * 3)}]
    path = tmp_path / "state.npz"
    checkpoint(path, states, np.nan, config)
    with pytest.raises(ValueError, match="time"):
        restore(path, states, config, "numpy")
    states[0]["W"][0, 0, 0] = np.nan
    checkpoint(path, states, 2.5, config)
    with pytest.raises(ValueError, match="field"):
        restore(path, states, config, "numpy")
