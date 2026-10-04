"""Independent action gradients and the sampling contract of the bosonic IKKT model."""

import jax.numpy as jnp
import numpy as np
import pytest

from particlesim.solvers.matrix.ikkt import (
    BosonicIKKT,
    ComplexLangevin,
    IKKTObservations,
    temporal_eigenvalues,
)


def test_commuting_matrices_leave_only_the_mass_and_gauge_measure():
    model = BosonicIKKT(size=4, gamma=0.7)
    tau = jnp.log(jnp.array([0.7, 1.2, 0.6], dtype=jnp.complex128))
    alpha = np.asarray(temporal_eigenvalues(tau))
    spatial = np.zeros((9, 4, 4), dtype=complex)
    spatial[0] = np.diag([0.2, -0.2, 0.4, -0.4])
    pairs = [alpha[a] - alpha[b] for a in range(4) for b in range(a)]
    expected = -0.5j * 4 * 0.7 * (sum(alpha**2) - 0.4)
    expected -= 2 * sum(np.log(pairs)) + sum(np.asarray(tau))
    assert complex(model.action(tau, jnp.asarray(spatial))) == pytest.approx(expected)


@pytest.mark.parametrize("gamma", [0.0, 3.0])
@pytest.mark.parametrize("complex_direction", [False, True])
def test_drift_is_the_directional_derivative_of_the_holomorphic_action(gamma, complex_direction):
    model = BosonicIKKT(size=4, gamma=gamma)
    state = model.initialize(seed=9)
    tau = state.tau + jnp.array([0.03j, -0.04j, 0.02j])
    spatial = state.spatial * (1 + 0.07j)
    temporal_drift, spatial_drift = model.drift(tau, spatial)
    direction = np.array([0.3, -0.2, 0.1], complex)
    matrix_direction = np.zeros((9, 4, 4), complex)
    matrix_direction[2, 1, 3] = 0.2 + 0.3j
    matrix_direction[2, 3, 1] = 0.2 - 0.3j
    matrix_direction[1] = np.diag([0.4, -0.1, -0.2, -0.1])
    if complex_direction:
        direction *= 1j
        matrix_direction *= 1j
    h = 1e-6
    measured = (
        model.action(tau + h * direction, spatial + h * matrix_direction)
        - model.action(tau - h * direction, spatial - h * matrix_direction)
    ) / (2 * h)
    expected = -np.dot(temporal_drift, direction) - np.einsum(
        "iab,iba->", spatial_drift, matrix_direction
    )
    assert complex(measured) == pytest.approx(complex(expected), rel=2e-7, abs=1e-7)


def test_initialization_is_traceless_and_reproducible():
    model = BosonicIKKT(size=8)
    left, right = model.initialize(3), model.initialize(3)
    assert np.array_equal(left.spatial, right.spatial)
    assert np.max(np.abs(np.trace(left.spatial, axis1=-1, axis2=-2))) < 1e-15
    alpha = np.asarray(temporal_eigenvalues(left.tau))
    assert abs(alpha.sum()) < 1e-14
    assert np.all(np.diff((alpha * np.exp(3j * np.pi / 8)).real) > 0)


def test_adaptive_sampling_uses_equal_langevin_times_and_retains_raw_drift():
    model = BosonicIKKT(size=4)
    sampler = ComplexLangevin(model, drift_limit=0.1)
    state, observations = sampler.sample(model.initialize(2), 3, interval=1e-4)
    assert observations.langevin_times == pytest.approx([1e-4, 2e-4, 3e-4], abs=1e-15)
    assert observations.hermitian_eigenvalues.shape == (3, 1, 9)
    assert np.max(np.abs(np.trace(state.spatial, axis1=-1, axis2=-2))) < 1e-14
    for record in observations.drifts:
        assert record.steps > 5
        assert record.maximum_drift > sampler.drift_limit
        assert record.time_in_bins.sum() == pytest.approx(1e-4, rel=1e-12)
    report = observations.summary()
    assert "bosonic" in report["provenance"]
    assert "debated" in report["validity_statement"]
    assert "Pfaffian is omitted" in report["validity_statement"]
    assert np.min(observations.hermitian_eigenvalues) > -1e-14


def test_physical_time_averages_the_eigenvalues_before_taking_absolute_gaps():
    alpha = np.array([[-1, 0, 1], [-1j, 0, 1j]])
    observations = IKKTObservations(
        alpha,
        np.zeros((2, 3)),
        np.zeros((2, 3, 9)),
        np.zeros((2, 3)),
        np.array([1, 2]),
        (),
        3,
        0,
        1,
        2e-5,
        0,
    )
    assert observations.physical_times() == pytest.approx([-np.sqrt(0.5), 0, np.sqrt(0.5)])


@pytest.mark.parametrize("kwargs", [{"size": 1}, {"size": 3.5}, {"gamma": -1}, {"gamma": np.nan}])
def test_invalid_models_are_refused(kwargs):
    with pytest.raises(ValueError):
        BosonicIKKT(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"step_size": 0}, {"drift_limit": np.inf}, {"stabilization": 0.01}, {"max_steps": 0}],
)
def test_invalid_sampler_settings_are_refused(kwargs):
    with pytest.raises(ValueError):
        ComplexLangevin(BosonicIKKT(size=4), **kwargs)


def test_invalid_trajectories_and_exhausted_step_budgets_are_not_results():
    model = BosonicIKKT(size=4)
    state = model.initialize(1)
    with pytest.raises(RuntimeError, match="maximum Langevin steps"):
        ComplexLangevin(model, max_steps=1).advance(state, 0.1)
    bad = state._replace(tau=jnp.full_like(state.tau, jnp.nan))
    with pytest.raises(RuntimeError, match="trajectory is invalid"):
        ComplexLangevin(model).advance(bad, 0.01)
    with pytest.raises(ValueError, match="finite Langevin clock"):
        ComplexLangevin(model).advance(state._replace(time=jnp.nan), 0.01)
