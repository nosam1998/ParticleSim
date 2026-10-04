"""Radiation acts on perturbations of an explicitly supplied stationary exterior."""

import numpy as np
import pytest

from particlesim.solvers.nr import boundary, puncture


def _data():
    axis = (np.arange(16) - 7.5) / 4
    coords = tuple(np.meshgrid(axis, axis, axis, indexing="ij"))
    x, y, z = coords
    reference = {"alpha": 1 + 0.1 * np.sin(x) + 0.2 * y**2, "phi": 0.3 * np.cos(y) * z}
    pulse = 0.01 * np.exp(-(x**2 + y**2 + z**2))
    return coords, reference, pulse


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("order", [4, 6])
def test_a_nonconstant_stationary_background_has_zero_radiation_rate(backend, order):
    coords, reference, _ = _data()
    condition = boundary.Radiative(
        coords,
        (0.25,) * 3,
        (0, 1, 2),
        3,
        backend=backend,
        order=order,
        background=reference,
    )
    for value in condition.rates(reference).values():
        assert np.all(np.asarray(value) == 0)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_the_same_outgoing_departure_has_the_same_rate_on_either_background(backend):
    coords, reference, pulse = _data()
    plain = boundary.Radiative(coords, (0.25,) * 3, (0, 1, 2), 3, backend=backend)
    curved = boundary.Radiative(
        coords, (0.25,) * 3, (0, 1, 2), 3, backend=backend, background=reference
    )
    expected = plain.rates({name: boundary.ASYMPTOTIC[name] + pulse for name in reference})
    actual = curved.rates({name: value + pulse for name, value in reference.items()})
    for name in reference:
        assert np.asarray(actual[name]) == pytest.approx(np.asarray(expected[name]), abs=2e-12)


def test_second_order_background_rates_agree_between_backends_and_keep_interior_rates():
    coords, reference, _ = _data()

    class Rates:
        def right_hand_side(self, state):
            return {name: np.full_like(value, 0.03) for name, value in state.items()}

    results = []
    for backend in ("numpy", "jax"):
        condition = boundary.Radiative(
            coords, (0.25,) * 3, (0, 1, 2), 3, backend=backend, background=reference
        )
        evolution = boundary.SecondOrder(Rates(), condition)
        rates = evolution.right_hand_side(evolution.start(reference))
        zone = condition.mask()
        for name in reference:
            assert np.all(np.asarray(rates[name])[zone] == 0)
            assert np.asarray(rates[name])[~zone] == pytest.approx(0.03)
        results.append(rates)
    for name in results[0]:
        assert np.asarray(results[1][name]) == pytest.approx(
            np.asarray(results[0][name]), abs=2e-12
        )


def test_reference_fields_must_have_the_grid_shape():
    coords, _, _ = _data()
    with pytest.raises(ValueError, match="grid shape"):
        boundary.Radiative(coords, (0.25,) * 3, (0, 1, 2), 3, background={"alpha": np.ones(3)})


@pytest.mark.parametrize("kwargs", [{}, {"data": "trumpet", "advect": True}])
def test_a_stationary_trumpet_boundary_refuses_incompatible_initial_data_or_gauge(kwargs):
    with pytest.raises(ValueError, match="trumpet.*(requires|needs)"):
        puncture.NestedPuncture.build(boundary_background="trumpet", **kwargs)
