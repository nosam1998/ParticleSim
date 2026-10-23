"""Two Brill-Lindquist punctures: exact constraints, and an ADM mass that is the bare masses."""

import numpy as np
import pytest

from particlesim.solvers.nr import bssn, puncture

MASSES = (0.5, 0.5)


def _pair_state(points, extent=4.0, z0=0.484375, backend="jax"):
    spacing = extent / points
    axis = (np.arange(points) + 0.5) * spacing - extent / 2
    grid = np.meshgrid(axis, axis, axis, indexing="ij")
    centres = [(0.0, 0.0, z0), (0.0, 0.0, -z0)]
    state = puncture.brill_lindquist_state(grid, centres, MASSES, backend)
    return state, grid, spacing, centres


def test_the_pair_satisfies_the_constraints_to_truncation_error():
    """``psi - 1`` is harmonic away from the punctures, and ``K_ij = 0``.

    So the discrete Hamiltonian constraint is pure truncation error,
    falling at fourth order or faster 1 M or more from either hole.
    Measured: 7.8e-05 at 32 points over 4 M and 3.9e-06 at 64, a ratio of
    20. A wrong superposition, the two conformal factors multiplied instead
    of added, is not a solution, and leaves 3.2e-02 at 32 points.
    """
    kernel = bssn.constraint_kernel(order=4)
    norms = []
    for points in (32, 64):
        state, grid, spacing, centres = _pair_state(points)
        h = np.asarray(kernel(bssn.physical_slice_arrays(state), (spacing,) * 3)["hamiltonian"])
        far = np.ones(h.shape, dtype=bool)
        for centre in centres:
            far &= sum((g - c) ** 2 for g, c in zip(grid, centre, strict=True)) > 1.0
        inner = (slice(4, -4),) * 3
        norms.append(float(np.sqrt(np.mean(h[inner][far[inner]] ** 2))))
        if points == 32:
            psi = np.exp(np.asarray(state["phi"]))
            each = [
                1 + m / (2 * np.sqrt(sum((g - c) ** 2 for g, c in zip(grid, centre, strict=True))))
                for m, centre in zip(MASSES, centres, strict=True)
            ]
            assert psi == pytest.approx(each[0] + each[1] - 1, rel=1e-13)
            wrong = dict(state, phi=np.log(each[0] * each[1]))
            wrong_h = np.asarray(
                kernel(bssn.physical_slice_arrays(wrong), (spacing,) * 3)["hamiltonian"]
            )
            wrong_norm = float(np.sqrt(np.mean(wrong_h[inner][far[inner]] ** 2)))
            assert wrong_norm > 10 * norms[0]
    assert 12 < norms[0] / norms[1] < 32, norms


def test_the_adm_mass_is_the_sum_of_the_bare_masses():
    """``M_ADM = -(1/2 pi) oint d_r psi dS``: by Gauss's law, ``sum m`` on any enclosing sphere.

    Read off the evolved ``W = psi^-2`` with second-order differences at
    spacing ``M/16`` on a sphere of radius 1.5 M: 1 to 4e-08. The same
    integral of one hole of mass 0.5, off centre, gives 0.5 to 2e-08.
    """
    from particlesim.analysis import extraction

    assert puncture.brill_lindquist_adm_mass(MASSES) == 1.0
    state, grid, spacing, _ = _pair_state(64, extent=4.0)
    data = bssn.with_conformal(state, "W", "jax")
    psi = np.asarray(data["W"]) ** -0.5
    axis = grid[0][:, 0, 0]
    gradient = np.gradient(psi, spacing, edge_order=2)
    radius = 1.5
    radial = sum(
        extraction.sphere_modes(g * n_field, (axis,) * 3, radius, ((0, 0),), spin=0)[(0, 0)]
        for g, n_field in zip(gradient, _direction_fields(grid), strict=True)
    )
    # sphere_modes integrates against Y00 = 1/sqrt(4 pi): undo it, then oint dS = r^2 dOmega.
    flux = radial.real * np.sqrt(4 * np.pi) * radius**2
    assert -flux / (2 * np.pi) == pytest.approx(1.0, abs=1e-6)
    single = puncture.brill_lindquist_state(grid, [(0.0, 0.0, 0.3)], [0.5], "numpy")
    psi_single = np.exp(np.asarray(single["phi"]))
    gradient = np.gradient(psi_single, spacing, edge_order=2)
    radial = sum(
        extraction.sphere_modes(g * n_field, (axis,) * 3, radius, ((0, 0),), spin=0)[(0, 0)]
        for g, n_field in zip(gradient, _direction_fields(grid), strict=True)
    )
    flux = radial.real * np.sqrt(4 * np.pi) * radius**2
    assert -flux / (2 * np.pi) == pytest.approx(0.5, abs=1e-6)


def _direction_fields(grid):
    radius = np.sqrt(sum(g**2 for g in grid))
    return [g / radius for g in grid]


def test_a_puncture_on_a_sample_and_mismatched_masses_are_refused():
    axis = (np.arange(8) + 0.5) * 0.5 - 2.0
    grid = np.meshgrid(axis, axis, axis, indexing="ij")
    with pytest.raises(ValueError, match="lands on a puncture"):
        puncture.brill_lindquist_state(grid, [(0.25, 0.25, 0.25)], [1.0], "numpy")
    with pytest.raises(ValueError, match="one mass"):
        puncture.brill_lindquist_state(grid, [(0.0, 0.0, 0.1)], [0.5, 0.5], "numpy")
