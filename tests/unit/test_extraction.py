"""Psi_4 and its spin-weighted modes, held to things that are exact (issue #136)."""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.analysis import extraction as ex

EXTENT, POINTS = 20.0, 64


@pytest.fixture(scope="module")
def grid():
    axis = (np.arange(POINTS) + 0.5) * EXTENT / POINTS - EXTENT / 2
    return axis, np.meshgrid(axis, axis, axis, indexing="ij"), (EXTENT / POINTS,) * 3


def _flat(shape):
    return [[np.full(shape, 1.0 if i == j else 0.0) for j in range(3)] for i in range(3)]


def _project(tensor, u, v):
    return sum(tensor[i][j] * u[i] * v[j] for i in range(3) for j in range(3))


def test_the_harmonics_are_the_closed_forms_and_orthonormal():
    theta = np.linspace(0.1, 3.0, 7)
    phi = np.linspace(0.0, 6.0, 7)
    y20 = ex.spin_weighted_harmonic(-2, 2, 0, theta, phi)
    assert np.allclose(y20, 0.25 * np.sqrt(15 / (2 * np.pi)) * np.sin(theta) ** 2, atol=1e-14)
    y22 = ex.spin_weighted_harmonic(-2, 2, 2, theta, phi)
    expected = 0.5 * np.sqrt(5 / np.pi) * np.cos(theta / 2) ** 4 * np.exp(2j * phi)
    assert np.allclose(y22, expected, atol=1e-14)

    th, ph, weights = ex.sphere_nodes(24, 48)
    modes = [(2, 0), (2, 1), (2, -2), (3, 0), (3, -2), (4, 3)]
    table = [ex.spin_weighted_harmonic(-2, l, m, th, ph) for l, m in modes]
    gram = np.array([[np.sum(weights * a * np.conj(b)) for b in table] for a in table])
    assert np.allclose(gram, np.eye(len(modes)), atol=1e-12)
    with pytest.raises(ValueError):
        ex.spin_weighted_harmonic(-2, 1, 0, theta, phi)


def test_schwarzschild_has_the_coulomb_weyl_tensor_and_no_psi4(grid):
    """Isotropic, time symmetric: ``E = diag(-2M/R^3, M/R^3, M/R^3)`` with ``R`` areal.

    Measured to 1e-3 of ``M/R^3`` at spacing ``M/3.2`` between ``r = 4`` and
    7, the stencil's error. ``B`` vanishes identically and ``Psi_4`` to 5e-4.
    """
    _, coords, spacing = grid
    x, y, z = coords
    r = np.sqrt(x**2 + y**2 + z**2)
    psi = 1 + 0.5 / r
    metric = [[psi**4 * (1.0 if i == j else 0.0) for j in range(3)] for i in range(3)]
    curvature = [[np.zeros_like(r) for _ in range(3)] for _ in range(3)]
    electric, magnetic = ex.weyl_electric_magnetic(metric, curvature, spacing)
    radial, polar, azimuthal = ex.radial_tetrad(metric, coords)
    areal = r * psi**2
    shell = (r > 4) & (r < 7)
    scale = areal[shell] ** 3
    assert np.max(np.abs(_project(electric, radial, radial)[shell] * scale + 2)) < 2e-3
    assert np.max(np.abs(_project(electric, polar, polar)[shell] * scale - 1)) < 2e-3
    assert np.max(np.abs(_project(electric, azimuthal, azimuthal)[shell] * scale - 1)) < 2e-3
    assert all(np.all(magnetic[i][j] == 0.0) for i in range(3) for j in range(3))
    psi4 = ex.psi4(electric, magnetic, metric, coords)
    assert np.max(np.abs(psi4[shell]) * scale) < 1e-3


@pytest.mark.parametrize("direction", ["outgoing", "ingoing"])
def test_a_plane_wave_leaving_is_its_second_derivative_and_one_arriving_is_nothing(grid, direction):
    """``h_yy = -h_zz = A sin(k(t -+ x))``, read on the ``+x`` axis where ``e_r`` is ``x``.

    Linear in ``A``, so ``K_ij = -h'_ij/2`` and ``E - iB`` either doubles or
    cancels. Leaving, ``Psi_4 = h''`` to 0.5%, the stencil's error; arriving,
    it is that error alone.
    """
    _, coords, spacing = grid
    x, y, z = coords
    amplitude, wavenumber = 1e-6, 2 * np.pi / 5.0
    phase = wavenumber * (-x if direction == "outgoing" else x)
    h = amplitude * np.sin(phase)
    h_dot = amplitude * wavenumber * np.cos(phase)
    h_ddot = -amplitude * wavenumber**2 * np.sin(phase)
    metric = _flat(x.shape)
    metric[1][1] = 1 + h
    metric[2][2] = 1 - h
    curvature = [[np.zeros_like(x) for _ in range(3)] for _ in range(3)]
    curvature[1][1] = -h_dot / 2
    curvature[2][2] = h_dot / 2
    electric, magnetic = ex.weyl_electric_magnetic(metric, curvature, spacing)
    psi4 = ex.psi4(electric, magnetic, metric, coords)
    axis = (np.abs(y) < 0.2) & (np.abs(z) < 0.2) & (x > 2) & (x < 8)
    scale = amplitude * wavenumber**2
    residual = psi4[axis] - (h_ddot[axis] if direction == "outgoing" else 0.0)
    assert np.max(np.abs(residual)) < 0.01 * scale


def test_the_modes_of_a_known_field_come_back(grid):
    """A field built from two harmonics times ``r`` is decomposed back into them."""
    axis, coords, _ = grid
    x, y, z = coords
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1, 1))
    phi = np.arctan2(y, x)
    field = (0.7 - 0.2j) * ex.spin_weighted_harmonic(
        -2, 2, 0, theta, phi
    ) + 0.3j * ex.spin_weighted_harmonic(-2, 3, 2, theta, phi)
    modes = ex.sphere_modes(field, (axis, axis, axis), 6.0, modes=((2, 0), (3, 2), (2, 2)))
    assert abs(modes[(2, 0)] - (0.7 - 0.2j)) < 2e-3
    assert abs(modes[(3, 2)] - 0.3j) < 2e-3
    assert abs(modes[(2, 2)]) < 2e-3
