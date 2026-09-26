"""Gravitational waves from a 3-D slice: ``Psi_4`` and its spin-weighted modes (issue #136).

A ringdown is read from the Weyl scalar ``Psi_4`` on a sphere around the
hole, decomposed into spin-weight ``-2`` spherical harmonics. Everything here
works on one slice at a time, so a run records the modes as it goes and the
fit happens afterwards.

**The Weyl tensor from the slice.** In vacuum the Weyl tensor's electric and
magnetic parts are the slice's own curvature:

    E_ij = R_ij + K K_ij - K_ik K^k_j
    B_ij = eps_(i|kl| D^k K^l_j)

with ``R_ij`` the Ricci tensor of the spatial metric and ``eps`` the volume
form. The Ricci tensor and the connection come from
:func:`particlesim.symbolic.threeplusone.grid_slice`, the same algebra the
constraint kernel is derived from, applied here to arrays.

**The scalar.** With a tetrad built from the coordinate radial, polar and
azimuthal directions, orthonormalised against ``gamma_ij``, and
``m = (e_theta + i e_phi)/sqrt(2)``,

    Psi_4 = (E_ij - i B_ij) mbar^i mbar^j

This is the convention of Baker, Campanelli and Lousto (2002) up to an overall
constant and sign, which a frequency does not see. For a plane wave leaving
along the radial direction ``|Psi_4| = |h''|``; for one arriving it is zero.
The tests hold it to both.

**The modes.** ``Psi_4`` is interpolated onto a sphere, cubic in each
direction, and integrated against ``conj(_{-2}Y_lm)`` with Gauss-Legendre
nodes in ``cos(theta)`` and a uniform grid in ``phi``. The harmonics come from
Goldberg's formula, and are held to the closed forms for ``l = 2``.
"""

from __future__ import annotations

from math import comb, factorial

import numpy as np
from scipy.ndimage import map_coordinates

from particlesim.symbolic.threeplusone import (
    INDICES,
    christoffel,
    determinant,
    grid_slice,
    inverse_metric,
    raise_index,
    ricci,
    trace,
)


def _levi_civita(i: int, j: int, k: int) -> int:
    return (i - j) * (j - k) * (k - i) // 2


def weyl_electric_magnetic(metric, curvature, spacing, order: int = 4):
    """``(E_ij, B_ij)`` of a vacuum slice, each a 3x3 nested list of arrays.

    ``metric`` and ``curvature`` are ``gamma_ij`` and ``K_ij`` as 3x3 nested
    lists of arrays on a grid of the given ``spacing``. Derivatives are
    fourth order in the interior and fall back to second order within two
    points of an edge, so extraction belongs well inside the grid.
    """
    zero = np.zeros_like(np.asarray(metric[0][0], dtype=float))
    slice_ = grid_slice(zero + 1.0, [zero, zero, zero], metric, curvature, spacing, order=order)
    inverse = inverse_metric(slice_.metric)
    curvature = slice_.curvature
    ricci_tensor = ricci(slice_, inverse)
    trace_k = trace(inverse, curvature)
    mixed = raise_index(inverse, curvature)  # K^k_j
    electric = [
        [
            ricci_tensor[i][j]
            + trace_k * curvature[i][j]
            - sum(curvature[i][k] * mixed[k][j] for k in INDICES)
            for j in INDICES
        ]
        for i in INDICES
    ]

    gamma = christoffel(slice_, inverse)
    d_k = slice_.d_curvature  # [k][i][j] = d_k K_ij
    # D_k K_lj = d_k K_lj - Gamma^m_kl K_mj - Gamma^m_kj K_lm
    covariant = [
        [
            [
                d_k[k][l][j]
                - sum(
                    gamma[m][k][l] * curvature[m][j] + gamma[m][k][j] * curvature[l][m]
                    for m in INDICES
                )
                for j in INDICES
            ]
            for l in INDICES
        ]
        for k in INDICES
    ]
    volume = np.sqrt(determinant(slice_.metric))
    # eps_i^{kl} = gamma_im [mkl] / sqrt(gamma)
    curl = [
        [
            sum(
                slice_.metric[i][m] * _levi_civita(m, k, l) * covariant[k][l][j]
                for m in INDICES
                for k in INDICES
                for l in INDICES
                if _levi_civita(m, k, l)
            )
            / volume
            for j in INDICES
        ]
        for i in INDICES
    ]
    magnetic = [[(curl[i][j] + curl[j][i]) / 2 for j in INDICES] for i in INDICES]
    return electric, magnetic


def radial_tetrad(metric, coords, centre=(0.0, 0.0, 0.0)):
    """``(e_r, e_theta, e_phi)``, orthonormal against ``gamma_ij``, as lists of three arrays.

    Gram-Schmidt from the coordinate radial direction, then the polar
    ``(xz, yz, -(x^2 + y^2))`` and the azimuthal ``(-y, x, 0)``. On the
    polar axis the last two vanish, so the axis itself is not a place to
    extract.
    """
    x, y, z = (np.asarray(c, dtype=float) - c0 for c, c0 in zip(coords, centre, strict=True))
    g = metric

    def dot(u, v):
        return sum(g[i][j] * u[i] * v[j] for i in INDICES for j in INDICES)

    def normalise(u):
        size = np.sqrt(dot(u, u))
        return [component / size for component in u]

    def remove(u, along):
        overlap = dot(u, along)
        return [u[i] - overlap * along[i] for i in INDICES]

    radial = normalise([x, y, z])
    polar = normalise(remove([x * z, y * z, -(x**2 + y**2)], radial))
    azimuthal = normalise(remove(remove([-y, x, np.zeros_like(x)], radial), polar))
    return radial, polar, azimuthal


def psi4(electric, magnetic, metric, coords, centre=(0.0, 0.0, 0.0)) -> np.ndarray:
    """``(E_ij - i B_ij) mbar^i mbar^j`` with ``m = (e_theta + i e_phi)/sqrt(2)``."""
    _, polar, azimuthal = radial_tetrad(metric, coords, centre)
    mbar = [(polar[i] - 1j * azimuthal[i]) / np.sqrt(2.0) for i in INDICES]
    return sum(
        (electric[i][j] - 1j * magnetic[i][j]) * mbar[i] * mbar[j] for i in INDICES for j in INDICES
    )


def spin_weighted_harmonic(s: int, l: int, m: int, theta, phi) -> np.ndarray:
    """``_sY_lm(theta, phi)`` by Goldberg's formula.

    ``(-1)^m sqrt((l+m)!(l-m)!(2l+1) / (4 pi (l+s)!(l-s)!)) sin^(2l)(theta/2)
    sum_r C(l-s, r) C(l+s, r+s-m) (-1)^(l-r-s) e^(i m phi) cot^(2r+s-m)(theta/2)``
    """
    if l < abs(s) or abs(m) > l:
        raise ValueError(f"no harmonic with s={s}, l={l}, m={m}")
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    norm = (-1) ** m * np.sqrt(
        factorial(l + m)
        * factorial(l - m)
        * (2 * l + 1)
        / (4 * np.pi * factorial(l + s) * factorial(l - s))
    )
    half = theta / 2
    total = np.zeros(np.broadcast(theta, phi).shape, dtype=complex)
    for r in range(0, l - s + 1):
        k = r + s - m
        if k < 0 or k > l + s:
            continue
        power = 2 * r + s - m
        # sin^(2l) cot^power = sin^(2l - power) cos^power, finite at the poles.
        term = np.sin(half) ** (2 * l - power) * np.cos(half) ** power
        total = total + comb(l - s, r) * comb(l + s, k) * (-1) ** (l - r - s) * term
    return norm * total * np.exp(1j * m * phi)


def sphere_nodes(count_theta: int = 24, count_phi: int = 48):
    """``(theta, phi, weights)`` on a Gauss-Legendre x uniform grid, weights summing to ``4 pi``."""
    nodes, legendre = np.polynomial.legendre.leggauss(count_theta)
    theta = np.arccos(nodes)
    phi = 2 * np.pi * np.arange(count_phi) / count_phi
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    weights = np.outer(legendre, np.full(count_phi, 2 * np.pi / count_phi))
    return th, ph, weights


def sphere_modes(
    field,
    axes,
    radius: float,
    modes=((2, 0),),
    spin: int = -2,
    centre=(0.0, 0.0, 0.0),
    count_theta: int = 24,
    count_phi: int = 48,
) -> dict[tuple[int, int], complex]:
    """``int field conj(_sY_lm) dOmega`` on the sphere of ``radius`` about ``centre``.

    ``axes`` are the grid's coordinate vectors, one per dimension, evenly
    spaced. The field is interpolated with cubic splines, real and imaginary
    parts separately.
    """
    th, ph, weights = sphere_nodes(count_theta, count_phi)
    points = [
        centre[0] + radius * np.sin(th) * np.cos(ph),
        centre[1] + radius * np.sin(th) * np.sin(ph),
        centre[2] + radius * np.cos(th),
    ]
    index = np.array(
        [(p - a[0]) / (a[1] - a[0]) for p, a in zip(points, axes, strict=True)]
    ).reshape(3, -1)
    field = np.asarray(field)
    sampled = map_coordinates(field.real, index, order=3, mode="nearest").reshape(th.shape)
    if np.iscomplexobj(field):
        sampled = sampled + 1j * map_coordinates(
            field.imag, index, order=3, mode="nearest"
        ).reshape(th.shape)
    return {
        (l, m): complex(
            np.sum(weights * sampled * np.conj(spin_weighted_harmonic(spin, l, m, th, ph)))
        )
        for l, m in modes
    }


def bssn_weyl(state, spacing, order: int = 4):
    """``(E, B, gamma)`` from a BSSN state, via its physical slice."""
    factor = np.exp(4 * np.asarray(state["phi"]))
    trace_k = np.asarray(state["trK"])
    metric = [[None] * 3 for _ in INDICES]
    curvature = [[None] * 3 for _ in INDICES]
    for i in INDICES:
        for j in range(i, 3):
            g = factor * np.asarray(state[f"gt{i}{j}"])
            k = factor * np.asarray(state[f"At{i}{j}"]) + g * trace_k / 3
            metric[i][j] = metric[j][i] = g
            curvature[i][j] = curvature[j][i] = k
    electric, magnetic = weyl_electric_magnetic(metric, curvature, spacing, order)
    return electric, magnetic, metric


__all__ = [
    "bssn_weyl",
    "psi4",
    "radial_tetrad",
    "sphere_modes",
    "sphere_nodes",
    "spin_weighted_harmonic",
    "weyl_electric_magnetic",
]
