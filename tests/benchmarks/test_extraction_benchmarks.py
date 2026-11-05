"""Psi_4 extrapolated to infinite radius, held to Teukolsky's ``l = 2`` wave (issue #51)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import cumulative_simpson

from particlesim.analysis import extraction as ex
from particlesim.solvers.nr import teukolsky

AMPLITUDE, WIDTH = 1e-4, 1.0
RADII = (6.0, 12.0, 24.0)
#: Every sample from 6 widths before the pulse to 6 after, where its tail is ``exp(-36)``.
REACH, STEP = 6.0, 0.05
#: The outgoing pulse, in retarded time, over which the radii are compared.
PULSE = 3.0
#: Gauss-Legendre nodes in ``cos(theta)``: the field is ``sin^2 theta`` times one of ``(t, r)``.
NODES = 8
#: Points either side of each node: a mixed second derivative composes two stencils of reach 2.
HALF = 4


def node_boxes(radius: float, spacing: float):
    """Coordinates of one ``(2 HALF + 1)^3`` box per node, stacked along ``x``, and the nodes.

    The nodes are the sphere's Gauss-Legendre points in the ``x``-``z`` half
    plane; the ``(2, 0)`` field is axisymmetric, so one azimuth integrates it
    exactly. Every stencil the Weyl tensor uses at a box's centre stays
    inside that box, so stacking them changes nothing there.
    """
    theta, phi, weights = ex.sphere_nodes(NODES, 1)
    offsets = spacing * np.arange(-HALF, HALF + 1)
    ox, oy, oz = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    x = np.concatenate([radius * np.sin(t) + ox for t in theta[:, 0]])
    y = np.concatenate([oy for _ in theta[:, 0]])
    z = np.concatenate([radius * np.cos(t) + oz for t in theta[:, 0]])
    return (x, y, z), theta, phi, weights


def psi4_mode(radius: float, times, spacing: float) -> np.ndarray:
    """The ``(2, 0)`` mode of ``Psi_4`` on the sphere of ``radius``, at each of ``times``.

    ``gamma_ij`` and ``K_ij = -d_t gamma_ij / 2`` as ``teukolsky.physical``
    builds them, then this module's Weyl tensor and ``Psi_4`` on the boxes,
    read at their centres and projected on ``conj(_{-2}Y_20)``.
    """
    coords, theta, phi, weights = node_boxes(radius, spacing)
    harmonic = np.conj(ex.spin_weighted_harmonic(-2, 2, 0, theta, phi))[:, 0]
    centre = slice(HALF, None, 2 * HALF + 1)
    out = []
    for time in times:
        h = teukolsky.perturbation(time, coords, AMPLITUDE, WIDTH)
        dh = teukolsky.perturbation(time, coords, AMPLITUDE, WIDTH, rate=True)
        metric = [[(1.0 if i == j else 0.0) + h[i][j] for j in range(3)] for i in range(3)]
        curvature = [[-0.5 * dh[i][j] for j in range(3)] for i in range(3)]
        electric, magnetic = ex.weyl_electric_magnetic(metric, curvature, (spacing,) * 3)
        field = ex.psi4(electric, magnetic, metric, coords)[centre, HALF, HALF]
        out.append(np.sum(weights[:, 0] * field * harmonic))
    return np.asarray(out)


def modes(spacing: float):
    """``(u, {radius: (times, Psi_4 mode)})`` on one grid of retarded time ``|u| <= REACH``.

    Each radius is sampled from ``t = max(0, r - REACH)``, so the integral
    in the correction starts where the news vanishes.
    """
    retarded = np.arange(-REACH, REACH + STEP / 2, STEP)
    return retarded, {r: (r + retarded, psi4_mode(r, r + retarded, spacing)) for r in RADII}


def raw(series):
    return {radius: radius * mode for radius, (_, mode) in series.items()}


def corrected(series):
    return {
        radius: ex.perturbative_extrapolation(times, mode, radius, 0.0, 2)
        for radius, (times, mode) in series.items()
    }


def half_corrected(series):
    """The control: the correction with ``1`` in place of ``(l-1)(l+2)/2 = 2``."""
    return {
        radius: radius * mode - cumulative_simpson(mode, x=times, initial=0.0)
        for radius, (times, mode) in series.items()
    }


def discrepancies(retarded, curves):
    """``D(r, 2r) = max |X(r) - X(2r)|`` over the pulse, over the outermost peak."""
    pulse = np.abs(retarded) <= PULSE
    scale = np.max(np.abs(curves[RADII[-1]][pulse]))
    return [
        float(np.max(np.abs(curves[inner][pulse] - curves[outer][pulse])) / scale)
        for inner, outer in zip(RADII[:-1], RADII[1:], strict=True)
    ]


def falls_as_inverse_square(candidate, reference) -> bool:
    """The pre-registered rule's (a) to (c): smaller by 4, and faster by 1.5 per doubling."""
    (near, far), (near_ref, far_ref) = candidate, reference
    smaller = near <= near_ref / 4 and far <= far_ref / 4
    rate = near / far
    return smaller and rate >= 3 and rate / (near_ref / far_ref) >= 1.5


@pytest.mark.benchmark
@pytest.mark.slow
def test_the_correction_turns_the_one_over_r_discrepancy_into_one_over_r_squared():
    """Teukolsky's flat-space ``l = 2`` wave, ``lambda = 1``, read at 6, 12 and 24 M.

    The rule was fixed before the run and is repeated in ``docs/benchmarks.md``.
    Measured over the pulse ``|u| <= 3``, relative to the peak at 24 M: raw
    ``r Psi_4`` differs by 4.95e-02 between 6 and 12 M and 2.48e-02 between
    12 and 24 M, a factor 1.99 per doubling; corrected, by 1.78e-03 and
    4.35e-04, a factor 4.09. Halving the spacing from ``M/20`` moves each
    waveform by 3.4e-05, under a tenth of the last. With the correction
    removed the rule fails, and so it does with half the coefficient.
    ``M = 0``, so the ``1 - 2M/r`` factor is not tested.
    """
    retarded, fine = modes(0.025)
    _, coarse = modes(0.05)
    d_raw = discrepancies(retarded, raw(fine))
    after, check = corrected(fine), corrected(coarse)
    d_cor = discrepancies(retarded, after)

    assert 1.5 <= d_raw[0] / d_raw[1] <= 2.5, d_raw  # the raw data go as 1/r
    assert falls_as_inverse_square(d_cor, d_raw), (d_cor, d_raw)
    pulse = np.abs(retarded) <= PULSE
    scale = np.max(np.abs(after[RADII[-1]][pulse]))
    for radius in RADII:
        floor = np.max(np.abs(after[radius][pulse] - check[radius][pulse])) / scale
        assert floor < d_cor[1] / 10, (radius, floor, d_cor)
    # The control: half the coefficient leaves a 1/r term, and the rule sees it.
    assert not falls_as_inverse_square(discrepancies(retarded, half_corrected(fine)), d_raw)
