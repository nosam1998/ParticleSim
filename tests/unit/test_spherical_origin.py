"""Regularity at the origin (design doc Section 5.4, Milestone 1).

The origin is not a boundary. The solution continues through it with a
definite parity, so nothing there needs one-sided treatment and nothing
there may be left undamped. Two bugs that both hid here are pinned below.
"""

import numpy as np
import pytest

from particlesim.core.spherical import SphericalGrid
from particlesim.solvers.nr.spherical import (
    ScalarCollapse,
    _d_dr,
    _dissipate,
    gaussian_pulse,
)


def _radii(n: int, r_max: float) -> np.ndarray:
    return SphericalGrid(r_max=r_max, n=n).radii()


class ExpandedFlux(ScalarCollapse):
    """The ``Pi`` equation as it was first written, for contrast.

    ``d(f Phi)/dr + 2 f Phi / r`` is the conservative ``(1/r^2) d(r^2 f
    Phi)/dr`` expanded by the product rule -- equal in the continuum, and
    not in the discrete system, where only the conservative form is the
    adjoint of the ``Phi`` equation's derivative.
    """

    def rhs(self, state, metric=None):
        a, alpha = self.solve_metric(state.Phi, state.Pi) if metric is None else metric
        f, r = alpha / a, self.r
        dPhi, dPi = super().rhs(state, (a, alpha))
        conservative = _d_dr(r**2 * f * state.Phi, r, parity=-1) / r**2
        expanded = _d_dr(f * state.Phi, r, parity=-1) + 2.0 * f * state.Phi / r
        dPi[:-2] += expanded[:-2] - conservative[:-2]
        return dPhi, dPi


def test_the_flux_pair_conserves_the_discrete_energy_exactly():
    """The root cause of every origin instability this solver has had.

    With ``f = 1`` the principal part is ``dPhi/dt = D Pi`` and ``dPi/dt =
    (1/r^2) D(r^2 Phi)``, and ``sum r^2 (Phi^2 + Pi^2)`` is conserved when
    the second operator is minus the adjoint of the first in that weighted
    sum. With the reflection ghosts of a cell-centred grid, the odd-parity
    stencil is exactly minus the transpose of the even-parity one, so the
    conservative form is that adjoint and the rate is zero to round-off, for
    any data at all. Expanded by the product rule it is not, and the
    difference sits at the origin, where ``2 Phi / r`` divides by ``dr / 2``:
    a growth rate proportional to ``1 / dr`` that dissipation, also
    ``epsilon / dr``, could only ever hold at a fixed ratio.
    """
    rates = {}
    for n in (64, 128):
        r = _radii(n, 4.0)
        rng = np.random.default_rng(3)
        Phi, Pi = np.zeros(n), np.zeros(n)
        Phi[:8], Pi[:8] = rng.normal(size=8), rng.normal(size=8)  # at the origin
        dPhi = _d_dr(Pi, r, parity=1)
        energy = float(np.sum(r**2 * (Phi**2 + Pi**2)))

        def rate(dPi, r=r, Phi=Phi, Pi=Pi, dPhi=dPhi, energy=energy):
            return float(np.sum(r**2 * (Phi * dPhi + Pi * dPi))) / energy

        assert abs(rate(_d_dr(r**2 * Phi, r, parity=-1) / r**2)) < 1e-13
        rates[n] = rate(_d_dr(Phi, r, parity=-1) + 2.0 * Phi / r)
    # Bilinear in Phi and Pi, so flipping the sign of Pi turns this loss
    # into growth; either way it doubles when dr halves.
    assert abs(rates[64]) > 1.0
    assert rates[128] / rates[64] == pytest.approx(2.0, rel=1e-9)


@pytest.mark.parametrize("parity", (1, -1))
def test_parity_stencils_are_fourth_order_at_the_innermost_cells(parity):
    """The two cells nearest the origin must carry the interior's accuracy.

    Reading their inner neighbours from the reflection makes the centred
    fourth-order stencil available there, so the error at r_0 and r_1 has
    to fall by at least 16 per halving. It falls faster in practice, near
    32, because the leading error term is itself odd and nearly cancels
    against the origin; the bound is one-sided for that reason. A one-sided
    second-order fallback falls by 4, which this separates either way.
    """
    # An even test function and an odd one, each analytic through r = 0.
    if parity == 1:
        f, df = (
            (lambda r: np.cos(1.3 * r) + 0.4 * r**2),
            (lambda r: -1.3 * np.sin(1.3 * r) + 0.8 * r),
        )
    else:
        f, df = (
            (lambda r: np.sin(1.3 * r) + 0.4 * r**3),
            (lambda r: 1.3 * np.cos(1.3 * r) + 1.2 * r**2),
        )

    errors = []
    for n in (200, 400, 800):
        r = _radii(n, 4.0)
        got = _d_dr(f(r), r, parity=parity)
        errors.append(float(np.abs(got[:2] - df(r)[:2]).max()))
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert all(o > 3.7 for o in orders), f"orders were {orders}, errors {errors}"

    # The one-sided fallback on the same data is second order, so its error
    # is worse by orders of magnitude at these resolutions.
    r = _radii(800, 4.0)
    one_sided = float(np.abs(_d_dr(f(r), r)[:2] - df(r)[:2]).max())
    assert one_sided > 1e3 * errors[-1], f"one-sided {one_sided:.3e} vs parity {errors[-1]:.3e}"


def test_parity_stencil_rejects_a_nonsense_parity():
    r = _radii(50, 4.0)
    with pytest.raises(ValueError, match="parity"):
        _d_dr(np.ones_like(r), r, parity=0)


def test_derivative_of_an_odd_field_comes_out_even_across_the_origin():
    """The property the ``2 f Phi / r`` term depends on.

    A one-sided stencil has no parity, so it returns a small *even*
    component for an odd field. Dividing that by ``r ~ dr/2`` is what turns
    a local accuracy loss into an instability, so the parity itself is
    worth pinning rather than only the order.
    """
    r = _radii(400, 4.0)
    odd = np.sin(1.3 * r) + 0.4 * r**3
    d = _d_dr(odd, r, parity=-1)
    # d is even, so its exact continuation satisfies d(-r_0) = d(r_0). The
    # reflected stencil at r_0 and r_1 must reproduce that to the scheme's
    # accuracy against the analytic even derivative.
    exact = 1.3 * np.cos(1.3 * r) + 1.2 * r**2
    assert abs(d[0] - exact[0]) < 1e-8
    assert abs(d[1] - exact[1]) < 1e-8

    one_sided = _d_dr(odd, r)
    assert abs(one_sided[0] - exact[0]) > 10 * abs(d[0] - exact[0])


def test_dissipation_reaches_the_origin_and_preserves_the_scheme_order():
    """Kreiss-Oliger must damp the innermost cells, not skip them.

    The stock operator returns zero within its stencil radius of either end.
    At the origin that leaves the shortest grid wavelength undamped exactly
    where the ``1/r`` terms amplify it. It must still annihilate the
    polynomials the scheme's order depends on.
    """
    r = _radii(200, 4.0)

    # A Nyquist-frequency ripple on an odd field is damped at every cell.
    ripple = (-1.0) ** np.arange(len(r))
    damped = _dissipate(ripple, float(r[1] - r[0]), -1, 0.1)
    assert np.all(np.abs(damped[:3]) > 0), "innermost cells were left undamped"

    # Degree-5 polynomials of the right parity are annihilated, so the
    # fourth-order scheme keeps its order.
    dr = float(r[1] - r[0])
    for parity, powers in ((1, (0, 2, 4)), (-1, (1, 3, 5))):
        for k in powers:
            out = _dissipate(r**k, dr, parity, 0.1)
            assert np.abs(out).max() < 1e-6 * max(1.0, np.abs(r**k).max()), (
                f"parity {parity} degree {k} was not annihilated"
            )


def test_ingoing_data_moves_toward_the_origin():
    """``ingoing=True`` must actually send the shell inward.

    In characteristic variables ``w+- = Phi +- Pi``, ``w+`` is the ingoing
    mode, so a purely ingoing pulse is ``Pi = +Phi``. The opposite sign
    sends the shell out through the boundary instead, and a collapse search
    built on it reports every amplitude as subcritical because the energy
    leaves before it can focus.
    """
    sim = ScalarCollapse(SphericalGrid(r_max=12.0, n=300), courant=0.25)
    r = sim.r

    def centroid(state):
        w = r**2 * (state.Phi**2 + state.Pi**2)
        return float((r * w).sum() / w.sum())

    st = gaussian_pulse(sim.grid, amplitude=1e-4, r0=5.0, width=1.0, ingoing=True)
    start = centroid(st)
    for _ in range(int(3.0 / sim.dt)):
        st = sim.step(st, sim.dt)
    moved = start - centroid(st)
    # Three units of time at the speed of light, less the spreading of the
    # tail; anything near zero or negative means the wrong characteristic.
    assert 2.0 < moved < 3.2, f"centroid moved {moved:+.3f}, expected inward by ~3"


def test_time_symmetric_data_has_no_initial_flux():
    sim = ScalarCollapse(SphericalGrid(r_max=12.0, n=200))
    st = gaussian_pulse(sim.grid, amplitude=1e-4, r0=5.0, width=1.0)
    np.testing.assert_array_equal(st.Pi, 0.0)


@pytest.mark.slow
@pytest.mark.parametrize("dissipation", (None, 0.0))
def test_the_origin_stays_quiet_long_after_the_pulse_has_left(dissipation):
    """Regression: an origin instability that only shows up late.

    A weak pulse passes through the origin and disperses through the outer
    boundary, and must leave essentially nothing behind. Three versions of
    this solver did not. With one-sided stencils at the two innermost cells a
    mode grew without bound after about two light-crossing times. With the
    stencils fixed, it still grew below a dissipation coefficient of about
    0.1, and at 0.02 the run ended with three times the ADM mass it began
    with. At 0.1 a residue of the pulse was left at the origin that fell with
    resolution, 1.6e-2 of the peak at 200 cells.

    All three were one defect, the ``Pi`` equation written as ``d(f Phi)/dr +
    2 f Phi / r`` rather than as ``(1/r^2) d(r^2 f Phi)/dr``; see
    :func:`test_the_flux_pair_conserves_the_discrete_energy_exactly`. In the
    conservative form nothing here depends on the dissipation: without any,
    the late activity is 3.9e-7 of the peak at 200 cells and at 400, the
    same to three figures, which is what says it is the solution's own tail
    rather than error. So the assertion is that it is small, that it is
    converged, and that no mass is made.
    """
    from particlesim.analysis.spherical_diagnostics import ricci_scalar

    def run(n: int):
        sim = ScalarCollapse(SphericalGrid(r_max=12.0, n=n), courant=0.25, dissipation=dissipation)
        st = gaussian_pulse(sim.grid, amplitude=6e-4, r0=5.0, width=1.0, ingoing=True)
        inner = sim.r < 1.0
        a0, _ = sim.solve_metric(st.Phi, st.Pi)
        m0 = sim.adm_mass(a0)

        central, peak = [], 0.0
        for k in range(int(22.0 / sim.dt)):
            st = sim.step(st, sim.dt)
            a, _ = sim.solve_metric(st.Phi, st.Pi)
            value = float(np.abs(ricci_scalar(a, st.Phi, st.Pi))[inner].max())
            peak = max(peak, value)
            if k % 100 == 0:
                central.append((st.t, value))

        assert np.isfinite(st.Phi).all() and np.isfinite(st.Pi).all()
        late = [v for t, v in central if t > 12.0]
        assert late, f"no late samples at n = {n}"
        a, _ = sim.solve_metric(st.Phi, st.Pi)
        final = sim.adm_mass(a)
        assert final < 0.05 * m0, f"mass was manufactured: {m0:.4f} -> {final:.4f}"
        return max(late) / peak

    coarse, fine = run(200), run(400)
    assert coarse < 1e-5, f"origin reawakened at 200 cells: {coarse:.3e} of peak"
    assert fine == pytest.approx(coarse, rel=0.05), (
        f"late activity {coarse:.3e} of peak at 200 cells, {fine:.3e} at 400: the "
        "solution's own tail converges, and anything else is error"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("solver", "conservative"), ((ScalarCollapse, True), (ExpandedFlux, False))
)
def test_a_near_critical_bounce_does_not_manufacture_mass(solver, conservative):
    """The same defect where it did the most damage: just below threshold.

    The thin shell of the collapse search, at 400 cells and 0.4% below its
    threshold, bounces through the origin on structure the grid barely
    resolves. Written expanded, the ``Pi`` equation then turned the
    grid-scale remains into a sawtooth across the forty innermost cells,
    holding a curvature near 9e3 and a lapse near 0.12 until the run ended,
    and finished with 75% more mass than it started with -- 43% with twice
    the dissipation. That is what the uniform-grid peaks near threshold had
    been measuring. In conservative form the mass can only leave, and does.
    """
    sim = solver(SphericalGrid(r_max=10.0, n=400), courant=0.25)
    st = gaussian_pulse(sim.grid, amplitude=8.45e-4, r0=4.0, width=0.5, ingoing=True)
    a0, _ = sim.solve_metric(st.Phi, st.Pi)
    m0 = sim.adm_mass(a0)
    for _ in range(int(12.0 / sim.dt)):
        st = sim.step(st, sim.dt)
    a, _ = sim.solve_metric(st.Phi, st.Pi)
    ratio = sim.adm_mass(a) / m0
    if conservative:
        assert ratio < 1.0, f"mass made: {ratio:.4f} of the initial"
    else:
        assert ratio > 1.2, f"the contrast should create mass, got {ratio:.4f}"
