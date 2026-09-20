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
def test_the_origin_stays_quiet_long_after_the_pulse_has_left():
    """Regression: an origin instability that only shows up late.

    With one-sided stencils at the two innermost cells and no dissipation
    on the innermost three, a weak pulse would pass through the origin,
    disperse, and then leave behind a mode that grew without bound. It took
    roughly two light-crossing times to become visible, so every short test
    passed while any long run -- which is what a threshold search needs --
    was destroyed.

    The sharpest symptom is the ADM mass. This pulse disperses through the
    outer boundary and must leave essentially nothing behind, so a run that
    ends heavier than it started has manufactured mass out of grid noise.
    At the old default coefficient of 0.02 it ended with three times the
    mass it began with.

    **The criterion is convergence, not a threshold at one resolution, and
    that distinction was bought the hard way.** An earlier version of this
    test ran only at 200 cells and required the late origin activity to sit
    below a hundredth of the peak. It did, by a factor of four and a half --
    but a residue of the pulse grows there at that resolution whatever the
    metric solve does, so the margin was measuring how large the transient
    happened to be seeded, not whether anything was unstable. Correcting an
    unrelated second-order error in the lapse's midpoint mass moved the seed
    by a factor of ten and the test failed, with nothing wrong.

    What separates a transient from an instability is refinement. At 400
    cells the late activity is eight orders below the peak and still falling,
    and the two metric solves agree to three figures. An instability does not
    do that: the original one grew without bound and refining did not touch
    it. So the run is done twice and the assertion is that the activity
    collapses when the grid is refined.
    """
    from particlesim.analysis.spherical_diagnostics import ricci_scalar

    def run(n: int):
        sim = ScalarCollapse(SphericalGrid(r_max=12.0, n=n), courant=0.25)
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
        return max(late) / peak, m0, sim.adm_mass(a)

    coarse, m0, final = run(200)
    assert final < 0.05 * m0, f"mass was manufactured: {m0:.4f} -> {final:.4f}"
    # Bounded at the coarse resolution: the instability this guards against
    # reached three times the initial mass, not a few percent of the peak.
    assert coarse < 0.1, f"origin reawakened at 200 cells: {coarse:.3e} of peak"

    fine, m0, final = run(400)
    assert final < 0.05 * m0, f"mass was manufactured: {m0:.4f} -> {final:.4f}"
    assert fine < coarse / 20.0, (
        f"the late origin activity did not converge away: {coarse:.3e} of peak at 200 "
        f"cells, {fine:.3e} at 400. A transient falls steeply here; an instability does not"
    )
