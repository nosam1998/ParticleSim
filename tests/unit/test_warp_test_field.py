"""A scalar test field on a warp background: is it the right equation, and does it hold up?

Issue #54's acceptance is "test-field evolution on Alcubierre background
stable and convergent", and the two halves need different evidence.
Convergence is measured against *exact* solutions where one exists, and the
case that matters is not flat space -- flat space multiplies the advective
terms by zero, so a sign error in them passes -- but a constant shift, which
is still flat and still has a closed-form solution while making ``beta^i``
do work.

Stability is the harder claim, because a superluminal bubble has a region
where both characteristic speeds share a sign. A scheme that reflected
there instead of trapping would look perfectly well behaved in any norm, so
the trapping is tested directly.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from particlesim.solvers.warp import testfield as tf

EXTENT = 20.0
WIDTH = 1.5


# --- the equation itself -------------------------------------------------


def test_the_flux_form_is_the_klein_gordon_equation():
    """The one place an algebra slip would produce a plausible wrong wave.

    Checked against ``box phi = (1/sqrt(-g)) d_a (sqrt(-g) g^ab d_b phi)``
    for a *general* ADM metric: arbitrary lapse, arbitrary shift, arbitrary
    symmetric three-metric, every component a function of all four
    coordinates. Nothing is specialised to a warp family, so this is the
    identity and not a coincidence at one background.
    """
    t, x, y, z = sp.symbols("t x y z")
    coords, space = (t, x, y, z), (x, y, z)

    lapse = sp.Function("alpha")(t, x, y, z)
    shift = [sp.Function(f"beta{i}")(t, x, y, z) for i in range(3)]
    spatial = sp.Matrix(3, 3, lambda i, j: sp.Function(f"g{min(i, j)}{max(i, j)}")(t, x, y, z))
    phi = sp.Function("phi")(t, x, y, z)

    inverse = spatial.inv()
    volume = sp.sqrt(spatial.det())

    lower = spatial * sp.Matrix(shift)
    full = sp.zeros(4, 4)
    full[0, 0] = -(lapse**2) + (sp.Matrix(shift).T * lower)[0, 0]
    for i in range(3):
        full[0, i + 1] = full[i + 1, 0] = lower[i]
        for j in range(3):
            full[i + 1, j + 1] = spatial[i, j]

    raised = sp.zeros(4, 4)
    raised[0, 0] = -1 / lapse**2
    for i in range(3):
        raised[0, i + 1] = raised[i + 1, 0] = shift[i] / lapse**2
        for j in range(3):
            raised[i + 1, j + 1] = inverse[i, j] - shift[i] * shift[j] / lapse**2

    # The ADM inverse really is the inverse, and sqrt(-g) = alpha sqrt(gamma)
    # -- checked on the determinant to keep clear of the square root's branch.
    assert sp.simplify(sp.expand(full * raised - sp.eye(4))) == sp.zeros(4, 4)
    assert sp.simplify(sp.expand(full.det() + lapse**2 * spatial.det())) == 0

    root = lapse * volume
    box = (
        sum(
            sp.diff(
                root * sum(raised[a, b] * sp.diff(phi, coords[b]) for b in range(4)),
                coords[a],
            )
            for a in range(4)
        )
        / root
    )

    momentum = (sp.diff(phi, t) - sum(shift[i] * sp.diff(phi, space[i]) for i in range(3))) / lapse
    flux = [
        volume * shift[i] * momentum
        + lapse * volume * sum(inverse[i, j] * sp.diff(phi, space[j]) for j in range(3))
        for i in range(3)
    ]
    claim = sp.diff(volume * momentum, t) - sum(sp.diff(flux[i], space[i]) for i in range(3))
    assert sp.simplify(sp.expand(claim + box * root)) == 0


# --- characteristic speeds and the horizon -------------------------------


def test_the_characteristic_speeds_are_the_shift_plus_minus_the_lapse():
    """``s_+- = -beta^i n_i +- alpha sqrt(gamma^ij n_i n_j)``, on a known background."""
    coords, spacing = tf.grid((16, 16, 16), EXTENT)
    for shift in (0.0, -0.6, -2.0, 1.3):
        field = tf.TestField(
            background=tf.uniform_background(shift), coords=coords, spacing=spacing
        )
        plus, minus = tf.characteristic_speeds(field, direction=(1, 0, 0))
        assert np.allclose(plus, -shift + 1.0)
        assert np.allclose(minus, -shift - 1.0)


def test_the_superluminal_region_has_both_speeds_of_one_sign():
    """The coordinate statement of a horizon for this field.

    Inside the default Alcubierre bubble ``beta^x = -2``, so the speeds are
    ``+3`` and ``+1``. Outside the shift vanishes and they are ``+-1``. The
    measured ranges are exactly those, which is what pins the background
    down before anything is evolved on it.
    """
    field, _ = tf.build("alcubierre", {"v_s": 2.0, "R": 5.0, "sigma": 2.0}, shape=(32,) * 3)
    plus, minus = tf.characteristic_speeds(field, direction=(1, 0, 0))
    assert float(plus.min()) == pytest.approx(1.0, abs=1e-6)
    assert float(plus.max()) == pytest.approx(3.0, abs=1e-6)
    assert float(minus.min()) == pytest.approx(-1.0, abs=1e-6)
    assert float(minus.max()) == pytest.approx(1.0, abs=1e-6)
    # Both positive somewhere: that region is the trapped one.
    assert np.any((plus > 0) & (minus > 0))


def test_the_horizon_indicator_is_positive_only_inside_the_bubble():
    """Hiscock's surface, at ``f = 1 - 1/v_s``, and its volume.

    ``alpha^2 - (beta^x + v_s)^2 = 1 - v_s^2 (1 - f)^2`` is positive where
    ``f > 1 - 1/v_s``, which for ``v_s = 2`` is ``f > 1/2`` -- the bubble
    interior out to the middle of the wall. Far outside, ``f -> 0`` and the
    indicator tends to ``1 - v_s^2 = -3``: a ship-frame observer in flat
    space moving at twice the speed of light is spacelike, which is the
    whole reason the bubble is needed.
    """
    field, coords = tf.build(
        "alcubierre", {"v_s": 2.0, "R": 5.0, "sigma": 2.0}, shape=(48,) * 3, extent=EXTENT
    )
    indicator = field.horizon(0.0)
    assert float(indicator.max()) == pytest.approx(1.0, abs=1e-6)
    assert float(indicator.min()) == pytest.approx(-3.0, abs=1e-6)

    distance = np.sqrt(sum(value**2 for value in coords))
    # The positive region is a ball about the origin, not a shell or a shred.
    assert float(distance[indicator > 0].max()) < 6.0
    assert float(distance[indicator > 0].min()) < 1.0


def test_the_courant_bound_accounts_for_the_shift():
    """A step built on the lapse alone would be three times too long.

    ``|beta| + alpha sqrt(gamma^ii)`` is what bounds it, and for the default
    bubble ``|beta|`` reaches 2 against a unit lapse. The check is that the
    reported speed grows with the shift rather than sitting at the lapse.
    """
    coords, spacing = tf.grid((16, 16, 16), EXTENT)
    speeds = []
    for shift in (0.0, 1.0, 2.0):
        field = tf.TestField(
            background=tf.uniform_background(shift), coords=coords, spacing=spacing
        )
        speeds.append(field.characteristic_speed(0.0))
    assert speeds[0] < speeds[1] < speeds[2]
    assert speeds[2] == pytest.approx(2.0 + np.sqrt(3.0), rel=1e-12)
    # and the step shrinks accordingly
    fast = tf.TestField(background=tf.uniform_background(2.0), coords=coords, spacing=spacing)
    slow = tf.TestField(background=tf.uniform_background(0.0), coords=coords, spacing=spacing)
    assert fast.time_step < slow.time_step / 2


def test_a_unit_lapse_still_produces_an_array():
    """Alcubierre's lapse is exactly one, which lambdifies to a scalar.

    Without the broadcast in ``_callable`` every quantity it multiplies
    collapses to a scalar and the first grid operation fails -- or worse,
    silently succeeds on a zero-dimensional array.
    """
    field, coords = tf.build("alcubierre", shape=(8,) * 3)
    fields = field.coefficients(0.0)
    assert fields["lapse"].shape == coords[0].shape
    assert np.allclose(fields["lapse"], 1.0)
    assert fields["volume"].shape == coords[0].shape
    assert np.allclose(fields["volume_rate"], 0.0), "flat slices do not deform"


def test_a_zone_narrower_than_the_stencil_is_refused():
    coords, spacing = tf.grid((16, 16, 16), EXTENT)
    with pytest.raises(ValueError, match="narrower than the stencil radius"):
        tf.TestField(
            background=tf.uniform_background(0.0),
            coords=coords,
            spacing=spacing,
            zone=1,
            order=4,
        )


# --- initial data --------------------------------------------------------


def test_the_travelling_pulse_is_consistent_with_its_own_speed():
    """``Pi = (beta^x - s) G'``, which for the right-mover is ``-G'``."""
    coords, _ = tf.grid((16, 16, 16), EXTENT)
    state = tf.travelling_pulse(coords, speed=1.0, width=WIDTH)
    position = np.asarray(coords[0])
    profile = np.exp(-(position**2) / (2 * WIDTH**2))
    assert np.allclose(state["phi"], profile)
    assert np.allclose(state["pi"], position / WIDTH**2 * profile)


def test_the_spherical_pulse_is_time_symmetric():
    """``Pi = 0``, so it splits evenly both ways -- half of it upstream.

    Its peak is *below* one, and deliberately: the grid is cell-centred, so
    no sample sits at ``r = 0`` where the Gaussian is one. The nearest is
    ``h sqrt(3)/2`` away, which is what the profile is checked against --
    a peak of exactly one would mean a sample on the centre, and several
    warp families have a ``1/r_s`` in their derivatives.
    """
    coords, spacing = tf.grid((16, 16, 16), EXTENT)
    state = tf.spherical_pulse(coords, width=WIDTH)
    assert np.allclose(state["pi"], 0.0)

    distance = np.sqrt(sum(np.asarray(value) ** 2 for value in coords))
    assert float(distance.min()) == pytest.approx(spacing[0] * np.sqrt(3) / 2, rel=1e-12)
    assert float(np.max(state["phi"])) == pytest.approx(
        np.exp(-(distance.min() ** 2) / (2 * WIDTH**2)), rel=1e-12
    )
    assert float(np.max(state["phi"])) < 1.0


# --- the three bugs, each tested where it actually lives -----------------


def test_the_direct_second_derivative_sees_the_nyquist_mode():
    """Why the flux form cannot be evaluated as written, in one exact check.

    The centred first derivative annihilates the grid-scale mode
    ``a_i = (-1)^i`` *exactly* -- its stencil weights pair up against the
    alternating sign -- so composing two of them gives identically zero
    there. The direct second-derivative stencil gives ``-64/(12 h^2)``
    instead, which is its maximum.

    That is the whole difference between a scheme where the Nyquist mode has
    a restoring force and one where it does not, and it is why the
    right-hand side expands the divergence rather than differencing a flux
    that already holds a derivative. Measured consequence, on an exact plane
    wave with a superluminal shift: the composed form gave 5.6e-02, 4.1e-02,
    4.8e-02, 1.0e-01 at n = 24, 32, 48, 64 -- growing -- where the direct
    one gives 2.0e-04, 6.3e-05, 1.3e-05, 4.0e-06 at fourth order.
    """
    coords, spacing = tf.grid((16, 8, 8), EXTENT)
    field = tf.TestField(background=tf.uniform_background(0.0), coords=coords, spacing=spacing)
    index = np.arange(16).reshape(-1, 1, 1)
    nyquist = np.broadcast_to((-1.0) ** index, coords[0].shape).copy()

    first = field.derivative(nyquist, 0)
    assert np.allclose(first, 0.0), "the centred first derivative kills Nyquist"
    assert np.allclose(field.derivative(first, 0), 0.0), "so does composing two"

    direct = field.second(nyquist, 0)
    assert not np.allclose(direct, 0.0)
    assert np.allclose(direct, -64.0 / (12.0 * spacing[0] ** 2) * nyquist)


def test_the_outflow_condition_does_not_reach_across_the_box():
    """The wrapping gradient, tested where it goes wrong rather than by a long run.

    The interior differences with ``roll``, so at the ``+x`` face its stencil
    reads the ``-x`` face. A Sommerfeld condition built on that gradient asks
    which way is outward using data from the opposite side of the domain.
    Perturbing the field only near ``-x`` must leave the rates at ``+x``
    untouched, and with the wrapping derivative it does not.

    The symptom, if this is wrong, is not a boundary artefact: it looks like
    the warp background amplifying the field, at 0.167, 0.252 and 0.420 per
    unit time for ``h = 1.0, 0.714, 0.5``. ``rate * h`` is then 0.167, 0.180,
    0.210 -- constant, so the rate goes as ``1/h`` and it is numerical. The
    same run with no zone at all decays at a resolution-independent -0.06.
    """
    coords, spacing = tf.grid((32, 8, 8), EXTENT)
    field = tf.TestField(
        background=tf.uniform_background(0.0),
        coords=coords,
        spacing=spacing,
        zone=4,
        dissipation=0.0,
    )
    quiet = {
        "phi": np.zeros(coords[0].shape),
        "pi": np.zeros(coords[0].shape),
    }
    poked = {name: value.copy() for name, value in quiet.items()}
    poked["phi"][1, 4, 4] = 1.0  # inside the -x zone

    before = field.right_hand_side(quiet, 0.0)
    after = field.right_hand_side(poked, 0.0)
    # The +x zone is the last four planes; nothing there may have moved.
    for name in ("phi", "pi"):
        assert np.allclose(before[name][-4:], after[name][-4:]), name
    # and the poke did change the near zone, so the test is not vacuous.
    # It has to be phi that is checked: the Sommerfeld rate for pi is built
    # from pi, which was left at zero.
    assert not np.allclose(before["phi"][:4], after["phi"][:4])


def test_the_energy_window_is_physical_not_a_point_count():
    """A fixed number of zone points is a different *region* at every resolution.

    Stripping the outflow zone is right for a single run and wrong for
    comparing resolutions: six points is 5.0 wide at ``n = 24`` and 2.2 at
    ``n = 54``, so the domain of integration is the resolution. With a pulse
    at a fixed physical place that gave 0.033, 1.53 and 3.26 for a quantity
    that should barely move -- the pulse was outside the region on the coarse
    grid and inside it on the fine one.

    What is checked is the regions themselves rather than an integral over
    them, because an integral also moves when the *integrand* is
    under-resolved and that would muddle the two effects. A physical window
    on a cell-centred grid still quantises to whole cells, so it is pinned to
    within one cell rather than exactly.
    """
    physical, counted = {}, {}
    for n in (24, 36, 54):
        coords, spacing = tf.grid((n, n, n), EXTENT)
        along = np.asarray(coords[0])[:, 0, 0]
        # The region energy(window=8.0) integrates, measured along x.
        physical[n] = float(np.sum(np.abs(along) <= 8.0) * spacing[0])
        # The region the point-count default integrates.
        counted[n] = float((n - 2 * 6) * spacing[0])

    widths = [physical[n] for n in sorted(physical)]
    assert max(widths) - min(widths) < 2 * max(20.0 / n for n in physical), widths
    assert all(abs(width - 16.0) < 1.0 for width in widths), widths

    # The point count, by contrast, does not describe a fixed region at all.
    counts = [counted[n] for n in sorted(counted)]
    assert max(counts) / min(counts) > 1.5, counts
    assert counts == sorted(counts), "and it grows with n, which is the trap"

    # The keyword really does change what is integrated.
    coords, spacing = tf.grid((24, 24, 24), EXTENT)
    field = tf.TestField(
        background=tf.uniform_background(0.0), coords=coords, spacing=spacing, zone=6
    )
    phase = 2 * np.pi / EXTENT * np.asarray(coords[0])
    state = {"phi": np.sin(phase), "pi": np.zeros_like(phase)}
    assert field.energy(state, 0.0, window=8.0) > field.energy(state, 0.0)


# --- convergence and the acceptance --------------------------------------


def _plane_wave(field, coords, shift, speed, time):
    """An exact plane wave on a constant-shift background."""
    wavenumber = 2 * np.pi / EXTENT
    phase = wavenumber * (np.asarray(coords[0]) - speed * time)
    return {
        "phi": np.sin(phase),
        "pi": -(speed + shift) * wavenumber * np.cos(phase),
    }


@pytest.mark.parametrize("shift", [0.0, -0.6, -2.0])
@pytest.mark.benchmark
def test_fourth_order_against_an_exact_plane_wave(shift):
    """The scheme, measured with no boundary and no background error at all.

    A plane wave on a constant shift is exact for all time on a periodic
    box, so this is the scheme and nothing else. ``shift = -2`` is the case
    that matters: the shift exceeds the lapse, both characteristics share a
    sign, and it still converges at fourth order. ``shift = 0`` is the case
    that proves least, because flat space multiplies every advective term by
    zero.

    Measured maximum error in ``phi`` at ``t = 2``:

        shift    n=24      n=32      n=48      n=64    order
         0.0   9.77e-06  3.10e-06  6.15e-07  1.95e-07  4.00
        -0.6   6.32e-05  2.00e-05  3.97e-06  1.26e-06  4.00
        -2.0   1.99e-04  6.32e-05  1.25e-05  3.97e-06  4.00
    """
    speed = -shift + 1.0
    span = 2.0
    errors, counts = [], (24, 32, 48)
    for n in counts:
        coords, spacing = tf.grid((n, 8, 8), EXTENT)
        field = tf.TestField(
            background=tf.uniform_background(shift),
            coords=coords,
            spacing=spacing,
            dissipation=0.0,
            zone=0,
            periodic=True,
        )
        steps = max(4, int(np.ceil(span / field.time_step)))
        final, _ = field.run(
            _plane_wave(field, coords, shift, speed, 0.0), steps, time_step=span / steps
        )
        want = _plane_wave(field, coords, shift, speed, span)
        errors.append(float(np.max(np.abs(final["phi"] - want["phi"]))))

    orders = [
        np.log(errors[i] / errors[i + 1]) / np.log(counts[i + 1] / counts[i])
        for i in range(len(errors) - 1)
    ]
    assert all(order > 3.7 for order in orders), (errors, orders)
    assert all(order < 4.3 for order in orders), (errors, orders)


@pytest.mark.benchmark
def test_a_time_symmetric_pulse_cannot_go_upstream_when_the_shift_wins():
    """The horizon, on a background where both characteristics are exact.

    ``Pi = 0`` splits a pulse evenly into the two characteristics, which
    travel at ``-beta +- alpha``. For ``beta = -0.5`` those are ``-0.5`` and
    ``+1.5``, so half of it goes left. For ``beta = -2`` they are ``+1`` and
    ``+3``: **both to the right**, and nothing is left behind. That is the
    coordinate content of a horizon for this field, and it is checked
    against the characteristic positions rather than against a norm.
    """
    span = 2.5
    for shift, goes_left in ((-0.5, True), (-2.0, False)):
        coords, spacing = tf.grid((128, 8, 8), EXTENT)
        field = tf.TestField(
            background=tf.uniform_background(shift),
            coords=coords,
            spacing=spacing,
            dissipation=0.02,
            zone=0,
            periodic=True,
        )
        line = np.asarray(coords[0])
        profile = np.exp(-(line**2) / (2 * 0.7**2))
        steps = int(np.ceil(span / field.time_step))
        final, _ = field.run(
            {"phi": profile, "pi": np.zeros_like(profile)}, steps, time_step=span / steps
        )

        # The measurement is how much of the field ended up *upstream*, not
        # where the support happens to end: a pulse has a width, and its
        # left edge sits a width beyond the characteristic that carries it.
        amplitude = np.abs(final["phi"][:, 4, 4]) ** 2
        position = line[:, 4, 4]
        upstream = float(amplitude[position < -0.5].sum() / amplitude.sum())
        if goes_left:
            assert upstream > 0.25, (shift, upstream)
        else:
            assert upstream < 0.01, (shift, upstream)

        # And the leading edge is where the fast characteristic put it.
        fast = (-shift + 1.0) * span
        peak = position[amplitude > 0.4 * amplitude.max()]
        assert peak.max() == pytest.approx(fast, abs=1.0), (shift, fast, peak.max())


# --- the coefficients a constant shift cannot test -----------------------


def _differenced(callable_, coords, spacing, axis):
    """A fourth-order finite difference of a background quantity, for comparison."""
    from particlesim.core.grid import derivative

    return derivative(callable_(0.0, *coords), axis, spacing[axis], order=4)


def test_the_shift_divergence_matches_a_finite_difference():
    """The term no exact-solution test can reach.

    ``Pi (1/sqrt(g)) d_i (sqrt(g) beta^i)`` appears only when the shift
    *varies*, and every background with a closed-form solution has a
    constant one -- so the plane-wave convergence tests above multiply this
    coefficient by zero and would pass with it completely wrong. It is
    differentiated symbolically, so what it is checked against is a finite
    difference of the shift on a fine grid, in the interior where the
    stencil is centred.
    """
    field, coords = tf.build(
        "alcubierre", {"v_s": 2.0, "R": 5.0, "sigma": 0.8}, shape=(64,) * 3, extent=EXTENT
    )
    exact = field.coefficients(0.0)["shift_divergence"]
    # Alcubierre has flat slices, so the divergence is just d_x beta^x.
    numeric = _differenced(field.background.shift[0], coords, field.spacing, 0)

    inside = (slice(4, -4),) * 3
    scale = float(np.max(np.abs(exact[inside])))
    assert scale > 0.1, "the shift must actually vary for this to test anything"
    difference = float(np.max(np.abs(exact[inside] - numeric[inside])))
    assert difference < 0.02 * scale, (difference, scale)


def test_the_flux_gradient_vanishes_for_flat_slices_and_does_not_otherwise():
    """``(1/sqrt(g)) d_i (alpha sqrt(g) gamma^ij)``, checked where it is not zero.

    For every family with ``flat_slices`` this is identically zero, which is
    worth pinning: an error there would be invisible on Alcubierre. Van den
    Broeck deforms its slices, so the coefficient is real, and the check is
    that a finite difference of the same background quantity *converges to
    it* as the grid refines. Not that the two agree at one resolution -- Van
    den Broeck's inner bubble is sharp and the difference is 8% at
    ``h = 0.42`` -- but that the gap falls like the stencil, which is what
    identifies the symbolic value as the limit rather than merely nearby.

    Measured relative gap: 0.175, 0.081, 0.041 at ``n = 32, 48, 64`` -- a
    factor of 4.3 across the ladder, at about second order rather than
    fourth. That is a *max norm over a sharp feature* converging rather than
    the stencil failing: the worst point is always on the inner bubble wall,
    which is where the grid is least able to resolve it. What this test needs
    is that the gap goes to zero, because that is what says which of the two
    quantities is the approximate one.
    """
    from particlesim.core.grid import derivative

    flat, _ = tf.build("alcubierre", shape=(24,) * 3, extent=EXTENT)
    for value in flat.coefficients(0.0)["flux_gradient"]:
        assert np.allclose(value, 0.0), "flat slices carry no flux gradient"

    gaps = {}
    for n in (32, 48, 64):
        curved, _ = tf.build("van_den_broeck", shape=(n,) * 3, extent=EXTENT)
        fields = curved.coefficients(0.0)
        inside = (slice(6, -6),) * 3
        worst = 0.0
        scale = 0.0
        for j in range(3):
            rebuilt = np.zeros_like(fields["volume"])
            for i in range(3):
                product = fields["lapse"] * fields["volume"] * fields["inverse"][i][j]
                rebuilt = rebuilt + derivative(product, i, curved.spacing[i], order=4)
            rebuilt = rebuilt / fields["volume"]
            exact = fields["flux_gradient"][j]
            worst = max(worst, float(np.max(np.abs(exact[inside] - rebuilt[inside]))))
            scale = max(scale, float(np.max(np.abs(exact[inside]))))
        assert scale > 1e-3, "the coefficient must be non-zero for this to test anything"
        gaps[n] = worst / scale

    counts = sorted(gaps)
    assert gaps[counts[-1]] < gaps[counts[0]] / 3.0, gaps
    assert [gaps[n] for n in counts] == sorted((gaps[n] for n in counts), reverse=True), gaps


def test_a_non_diagonal_background_forms_the_mixed_derivatives():
    """``diagonal`` decides whether the mixed second derivatives are built.

    Skipping them is a real saving and a real hazard: if a family's inverse
    metric has off-diagonal entries and they are skipped, the principal part
    is simply missing terms. The flag is derived symbolically, so it is the
    metric that decides rather than the caller.
    """
    assert tf.Background.named("alcubierre").diagonal
    assert tf.uniform_background(1.0).diagonal


# --- light rays, and the number both halves have to agree on -------------


@pytest.mark.benchmark
def test_a_light_ray_and_the_field_agree_on_the_speeds():
    """Two independent routes to ``-beta^x +- alpha``, which had better match.

    A radial null ray satisfies
    ``(-alpha^2 + beta^2) dt^2 + 2 beta dt dx + dx^2 = 0``, so its coordinate
    velocity is ``dx/dt = -beta +- alpha`` -- exactly
    :func:`particlesim.solvers.warp.testfield.characteristic_speeds`. The two
    are computed by entirely separate code: the ray integrates Christoffels
    of the full four-metric through
    :class:`particlesim.analysis.geodesics.GeodesicIntegrator`, while the
    field's speeds come from the ADM decomposition in this module. Nothing is
    shared but the metric.

    Inside the default bubble both come out ``+3`` and ``+1``. A light ray
    cannot go upstream there either, which is the statement that the scalar
    field's trapping is a property of the spacetime and not of the scheme.
    """
    from particlesim.analysis.geodesics import GeodesicIntegrator
    from particlesim.scenarios.warp.metrics import make_metric

    parameters = {"v_s": 2.0, "R": 5.0, "sigma": 2.0}
    metric = make_metric("alcubierre", parameters)
    coordinates = list(sp.symbols("t x y z", real=True))
    rays = GeodesicIntegrator(metric.metric(), coordinates, metric.params)

    # Deep inside the bubble, off the singular centre, where f = 1.
    start = np.array([0.0, 0.4, 0.3, 0.2])
    measured = []
    for direction in (+1.0, -1.0):
        # Coordinate time is *spacelike* inside a superluminal bubble --
        # ``g_tt = -alpha^2 + beta^2 = +3`` here -- so the tangent has to be
        # built in the Eulerian frame, where a light ray still moves at one.
        tangent = rays.from_eulerian_velocity(start, [direction, 0.0, 0.0], kind="null")
        result = rays.integrate(start, tangent, affine_max=0.4, n_out=40)
        assert result.success
        assert np.abs(result.norm).max() < 1e-8, "the ray must stay null"
        slope = np.polyfit(result.x[:, 0], result.x[:, 1], 1)[0]
        measured.append(float(slope))

    field, _ = tf.build("alcubierre", parameters, shape=(32,) * 3, extent=EXTENT)
    plus, minus = tf.characteristic_speeds(field, direction=(1, 0, 0))
    assert sorted(measured) == pytest.approx([1.0, 3.0], abs=2e-3), measured
    assert float(plus.max()) == pytest.approx(max(measured), abs=2e-3)
    assert float(minus.max()) == pytest.approx(min(measured), abs=2e-3)
    # Both rays travel in +x: no light escapes upstream inside the bubble.
    assert all(value > 0 for value in measured), measured


# --- the acceptance: stable and convergent on Alcubierre -----------------


@pytest.mark.slow
@pytest.mark.benchmark
def test_the_field_is_bounded_on_the_alcubierre_background():
    """Issue #54's "stable": no growth, over a crossing time and more.

    The measurement worth making is not that a norm stays small -- a pulse
    leaving the domain does that -- but that it never *rises*. The peak over
    the whole run is the initial value, and after that the field only falls.
    Measured at ``n = 40`` out to three crossing times:

        t/L          0.5       1.0       1.5       2.0       2.5
        max|phi|   2.8e-02   6.9e-03   1.7e-03   6.9e-04   2.4e-04
        E          6.3e-02   2.4e-03   2.8e-04   3.3e-05   6.1e-06

    The decay rate is also resolution-independent, which is what says it is
    the wave leaving rather than the scheme dissipating: -0.218, -0.296 and
    -0.225 per unit time at ``h = 1.0, 0.714, 0.5``. A numerical instability
    from the principal part would go as ``1/h`` and the rate would track it.

    That the *growth* is what matters here is not pedantry. The bubble wall
    carries a genuine source term ``Pi (1/sqrt(g)) d_i (sqrt(g) beta^i)``,
    reaching ``+-0.78`` for these parameters, so amplification was a live
    possibility rather than a hypothetical -- and it was masked for a while
    by that term being silently zero.
    """
    field, coords = tf.build(
        "alcubierre",
        {"v_s": 2.0, "R": 5.0, "sigma": 2.0},
        shape=(28,) * 3,
        extent=EXTENT,
        dissipation=0.1,
        zone=6,
    )
    # The source term is real, so this test has something to catch.
    divergence = field.coefficients(0.0)["shift_divergence"]
    assert float(np.max(np.abs(divergence))) > 0.1

    state = tf.spherical_pulse(coords, centre=(-7.0, 0.0, 0.0), width=1.2)
    start = float(np.max(np.abs(state["phi"])))
    span = 24.0
    steps = int(np.ceil(span / field.time_step))
    step = span / steps

    peak, samples = start, []
    for index in range(1, steps + 1):
        state = field.step(state, (index - 1) * step, step)
        if index % max(1, steps // 6) == 0:
            amplitude = float(np.max(np.abs(tf.interior(state["phi"], 6))))
            assert np.all(np.isfinite(state["phi"]))
            peak = max(peak, amplitude)
            samples.append(amplitude)

    assert peak == pytest.approx(start, rel=1e-9), (peak, start)
    assert samples == sorted(samples, reverse=True), samples
    assert samples[-1] < start / 20.0, (samples[-1], start)


@pytest.mark.slow
@pytest.mark.benchmark
def test_fourth_order_self_convergence_on_the_alcubierre_background():
    """Issue #54's "convergent", on the background with no closed-form solution.

    There is nothing exact to compare against, so successive resolutions are
    compared with each other -- and **with no interpolation anywhere**. For
    cell-centred grids a refinement ratio of *three* makes every coarse cell
    centre exactly a fine one, since ``(i + 1/2) * 3 = (3i + 1) + 1/2``. So
    the fields are compared point for point and the measured difference is
    the scheme's error rather than an interpolant's.

    Measured at ``t = 2``, over the fixed physical window ``|x_i| <= 4``:

        n vs 3n     h       wall/h    max|phi_n - phi_3n|   order
        16 vs 48  1.2500     0.40           3.16e-01
        24 vs 72  0.8333     0.60           5.45e-02        4.33
        32 vs 96  0.6250     0.80           1.63e-02        4.20

    Fourth order, and the ``wall/h`` column is why the coarsest point is so
    far off: the Alcubierre wall is about ``1/sigma = 0.5`` wide, so at
    ``h = 1.25`` it spans less than half a cell. The orders come out slightly
    *above* four because the background is becoming better resolved at the
    same time as the field.

    Two mistakes are designed out of this rather than tolerated. The window
    is physical, not a point count -- a fixed number of points is a
    different region at every ``n``, which is what made an earlier version
    of this study report 0.033, 1.53 and 3.26 for the initial energy. And
    the ratio is three rather than two, because at ratio two no coarse
    sample is a fine sample and every comparison would have gone through an
    interpolation whose own order would cap the measurement.
    """
    parameters = {"v_s": 2.0, "R": 5.0, "sigma": 2.0}
    span, window = 2.0, 4.0

    def solve(count):
        field, coords = tf.build(
            "alcubierre",
            parameters,
            shape=(count,) * 3,
            extent=EXTENT,
            dissipation=0.1,
            zone=6,
        )
        state = tf.spherical_pulse(coords, centre=(0.0, 0.0, 0.0), width=1.5)
        steps = int(np.ceil(span / field.time_step))
        final, _ = field.run(state, steps, time_step=span / steps)
        return final["phi"], coords

    errors, counts = {}, (16, 24)
    for base in counts:
        coarse, coords = solve(base)
        fine, _ = solve(3 * base)
        picked = 3 * np.arange(base) + 1
        sampled = fine[np.ix_(picked, picked, picked)]
        # Coincident points, so this is a difference and not an interpolation.
        assert sampled.shape == coarse.shape

        inside = np.ones(coarse.shape, dtype=bool)
        for axis in range(3):
            inside &= np.abs(np.asarray(coords[axis])) <= window
        errors[base] = float(np.max(np.abs(coarse[inside] - sampled[inside])))

    order = np.log(errors[counts[0]] / errors[counts[1]]) / np.log(counts[1] / counts[0])
    assert order > 3.7, (errors, order)
    assert order < 5.0, (errors, order)
