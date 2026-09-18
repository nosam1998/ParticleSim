"""Horizons on a 3-D slice, against a spacetime whose answers are exact.

Schwarzschild in isotropic coordinates gives three numbers with no error
bars: the apparent horizon sits at ``r = M/2``, its area is ``16 pi M^2``,
and so its irreducible mass is exactly ``M``. The event horizon is the same
surface, and outgoing null surfaces converge onto it backwards in time at
the surface gravity ``1/(4M)`` -- which is a fourth exact number, and the
only one here that is not a statement about a single slice.

The cheap tests use the analytic expansion ``psi^-2 (2/r + 4 psi'/psi)`` for
a coordinate sphere, which is worth having written down: it is what caught
that the expansion itself was right while the *flow* around it was not.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.analysis import horizon
from particlesim.analysis.horizon import Slice

MASS = 1.0
EXTENT = 4.0


def _analytic_expansion(radius: float, mass: float = MASS) -> float:
    """``Theta`` of a coordinate sphere in isotropic Schwarzschild.

    ``Theta = psi^-2 (2/r + 4 psi'/psi)`` with ``psi = 1 + M/2r``, which
    vanishes at ``r = M/2`` -- set ``2/r = 2M/(r^2 psi)`` and the ``psi``
    cancels to leave ``r = M - M/2``.
    """
    conformal = 1.0 + mass / (2.0 * radius)
    slope = -mass / (2.0 * radius**2)
    return conformal**-2 * (2.0 / radius + 4.0 * slope / conformal)


def _sphere(slice_: horizon.Slice, radius: float):
    """The level set of a coordinate sphere, and its points on a ray grid."""
    coords = slice_.coordinates()
    field = np.sqrt(sum(value**2 for value in coords)) - radius
    theta, phi, _, _ = horizon.angular_grid(12, 24)
    points = [radius * value for value in horizon._directions(theta, phi)]
    return field, points


# --- the slice ----------------------------------------------------------


def test_the_exact_slice_has_the_geometry_it_claims():
    """Conformally flat, time-symmetric, with the static lapse."""
    slice_ = horizon.schwarzschild_slice(shape=(24, 24, 24), extent=EXTENT, mass=MASS)
    coords = slice_.coordinates()
    radius = np.sqrt(sum(value**2 for value in coords))
    conformal = 1.0 + MASS / (2.0 * radius)

    assert np.allclose(slice_.fields["gamma00"], conformal**4)
    assert np.allclose(slice_.fields["gamma01"], 0.0)
    assert np.allclose(slice_.fields["K00"], 0.0), "time-symmetric data"
    assert np.allclose(
        slice_.fields["alpha"],
        (1.0 - MASS / (2.0 * radius)) / (1.0 + MASS / (2.0 * radius)),
    )
    # The inverse really is the inverse.
    inverse = slice_.inverse_metric()
    assert np.allclose(inverse[0][0] * conformal**4, 1.0)
    assert np.allclose(inverse[0][1], 0.0)


def test_a_slice_missing_the_curvature_is_refused():
    """A BSSN state is not a slice, and the error says which to pass."""
    with pytest.raises(ValueError, match="physical_slice_arrays"):
        Slice(fields={"gamma00": np.ones((4, 4, 4))}, spacing=(0.1, 0.1, 0.1))


def test_a_puncture_on_a_grid_point_is_refused():
    """``psi`` is infinite there, and an *even* shape unstaggered does it.

    The grid runs from zero, so the box centre ``extent/2`` is
    ``(n/2) * spacing`` -- a sample exactly when ``n`` is even. An odd shape
    puts the centre half a cell off on its own, which is why the refusal
    names staggering *or* the shape.
    """
    with pytest.raises(ValueError, match="exactly on the puncture"):
        horizon.schwarzschild_slice(shape=(24, 24, 24), extent=EXTENT, stagger=False)
    # Odd and unstaggered is fine: 2.0 is not a multiple of 4/25.
    horizon.schwarzschild_slice(shape=(25, 25, 25), extent=EXTENT, stagger=False)


# --- the expansion ------------------------------------------------------


@pytest.mark.parametrize("radius", [0.6, 0.8, 1.2, 1.5])
def test_the_expansion_matches_the_analytic_value_outside(radius):
    """Within a percent at a spacing of 1/24, and it is the sign that matters.

    Outside the horizon the outward null expansion is positive: the surface
    is expanding, light gets away. A sign error here inverts the flow and
    every trial surface leaves the grid.
    """
    slice_ = horizon.schwarzschild_slice(shape=(96, 96, 96), extent=EXTENT, mass=MASS)
    field, points = _sphere(slice_, radius)
    sampled = horizon._sample(horizon.expansion(field, slice_), points, slice_)
    expected = _analytic_expansion(radius)
    assert expected > 0.0
    assert float(np.mean(sampled)) == pytest.approx(expected, rel=2e-3)


@pytest.mark.parametrize("radius", [0.3, 0.4])
def test_the_expansion_is_negative_inside_the_horizon(radius):
    """Trapped: even the outgoing congruence converges."""
    slice_ = horizon.schwarzschild_slice(shape=(96, 96, 96), extent=EXTENT, mass=MASS)
    field, points = _sphere(slice_, radius)
    sampled = horizon._sample(horizon.expansion(field, slice_), points, slice_)
    expected = _analytic_expansion(radius)
    assert expected < 0.0
    assert float(np.mean(sampled)) == pytest.approx(expected, rel=1e-2)


def test_the_expansion_vanishes_on_the_exact_horizon():
    """The number the whole module exists to find, evaluated directly.

    At ``r = M/2`` the analytic expansion is exactly zero, and on the grid it
    converges there: 1.7e-3, 5.9e-4, 1.1e-4, 3.7e-5 at n = 48, 64, 96, 128.
    That is what sets the floor on any tolerance the flow can be asked for.
    """
    previous = None
    for count in (48, 64, 96):
        slice_ = horizon.schwarzschild_slice(shape=(count,) * 3, extent=EXTENT, mass=MASS)
        field, points = _sphere(slice_, MASS / 2)
        sampled = horizon._sample(horizon.expansion(field, slice_), points, slice_)
        error = abs(float(np.mean(sampled)))
        assert error < 3e-3, (count, error)
        if previous is not None:
            assert error < previous
        previous = error


def test_a_level_set_with_no_gradient_is_refused():
    """A constant field has no normal, so it has no expansion."""
    slice_ = horizon.schwarzschild_slice(shape=(16, 16, 16), extent=EXTENT, mass=MASS)
    with pytest.raises(ValueError, match="vanishing gradient"):
        horizon.expansion(np.ones(slice_.shape), slice_)


# --- the surface and its area -------------------------------------------


def test_the_area_of_a_coordinate_sphere_is_the_conformal_one():
    """``psi^4 r^2 4 pi``, which at ``r = M/2`` is ``16 pi M^2``."""
    slice_ = horizon.schwarzschild_slice(shape=(96, 96, 96), extent=EXTENT, mass=MASS)
    theta, phi, weights, phi_weight = horizon.angular_grid(20, 40)
    for radius in (MASS / 2, 1.0, 1.5):
        radii = np.full((20, 40), radius)
        area = horizon.surface_area(radii, slice_, theta, phi, weights, phi_weight)
        conformal = 1.0 + MASS / (2.0 * radius)
        assert area == pytest.approx(4 * np.pi * conformal**4 * radius**2, rel=2e-3)


def test_the_quadrature_is_exact_on_a_flat_sphere():
    """Zero mass makes ``psi = 1``, so the area must be ``4 pi r^2`` exactly."""
    slice_ = horizon.schwarzschild_slice(shape=(32, 32, 32), extent=EXTENT, mass=0.0)
    theta, phi, weights, phi_weight = horizon.angular_grid(16, 32)
    radii = np.full((16, 32), 0.8)
    area = horizon.surface_area(radii, slice_, theta, phi, weights, phi_weight)
    assert area == pytest.approx(4 * np.pi * 0.8**2, rel=1e-10)


def test_the_monomial_basis_spans_the_harmonics_it_claims():
    """Degree ``d`` in the direction components is ``l <= d`` on the sphere.

    Checked by counting: the monomials of total degree at most ``d`` in three
    variables number ``(d+1)(d+2)(d+3)/6``, and restricted to the unit sphere
    they collapse onto the ``(d+1)^2`` independent harmonics. The rank of the
    design matrix is what has to come out right, because that is what the
    least-squares projection actually uses.
    """
    theta, phi, _, _ = horizon.angular_grid(16, 32)
    direction = horizon._directions(theta, phi)
    for degree in (0, 1, 2, 3, 4):
        exponents = horizon._monomial_exponents(degree)
        assert len(exponents) == (degree + 1) * (degree + 2) * (degree + 3) // 6
        design = np.stack(
            [term.ravel() for term in horizon._monomials(direction, exponents)], axis=-1
        )
        assert np.linalg.matrix_rank(design, tol=1e-10) == (degree + 1) ** 2, degree


def test_ray_casting_finds_the_outermost_crossing():
    """Which is the one that matters: an inner MOTS is a different surface."""
    slice_ = horizon.schwarzschild_slice(shape=(64, 64, 64), extent=EXTENT, mass=MASS)
    theta, phi, _, _ = horizon.angular_grid(8, 16)
    coords = slice_.coordinates()
    radius = np.sqrt(sum(value**2 for value in coords))
    # A field with zeros at 0.4 and 1.0; the outer one should win.
    field = (radius - 0.4) * (radius - 1.0) * (radius + 1.0)
    found = horizon.ray_radii(field, slice_, theta, phi)
    assert np.all(np.isfinite(found))
    assert float(np.mean(found)) == pytest.approx(1.0, abs=0.02)


# --- the apparent horizon -----------------------------------------------


@pytest.mark.slow
@pytest.mark.benchmark
def test_the_flow_finds_the_schwarzschild_horizon_from_either_side():
    """``r = M/2``, area ``16 pi M^2``, irreducible mass ``M``.

    From 0.9 and from 1.3, both of which are outside, and the answer does
    not depend on which. Measured at ``n = 64``: radius 0.5075, area
    0.99952 of ``16 pi M^2``, irreducible mass 0.99976.
    """
    slice_ = horizon.schwarzschild_slice(shape=(64, 64, 64), extent=EXTENT, mass=MASS)
    results = []
    for guess in (0.9, 1.3):
        found = horizon.find_apparent_horizon(
            slice_, guess, theta_count=16, phi_count=32, tolerance=1e-2, max_iterations=300
        )
        assert found.converged, (guess, found.summary())
        assert found.mean_radius == pytest.approx(MASS / 2, abs=0.02)
        assert found.area == pytest.approx(16 * np.pi * MASS**2, rel=5e-3)
        assert found.irreducible_mass == pytest.approx(MASS, rel=3e-3)
        # Spherically symmetric data gives a spherical surface.
        assert found.distortion < 1e-3, found.distortion
        results.append(found.irreducible_mass)
    assert results[0] == pytest.approx(results[1], rel=1e-3)


@pytest.mark.slow
@pytest.mark.benchmark
def test_the_area_is_right_even_when_the_radius_is_not_constant():
    """Rays cast off centre: the same surface, an angle-dependent radius.

    This is the test that exercises the angular derivative terms in
    :func:`surface_area`, which a coordinate sphere leaves at zero. Measured
    at an offset of ``0.2M``: the radius varies by 83% across directions and
    the area still comes back to 0.99958 of ``16 pi M^2``.
    """
    base = horizon.schwarzschild_slice(shape=(64, 64, 64), extent=EXTENT, mass=MASS)
    offset = 0.2
    shifted = Slice(
        fields=base.fields,
        spacing=base.spacing,
        centre=(base.centre[0] + offset, base.centre[1], base.centre[2]),
    )
    found = horizon.find_apparent_horizon(
        shifted,
        0.5,
        theta_count=20,
        phi_count=40,
        tolerance=3e-2,
        max_iterations=400,
        degree=4,
    )
    assert found.distortion > 0.5, "the point of this test is a non-spherical radius"
    assert found.area == pytest.approx(16 * np.pi * MASS**2, rel=5e-3)
    assert found.irreducible_mass == pytest.approx(MASS, rel=3e-3)


def test_a_guess_outside_the_grid_is_refused_with_a_reason():
    slice_ = horizon.schwarzschild_slice(shape=(24, 24, 24), extent=EXTENT, mass=MASS)
    with pytest.raises(RuntimeError, match="left the grid"):
        horizon.find_apparent_horizon(
            slice_, 1.95, theta_count=8, phi_count=16, max_iterations=2, safety=50.0
        )


def test_an_unconverged_search_returns_its_best_surface():
    """A caller polling during an evolution wants the best so far, not an exception."""
    slice_ = horizon.schwarzschild_slice(shape=(32, 32, 32), extent=EXTENT, mass=MASS)
    found = horizon.find_apparent_horizon(
        slice_, 0.9, theta_count=8, phi_count=16, tolerance=1e-12, max_iterations=3
    )
    assert not found.converged
    assert found.iterations == 3
    assert np.isfinite(found.residual)
    assert found.area > 0.0


def test_the_summary_is_json_shaped():
    """Floats only, so a run log can carry it."""
    found = horizon.ApparentHorizon(
        radii=np.full((4, 8), 0.5),
        area=16 * np.pi,
        residual=1e-4,
        iterations=7,
        converged=True,
    )
    summary = found.summary()
    assert set(summary) == {
        "area",
        "irreducible_mass",
        "mean_radius",
        "distortion",
        "residual",
        "iterations",
        "converged",
    }
    assert all(isinstance(value, float) for value in summary.values())
    assert summary["irreducible_mass"] == pytest.approx(1.0)
    assert summary["distortion"] == 0.0


# --- the event horizon --------------------------------------------------


def test_the_null_rate_vanishes_on_the_horizon_and_is_outgoing_outside():
    """``d_t h = alpha/psi^2``: zero at ``r = M/2``, positive beyond it.

    The lapse is what makes this work. With ``alpha = 1`` there is nothing
    for a null surface to converge to, because the slicing is not static and
    one slice does not determine the spacetime.
    """
    slice_ = horizon.schwarzschild_slice(shape=(96, 96, 96), extent=EXTENT, mass=MASS)
    for radius, expected_sign in ((MASS / 2, 0), (0.8, 1), (1.4, 1)):
        field, points = _sphere(slice_, radius)
        # null_rate is d_t F; the radius moves the other way.
        rate = -horizon._sample(horizon.null_rate(field, slice_), points, slice_)
        mean = float(np.mean(rate))
        if expected_sign == 0:
            assert abs(mean) < 5e-3, mean
        else:
            assert mean > 0.0, (radius, mean)
            conformal = 1.0 + MASS / (2.0 * radius)
            lapse = (1.0 - MASS / (2.0 * radius)) / conformal
            assert mean == pytest.approx(lapse / conformal**2, rel=5e-3)


@pytest.mark.slow
@pytest.mark.benchmark
def test_backward_null_surfaces_converge_at_the_surface_gravity():
    """The event horizon, and an analytic rate to check it against.

    ``d/dr(alpha/psi^2)`` at ``r = M/2`` is exactly ``1/(4M)``, the surface
    gravity, so the gap between a backward null surface and the horizon must
    fall as ``e^(-kappa t)``. Measured from four starting radii on both
    sides, the rate comes out between 0.243 and 0.262 against 0.25.

    Forwards this is hopeless -- the same exponential is a divergence -- and
    that asymmetry is the whole reason event horizons are found in
    post-processing.
    """
    slice_ = horizon.schwarzschild_slice(shape=(64, 64, 64), extent=EXTENT, mass=MASS)
    kappa = 1.0 / (4.0 * MASS)
    for guess in (0.70, 1.00, 0.42):
        history = horizon.event_horizon_flow(
            slice_, guess, time_step=0.1, steps=150, theta_count=8, phi_count=16, degree=0
        )
        times = np.array([time for time, _ in history])
        radii = np.array([float(np.mean(value)) for _, value in history])
        gap = np.abs(radii - MASS / 2)

        assert gap[-1] < gap[0] / 20.0, (guess, gap[0], gap[-1])
        assert radii[-1] == pytest.approx(MASS / 2, abs=0.01)

        window = (np.abs(times) > 2.0) & (np.abs(times) < 12.0) & (gap > 1e-10)
        rate = float(-np.polyfit(np.abs(times[window]), np.log(gap[window]), 1)[0])
        assert rate == pytest.approx(kappa, rel=0.08), (guess, rate)


def test_the_flow_accepts_a_sequence_of_slices():
    """A real spacetime is a list of slices, latest first."""
    slice_ = horizon.schwarzschild_slice(shape=(32, 32, 32), extent=EXTENT, mass=MASS)
    history = horizon.event_horizon_flow(
        [slice_] * 6, 0.8, time_step=0.1, steps=5, theta_count=8, phi_count=16, degree=0
    )
    assert len(history) == 6
    assert history[0][0] == 0.0
    assert history[-1][0] == pytest.approx(-0.5)
    radii = [float(np.mean(value)) for _, value in history]
    assert radii == sorted(radii, reverse=True), "started outside, so it moves in"
