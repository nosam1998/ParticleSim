"""A radiative outer boundary: does a pulse leave, and what does it cost?

The question a boundary has to answer is whether it is a boundary or a wall,
and a norm that only goes down does not answer it -- a pulse spreading out
of the region being measured looks the same as a pulse leaving the domain.
The comparison is against the *same run with periodic boundaries*, where the
pulse is known to come back. Two curves that agree until the pulse reaches
the edge and then part company is the signature worth testing.

Measured, amplitude of the traceless curvature in the interior, on a domain
of extent 4 with a 6-point zone:

    t/L          0.00     0.62     0.88     1.50     2.50
    periodic     1.0e-2   4.8e-4   1.3e-3   2.1e-3   2.2e-3
    radiative    1.0e-2   5.0e-4   6.4e-5   4.7e-6   2.2e-7

Identical until 0.62 L, which says the boundary does not disturb the
interior before the pulse arrives. Then the periodic run rises as the pulse
re-enters and settles into recirculating around 1e-3 forever, while the
radiative one keeps falling: 4.7 orders of magnitude by 2.5 crossing times,
and no recurrence.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.solvers.nr import boundary, bssn

EXTENT = 4.0
WIDTH = 6
AXES = (0, 1, 2)


def _pulse(n: int):
    """Flat space with a small Gaussian bump on the traceless curvature.

    Constraint-violating on purpose. What is being tested is whether a
    disturbance leaves, and a bump is the cleanest disturbance to follow;
    the amplitude is small enough that it stays in the linear regime.
    """
    state, spacing = bssn.gauge_wave(shape=(n, n, n), amplitude=0.0, extent=EXTENT)
    axis = np.linspace(0.0, EXTENT, n, endpoint=False) - EXTENT / 2
    mesh = np.meshgrid(axis, axis, axis, indexing="ij")
    bump = 0.01 * np.exp(-sum(value**2 for value in mesh) / (2 * 0.35**2))
    out = dict(state)
    out["At00"] = np.asarray(state["At00"]) + bump
    out["At11"] = np.asarray(state["At11"]) - bump / 2
    out["At22"] = np.asarray(state["At22"]) - bump / 2
    return out, spacing, tuple(mesh)


def _amplitude(state) -> float:
    return max(
        float(np.max(np.abs(boundary.interior(state[f"At{i}{i}"], WIDTH, AXES)))) for i in range(3)
    )


# --- the zone -----------------------------------------------------------


def test_the_zone_is_the_outer_points_of_the_bounded_axes_only():
    """A periodic axis keeps no zone, which is what allows a 1-D boundary."""
    _, spacing, mesh = _pulse(16)
    along_x = boundary.Radiative(coords=mesh, spacing=spacing, axes=(0,), width=3, backend="numpy")
    zone = along_x.mask()
    assert zone[:3].all() and zone[-3:].all()
    assert not zone[3:-3].any()

    everywhere = boundary.Radiative(
        coords=mesh, spacing=spacing, axes=AXES, width=3, backend="numpy"
    )
    # The union over three axes, corners included: 1 - (1 - 2*3/16)^3.
    expected = 1.0 - (1.0 - 6.0 / 16.0) ** 3
    assert everywhere.mask().mean() == pytest.approx(expected, rel=1e-12)


def test_a_zone_narrower_than_the_stencil_is_refused():
    """Otherwise the kernel's wrapped rates reach the interior."""
    _, spacing, mesh = _pulse(16)
    with pytest.raises(ValueError, match="narrower than the stencil radius"):
        boundary.Radiative(coords=mesh, spacing=spacing, axes=AXES, width=1, order=4)


def test_the_condition_leaves_minkowski_alone():
    """Flat space is the background the condition is written against.

    ``d_t f = -v (x^i/r) d_i f - v (f - f_0)/r`` vanishes term by term when
    the state *is* the background: no gradient, no departure. A wrong entry
    in ``ASYMPTOTIC`` shows up here at the amplitude of the error rather
    than as a slow drift in a long run.
    """
    state, spacing, mesh = _pulse(16)
    flat, _ = bssn.gauge_wave(shape=(16, 16, 16), amplitude=0.0, extent=EXTENT)
    condition = boundary.Radiative(
        coords=mesh, spacing=spacing, axes=AXES, width=3, backend="numpy"
    )
    rates = condition.rates(flat)
    for name, value in rates.items():
        assert float(np.max(np.abs(np.asarray(value)))) < 1e-12, name


def test_every_evolved_variable_has_a_background():
    """A variable missing from ASYMPTOTIC would be pulled towards zero.

    For the conformal metric's diagonal that is a one, not a zero, and
    getting it wrong reflects at the amplitude of the difference -- which is
    to say, at order one.
    """
    from particlesim.symbolic.bssn import STATE_NAMES

    for name in STATE_NAMES:
        assert name in boundary.ASYMPTOTIC, name
    for i in range(3):
        assert boundary.ASYMPTOTIC[f"gt{i}{i}"] == 1.0
    assert boundary.ASYMPTOTIC["alpha"] == 1.0
    assert boundary.ASYMPTOTIC["gt01"] == 0.0


# --- the measurement -----------------------------------------------------


@pytest.mark.slow
@pytest.mark.benchmark
def test_a_pulse_leaves_instead_of_coming_back():
    """The test that tells a boundary from a wall.

    Same initial pulse, same kernel, same integrator, same dissipation --
    only the boundary differs. The periodic run is the control, and it is
    *known* to bring the pulse back, which is what makes the comparison
    meaningful: a radiative curve that merely decreases could be a pulse
    spreading out of the measured region.

    Measured at 2.5 crossing times: periodic peaks at 2.23e-03 after the
    pulse re-enters, radiative reaches 2.2e-07. Run here to 1.5 crossings to
    keep the test affordable, which is past the first recurrence.

    **The window is in time, not in sample indices.** The first version of
    this test took the maximum from the fortieth percentile of the run
    onwards, which is ``t = 0.5 L`` -- *before* the pulse reaches the edge,
    where both curves still read 6.6e-04 because nothing has left yet. It
    failed at a separation of 3.2 while the physics was giving 35, and the
    bug was entirely in where the window started.
    """
    n, crossings = 32, 1.5
    state, spacing, mesh = _pulse(n)
    evolution = bssn.Evolution.build(spacing, slicing="harmonic", shift_condition="frozen")
    bounded = boundary.Bounded(
        evolution,
        boundary.Radiative(coords=mesh, spacing=spacing, axes=AXES, width=WIDTH),
    )

    steps = int(round(crossings * EXTENT / evolution.time_step))
    step = crossings * EXTENT / steps
    every = max(1, steps // 12)

    curves = {}
    for label, stepper in (("periodic", evolution), ("radiative", bounded)):
        current = dict(state)
        series = [(0.0, _amplitude(current))]
        for index in range(1, steps + 1):
            current = stepper.step(current, step)
            if index % every == 0:
                series.append((index * step, _amplitude(current)))
        curves[label] = series
        assert all(np.isfinite(value) for _, value in series), label

    # Both start from the same pulse and shed it at the same rate while it is
    # still in flight, which is the check that the boundary is not simply
    # damping everything everywhere.
    assert curves["radiative"][0][1] == pytest.approx(curves["periodic"][0][1])
    assert curves["radiative"][1][1] == pytest.approx(curves["periodic"][1][1], rel=0.2)

    # And after the pulse has had time to leave -- 0.9 crossing times, past
    # where the periodic run's recurrence begins -- they part company.
    def late(label):
        return max(value for time, value in curves[label] if time / EXTENT >= 0.9)

    periodic_late, radiative_late = late("periodic"), late("radiative")
    assert radiative_late < periodic_late / 5.0, (radiative_late, periodic_late)
    assert radiative_late < curves["radiative"][0][1] / 50.0, curves["radiative"]


@pytest.mark.slow
@pytest.mark.benchmark
def test_the_boundary_lowers_the_constraint_rather_than_raising_it():
    """The opposite of the guess, and the reason is worth stating.

    A condition applied variable by variable to things that are not
    characteristic variables has no reason to respect the constraints, so the
    expectation was that it would make them worse. It makes them much better,
    because the violation leaves with the pulse instead of recirculating:

        n     periodic |H|   radiative |H|   ratio
        32      1.297e-03      3.806e-05      0.03
        48      1.224e-03      1.938e-05      0.02
        64      1.243e-03      1.088e-05      0.01

    The periodic numbers do not converge at all, since whatever the pulse
    deposits stays in the domain. The radiative ones converged at order 1.7
    to 2.0 with a second-order stencil at the edge. With
    :func:`~particlesim.solvers.nr.boundary.edge_derivative` they are
    9.9e-06, 5.4e-06 and 4.4e-06. What is left there is the bump's own
    violation, parked at the centre by the frozen shift.
    """
    from particlesim.solvers.nr.bssn import constraint_kernel, physical_slice_arrays

    n, crossings = 32, 1.25
    state, spacing, mesh = _pulse(n)
    evolution = bssn.Evolution.build(spacing, slicing="harmonic", shift_condition="frozen")
    bounded = boundary.Bounded(
        evolution,
        boundary.Radiative(coords=mesh, spacing=spacing, axes=AXES, width=WIDTH),
    )
    steps = int(round(crossings * EXTENT / evolution.time_step))
    step = crossings * EXTENT / steps

    def hamiltonian(current):
        out = constraint_kernel()(physical_slice_arrays(current, "jax"), tuple(spacing))
        inside = boundary.interior(out["hamiltonian"], WIDTH, AXES)
        return float(np.sqrt(np.mean(np.asarray(inside) ** 2)))

    periodic, radiative = dict(state), dict(state)
    for _ in range(steps):
        periodic = evolution.step(periodic, step)
        radiative = bounded.step(radiative, step)

    open_domain, closed_domain = hamiltonian(radiative), hamiltonian(periodic)
    assert open_domain < closed_domain / 10.0, (open_domain, closed_domain)


# --- the Teukolsky wave -----------------------------------------------------


TEUKOLSKY_EXTENT = 10.0
TEUKOLSKY_ZONE = 1.0
TEUKOLSKY_CUBE = 2.5


def _teukolsky_measure(n: int, final: float, every: float = 0.5):
    """Evolve a linear Teukolsky wave through the radiative boundary.

    Returns ``(t, amplitude, error)`` samples over the cube ``|x| <= 2.5``,
    both divided by the wave's amplitude: the largest departure of the
    conformal metric from flat, and the RMS difference from Teukolsky's
    closed form at the same time. The closed form is the control a Gaussian
    pulse did not have -- it says what the interior *should* hold at every
    moment, including after the wave has gone, so a reflection is measured
    as error rather than inferred from a norm that failed to fall.
    """
    from particlesim.solvers.nr import teukolsky

    amplitude = 1e-6
    shape = (n, n, n)
    state, spacing = teukolsky.teukolsky_wave(
        shape=shape, amplitude=amplitude, width=1.0, extent=TEUKOLSKY_EXTENT
    )
    axis = np.linspace(0.0, TEUKOLSKY_EXTENT, n, endpoint=False) - TEUKOLSKY_EXTENT / 2
    mesh = tuple(np.meshgrid(axis, axis, axis, indexing="ij"))
    evolution = bssn.Evolution.build(spacing, slicing="harmonic", shift_condition="frozen")
    width = int(round(TEUKOLSKY_ZONE / spacing[0]))
    bounded = boundary.Bounded(
        evolution, boundary.Radiative(coords=mesh, spacing=spacing, axes=AXES, width=width)
    )
    inside = np.abs(axis) <= TEUKOLSKY_CUBE + 1e-9
    cube = np.ix_(inside, inside, inside)
    names = [f"gt{i}{j}" for i in range(3) for j in range(i, 3)]

    def sample(time, current):
        exact, _ = teukolsky.teukolsky_wave(
            shape=shape,
            amplitude=amplitude,
            width=1.0,
            extent=TEUKOLSKY_EXTENT,
            time=time,
            backend="numpy",
        )
        flat = {name: 1.0 if name[2] == name[3] else 0.0 for name in names}
        largest = max(
            float(np.max(np.abs(np.asarray(current[name])[cube] - flat[name]))) for name in names
        )
        error = np.sqrt(
            sum(
                np.mean((np.asarray(current[name])[cube] - exact[name][cube]) ** 2)
                for name in names
            )
        )
        return time, largest / amplitude, float(error) / amplitude

    steps = int(round(final / evolution.time_step))
    step = final / steps
    stride = max(1, int(round(every / step)))
    current = dict(state)
    samples = [sample(0.0, current)]
    for index in range(1, steps + 1):
        current = bounded.step(current, step)
        if index % stride == 0:
            samples.append(sample(index * step, current))
    return samples


def test_the_edge_derivative_keeps_fourth_order_to_the_last_point():
    """The condition acts at the outermost points, so their stencil sets its order.

    :func:`particlesim.core.grid.derivative` is second order there; the
    boundary's own derivative is fourth order everywhere, at both edges.
    """
    errors = []
    for n in (20, 40, 80):
        x = np.linspace(0.0, 1.0, n)
        field = np.sin(3 * x)[:, None] * np.ones((1, 4))
        exact = 3 * np.cos(3 * x)[:, None]
        errors.append(np.max(np.abs(boundary.edge_derivative(field, 0, x[1] - x[0]) - exact)))
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(orders > 3.9), orders


@pytest.mark.slow
@pytest.mark.benchmark
def test_a_teukolsky_wave_leaves_without_a_reflection_above_truncation():
    """Issue #132's acceptance test, on the wave it names.

    Linear Teukolsky wave, ``λ = 1``, box of 10 at 40 points, a zone one
    unit deep starting at ``r = 4``. Measured in the cube ``|x| <= 2.5``
    against the closed form, as multiples of the amplitude:

        t                 0      2.5    3.5    6.5    8.0    8.5    10
        largest |h|       48     5.47   3.74   0.142  0.163  0.244  0.129
        error, fourth     0      0.071  0.063  0.062  0.071  0.064  0.056
        error, second     0      0.071  0.062  0.114  0.134  0.139  0.109

    Until ``t = 3.5`` the error is the interior's truncation error and the
    boundary is invisible in it. After the wave has gone, what is left is
    what came back, and with the fourth-order edge it is no larger than the
    truncation error was: 0.071 against 0.071. With the second-order edge
    :func:`particlesim.core.grid.derivative` would have supplied, it is
    twice that -- which is why the bound below is 1.5 and not 1, and why it
    is a bound on this boundary's stencil and not a formality.

    Whether that reflection is discretisation or a property of Sommerfeld
    itself is not something one resolution can say; a box of 12 at 48 and
    72 points can, and it converges at order 3.2 (``docs/benchmarks.md``).
    """
    samples = _teukolsky_measure(40, 10.0)
    assert all(np.isfinite(value) for _, amp, err in samples for value in (amp, err))
    initial = samples[0][1]
    truncation = max(err for time, _, err in samples if time <= 3.5)
    after = [(amp, err) for time, amp, err in samples if time >= 7.5]

    assert max(err for _, err in after) < 1.5 * truncation, samples
    # Two orders of magnitude, from 48 to at most 0.24 at this resolution.
    assert max(amp for amp, _ in after) < initial / 100.0, samples
