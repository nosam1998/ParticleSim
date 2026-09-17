"""Laser injection, the moving window and ADK ionization (Section 3.3, M2)."""

import numpy as np
import pytest

from particlesim.solvers.pic import (
    Conducting,
    Fields,
    LaserPulse,
    MovingWindow,
    PerfectlyMatchedLayer,
    Periodic,
    PlaneWaveSource,
    Species,
    YeeGrid,
    YeeSolver,
    adk_rate,
    barrier_suppression_field,
    ionization_probability,
    keldysh_parameter,
    numeric_pulse_energy,
    tunnel_ionize,
)
from particlesim.solvers.pic.ionization import HARTREE_EV, electronvolts_to_hartree

WAVELENGTH = 1.0


def _pulse(cycles: float = 2.0, **kw) -> LaserPulse:
    """A pulse of ``cycles`` field-envelope widths per carrier period."""
    return LaserPulse(
        a0=0.5,
        wavelength=WAVELENGTH,
        duration=cycles * WAVELENGTH * np.sqrt(2 * np.log(2)),
        **kw,
    )


def _inject(pulse, cells_per_wavelength=40, nx=3000, index=200, thickness=16):
    dx = WAVELENGTH / cells_per_wavelength
    grid = YeeGrid((nx,), (dx,))
    solver = YeeSolver(grid, courant=0.5, boundary=PerfectlyMatchedLayer(thickness=thickness))
    source = PlaneWaveSource(pulse, index=index)
    source.attach(solver)
    fields = Fields.zeros(grid)
    for _ in range(int(round((pulse.start + 5 * pulse.tau) / solver.dt))):
        fields = source.step(fields)
    return grid, solver, source, fields


def _energy(fields, dx, region=slice(None), polarization="z"):
    electric = fields.Dz if polarization == "z" else fields.Dy
    magnetic = fields.By if polarization == "z" else fields.Bz
    return 0.5 * float(np.sum(electric[region] ** 2 + magnetic[region] ** 2)) * dx


# --- the pulse specification ------------------------------------------------


def test_pulse_validates_its_arguments():
    with pytest.raises(ValueError, match="must be positive"):
        LaserPulse(wavelength=0.0)
    with pytest.raises(ValueError, match="polarization"):
        LaserPulse(polarization="x")
    with pytest.raises(ValueError, match="envelope"):
        LaserPulse(envelope="lorentzian")


def test_peak_field_follows_from_the_normalized_vector_potential():
    pulse = LaserPulse(a0=2.5, wavelength=0.8)
    assert pulse.peak_field == pytest.approx(2.5 * 2 * np.pi / 0.8)


def test_the_closed_form_energy_is_the_many_cycle_limit():
    """The ``E0^2 tau sqrt(pi/2) / 2`` formula assumes the carrier's ``cos^2``
    averages to a half. It does, once the pulse is long enough, and this is
    where that stops being true."""
    long_pulse = _pulse(cycles=20.0)
    numeric = numeric_pulse_energy(long_pulse, 1e-4)
    assert long_pulse.energy == pytest.approx(numeric, rel=1e-6)

    short = _pulse(cycles=0.5)
    assert abs(short.energy / numeric_pulse_energy(short, 1e-4) - 1) > 1e-3


def test_the_closed_form_refuses_a_non_gaussian_envelope():
    with pytest.raises(ValueError, match="closed form"):
        _ = _pulse(envelope="sin2").energy


# --- injection --------------------------------------------------------------


def test_source_refuses_a_two_dimensional_grid():
    solver = YeeSolver(YeeGrid((32, 32), (0.05, 0.05)))
    with pytest.raises(ValueError, match="one-dimensional"):
        PlaneWaveSource(_pulse()).attach(solver)


def test_source_refuses_a_plane_outside_the_grid():
    solver = YeeSolver(YeeGrid((32,), (0.05,)))
    with pytest.raises(ValueError, match="must lie inside"):
        PlaneWaveSource(_pulse(), index=40).attach(solver)


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.parametrize("polarization", ("z", "y"))
def test_injected_energy_matches_the_specification(polarization):
    """Acceptance for issue #32: 1%.

    Measured against quadrature on the specified waveform rather than the
    closed form, so no assumption about the number of cycles enters.
    """
    pulse = _pulse(cycles=5.0, polarization=polarization)
    grid, solver, source, fields = _inject(pulse)
    forward = _energy(fields, grid.spacing[0], slice(source.index, None), polarization)
    assert forward == pytest.approx(numeric_pulse_energy(pulse, solver.dt), rel=0.01)


@pytest.mark.slow
def test_nothing_travels_backward_from_the_source():
    """What total-field/scattered-field buys over a current sheet.

    A sheet radiates symmetrically and throws half the energy the wrong way.
    Here the backward region holds twelve orders of magnitude less than the
    forward one, and what remains is the lattice dispersion: a pulse spans
    wavenumbers and no single incident phase velocity is right for all.
    """
    pulse = _pulse(cycles=5.0)
    grid, solver, source, fields = _inject(pulse)
    dx = grid.spacing[0]
    forward = _energy(fields, dx, slice(source.index, None))
    backward = _energy(fields, dx, slice(0, source.index - 1))
    assert backward / forward < 1e-8


@pytest.mark.slow
@pytest.mark.benchmark
def test_injection_error_is_second_order_in_the_spacing():
    """Confirms the residue is the discretization and not a systematic.

    A constant offset would not shrink; this one falls by four per doubling.
    """
    pulse = _pulse(cycles=4.0)
    errors = []
    for cells_per_wavelength in (20, 40, 80):
        nx = int(1200 * cells_per_wavelength / 20)
        grid, solver, source, fields = _inject(
            pulse, cells_per_wavelength=cells_per_wavelength, nx=nx, index=100
        )
        forward = _energy(fields, grid.spacing[0], slice(source.index, None))
        errors.append(abs(forward / numeric_pulse_energy(pulse, solver.dt) - 1))
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert all(1.7 < o < 2.3 for o in orders), f"orders were {orders}, errors {errors}"


# --- the moving window ------------------------------------------------------


def test_window_owes_whole_cells_only():
    grid = YeeGrid((64,), (0.1,))
    window = MovingWindow(velocity=1.0)
    fields = Fields.zeros(grid)
    assert window.due(grid, 0.05) == 0
    assert window.due(grid, 0.35) == 3
    window.shifted = 3
    assert window.due(grid, 0.35) == 0
    del fields


def test_window_refuses_to_shift_past_the_whole_box():
    grid = YeeGrid((16,), (0.1,))
    window = MovingWindow(velocity=1.0)
    fields = Fields.zeros(grid)
    fields = fields.with_D(fields.D, time=100.0)
    with pytest.raises(ValueError, match="owes"):
        window.advance(grid, fields)


def test_window_carries_particles_and_drops_what_falls_behind():
    grid = YeeGrid((32,), (0.1,))
    species = Species.create(-1.0, 1.0, [[0.05], [0.35], [2.0]])
    fields = Fields.zeros(grid).with_D(Fields.zeros(grid).D, time=0.25)
    _, moved = MovingWindow(velocity=1.0).advance(grid, fields, species)
    # Two cells of shift is 0.2, so the particle at 0.05 falls out of the box.
    assert moved.count == 2
    np.testing.assert_allclose(moved.position[:, 0], [0.15, 1.8])


def test_shifting_is_a_no_op_for_stateless_boundaries():
    for boundary in (Periodic(), Conducting()):
        boundary.attach(YeeSolver(YeeGrid((32,), (0.1,))))
        boundary.shift(3)  # must not raise


def test_the_layer_carries_its_convolution_history():
    """Regression: a window that leaves the history behind pairs each field
    with another cell's memory, and the layer stops absorbing. Measured, a
    pulse followed for two thousand cells gained five orders of magnitude in
    energy instead of holding it."""
    grid = YeeGrid((64,), (0.02,))
    layer = PerfectlyMatchedLayer(thickness=8)
    solver = YeeSolver(grid, boundary=layer)
    x = grid.coordinates((0.0,))[0]
    zero = np.zeros(64)
    fields = Fields(
        zero.copy(),
        zero.copy(),
        np.exp(-(((x - 0.2) / 0.05) ** 2)),
        zero.copy(),
        zero.copy(),
        zero.copy(),
    )
    solver.run(fields, 60)
    before = {k: v.copy() for k, v in layer._psi.items()}
    assert before, "the layer should have history to carry"
    layer.shift(2)
    for key, psi in before.items():
        np.testing.assert_allclose(layer._psi[key][:-2], psi[2:])
        assert (layer._psi[key][-2:] == 0).all()


@pytest.mark.slow
@pytest.mark.benchmark
def test_a_followed_pulse_keeps_its_energy():
    """Shifting by whole cells is a relabelling, so it costs nothing.

    A continuous shift would interpolate, and interpolating a pulse on every
    step of a run that lasts thousands of steps filters it away. Here the
    energy is unchanged to five decimals over three thousand shifts, which
    is a hundred and fifty box lengths of travel.
    """
    pulse = _pulse(cycles=2.0)
    cells_per_wavelength, nx, thickness = 20, 1200, 16
    dx = WAVELENGTH / cells_per_wavelength
    grid = YeeGrid((nx,), (dx,))
    boundary = PerfectlyMatchedLayer(thickness=thickness)
    solver = YeeSolver(grid, courant=0.5, boundary=boundary)
    source = PlaneWaveSource(pulse, index=100)
    source.attach(solver)

    fields = Fields.zeros(grid)
    # Inject, and let the peak reach mid-box before following it. Starting
    # while the pulse still overlaps the absorbing layer would measure the
    # layer rather than the window.
    target = pulse.start + (nx // 2 - 100) * dx
    while fields.time < target:
        fields = source.step(fields)

    reference = _energy(fields, dx)
    peak = float(np.abs(fields.Dz).max())
    window = MovingWindow(velocity=1.0, start=fields.time)
    for _ in range(int(round(150.0 / solver.dt))):
        fields = solver.step(fields)
        fields, _ = window.advance(grid, fields, None, boundary=boundary)

    assert window.shifted > 2500
    assert _energy(fields, dx) == pytest.approx(reference, rel=1e-4)
    # The peak spreads a little: that is the lattice's dispersion, not loss.
    assert 0.97 < float(np.abs(fields.Dz).max()) / peak <= 1.01


# --- ADK ionization ---------------------------------------------------------


def test_adk_reproduces_the_hydrogen_closed_form():
    """For hydrogen the whole formula collapses to ``(4/F) exp(-2/3F)``, so
    the coefficients and the exponents are checked against an exact result
    rather than against a table someone typed in."""
    field = np.array([0.02, 0.05, 0.1, 0.2])
    exact = 4.0 / field * np.exp(-2.0 / (3.0 * field))
    np.testing.assert_allclose(adk_rate(field, 0.5, 1), exact, rtol=1e-12)


def test_zero_field_gives_zero_rate_rather_than_an_overflow():
    """A cycle-resolved field passes through zero twice a period, so this is
    the common case rather than an edge one."""
    rates = adk_rate(np.array([0.0, 0.05]), 0.5, 1)
    assert rates[0] == 0.0
    assert rates[1] > 0.0


def test_rate_rises_with_field_and_falls_with_binding():
    fields = np.array([0.03, 0.06, 0.12])
    rates = adk_rate(fields, 0.5, 1)
    assert (np.diff(rates) > 0).all()
    helium = float(electronvolts_to_hartree(24.5874))
    hydrogen = float(electronvolts_to_hartree(13.5984))
    assert adk_rate(np.array([0.05]), helium, 1)[0] < adk_rate(np.array([0.05]), hydrogen, 1)[0]


def test_adk_validates_its_quantum_numbers():
    with pytest.raises(ValueError, match=r"\|m\| <= l"):
        adk_rate(np.array([0.05]), 0.5, 1, l=0, m=1)
    with pytest.raises(ValueError, match="ionization potential must be positive"):
        adk_rate(np.array([0.05]), 0.0)
    with pytest.raises(ValueError, match="charge left behind"):
        adk_rate(np.array([0.05]), 0.5, charge_state=0)


def test_barrier_suppression_and_keldysh_mark_the_formula_s_limits():
    """Hydrogen's barrier stops existing at 0.0625 atomic units, and a
    laser-plasma run passes that early rather than as an edge case."""
    assert barrier_suppression_field(0.5, 1) == pytest.approx(0.0625)
    # 800 nm light, omega = 0.05695 atomic units.
    assert keldysh_parameter(0.05695, 0.01, 0.5) > 2.0  # multiphoton, not tunnelling
    assert keldysh_parameter(0.05695, 0.2, 0.5) < 0.5  # solidly tunnelling


def test_probability_stays_a_probability_when_the_rate_is_large():
    """``W dt`` routinely exceeds one in a laser focus, and an unbounded
    expression there ionizes more than everything."""
    assert ionization_probability(1e6, 1.0) == pytest.approx(1.0)
    assert ionization_probability(1e-6, 1.0) == pytest.approx(1e-6, rel=1e-5)
    assert 0.0 <= ionization_probability(3.0, 0.5) <= 1.0


def test_ionized_fraction_follows_the_rate():
    """Drawn per macro-particle, so the fraction is binomial around the rate.

    The field here is chosen to put the probability near a third rather than
    near a thousandth: at a thousandth the counting noise alone is seven per
    cent, and a test at that tolerance would pass whatever the rate said.
    """
    rng = np.random.default_rng(4)
    count, field_strength = 200_000, 0.15
    mask = tunnel_ionize(np.full(count, field_strength), 0.5, dt=1.0, rng=rng)
    expected = float(ionization_probability(adk_rate(np.array([field_strength]), 0.5)[0], 1.0))
    assert 0.2 < expected < 0.4, expected
    assert mask.mean() == pytest.approx(expected, rel=0.02)


def test_hartree_conversion_is_the_accepted_value():
    assert HARTREE_EV == pytest.approx(27.2113862, abs=1e-6)
    assert float(electronvolts_to_hartree(27.211386245988)) == pytest.approx(1.0)
