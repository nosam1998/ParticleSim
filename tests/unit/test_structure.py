"""Structure observables: an estimator held to identities, not to tolerances.

The power spectrum estimator has an exact property -- Parseval -- that ties
its normalisation and its mode weighting together in one number, so that is
what is asserted rather than a round trip through a random field. What the
random field is for is the corrections: shot noise and the deposition
window are each checked in the regime where they apply *and* in the regime
where they do not, because both are silent when misapplied.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.analysis.structure import (
    Spectrum,
    cloud_in_cell_shot_noise,
    cloud_in_cell_window,
    field_power_spectrum,
    friends_of_friends,
    halo_catalogue,
    mode_weights,
    power_spectrum,
    transfer_ratio,
)
from particlesim.cosmo.nbody import (
    Mesh,
    ParticleMesh,
    deposit,
    gaussian_field,
    zeldovich_from_field,
)

BOX = 100.0


@pytest.fixture(scope="module")
def mesh() -> Mesh:
    return Mesh(size=BOX, cells=32)


def wide_bins(mesh: Mesh) -> np.ndarray:
    """Edges reaching the grid's corner, ``sqrt(3)`` times Nyquist."""
    nyquist = np.pi / mesh.spacing
    return np.linspace(0.0, np.sqrt(3.0) * nyquist * 1.001, 300)


# --- the identity that pins the normalisation ----------------------------


@pytest.mark.parametrize("seed", [3, 11])
def test_the_spectrum_satisfies_parseval_exactly(mesh, seed):
    """``<delta^2> = (1/V) sum_k w_k P(k)``, to round-off.

    This is the whole normalisation in one number, and it is only true if
    the mode *weights* are right as well: the half-grid stores one entry
    per conjugate pair except on two self-conjugate planes, and counting
    those twice breaks the identity by several percent. Bins are widened to
    the grid's corner so no mode is left out of the sum.
    """
    field = gaussian_field(mesh, lambda k: 50.0 * k**-2.0, seed=seed)
    spectrum = field_power_spectrum(field, mesh, bins=wide_bins(mesh))

    good = np.isfinite(spectrum.power)
    total = np.sum(spectrum.power[good] * spectrum.modes[good]) / mesh.size**3
    assert total == pytest.approx(float(np.mean(field**2)), rel=1e-12)


def test_a_single_mode_lands_in_a_single_bin(mesh):
    """All the power in one shell, and nothing anywhere else."""
    wavenumber = 2.0 * np.pi * 2 / mesh.size
    x = np.arange(mesh.cells) * mesh.spacing
    field = 0.3 * np.cos(wavenumber * x)[:, None, None] * np.ones(mesh.shape)

    spectrum = field_power_spectrum(field, mesh)
    here = int(np.nanargmin(np.abs(spectrum.wavenumber - wavenumber)))
    elsewhere = np.nansum(np.delete(spectrum.power, here))
    assert elsewhere < 1e-25 * spectrum.power[here]


def test_mode_weights_count_the_real_degrees_of_freedom(mesh):
    """They must sum to ``cells^3``: a real field has exactly that many."""
    assert mode_weights(mesh).sum() == pytest.approx(mesh.cells**3, abs=1e-9)


def test_the_window_is_one_at_zero_and_four_over_pi_squared_at_nyquist(mesh):
    window = cloud_in_cell_window(mesh)
    assert window[0, 0, 0] == pytest.approx(1.0, abs=1e-14)
    assert window[0, 0, -1] == pytest.approx((2.0 / np.pi) ** 2, abs=1e-12)


def test_field_power_spectrum_rejects_the_wrong_shape(mesh):
    with pytest.raises(ValueError, match="must have shape"):
        field_power_spectrum(np.zeros((4, 4, 4)), mesh)


# --- shot noise, in both regimes -----------------------------------------


def test_shot_noise_is_not_white(mesh):
    """``V/N`` is the ``k -> 0`` limit, and a quarter of the truth at Nyquist.

    Subtracting the flat value is the obvious thing to do and over-subtracts
    fourfold at the high-``k`` end, which drives the estimate negative --
    a failure with a sign, not a small bias.
    """
    noise = cloud_in_cell_shot_noise(mesh, 1000)
    flat = mesh.size**3 / 1000
    assert noise[0, 0, 0] == pytest.approx(flat, rel=1e-12)
    assert noise[0, 0, -1] == pytest.approx(flat / 3.0, rel=1e-12)

    corner = noise[mesh.cells // 2, mesh.cells // 2, -1]
    assert corner == pytest.approx(flat / 27.0, rel=1e-12)


def test_shot_noise_refuses_an_empty_catalogue(mesh):
    with pytest.raises(ValueError, match="at least one particle"):
        cloud_in_cell_shot_noise(mesh, 0)


def test_poisson_particles_subtract_to_zero(mesh):
    """A fair sample is pure shot noise, and the subtraction removes it.

    Averaged over realisations so the tolerance can be the sampling error
    of the check rather than a guess.
    """
    trials, count = 6, 40000
    mean = np.zeros(mesh.cells // 2)
    for seed in range(trials):
        positions = np.random.default_rng(seed).uniform(0.0, BOX, size=(3, count))
        mean = mean + power_spectrum(positions, mesh).power / trials

    reference = power_spectrum(positions, mesh, subtract_shot_noise=False)
    scale = mesh.size**3 / count
    for index in (0, 5, 10, 15):
        sigma = scale * reference.scatter[index] / np.sqrt(trials)
        assert abs(mean[index]) < 3.0 * sigma


def test_subtracting_shot_noise_from_a_lattice_is_wrong(mesh):
    """The correction has a domain, and outside it the answer changes sign.

    Zel'dovich particles are a *displaced lattice*, which is sub-Poisson:
    at low ``k`` its discreteness power is nothing like ``V/N``. Subtracting
    the Poisson value removes noise that was never there and the estimate
    goes negative -- here by a factor of fifty. Documented as a number so
    that turning the flag on for lattice initial conditions is a test
    failure and not a plausible plot.
    """
    field = gaussian_field(mesh, lambda k: 0.5 * k**-2.0, seed=4)
    state = zeldovich_from_field(mesh, field, scale=0.1)

    honest = power_spectrum(state.positions, mesh, subtract_shot_noise=False)
    spoiled = power_spectrum(state.positions, mesh, subtract_shot_noise=True)
    assert honest.power[0] > 0.0
    assert spoiled.power[0] < -10.0 * honest.power[0]


# --- the acceptance ------------------------------------------------------


def test_the_estimator_recovers_the_reference_at_low_k(mesh):
    """Issue #80: ``P(k)`` at low ``k`` within 5% of reference.

    The reference is this realisation's *own* input spectrum, not the
    ensemble ``P(k)``. That matters: the box's lowest bin holds 18 modes,
    whose sample scatter is ``sqrt(2/18)`` = 33%, so no 5% statement about
    an ensemble survives one realisation. Compared against the field the
    particles were made from, sample variance cancels and what is left is
    the estimator.

    Measured: 0.07% in the lowest bin, and inside 5% out to ``k h = 1.8``,
    which is 56% of the way to Nyquist. ``subtract_shot_noise=False``
    because these are lattice initial conditions -- see the test above for
    what the other choice does.

    Past that the estimate runs *high*, reaching +15% near Nyquist, and the
    sign is informative: the deconvolution divides by the fair-sample
    window ``sinc^4(k h/2)``, and a near-lattice distribution is windowed
    less than that, so the correction overshoots. It is the same
    distinction that makes the force window ``sinc(k h)`` rather than the
    textbook value. The acceptance is stated at low ``k`` for real reasons,
    and both ends are asserted so that neither can drift unnoticed.
    """
    scale = 0.1
    field = gaussian_field(mesh, lambda k: 0.5 * k**-2.0, seed=4)
    state = zeldovich_from_field(mesh, field, scale=scale)

    reference = field_power_spectrum(field, mesh)
    estimate = power_spectrum(state.positions, mesh, subtract_shot_noise=False)

    good = np.isfinite(reference.power) & (reference.power > 0.0)
    ratio = estimate.power[good] / (scale**2 * reference.power[good])
    assert np.abs(ratio[:2] - 1.0).max() < 0.005
    assert np.abs(ratio[:9] - 1.0).max() < 0.05
    assert ratio[14] == pytest.approx(1.148, abs=0.02)  # and high near Nyquist


def test_without_the_window_the_estimate_is_low_and_gets_worse(mesh):
    """The deconvolution earns its place: it is what makes the 5% hold."""
    scale = 0.1
    field = gaussian_field(mesh, lambda k: 0.5 * k**-2.0, seed=4)
    state = zeldovich_from_field(mesh, field, scale=scale)

    reference = field_power_spectrum(field, mesh)
    raw = power_spectrum(state.positions, mesh, deconvolve=False, subtract_shot_noise=False)
    ratio = raw.power / (scale**2 * reference.power)
    assert ratio[0] < 1.0
    assert ratio[7] < ratio[0]  # monotonically worse with k
    assert ratio[7] < 0.75  # 32% low by k h = 1.6


# --- the regression estimator --------------------------------------------


def test_transfer_ratio_of_a_field_with_itself_is_one(mesh):
    field = gaussian_field(mesh, lambda k: 50.0 * k**-2.0, seed=2)
    ratio = transfer_ratio(field, field, mesh)
    good = np.isfinite(ratio.power)
    assert np.abs(ratio.power[good] - 1.0).max() < 1e-12


def test_transfer_ratio_reads_off_a_scaling_exactly(mesh):
    """Insensitive to the realisation, which is the point of using it."""
    field = gaussian_field(mesh, lambda k: 50.0 * k**-2.0, seed=2)
    ratio = transfer_ratio(0.37 * field, field, mesh)
    good = np.isfinite(ratio.power)
    assert np.abs(ratio.power[good] - 0.37).max() < 1e-12


def test_spectrum_scatter_is_the_mode_count(mesh):
    spectrum = Spectrum(wavenumber=np.array([1.0]), power=np.array([1.0]), modes=np.array([50.0]))
    assert spectrum.scatter[0] == pytest.approx(np.sqrt(2.0 / 50.0))


@pytest.mark.slow
def test_linear_growth_is_recovered_at_low_k(mesh):
    """The mesh grows the largest modes at the right rate, to a few percent.

    Ties the estimator back to issue #78: the force carries ``sinc(k h)``,
    which moves the growing exponent, so the recovered growth falls short
    by more and more as ``k h`` rises. At the box's longest modes it is
    1.4% and 4.1%; by ``k h = 0.8`` it is 13%. "Low ``k``" in the
    acceptance is not decoration.
    """
    start, final = 0.02, 0.10
    field = gaussian_field(mesh, lambda k: 0.5 * k**-2.0, seed=4)
    state = zeldovich_from_field(mesh, field, scale=start)
    evolved, _ = ParticleMesh(mesh).run(state, final, 120)

    ratio = transfer_ratio(deposit(evolved.positions, mesh), field, mesh)
    assert ratio.power[0] / final - 1.0 == pytest.approx(-0.014, abs=0.01)
    assert ratio.power[1] / final - 1.0 == pytest.approx(-0.041, abs=0.015)
    assert ratio.power[7] < ratio.power[0]  # suppression grows with k


# --- haloes --------------------------------------------------------------


@pytest.fixture(scope="module")
def small() -> Mesh:
    return Mesh(size=1.0, cells=8)


def lattice(mesh: Mesh) -> np.ndarray:
    axis = (np.arange(mesh.cells) + 0.5) * mesh.spacing
    grid = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([value.ravel() for value in grid])


def test_percolation_on_a_lattice_is_a_step(small):
    """Below the lattice spacing nothing links; above it, everything does.

    A sharp threshold rather than a statistical one, which is what makes it
    worth testing: the answer either side is an integer.
    """
    points = lattice(small)
    assert len(np.unique(friends_of_friends(points, small, 0.99 * small.spacing))) == 512
    assert len(np.unique(friends_of_friends(points, small, 1.01 * small.spacing))) == 1


def test_linking_is_transitive(small):
    """Ten particles in a line, each within ``b`` of the next, are one group.

    End to end they span 0.36, seven times the linking length. Friends of
    friends are friends, and a finder that only linked pairs directly would
    return ten groups here.
    """
    chain = np.zeros((3, 10))
    chain[0] = np.arange(10) * 0.04
    assert len(np.unique(friends_of_friends(chain, small, 0.05))) == 1


def test_a_halo_across_the_boundary_is_one_halo(small):
    """And its centre is at the boundary, not in the middle of the box."""
    rng = np.random.default_rng(0)
    far = rng.normal(0.25, 0.01, size=(3, 60))
    split = rng.normal(0.0, 0.01, size=(3, 60)) % small.size
    positions = np.concatenate([far, split], axis=1)

    catalogue = halo_catalogue(positions, small, 0.05, minimum=10)
    assert catalogue.count == 2
    assert list(catalogue.sizes) == [60, 60]

    centres = np.sort(catalogue.centres[0])
    assert centres[0] == pytest.approx(0.25, abs=0.01)
    wrapped = (centres[1] + 0.5) % 1.0 - 0.5
    assert wrapped == pytest.approx(0.0, abs=0.01)


def test_halo_catalogue_drops_groups_below_the_minimum(small):
    rng = np.random.default_rng(1)
    blob = rng.normal(0.5, 0.01, size=(3, 40))
    stray = np.array([[0.05], [0.05], [0.05]])
    catalogue = halo_catalogue(np.concatenate([blob, stray], axis=1), small, 0.05, minimum=10)

    assert catalogue.count == 1
    assert catalogue.sizes[0] == 40
    assert catalogue.labels[-1] == -1  # the stray belongs to nothing


@pytest.mark.parametrize("bad", [0.0, -0.1, 0.5, 1.0])
def test_friends_of_friends_refuses_a_bad_linking_length(small, bad):
    with pytest.raises(ValueError, match="linking length"):
        friends_of_friends(lattice(small), small, bad)


def test_friends_of_friends_refuses_the_wrong_shape(small):
    with pytest.raises(ValueError, match=r"shape \(3, N\)"):
        friends_of_friends(np.zeros((2, 5)), small, 0.1)


def test_halo_catalogue_refuses_a_zero_minimum(small):
    with pytest.raises(ValueError, match="at least one particle"):
        halo_catalogue(lattice(small), small, 0.1, minimum=0)
