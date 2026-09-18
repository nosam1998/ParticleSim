"""Structure observables: the power spectrum, and haloes.

Issue #80, Level C3. Two estimators run on the output of
:mod:`particlesim.cosmo.nbody`.

**The power spectrum estimator has to undo the mesh, and unlike the force
it can.** Depositing particles with cloud-in-cell multiplies the density by
``W(k) = prod_i sinc^2(k_i h / 2)``, so the measured power is ``W^2 P``
rather than ``P``. Dividing it back out is the right thing to do here,
which is worth saying because in :func:`~particlesim.cosmo.nbody.potential_gradient`
the same correction makes the *force* worse. The difference is what the
answer is used for: the force is evaluated at particle positions, where
deconvolution amplifies aliased power that the window was holding down,
whereas the spectrum is reported per mode, where the aliased contribution
is a separate additive term.

**That ``W`` is the window averaged over sub-cell phase**, which is what
particles sampling the box fairly give. A near-lattice distribution -- and
Zel'dovich initial conditions are exactly that -- sits at one phase and is
windowed *less*, so dividing by ``W^2`` overshoots. Measured against the
field the particles were made from, the recovered power is right to 0.07%
in the box's lowest bin and stays inside 5% out to ``k h = 1.8``, then runs
high, reaching +15% near Nyquist. It is the same distinction that makes the
force window ``sinc(k h)`` rather than the textbook ``sinc^4(k h/2)``. The
acceptance for issue #80 is stated at low ``k``, where the two agree.

**Shot noise is subtracted, and it is not white.** ``N`` particles in a
volume ``V`` carry ``V/N`` from their own discreteness, but only as
``k -> 0``. Depositing them aliases that term too, and the alias sum for
cloud-in-cell has a closed form,

    P_shot(k) = (V/N) prod_i (1 - (2/3) sin^2(k_i h / 2))

which falls to a quarter of ``V/N`` by Nyquist. Subtracting a flat ``V/N``
instead over-subtracts by a factor of four there and drives the estimate
negative, which is how this was caught. Measured against 12 Poisson
realisations the closed form holds to about 1% at every ``k``, which is the
sampling error of the check. Subtraction is exact in the mean; what it
cannot do is restore the information, so the variance stays and the
estimator stops being trustworthy well before Nyquist. The acceptance for
issue #80 is stated at low ``k`` for that reason.

**Modes are weighted, because half of them are missing.** A real field's
transform is stored on a half-grid, where the ``k_z = 0`` and
``k_z = Nyquist`` planes are self-conjugate and every other mode stands for
a conjugate pair. Counting the array's entries equally over-weights those
two planes and gets the mode counts -- and so the error bars -- wrong by up
to 6%. They are weighted 1 and everything else 2.

**Comparing two fields mode by mode beats comparing their spectra.** The
box's longest bin holds six modes, so its sample scatter is `sqrt(2/6)`,
about 58%: no 5% statement survives that. But two fields built from the
same realisation share their phases, so the regression coefficient
``Re(<d_a d_b*>) / <|d_b|^2>`` measures the *ratio* with the realisation
divided out. That is what :func:`transfer_ratio` returns, and it is how the
growth of a mode is checked against linear theory without needing an
ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from particlesim.cosmo.nbody import Mesh, deposit


@dataclass(frozen=True)
class Spectrum:
    """A binned spectrum and the number of independent modes behind it."""

    wavenumber: np.ndarray
    power: np.ndarray
    modes: np.ndarray

    @property
    def scatter(self) -> np.ndarray:
        """``sqrt(2/m)``: the fractional sample scatter of a Gaussian field.

        Returned so a test can size its tolerance from the statistics
        instead of guessing one.
        """
        return np.sqrt(2.0 / np.maximum(self.modes, 1.0))


def cloud_in_cell_window(mesh: Mesh) -> np.ndarray:
    """``W(k) = prod_i sinc^2(k_i h / 2)``, the deposition transfer function.

    One factor per dimension, squared because cloud-in-cell is the
    convolution of two top hats. This is the same function that appears to
    the *fourth* power in the force, where deposition and interpolation
    each contribute a copy; a spectrum is deposited once.
    """
    grids, _ = mesh.wavenumbers()
    window = np.ones_like(grids[0])
    for component in grids:
        window = window * np.sinc(component * mesh.spacing / (2.0 * np.pi)) ** 2
    return window


def cloud_in_cell_shot_noise(mesh: Mesh, count: int) -> np.ndarray:
    """``(V/N) prod_i (1 - (2/3) sin^2(k_i h/2))``: discreteness, aliased.

    The flat ``V/N`` is only the ``k -> 0`` limit. Deposition folds power in
    from beyond Nyquist, and for cloud-in-cell the sum over aliases closes
    into this product -- a quarter of ``V/N`` at the Nyquist frequency, so
    using the flat value would over-subtract fourfold at the high-``k`` end.
    """
    if count < 1:
        raise ValueError(f"a shot-noise estimate needs at least one particle, got {count}")
    grids, _ = mesh.wavenumbers()
    noise = np.full_like(grids[0], mesh.size**3 / float(count))
    for component in grids:
        noise = noise * (1.0 - (2.0 / 3.0) * np.sin(component * mesh.spacing / 2.0) ** 2)
    return noise


def mode_weights(mesh: Mesh) -> np.ndarray:
    """``1`` on the two self-conjugate planes of a half-grid, ``2`` elsewhere."""
    weights = np.full((mesh.cells, mesh.cells, mesh.cells // 2 + 1), 2.0)
    weights[:, :, 0] = 1.0
    if mesh.cells % 2 == 0:
        weights[:, :, -1] = 1.0
    return weights


def _default_bins(mesh: Mesh) -> np.ndarray:
    """Edges one fundamental wide, from half a fundamental up to Nyquist."""
    fundamental = 2.0 * np.pi / mesh.size
    return (np.arange(mesh.cells // 2 + 1) + 0.5) * fundamental


def _modes(density, mesh: Mesh) -> np.ndarray:
    """``delta_k = (V/N^3) rfftn(delta)``, the continuum convention."""
    array = np.asarray(density, dtype=float)
    if array.shape != mesh.shape:
        raise ValueError(f"the density must have shape {mesh.shape}, got {array.shape}")
    return np.fft.rfftn(array) * mesh.spacing**3


def _bin(values, weights, magnitude, edges):
    """Weighted mean of ``values`` in shells of ``magnitude``."""
    index = np.digitize(magnitude.ravel(), edges)
    count = len(edges) - 1
    flat_weights = weights.ravel()
    total = np.bincount(index, weights=flat_weights, minlength=count + 2)
    summed = np.bincount(index, weights=flat_weights * values.ravel(), minlength=count + 2)
    keep = slice(1, count + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(total[keep] > 0.0, summed[keep] / total[keep], np.nan)
    return mean, total[keep]


def field_power_spectrum(density, mesh: Mesh, bins=None) -> Spectrum:
    """``P(k)`` of a density contrast already on the mesh.

    No window and no shot noise: a field that was never made of particles
    has neither. :func:`power_spectrum` is the entry point for particles.
    """
    edges = _default_bins(mesh) if bins is None else np.asarray(bins, dtype=float)
    modes = _modes(density, mesh)
    _, squared = mesh.wavenumbers()
    magnitude = np.sqrt(squared)
    magnitude[0, 0, 0] = 0.0

    weights = mode_weights(mesh) * (magnitude > 0.0)
    power, counts = _bin(np.abs(modes) ** 2 / mesh.size**3, weights, magnitude, edges)
    centres, _ = _bin(magnitude, weights, magnitude, edges)
    return Spectrum(wavenumber=centres, power=power, modes=counts)


def power_spectrum(
    positions,
    mesh: Mesh,
    bins=None,
    deconvolve: bool = True,
    subtract_shot_noise: bool = True,
) -> Spectrum:
    """``P(k)`` from particle positions, with the mesh's two artefacts removed.

    The measured power is ``W(k)^2 P(k) + P_shot(k)``, so the shot noise
    comes off *before* the window is divided out -- the two do not commute,
    and doing it the other way inflates the discreteness term by ``1/W^2``
    near Nyquist. Both corrections can be turned off, because a test that
    cannot see the uncorrected number cannot show that the correction is
    the right size.

    The shot-noise term assumes the particles are a Poisson sample. A
    lattice or a glass is sub-Poisson and this will over-subtract; that is a
    property of the initial conditions, not of the estimator.
    """
    array = np.asarray(positions, dtype=float)
    if array.ndim != 2 or array.shape[0] != 3:
        raise ValueError(f"positions must have shape (3, N), got {array.shape}")

    edges = _default_bins(mesh) if bins is None else np.asarray(bins, dtype=float)
    modes = _modes(deposit(array, mesh), mesh)
    _, squared = mesh.wavenumbers()
    magnitude = np.sqrt(squared)
    magnitude[0, 0, 0] = 0.0

    measured = np.abs(modes) ** 2 / mesh.size**3
    if subtract_shot_noise:
        measured = measured - cloud_in_cell_shot_noise(mesh, array.shape[1])
    if deconvolve:
        measured = measured / cloud_in_cell_window(mesh) ** 2

    weights = mode_weights(mesh) * (magnitude > 0.0)
    power, counts = _bin(measured, weights, magnitude, edges)
    centres, _ = _bin(magnitude, weights, magnitude, edges)
    return Spectrum(wavenumber=centres, power=power, modes=counts)


def transfer_ratio(density, reference, mesh: Mesh, bins=None) -> Spectrum:
    """Mode-by-mode ratio of two fields, with the realisation divided out.

    ``Re(<d d_ref*>) / <|d_ref|^2>`` in shells: the regression of one field
    on the other. For two fields grown from the same seed this measures the
    growth factor between them with **no** sample variance, which is what
    makes a 5% statement possible in the box's lowest bins at all. The
    ``power`` field carries the ratio.
    """
    edges = _default_bins(mesh) if bins is None else np.asarray(bins, dtype=float)
    first = _modes(density, mesh)
    second = _modes(reference, mesh)
    _, squared = mesh.wavenumbers()
    magnitude = np.sqrt(squared)
    magnitude[0, 0, 0] = 0.0

    weights = mode_weights(mesh) * (magnitude > 0.0)
    cross, counts = _bin(np.real(first * np.conjugate(second)), weights, magnitude, edges)
    auto, _ = _bin(np.abs(second) ** 2, weights, magnitude, edges)
    centres, _ = _bin(magnitude, weights, magnitude, edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(auto > 0.0, cross / auto, np.nan)
    return Spectrum(wavenumber=centres, power=ratio, modes=counts)


# --- haloes ---------------------------------------------------------------


@dataclass(frozen=True)
class Catalogue:
    """Friends-of-friends groups, largest first."""

    labels: np.ndarray
    sizes: np.ndarray
    centres: np.ndarray

    @property
    def count(self) -> int:
        return int(self.sizes.size)


def friends_of_friends(positions, mesh: Mesh, linking_length: float) -> np.ndarray:
    """Group particles that are within ``linking_length`` of one another.

    Friends-of-friends is a connected-components problem on the graph of
    pairs closer than ``b``, and is written as one: a periodic k-d tree
    supplies the edges and ``connected_components`` supplies the grouping.
    Doing it by hand with a chaining mesh is the traditional route and adds
    a union-find to get wrong for no gain here.

    Linking is *transitive*, which is the whole character of the algorithm:
    two particles in one group may be arbitrarily far apart provided there
    is a chain of hops between them. That is why the percolation threshold
    is sharp on a lattice, and it is what a test should pin.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    array = np.asarray(positions, dtype=float)
    if array.ndim != 2 or array.shape[0] != 3:
        raise ValueError(f"positions must have shape (3, N), got {array.shape}")
    if not 0.0 < linking_length < 0.5 * mesh.size:
        raise ValueError(
            f"the linking length must lie in (0, {0.5 * mesh.size}), got {linking_length}; "
            "beyond half the box a particle would link to its own periodic image"
        )

    wrapped = array.T % mesh.size
    wrapped[wrapped >= mesh.size] = 0.0  # a tiny negative can round up to the box

    tree = cKDTree(wrapped, boxsize=mesh.size)
    pairs = tree.query_pairs(linking_length, output_type="ndarray")
    total = wrapped.shape[0]
    if pairs.size == 0:
        return np.arange(total)

    graph = coo_matrix((np.ones(pairs.shape[0]), (pairs[:, 0], pairs[:, 1])), shape=(total, total))
    _, labels = connected_components(graph, directed=False)
    return labels


def halo_catalogue(positions, mesh: Mesh, linking_length: float, minimum: int = 20) -> Catalogue:
    """Friends-of-friends groups of at least ``minimum`` particles.

    Centres are circular means -- ``atan2`` of the mean of ``exp(2 pi i x/L)``
    -- not arithmetic ones. A group straddling the boundary has particles at
    both ends of the box, whose arithmetic mean is the middle of the box:
    the one place the halo certainly is not.
    """
    if minimum < 1:
        raise ValueError(f"a halo needs at least one particle, got {minimum}")
    labels = friends_of_friends(positions, mesh, linking_length)
    array = np.asarray(positions, dtype=float)

    counts = np.bincount(labels)
    keep = np.flatnonzero(counts >= minimum)
    order = keep[np.argsort(-counts[keep], kind="stable")]

    centres = np.empty((3, order.size))
    for slot, group in enumerate(order):
        members = array[:, labels == group] * (2.0 * np.pi / mesh.size)
        angle = np.arctan2(np.sin(members).mean(axis=1), np.cos(members).mean(axis=1))
        centres[:, slot] = angle * (mesh.size / (2.0 * np.pi)) % mesh.size

    renumbered = np.full(counts.size, -1, dtype=np.int64)
    renumbered[order] = np.arange(order.size)
    return Catalogue(labels=renumbered[labels], sizes=counts[order], centres=centres)


__all__ = [
    "Catalogue",
    "Spectrum",
    "cloud_in_cell_shot_noise",
    "cloud_in_cell_window",
    "field_power_spectrum",
    "friends_of_friends",
    "halo_catalogue",
    "mode_weights",
    "power_spectrum",
    "transfer_ratio",
]
