"""Numerical Calabi-Yau metrics on the quintic, against what is exact (issue #87)."""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.theories.calabi_yau import (
    Points,
    Quintic,
    balanced_metric,
    fermat_omega_volume,
    fubini_study,
    laplacian_spectrum,
    monomials,
    sigma_measure,
)


@pytest.fixture(scope="module")
def fermat() -> Points:
    return Quintic().sample(60000, seed=1)


@pytest.fixture(scope="module")
def held_out() -> Points:
    return Quintic().sample(40000, seed=2)


@pytest.fixture(scope="module")
def balanced(fermat):
    """Donaldson's metrics at degrees 2 and 3, fitted once for the module."""
    return {k: balanced_metric(fermat, k) for k in (2, 3)}


@pytest.mark.parametrize("psi", [0.0, 0.5, 0.3 + 0.2j])
def test_sampled_points_lie_on_the_quintic(psi):
    quintic = Quintic(psi)
    points = quintic.sample(5000, seed=3)
    assert len(points) == 5000
    assert np.abs(quintic.polynomial(points.Z)).max() < 1e-12
    assert np.allclose(np.abs(points.Z).max(axis=1), 1.0)


def test_the_sampler_integrates_omega_exactly_at_the_fermat_point(fermat):
    """``int_X |Omega|^2`` against ``pi^3 gamma(1/5)^5 / 625``, a complex Selberg integral.

    This checks the sampler's measure. A wrong distribution of points would
    bias the Monte Carlo mean, and no amount of sampling would bring it back.
    """
    estimate, error = fermat.omega_volume()
    exact = fermat_omega_volume()
    assert exact == pytest.approx(47.2975609186563, rel=1e-12)
    assert abs(estimate - exact) < 3 * error
    assert error / exact < 2e-3


def test_eta_does_not_depend_on_the_chart(fermat, balanced):
    """``det(g) / |Omega|^2`` is a ratio of two densities on ``X``, so it is chart independent.

    Solving for the coordinate with the second-largest ``|dP/dz|`` instead of
    the largest changes ``Omega``, the Jacobian and ``det(g)``, but not
    their ratio.
    """
    points = fermat.subset(slice(0, 3000))
    grad = np.abs(Quintic().gradient(points.Z))
    patch = np.argmax(np.abs(points.Z), axis=1)
    grad[np.arange(len(grad)), patch] = -1.0
    second = np.argsort(grad, axis=1)[:, -2]
    other = Points.on(Quintic(), points.Z, dependent=second)
    metric = balanced[2]
    first_eta = metric.determinant(points) / points.omega
    other_eta = metric.determinant(other) / other.omega
    assert np.abs(first_eta / other_eta - 1).max() < 1e-9


def test_every_metric_has_the_degree_as_its_volume(fermat, held_out, balanced):
    """``int_X omega^3 = 5`` is fixed by the Kahler class, whatever the metric in it."""
    fs = fubini_study()
    assert fermat.kahler_volume(fs.determinant(fermat)) == pytest.approx(5.0, rel=1e-12)
    for metric in balanced.values():
        assert held_out.kahler_volume(metric.determinant(held_out)) == pytest.approx(5.0, rel=0.01)


def test_donaldsons_iteration_converges_to_a_balanced_metric(balanced):
    """The change in ``H`` falls by about 2.5 each step, to below 1e-5 in ten."""
    metric = balanced[3]
    history = np.array(metric.history)
    assert history[-1] < 1e-5
    assert np.all(history[1:] < history[:-1])
    assert np.allclose(metric.H, metric.H.conj().T)
    assert np.linalg.eigvalsh(metric.H).min() > 0


def test_sigma_falls_as_the_degree_rises(held_out, balanced):
    """Donaldson's theorem, measured: the balanced metrics approach Ricci-flat.

    The Fubini-Study value, 0.373, is the familiar starting point for the
    Fermat quintic. Each degree is measured on points it was not fitted on.
    """
    sigmas = [sigma_measure(held_out, fubini_study().determinant(held_out))]
    for k in (2, 3):
        metric = balanced[k]
        sigmas.append(sigma_measure(held_out, metric.determinant(held_out)))
    assert sigmas[0] == pytest.approx(0.373, abs=0.006)
    assert sigmas[1] == pytest.approx(0.272, abs=0.006)
    assert sigmas[2] == pytest.approx(0.194, abs=0.006)


def test_the_section_basis_drops_the_quintic_relation():
    assert [len(monomials(k)) for k in range(1, 7)] == [5, 15, 35, 70, 125, 205]


def test_the_laplacian_multiplicities_come_from_the_symmetry(held_out):
    """A zero mode, then 20 and 4: the ``z_a zbar_b`` with ``a != b``, then the traceless diagonal.

    The Fermat quintic's permutations and phases act on the 24 non-constant
    ``z_a zbar_b / |z|^2`` as two irreducible pieces. Any metric keeping the
    symmetry has those multiplicities. The eigenvalues themselves carry
    Monte Carlo noise of a percent or so.
    """
    spectrum = laplacian_spectrum(held_out, fubini_study().tensor(held_out))
    levels = spectrum.levels()
    assert abs(levels[0][0]) < 1e-9 and levels[0][1] == 1
    assert levels[1][1] == 20
    assert levels[2][1] == 4
    assert levels[1][0] == pytest.approx(42.0, rel=0.03)
    assert spectrum.volume == pytest.approx(5.0 / 6.0, rel=1e-12)


def test_the_laplacian_spectrum_moves_with_the_metric(held_out, balanced):
    """The 4-fold level rises as the metric approaches Ricci-flat; the 20-fold barely moves."""
    fs = laplacian_spectrum(held_out, fubini_study().tensor(held_out)).levels()
    near = laplacian_spectrum(held_out, balanced[3].tensor(held_out)).levels()
    assert near[1][1] == 20 and near[2][1] == 4
    assert near[2][0] > fs[2][0] * 1.08
    assert near[1][0] == pytest.approx(fs[1][0], rel=0.03)


def test_the_neural_metric_learns_ricci_flatness(fermat, held_out):
    """A short run already beats Fubini-Study, and the metric it learns stays a metric."""
    pytest.importorskip("jax")
    from particlesim.theories.calabi_yau import NeuralMetric

    network = NeuralMetric(widths=(32, 32), seed=0)
    start = sigma_measure(held_out, network.determinant(held_out))
    assert start == pytest.approx(sigma_measure(held_out, fubini_study().determinant(held_out)))
    network.fit(fermat.subset(slice(0, 20000)), epochs=3, batch=500, learning_rate=3e-3)
    test = held_out.subset(slice(0, 10000))
    tensor = network.tensor(test)
    assert np.linalg.eigvalsh(tensor).min() > 0
    determinant = np.linalg.det(tensor).real
    assert sigma_measure(test, determinant) < 0.3 * start
    assert test.kahler_volume(determinant) == pytest.approx(5.0, rel=0.02)


@pytest.mark.slow
@pytest.mark.benchmark
def test_a_trained_network_is_ten_times_closer_to_ricci_flat_than_donaldson_at_degree_three():
    """Ten epochs on 100,000 points: ``sigma`` 0.029, against 0.194 for the balanced ``k = 3``.

    The spectrum keeps the symmetry's 20-fold first level on the trained
    metric too, so what the network learned respects the symmetry the
    architecture never imposed.
    """
    pytest.importorskip("jax")
    from particlesim.theories.calabi_yau import NeuralMetric

    quintic = Quintic()
    train, test = quintic.sample(100000, seed=11), quintic.sample(50000, seed=12)
    network = NeuralMetric(seed=0)
    network.fit(train, epochs=10)
    tensor = network.tensor(test)
    determinant = np.linalg.det(tensor).real
    assert np.linalg.eigvalsh(tensor).min() > 0
    assert sigma_measure(test, determinant) < 0.035
    assert test.kahler_volume(determinant) == pytest.approx(5.0, rel=0.01)
    levels = laplacian_spectrum(test, tensor).levels()
    assert levels[1][1] == 20 and levels[2][1] == 4
