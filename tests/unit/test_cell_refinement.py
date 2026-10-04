"""Point-sampled cell centres preserve reflection symmetry on every level."""

import numpy as np
import pytest

from particlesim.solvers.nr import bssn, mesh, puncture, refined


@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("operation", ["prolong", "restrict"])
def test_cell_transfers_converge_at_the_requested_order(order, operation):
    errors = []
    for n in (32, 64, 128):
        coarse_x = (np.arange(n) + 0.5) / n
        fine_x = (np.arange(2 * n) + 0.5) / (2 * n)

        def wave(x):
            return np.sin(2 * np.pi * x) * np.cos(4 * np.pi * x)

        if operation == "prolong":
            got = mesh.prolong(wave(coarse_x), order, centering="cell")
            exact = wave(fine_x)
        else:
            got = mesh.restrict(wave(fine_x), order=order, centering="cell")
            exact = wave(coarse_x)
        errors.append(np.max(abs(got - exact)))
    assert errors[-2] / errors[-1] == pytest.approx(2**order, rel=0.1), errors


@pytest.mark.parametrize("order", [2, 4, 6])
def test_cell_transfers_are_exact_on_polynomials_below_their_order(order):
    coarse_x = (np.arange(32) + 0.5) / 32
    fine_x = (np.arange(64) + 0.5) / 64
    for degree in range(order):
        fine = mesh.prolong(coarse_x**degree, order, centering="cell")
        coarse = mesh.restrict(fine_x**degree, order=order, centering="cell")
        assert fine[12:-12] == pytest.approx(fine_x[12:-12] ** degree, abs=2e-15)
        assert coarse[6:-6] == pytest.approx(coarse_x[6:-6] ** degree, abs=2e-15)


@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("parity", [-1, 1])
def test_cell_transfers_preserve_reflection_parity(order, parity):
    values = np.random.default_rng(29).normal(size=(16, 8))
    values += parity * values[::-1]
    fine = mesh.prolong(values, order, centering="cell")
    coarse = mesh.restrict(fine, order=order, centering="cell")
    assert fine == pytest.approx(parity * fine[::-1], abs=2e-15)
    assert coarse == pytest.approx(parity * coarse[::-1], abs=2e-15)


@pytest.mark.parametrize("order", [2, 4, 6])
def test_cell_extract_agrees_with_prolonging_the_whole_parent(order):
    parent = np.random.default_rng(8).normal(size=(32, 12))
    box = mesh.Box((4, 0), (16, 12), centering="cell")
    whole = mesh.prolong(parent, order, centering="cell")
    assert np.array_equal(mesh.extract(parent, box, order), whole[8:40])


def test_cell_injection_uses_the_fine_interior_and_never_wraps_a_box_edge():
    parent = np.full(32, -10.0)
    box = mesh.Box((8,), (16,), centering="cell")
    fine_x = 8 + (np.arange(32) + 0.5) / 2
    fine = fine_x**3
    fine[:6], fine[-6:] = 1e9, -1e9
    got = mesh.inject(parent, fine, box, order=4, buffer=6)
    assert got[12:20] == pytest.approx((np.arange(12, 20) + 0.5) ** 3)
    assert np.array_equal(got[:12], parent[:12])
    assert np.array_equal(got[20:], parent[20:])
    assert np.all(parent == -10)


def test_bad_cell_transfer_conventions_are_rejected():
    with pytest.raises(ValueError, match="centering"):
        mesh.Box((0,), (8,), centering="unknown")
    with pytest.raises(ValueError, match="even axis"):
        mesh.restrict(np.zeros(9), centering="cell")


@pytest.mark.parametrize("centering", ["vertex", "cell"])
@pytest.mark.parametrize("order", [2, 4, 6])
def test_compiled_device_transfers_match_numpy(centering, order):
    import jax

    jnp = mesh._module("jax")

    parent = np.random.default_rng(22).normal(size=(24, 12))
    box = mesh.Box((6, 0), (12, 12), centering=centering)

    def transfer(values, backend):
        fine = mesh.extract(values, box, order, backend=backend)
        return fine, mesh.inject(values, fine, box, order=order, buffer=4, backend=backend)

    want = transfer(parent, "numpy")
    got = jax.jit(lambda v: transfer(v, "jax"))(jnp.asarray(parent))
    for actual, expected in zip(got, want, strict=True):
        assert np.asarray(actual) == pytest.approx(expected, abs=3e-15)


@pytest.mark.parametrize("centering", ["vertex", "cell"])
@pytest.mark.slow
def test_device_refined_evolution_matches_numpy(centering):
    results = []
    for backend in ("numpy", "jax"):
        state, spacing = bssn.gauge_wave(shape=(16, 4, 4), backend=backend)
        evolution = bssn.Evolution.build(
            spacing, slicing="harmonic", shift_condition="frozen", backend=backend
        )
        box = mesh.Box((4, 0, 0), (8, 4, 4), centering=centering)
        pair = refined.Hierarchy.build(evolution, box, (16, 4, 4), buffer=3)
        fine = pair.refine(state)
        results.append(pair.step(state, fine))
    for actual, expected in zip(results[1], results[0], strict=True):
        for name in expected:
            assert np.asarray(actual[name]) == pytest.approx(expected[name], abs=2e-13), name


@pytest.mark.parametrize("levels", [2, 3, 5])
@pytest.mark.slow
def test_cell_puncture_is_a_common_reflection_centre_without_a_singular_sample(levels):
    setup, states = puncture.NestedPuncture.build(
        n=32, extent=16, levels=levels, centering="cell", data="trumpet", conformal="W"
    )
    assert setup.position == (8.0, 8.0, 8.0)
    for axis, state in zip(setup.axes, states, strict=True):
        assert axis + axis[::-1] == pytest.approx(16.0)
        assert np.min(abs(axis - 8)) == pytest.approx((axis[1] - axis[0]) / 2)
        assert np.asarray(state["alpha"]) == pytest.approx(np.asarray(state["alpha"])[::-1])
        assert np.asarray(state["beta0"]) == pytest.approx(-np.asarray(state["beta0"])[::-1])
    with pytest.raises(ValueError, match="lands on the puncture"):
        puncture.NestedPuncture.build(n=32, levels=levels, centering="cell", offset=1)


@pytest.mark.slow
@pytest.mark.benchmark
def test_cell_refinement_preserves_gauge_wave_convergence():
    errors = []
    for n in (32, 64, 128):
        # For this x-directed wave, translating the samples by h/2 is
        # exactly the same as evaluating its analytic phase at t-h/2.
        initial, spacing = bssn.gauge_wave(shape=(n, 8, 8), time=-0.5 / n)
        evolution = bssn.Evolution.build(spacing, slicing="harmonic", shift_condition="frozen")
        box = mesh.Box((n // 8, 0, 0), (3 * n // 4, 8, 8), centering="cell")
        hierarchy = refined.Hierarchy.build(evolution, box, (n, 8, 8), buffer=6)
        fine, _ = bssn.gauge_wave(shape=(2 * n, 16, 16), time=-0.25 / n)
        lo, hi = 2 * box.origin[0], 2 * (box.origin[0] + box.shape[0])
        fine = {k: v[lo:hi] for k, v in fine.items()}
        steps = round(0.25 / hierarchy.coarse.time_step)
        _, result = hierarchy.run(initial, fine, steps, 0.25 / steps)
        exact, _ = bssn.gauge_wave(shape=(2 * n, 16, 16), time=0.25 - 0.25 / n)
        x = (np.arange(lo, hi) + 0.5) / (2 * n)
        keep = (x >= 0.35) & (x <= 0.65)
        errors.append(
            max(
                np.sqrt(np.mean((np.asarray(result[k])[keep] - np.asarray(v)[lo:hi][keep]) ** 2))
                for k, v in exact.items()
            )
        )
    ratios = np.array(errors[:-1]) / errors[1:]
    assert np.all(ratios > 11), (errors, ratios)
