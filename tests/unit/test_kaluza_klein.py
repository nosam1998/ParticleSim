"""Kaluza-Klein reduction against the spectrum it is supposed to produce.

Issue #73 asks that the reduction reproduce known KK spectra. Those are
closed forms, so comparing one formula to another would prove nothing: the
lattice path builds the Laplacian on the compact space and *diagonalises* it,
and what is asserted is that its eigenvalues are the lattice dispersion
exactly and the continuum tower in the limit.
"""

from __future__ import annotations

import numpy as np
import pytest

from particlesim.theories.kk import (
    KaluzaKlein,
    Torus,
    continuum_limit,
    lattice_cost,
    lattice_masses,
    lattice_spectrum,
    orbifold_spectrum,
)
from particlesim.theories.limits import check_gr_limit
from particlesim.theories.registry import list_theories

# --- the closed-form tower ------------------------------------------------


def test_a_circle_gives_the_n_over_r_tower():
    """Masses ``n/R``, and the degeneracy is two everywhere above zero."""
    spectrum = Torus(radii=(2.0,)).spectrum(levels=3)
    assert [mass for mass, _ in spectrum] == pytest.approx([0.0, 0.5, 1.0, 1.5])
    assert [count for _, count in spectrum] == [1, 2, 2, 2]


def test_a_square_torus_has_the_degeneracies_of_the_gaussian_integers():
    """4, 4, 4, 8: the count is the number of lattice points on a circle.

    This is the part a formula for the masses alone would miss. Level ``5``
    is eight-fold because ``(+-1, +-2)`` and ``(+-2, +-1)`` all have
    ``n^2 = 5``, while level ``4`` is only four-fold. Getting the
    multiplicities right is most of what "reproduces the spectrum" means --
    they are what a 4D observer would actually count.
    """
    spectrum = Torus(radii=(1.0, 1.0)).spectrum(levels=2)
    assert [mass for mass, _ in spectrum][:5] == pytest.approx(
        [0.0, 1.0, np.sqrt(2.0), 2.0, np.sqrt(5.0)]
    )
    assert [count for _, count in spectrum][:5] == [1, 4, 4, 4, 8]


def test_unequal_radii_split_the_degeneracy():
    """A rectangle resolves the square's four-fold level into two pairs."""
    spectrum = Torus(radii=(1.0, 2.0)).spectrum(levels=2)
    assert [mass for mass, _ in spectrum][:3] == pytest.approx([0.0, 0.5, 1.0])
    assert [count for _, count in spectrum][:3] == [1, 2, 4]


def test_the_radii_are_moduli_the_whole_tower_rides_on():
    """``dm/dR = -1/R^2``: growing a dimension lowers its tower."""
    torus = Torus(radii=(2.0, 4.0))
    assert torus.radion_coupling() == pytest.approx((-0.25, -0.0625))
    assert torus.volume == pytest.approx((2.0 * np.pi * 2.0) * (2.0 * np.pi * 4.0))


@pytest.mark.parametrize("radii", [(), (0.0,), (1.0, -2.0)])
def test_a_torus_refuses_a_degenerate_geometry(radii):
    with pytest.raises(ValueError):
        Torus(radii=radii)


def test_mass_squared_refuses_the_wrong_number_of_modes():
    with pytest.raises(ValueError, match="expected 2 mode numbers"):
        Torus(radii=(1.0, 1.0)).mass_squared((1,))


# --- the orbifold ---------------------------------------------------------


def test_the_orbifold_halves_the_tower():
    """``+n`` and ``-n`` are identified, so every level is non-degenerate."""
    circle = Torus(radii=(2.0,))
    even = orbifold_spectrum(circle, levels=3, parity="even")
    assert [mass for mass, _ in even] == pytest.approx([0.0, 0.5, 1.0, 1.5])
    assert [count for _, count in even] == [1, 1, 1, 1]


def test_an_odd_field_has_no_zero_mode():
    """The one state that distinguishes the two parities, and the point of both.

    Even fields are cosines and keep ``n = 0``; odd fields are sines and
    cannot. Projecting a zero mode away by boundary conditions is how the
    model-building literature breaks symmetries on an orbifold, and the
    difference here is exactly one massless state -- not a shifted mass, an
    absent one.
    """
    circle = Torus(radii=(2.0,))
    odd = orbifold_spectrum(circle, levels=3, parity="odd")
    assert [mass for mass, _ in odd] == pytest.approx([0.5, 1.0, 1.5])
    assert 0.0 not in [mass for mass, _ in odd]

    even = orbifold_spectrum(circle, levels=3, parity="even")
    assert len(even) == len(odd) + 1


def test_orbifold_refuses_an_unknown_parity():
    with pytest.raises(ValueError, match="parity must be"):
        orbifold_spectrum(Torus(), parity="sideways")


# --- the lattice, which is the actual computation -------------------------


@pytest.mark.parametrize("points", [16, 32, 64])
def test_the_diagonalised_lattice_is_the_lattice_dispersion(points):
    """Eigenvalues of the built matrix equal ``(4/h^2) sin^2(pi k/N)``, to 1e-13.

    The matrix is assembled and handed to a dense eigensolver, so this is a
    numerical result meeting a closed form rather than one formula meeting
    another. It also pins the *lattice's* answer, which is not the continuum
    one -- see the convergence test below.
    """
    radius = 2.0
    circle = Torus(radii=(radius,))
    masses = lattice_masses(circle, points)

    for level in (1, 2, 3):
        predicted = (level / radius) * continuum_limit(level, points)
        assert masses[2 * level - 1] == pytest.approx(predicted, rel=1e-13)


def test_the_lattice_converges_to_the_continuum_tower_at_second_order():
    """Error quarters as the lattice doubles: 6.4e-3, 1.6e-3, 4.0e-4.

    Asserted as a rate rather than a tolerance, because a tolerance loose
    enough to pass at 16 points says nothing, and one tight enough to be
    interesting at 64 would simply be a statement about 64.
    """
    radius = 2.0
    circle = Torus(radii=(radius,))
    errors = []
    for points in (16, 32, 64):
        measured = lattice_masses(circle, points)[1]
        errors.append(abs(measured / (1.0 / radius) - 1.0))

    ratios = [before / after for before, after in zip(errors[:-1], errors[1:], strict=True)]
    assert all(ratio == pytest.approx(4.0, rel=0.05) for ratio in ratios), ratios


def test_the_lattice_always_undershoots():
    """``sin(x)/x < 1``: a lattice mode is softer than its continuum twin.

    One-signed, which matters -- a scheme whose error changed sign with the
    level would be aliasing rather than discretising.
    """
    circle = Torus(radii=(1.0,))
    points = 32
    masses = lattice_masses(circle, points)
    for level in (1, 3, 7, 12):
        assert masses[2 * level - 1] < level / 1.0


def test_a_two_torus_lattice_reproduces_the_split_degeneracies():
    """The 2D case, where the eigenvalues have to arrive in the right groups."""
    torus = Torus(radii=(1.0, 1.0))
    spectrum = lattice_spectrum(torus, 8)
    assert spectrum[0] == pytest.approx(0.0, abs=1e-12)
    # four states at the first level, as in the continuum
    assert spectrum[1:5] == pytest.approx([spectrum[1]] * 4, rel=1e-12)
    assert spectrum[5] > spectrum[4] * 1.5


# --- the cost model -------------------------------------------------------


def test_the_cost_model_is_why_three_dimensions_are_refused():
    """``(points^n)^3`` operations, reported before they are paid.

    Three extra dimensions on a 16-point lattice is 6.9e10 operations for a
    dense eigendecomposition. The refusal quotes the number rather than
    declaring a policy, so the limit can be re-argued when the solver stops
    being dense.
    """
    cost = lattice_cost(Torus(radii=(1.0,) * 3), 16)
    assert cost["sites"] == 16.0**3
    assert cost["operations"] == pytest.approx(16.0**9)

    with pytest.raises(ValueError, match="impossible one"):
        lattice_spectrum(Torus(radii=(1.0,) * 3), 4)


def test_lattice_cost_refuses_a_degenerate_grid():
    with pytest.raises(ValueError, match="at least two points"):
        lattice_cost(Torus(), 1)


# --- the plugin -----------------------------------------------------------


def test_the_plugin_is_registered_and_reduces_to_gr():
    """Registered as a gravity theory, and it passes the shared limit harness."""
    assert list_theories()["string.kk"] is KaluzaKlein
    report = check_gr_limit(KaluzaKlein, check_action=False)
    assert report.checked
    assert report.passed, report.reasons


def test_the_limit_is_the_inverse_radius_so_a_test_can_set_it():
    """Decompactification is at infinity; the inverse radius puts it at zero.

    The same convention as the electromagnetic sector, and for the same
    reason: a limit you cannot evaluate is a claim, not a check.
    """
    assert KaluzaKlein().gr_limit() == {"inverse_radius": 0.0}
    assert KaluzaKlein(inverse_radius=0.0).observable_predictions()["decoupled"]
    assert not KaluzaKlein(inverse_radius=0.5).observable_predictions()["decoupled"]


def test_a_decompactified_theory_has_no_torus():
    """Zero inverse radius is the 4D limit, not a geometry of infinite size."""
    with pytest.raises(ValueError, match="no torus"):
        KaluzaKlein(inverse_radius=0.0).torus()

    torus = KaluzaKlein(inverse_radius=0.25, dimensions=2).torus()
    assert torus.radii == (4.0, 4.0)
    assert torus.spectrum(1)[1][0] == pytest.approx(0.25)
