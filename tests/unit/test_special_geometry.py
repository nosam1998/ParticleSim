"""The quintic's couplings from mirror symmetry, against published and exact values (issue #87)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from particlesim.theories.special_geometry import (
    ZETA3,
    QuinticModuli,
    chern_numbers,
    fundamental_period,
    instanton_numbers,
    mirror_map,
    yukawa_expansion,
)

#: Candelas, de la Ossa, Green and Parkes, Nucl. Phys. B359 (1991) 21.
PUBLISHED_INSTANTON_NUMBERS = (2875, 609250, 317206375, 242467530000, 229305888887625)


def test_instanton_numbers_are_the_published_ones():
    """The issue's acceptance: emitted couplings against a published numerical result.

    Nothing is fitted. The numbers come out of the Picard-Fuchs periods of
    the mirror quintic by exact rational arithmetic.
    """
    assert instanton_numbers(5) == PUBLISHED_INSTANTON_NUMBERS


def test_instanton_numbers_to_degree_ten_are_whole():
    """Integrality is not built in: each ``n_d`` is a rational the calculation could get wrong.

    ``instanton_numbers`` raises if any comes out fractional, so this only
    has to ask for them. They are also positive and grow roughly as
    ``5^(5d)``, the conifold's radius of convergence.
    """
    numbers = instanton_numbers(10)
    assert all(n > 0 for n in numbers)
    growth = [np.log(b / a) for a, b in zip(numbers[4:], numbers[5:], strict=False)]
    assert growth[-1] == pytest.approx(5 * np.log(5), rel=0.1)


def test_the_yukawa_expansion_and_mirror_map_start_as_published():
    assert yukawa_expansion(3) == (5, 2875, 4876875, 8564575000)
    assert mirror_map(3)[1:] == (1, -770, 171525)
    assert fundamental_period(3) == [1, 120, 113400, 168168000]


def test_the_topology_is_derived_from_the_chern_class():
    """``c(X) = (1 + J)^5 / (1 + 5J)``, and two independent counts of ``h^(2,1)`` agree."""
    topology = chern_numbers()
    assert topology["c1"] == 0  # Calabi-Yau
    assert (topology["c2"], topology["c3"]) == (10, -40)
    assert topology["euler"] == -200
    assert topology["c2_dot_J"] == 50
    assert (topology["h11"], topology["h21"]) == (1, 101)


def test_the_normalised_yukawa_tends_to_two_over_root_three():
    """``e^K |kappa| G^(-3/2) -> 2/sqrt(3)`` at large volume, whatever the intersection number.

    The classical ``e^-K = (20/3) y^3`` and ``G = 3 / (4 y^2)`` give exactly that.
    The ``zeta(3)`` correction then falls as ``1/y^3``, so the approach is
    monotone.
    """
    moduli = QuinticModuli()
    limit = 2 / np.sqrt(3)
    gaps = [moduli.normalized_yukawa(complex(0.1, y)) - limit for y in (3.0, 5.0, 10.0, 30.0)]
    assert all(g > 0 for g in gaps)
    assert all(a > b for a, b in itertools.pairwise(gaps))
    assert gaps[-1] < 1e-4


def test_the_alpha_prime_correction_is_the_euler_number_term():
    """At large volume ``e^-K - (20/3) y^3 -> -chi zeta(3) / (4 pi^3) = 50 zeta(3) / pi^3``."""
    moduli = QuinticModuli()
    y = 20.0
    correction = moduli.exp_minus_kahler(complex(0.3, y)) - 20.0 / 3.0 * y**3
    assert correction == pytest.approx(50 * ZETA3 / np.pi**3, rel=1e-9)


def test_the_metric_is_the_laplacian_of_the_kahler_potential():
    """``G = d_t d_tbar K = (1/4)(d_x^2 + d_y^2) K``, by finite differences, instantons included."""
    moduli = QuinticModuli()
    t, h = complex(0.23, 1.7), 1e-3
    K = moduli.kahler_potential
    laplacian = (K(t + h) + K(t - h) + K(t + 1j * h) + K(t - 1j * h) - 4 * K(t)) / h**2
    assert moduli.metric(t) == pytest.approx(laplacian / 4, rel=1e-5)
    assert moduli.metric(t) > 0


def test_the_prepotentials_derivatives_are_consistent():
    """``F_t``, ``F_tt`` and ``F_ttt`` are each the holomorphic derivative of the one before."""
    moduli = QuinticModuli()
    t, h = complex(0.1, 1.6), 1e-4
    for order in range(3):
        forward = moduli.prepotential(t + h)[order]
        backward = moduli.prepotential(t - h)[order]
        derivative = moduli.prepotential(t)[order + 1]
        assert (forward - backward) / (2 * h) == pytest.approx(derivative, rel=1e-7)


def test_instantons_matter_near_the_conifold_and_vanish_at_large_volume():
    moduli = QuinticModuli()
    near, far = moduli.yukawa(complex(0.0, 1.5)), moduli.yukawa(complex(0.0, 5.0))
    assert abs(near - 5) > 0.05
    assert abs(far - 5) < 1e-9


def test_a_volume_must_be_positive():
    with pytest.raises(ValueError, match="volume"):
        QuinticModuli().yukawa(complex(0.0, -1.0))
