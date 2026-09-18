"""Leaver's recursion: does the derivation hold together, and does the code match it?

The point of deriving a three-term recursion symbolically is that a typo in
one produces a continued fraction with roots, and those roots look exactly
like quasinormal frequencies. So every step is checked against something
that was not used to produce it: the cleared equation against the master
equation, the factored coefficients against their own reassembly, and the
recursion against a truncated series substituted back in.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from particlesim.analysis.qnm import leaver_coefficients
from particlesim.symbolic import reggewheeler as rw


def test_the_cleared_equation_is_r4_times_the_master_equation():
    """The starting point, so that everything after it starts right."""
    assert rw.check_polynomial_form()


def test_the_potential_has_the_right_spin_term():
    """``(1 - s^2)/r^3``: positive for a scalar, absent for light, -3 for gravity."""
    f = 1 - 1 / rw.r
    for spin, coefficient in ((0, 1), (1, 0), (2, -3)):
        expected = f * (rw.ell * (rw.ell + 1) / rw.r**2 + coefficient / rw.r**3)
        assert sp.simplify(rw.potential(s=spin) - expected) == 0, spin


def test_factoring_the_prefactor_out_leaves_polynomial_coefficients():
    """It divides out exactly: it is nowhere zero on ``r > 1``."""
    second, first, zeroth = rw.radial_coefficients()
    for coefficient in (second, first, zeroth):
        assert sp.Poly(sp.expand(coefficient), rw.r).total_degree() <= 4
    # The second-derivative coefficient is untouched by the substitution.
    assert sp.simplify(second - rw.r**2 * (rw.r - 1) ** 2) == 0


def test_the_horizon_exponents_are_zero_and_two_i_omega():
    """Which is *why* a power series is the ingoing solution.

    Divide the factored equation by ``(r-1)`` and set ``r = 1 + rho``: the
    second-derivative coefficient goes as ``rho`` and the first-derivative
    one tends to ``1 - 2 i w``, so the indicial polynomial is
    ``p(p - 1) + (1 - 2 i w) p = p (p - 2 i w)``. The analytic branch ``p=0``
    is the one a power series finds, and it is the ingoing one; ``p = 2 i w``
    is ``psi ~ (r-1)^(+i w)``, outgoing at the horizon, which no power series
    can represent. Were these the other way round the method would impose
    the wrong condition and still converge to something.
    """
    second, first, _ = rw.radial_coefficients()
    rho, exponent = sp.symbols("rho p")
    leading = sp.limit(sp.simplify(second / (rw.r - 1)).subs(rw.r, 1 + rho) / rho, rho, 0)
    slope = sp.limit(sp.simplify(first / (rw.r - 1)).subs(rw.r, 1 + rho), rho, 0)
    assert sp.simplify(leading - 1) == 0
    assert sp.simplify(slope - (1 - 2 * sp.I * rw.omega)) == 0

    indicial = sp.factor(sp.expand(leading * exponent * (exponent - 1) + slope * exponent))
    assert sp.simplify(indicial - exponent * (exponent - 2 * sp.I * rw.omega)) == 0


def test_the_compactified_degrees_are_what_make_it_three_term():
    """3, 2 and 1. A degree-4 second-derivative coefficient means no continued fraction."""
    parts = rw.compactified_coefficients()
    assert [part.degree() for part in parts] == [3, 2, 1]
    # And the g'' coefficient is u(1-u)^2, which vanishes at both endpoints:
    # u=0 is the horizon and u=1 is infinity, and both are singular points.
    assert sp.simplify(parts[0].as_expr() - rw.u * (rw.u - 1) ** 2) == 0


def test_the_recursion_matches_the_numerical_coefficients():
    """The cross-check that makes a typo in :mod:`particlesim.analysis.qnm` fail.

    The solver carries the coefficients written out, because evaluating a
    continued fraction a thousand terms deep through SymPy is not
    affordable. This is what keeps the two in step.
    """
    index = sp.symbols("k")
    derived = rw.recursion_coefficients(index=index)
    written = leaver_coefficients(index, rw.omega, rw.ell, rw.spin)
    for label, left, right in zip(("alpha", "beta", "gamma"), derived, written, strict=True):
        assert sp.simplify(sp.expand(left - right)) == 0, label


def test_the_recursion_coefficients_have_the_expected_closed_forms():
    """Factored shapes worth naming, because they are easy to mistype.

    ``gamma_k = (k - 2 i w)^2 - s^2`` is the one that carries the spin, and
    it is the only place the spin enters the recursion at all -- the rest of
    the ``s`` dependence sits in ``beta_k`` as a bare ``s^2``.
    """
    index = sp.symbols("k")
    alpha, beta, gamma = rw.recursion_coefficients(index=index)
    assert sp.simplify(alpha - (index + 1) * (index + 1 - 2 * sp.I * rw.omega)) == 0
    assert sp.simplify(gamma - ((index - 2 * sp.I * rw.omega) ** 2 - rw.spin**2)) == 0
    # beta is where l enters, and it does so only as l(l+1).
    assert sp.simplify(sp.expand(beta).coeff(rw.ell, 2) + 1) == 0
    assert sp.simplify(sp.expand(beta).coeff(rw.ell, 1) + 1) == 0


@pytest.mark.parametrize(
    ("l", "spin", "frequency"),
    [
        (2, 2, 0.747343368836 - 0.177924631378j),
        (3, 2, 1.198886577196 - 0.185406097602j),
        (0, 0, 0.220909878161 - 0.209791434174j),
        (1, 1, 0.496526528356 - 0.184975435906j),
    ],
)
def test_the_series_satisfies_the_equation_it_came_from(l, spin, frequency):
    """A truncated series put back into the ``u``-form ODE, at ``u = 0.3``.

    This is the check that does not go through the continued fraction at
    all: build forty terms from the recursion, evaluate the series and its
    two derivatives, and hit the polynomial coefficients with them. A wrong
    recursion fails here at order one. Measured residual is around 1e-16.

    The frequencies are in units ``2M = 1``, which is what the recursion
    uses.
    """
    parts = [
        [
            complex(sp.sympify(value).subs(rw.omega, frequency))
            for value in reversed(part.all_coeffs())
        ]
        for part in rw.compactified_coefficients(l=l, s=spin)
    ]
    terms = 40
    a = np.zeros(terms + 2, dtype=complex)
    a[0] = 1.0
    alpha, beta, _ = leaver_coefficients(0, frequency, l, spin)
    a[1] = -beta * a[0] / alpha
    for k in range(1, terms):
        alpha, beta, gamma = leaver_coefficients(k, frequency, l, spin)
        a[k + 1] = -(beta * a[k] + gamma * a[k - 1]) / alpha

    point = 0.3
    power = np.arange(terms + 2)
    value = np.sum(a * point**power)
    first = np.sum(a[1:] * power[1:] * point ** (power[1:] - 1))
    second = np.sum(a[2:] * power[2:] * (power[2:] - 1) * point ** (power[2:] - 2))

    def evaluate(coefficients):
        return sum(c * point**n for n, c in enumerate(coefficients))

    residual = abs(
        evaluate(parts[0]) * second + evaluate(parts[1]) * first + evaluate(parts[2]) * value
    )
    assert residual < 1e-12, residual


def test_a_four_term_recursion_is_refused_rather_than_truncated(monkeypatch):
    """The guard, exercised by handing it a degree-4 second-derivative term.

    A four-term recursion has no continued fraction, so silently dropping
    the extra term would give plausible and wrong frequencies. The guard is
    reached by patching the compactified coefficients to raise the degree of
    the ``g''`` term by one, which is exactly the mistake a slip in the
    change of variables would make.
    """
    real = rw.compactified_coefficients

    def one_degree_higher(*args, **kwargs):
        parts = real(*args, **kwargs)
        return (sp.Poly(parts[0].as_expr() * rw.u, rw.u), parts[1], parts[2])

    monkeypatch.setattr(rw, "compactified_coefficients", one_degree_higher)
    with pytest.raises(ValueError, match="more than three terms"):
        rw.recursion_coefficients()
