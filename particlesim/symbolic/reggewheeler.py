"""Regge-Wheeler perturbations of Schwarzschild, and Leaver's recursion.

Issue #50. The quasinormal frequencies of Schwarzschild are what a ringdown
extracted from a 3-D run has to be compared *against*, so they have to come
from somewhere trustworthy. They come from here: the master equation is
written down, both boundary conditions are factored out as exponents, and
the resulting power series gives a three-term recursion whose coefficients
this module derives rather than quotes.

**The master equation.** A single multipole of a perturbation of
Schwarzschild obeys

    f d/dr (f dpsi/dr) + (omega^2 - V_s) psi = 0,
    f = 1 - 1/r,   V_s = f (l(l+1)/r^2 + (1 - s^2)/r^3)

in units ``2M = 1``, with ``s = 2`` for axial gravitational perturbations
(Regge-Wheeler), ``s = 1`` electromagnetic and ``s = 0`` scalar. Polar
gravitational perturbations obey the Zerilli equation instead, whose
potential is different but which is isospectral to this one -- the same
quasinormal frequencies, by the Chandrasekhar-Detweiler transformation.

**Why the boundary conditions are the whole problem.** A quasinormal mode is
ingoing at the horizon and outgoing at infinity. Both are conditions at
singular points of the equation, and for complex ``omega`` the two solutions
at each end differ by a *growing* exponential, so integrating inward from
large ``r`` loses the mode being looked for. The way out is to put both
behaviours into the ansatz. Near ``r = 1`` the indicial exponents of the
master equation are ``+-i omega``, and ingoing picks ``(r-1)^(-i omega)``;
at large ``r`` the outgoing solution goes as ``e^(i omega r) r^(i omega)``.
So

    psi = (r-1)^(-i omega) r^(2 i omega) e^(i omega r) phi(r)

leaves ``phi`` with exponents ``0`` and ``2 i omega`` at the horizon. The
analytic branch is the ingoing one, which is the point: a *power series*
in ``u = 1 - 1/r`` now selects the right behaviour at the horizon
automatically, and the surviving condition is at ``u = 1``.

**What the recursion is for.** With ``phi = sum a_n u^n`` the equation
becomes a three-term recursion. Such a recursion has a two-dimensional
solution space, and only its *minimal* solution gives a series that
converges at ``u = 1``, which is infinity -- that convergence is the
outgoing condition, and it holds only at the quasinormal frequencies. The
condition is expressible as a continued fraction, which is what
:mod:`particlesim.analysis.qnm` roots.

**Why derive it here rather than copy it.** A wrong coefficient in a
three-term recursion produces a continued fraction with roots, and those
roots look like quasinormal frequencies. Every step below is checked: the
polynomial form against the master equation, the split into
``A phi'' + B phi' + C phi`` against its own reassembly, and the recursion
against the series put back into the equation. The tests then check the
frequencies the recursion produces against the light-ring limit, which is
analytic and independent of any of this.
"""

from __future__ import annotations

import sympy as sp

#: The radial coordinate, the compactified variable, and the parameters.
#:
#: ``u = 1 - 1/r`` maps the horizon to 0 and infinity to 1, which is what
#: makes a power series about the horizon reach all the way out.
r, u = sp.symbols("r u", positive=True)
omega = sp.symbols("omega")
ell, spin = sp.symbols("ell spin")


def potential(l=ell, s=spin):
    """``V_s``, the Regge-Wheeler potential for spin-weight ``s``.

    The ``(1 - s^2)/r^3`` term is the one that distinguishes the three
    fields: positive for a scalar, absent for electromagnetism, and
    ``-3/r^3`` for gravity, where it is what makes the potential dip below
    the centrifugal barrier.
    """
    f = 1 - 1 / r
    return f * (l * (l + 1) / r**2 + (1 - s**2) / r**3)


def master_equation(psi, l=ell, s=spin):
    """``f (f psi')' + (omega^2 - V) psi``, as an expression to set to zero."""
    f = 1 - 1 / r
    return f * sp.diff(f * sp.diff(psi, r), r) + (omega**2 - potential(l, s)) * psi


def polynomial_form(psi, l=ell, s=spin):
    """The same equation with the denominators cleared -- ``r^4`` times it.

    Written out rather than computed so that :func:`check_polynomial_form`
    has something independent to compare against; the coefficients here are
    what the substitution below is actually applied to.
    """
    return (
        r**2 * (r - 1) ** 2 * sp.diff(psi, r, 2)
        + r * (r - 1) * sp.diff(psi, r)
        + (omega**2 * r**4 - (r - 1) * (l * (l + 1) * r + 1 - s**2)) * psi
    )


def check_polynomial_form() -> bool:
    """Is the cleared form really ``r^4`` times the master equation?"""
    psi = sp.Function("psi")(r)
    difference = r**4 * master_equation(psi) - polynomial_form(psi)
    return sp.simplify(sp.expand(difference)) == 0


def leaver_prefactor(w=omega):
    """``(r-1)^(-i w) r^(2 i w) e^(i w r)``: both boundary conditions.

    Ingoing at the horizon and outgoing at infinity, as exponents. What
    multiplies it is analytic at the horizon and bounded at infinity, so the
    mode condition becomes a statement about a power series instead of about
    competing exponentials.
    """
    return (r - 1) ** (-sp.I * w) * r ** (2 * sp.I * w) * sp.exp(sp.I * w * r)


def radial_coefficients(l=ell, s=spin):
    """``(A, B, C)`` with ``A phi'' + B phi' + C phi = 0`` after factoring.

    The prefactor divides out exactly -- it is nowhere zero on ``r > 1`` --
    leaving polynomial coefficients. ``A = r^2 (r-1)^2`` is unchanged,
    because the prefactor does not touch the second-derivative coefficient.
    """
    phi = sp.Function("phi")
    prefactor = leaver_prefactor()
    substituted = polynomial_form(prefactor * phi(r), l, s).doit()
    reduced = sp.expand(sp.simplify(sp.expand(substituted / prefactor)))

    second = sp.simplify(reduced.coeff(sp.diff(phi(r), r, 2)))
    first = sp.simplify(reduced.coeff(sp.diff(phi(r), r)))
    zeroth = sp.simplify(reduced.coeff(phi(r)))
    residual = sp.simplify(
        reduced - (second * sp.diff(phi(r), r, 2) + first * sp.diff(phi(r), r) + zeroth * phi(r))
    )
    if residual != 0:
        raise ValueError(
            f"the factored equation did not split into second, first and "
            f"zeroth order parts: {residual} was left over"
        )
    return second, first, zeroth


def compactified_coefficients(l=ell, s=spin):
    """The same equation in ``u``, as three :class:`sympy.Poly` in ``u``.

    ``du/dr = 1/r^2 = (1-u)^2``, so ``d/dr = (1-u)^2 d/du`` and
    ``d^2/dr^2 = (1-u)^4 d^2/du^2 - 2 (1-u)^3 d/du``. A factor
    ``u/(1-u)^2`` is common to all three and is divided out, which is what
    leaves the degrees at 3, 2 and 1 -- and those degrees are why the
    recursion has three terms rather than four.
    """
    second, first, zeroth = radial_coefficients(l, s)
    g = sp.Function("g")
    to_u = {r: 1 / (1 - u)}

    in_u = (
        second.subs(to_u)
        * ((1 - u) ** 4 * sp.diff(g(u), u, 2) - 2 * (1 - u) ** 3 * sp.diff(g(u), u))
        + first.subs(to_u) * (1 - u) ** 2 * sp.diff(g(u), u)
        + zeroth.subs(to_u) * g(u)
    )
    in_u = sp.expand(sp.simplify(sp.expand(in_u * (1 - u) ** 2 / u)))

    parts = (
        sp.Poly(sp.expand(in_u.coeff(sp.diff(g(u), u, 2))), u),
        sp.Poly(sp.expand(in_u.coeff(sp.diff(g(u), u))), u),
        sp.Poly(sp.expand(in_u.coeff(g(u))), u),
    )
    residual = sp.simplify(
        in_u
        - (
            parts[0].as_expr() * sp.diff(g(u), u, 2)
            + parts[1].as_expr() * sp.diff(g(u), u)
            + parts[2].as_expr() * g(u)
        )
    )
    if residual != 0:
        raise ValueError(f"the u-form did not split cleanly: {residual} was left over")
    return parts


def recursion_coefficients(index=None, l=ell, s=spin):
    """``(alpha_k, beta_k, gamma_k)`` for ``alpha a_{k+1} + beta a_k + gamma a_{k-1} = 0``.

    Obtained by inserting ``sum a_n u^n`` and collecting the coefficient of
    ``u^k``. The polynomial degrees 3, 2 and 1 shift the three terms onto
    exactly ``a_{k+1}``, ``a_k`` and ``a_{k-1}``: a degree-4 coefficient of
    ``g''`` would have made this a four-term recursion with no continued
    fraction.

    At ``k = 0`` the ``a_{k-1}`` term is absent from the equation rather than
    merely multiplied by zero, and the convention ``a_{-1} = 0`` makes the
    general formula agree -- the ``k(k-1)`` weight kills the one
    second-derivative contribution that would otherwise appear.
    """
    k = sp.symbols("k") if index is None else index
    a = sp.Function("a")
    parts = compactified_coefficients(l, s)

    def contribution(poly, shift, weight):
        total = 0
        for power, coefficient in enumerate(reversed(poly.all_coeffs())):
            position = k + shift - power
            total += coefficient * weight(position) * a(position)
        return total

    row = sp.expand(
        contribution(parts[0], 2, lambda m: m * (m - 1))
        + contribution(parts[1], 1, lambda m: m)
        + contribution(parts[2], 0, lambda m: sp.Integer(1))
    )
    alpha = sp.simplify(row.coeff(a(k + 1)))
    beta = sp.simplify(row.coeff(a(k)))
    gamma = sp.simplify(row.coeff(a(k - 1)))
    residual = sp.simplify(row - alpha * a(k + 1) - beta * a(k) - gamma * a(k - 1))
    if residual != 0:
        raise ValueError(
            f"the series gave more than three terms: {residual} was left over, "
            "so there is no continued fraction"
        )
    return alpha, beta, gamma


__all__ = [
    "check_polynomial_form",
    "compactified_coefficients",
    "ell",
    "leaver_prefactor",
    "master_equation",
    "omega",
    "polynomial_form",
    "potential",
    "r",
    "radial_coefficients",
    "recursion_coefficients",
    "spin",
    "u",
]
