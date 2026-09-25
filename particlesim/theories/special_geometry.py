"""The quintic's Kahler moduli space, from mirror symmetry (issue #87).

A heterotic compactification on the quintic threefold with the standard
embedding has an ``E_6`` gauge group, with matter in the ``27`` and the
``27bar``. There is one ``27bar`` for each Kahler modulus, and the quintic
has one, ``t = B + i J``, the complexified size of its line class. The
``27bar^3`` Yukawa coupling is the classical intersection number ``5``,
corrected by worldsheet instantons, strings wrapping rational curves:

    kappa_ttt(t) = 5 + sum_d n_d d^3 q^d / (1 - q^d),    q = exp(2 pi i t)

``n_d`` counts the rational curves of degree ``d``. They are not computed by
counting curves. Mirror symmetry trades them for the periods of the mirror
quintic, and those are what this module computes:

- **The fundamental period** is ``w0(z) = sum (5n)! / (n!)^5 z^n``, the power
  series solution of the Picard-Fuchs equation
  ``theta^4 - 5 z (5 theta + 1)(5 theta + 2)(5 theta + 3)(5 theta + 4)``.
- **The mirror map.** The logarithmic solution
  ``w1 = w0 ln z + sum (5n)!/(n!)^5 5 (H_5n - H_n) z^n`` defines the flat
  coordinate ``t = w1 / (2 pi i w0)``. Here ``H_n`` is the harmonic number.
- **The coupling.** In the complex-structure coordinate ``z``, the mirror's
  Yukawa coupling is ``5 / (z^3 (1 - 5^5 z))``. Written in ``t``, in the
  gauge ``w0 = 1``, it is ``kappa_ttt(q)``.

Every step above is a power series with rational coefficients, so the
module works in exact arithmetic. The instanton numbers come out as
integers, which is a check in itself: nothing in the calculation forces a
rational ``n_d`` to be whole. They agree with Candelas, de la Ossa, Green and
Parkes (1991), the published numbers the issue's acceptance names:
``2875, 609250, 317206375, 242467530000, ...``. There are 2875 lines on a
quintic, which was known classically, and 609250 conics, which was not known
until this calculation.

**The rest of the four-dimensional data** comes from the prepotential

    F = -(5/6) t^3 + (c2.J / 24) t + zeta(3) chi / (2 (2 pi i)^3)
        - (2 pi i)^-3 sum_d n_d Li_3(q^d)

with ``chi = -200`` and ``c2.J = 50``, both derived below from the quintic's
Chern classes rather than quoted. ``F`` gives the Kahler potential
``e^-K = i [2 (F - Fbar) - (t - tbar)(F_t + Fbar_t)]``, the moduli-space
metric ``G = d_t d_tbar K``, and the coupling ``kappa_ttt = -F_ttt``. A real
quadratic term, which conventions differ on, cancels out of all three and is
left out.

The combination ``e^K |kappa_ttt| G^(-3/2)`` does not change under Kahler
transformations or reparametrisations of ``t``. It is the Yukawa coupling of
three ``27bar`` fields with canonically normalised kinetic terms, up to the
dilaton's overall factor. At large volume it tends to exactly ``2/sqrt(3)``,
whatever the intersection number. The tests hold it to that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import cache
from math import comb, factorial

import numpy as np

#: ``zeta(3)``, to double precision.
ZETA3 = 1.2020569031595942


def _multiply(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    """The product of two truncated power series of the same length."""
    size = len(a)
    out = [Fraction(0)] * size
    for i, x in enumerate(a):
        if x:
            for j in range(size - i):
                out[i + j] += x * b[j]
    return out


def _inverse(a: list[Fraction]) -> list[Fraction]:
    """``1 / a`` as a truncated power series; ``a[0]`` must be non-zero."""
    out = [Fraction(0)] * len(a)
    out[0] = 1 / a[0]
    for n in range(1, len(a)):
        out[n] = -sum(a[k] * out[n - k] for k in range(1, n + 1)) / a[0]
    return out


def _exponential(a: list[Fraction]) -> list[Fraction]:
    """``exp(a)`` for a series with ``a[0] = 0``, from ``e' = a' e``."""
    out = [Fraction(0)] * len(a)
    out[0] = Fraction(1)
    for n in range(1, len(a)):
        out[n] = sum(k * a[k] * out[n - k] for k in range(1, n + 1)) / n
    return out


def _compose(f: list[Fraction], g: list[Fraction]) -> list[Fraction]:
    """``f(g(q))`` for a series ``g`` with no constant term."""
    out = [Fraction(0)] * len(f)
    power = [Fraction(1)] + [Fraction(0)] * (len(f) - 1)
    for k, coefficient in enumerate(f):
        if k:
            power = _multiply(power, g)
        for n in range(len(f)):
            out[n] += coefficient * power[n]
    return out


def _harmonic(n: int) -> Fraction:
    return sum((Fraction(1, j) for j in range(1, n + 1)), Fraction(0))


def fundamental_period(order: int) -> list[int]:
    """``(5n)! / (n!)^5`` for ``n = 0 .. order``: ``w0``'s coefficients in ``z``."""
    return [factorial(5 * n) // factorial(n) ** 5 for n in range(order + 1)]


@cache
def mirror_map(order: int) -> tuple[Fraction, ...]:
    """``z(q)`` to ``q^order``: the complex structure, as a series in ``q = e^(2 pi i t)``.

    ``q = z exp(w1_regular / w0)`` follows from ``t = w1 / (2 pi i w0)``, and is
    inverted order by order. The leading coefficient is 1; the next is
    ``-770``.
    """
    size = order + 1
    w0 = [Fraction(c) for c in fundamental_period(order)]
    regular = [w0[n] * 5 * (_harmonic(5 * n) - _harmonic(n)) for n in range(size)]
    # q(z) = z * exp(regular / w0).
    q_of_z = [Fraction(0)] + _exponential(_multiply(regular, _inverse(w0)))[:order]
    # Invert by fixed-point iteration: each pass fixes one more coefficient.
    z_of_q = [Fraction(0), Fraction(1)] + [Fraction(0)] * (order - 1)
    for _ in range(order):
        residual = _compose(q_of_z, z_of_q)
        residual[1] -= 1
        z_of_q = [z - r for z, r in zip(z_of_q, residual, strict=True)]
    return tuple(z_of_q)


@cache
def yukawa_expansion(order: int) -> tuple[int, ...]:
    """``kappa_ttt``'s coefficients in ``q``, from ``q^0`` to ``q^order``.

    ``5 / (z^3 (1 - 3125 z))`` in the gauge ``w0 = 1`` and the coordinate
    ``t``: ``kappa = 5 / ((1 - 3125 z) w0^2) (q dz/dq / z)^3``. Starts
    ``5, 2875, 4876875, 8564575000``.
    """
    size = order + 2  # one spare order: q dz/dq / z divides a series by q
    z = list(mirror_map(size - 1))
    w0 = _compose([Fraction(c) for c in fundamental_period(size - 1)], z)
    conifold = _compose([Fraction(1), Fraction(-3125)] + [Fraction(0)] * (size - 2), z)
    numerator = [n * z[n] for n in range(1, size)] + [Fraction(0)]
    denominator = z[1:] + [Fraction(0)]
    jacobian = _multiply(numerator, _inverse(denominator))  # q dz/dq / z
    cube = _multiply(jacobian, _multiply(jacobian, jacobian))
    kappa = _multiply(_inverse(_multiply(conifold, _multiply(w0, w0))), cube)
    coefficients = [5 * c for c in kappa[: order + 1]]
    if any(c.denominator != 1 for c in coefficients):
        raise ArithmeticError("the Yukawa coupling's q-expansion is not integral")
    return tuple(int(c) for c in coefficients)


@cache
def instanton_numbers(order: int) -> tuple[int, ...]:
    """``n_1 .. n_order``, the numbers of rational curves of each degree.

    From ``kappa = 5 + sum_m q^m sum_(d | m) n_d d^3``: each coefficient of
    ``q^m`` gives ``n_m`` once the divisors below it are known. Raises if any
    comes out fractional, which would mean the calculation is wrong.
    """
    kappa = yukawa_expansion(order)
    found: dict[int, Fraction] = {}
    for m in range(1, order + 1):
        rest = kappa[m] - sum(found[d] * d**3 for d in found if m % d == 0)
        found[m] = Fraction(rest, m**3)
        if found[m].denominator != 1:
            raise ArithmeticError(f"n_{m} = {found[m]} is not an integer")
    return tuple(int(found[m]) for m in range(1, order + 1))


def chern_numbers() -> dict[str, int]:
    """The quintic's topology from its Chern class, ``c(X) = (1 + J)^5 / (1 + 5 J)``.

    The tangent bundle of ``P^4`` restricted to ``X`` is ``(1 + J)^5``, and the
    normal bundle is ``O(5)``. Expanding the ratio to third order in ``J``
    gives ``c_1 = 0``, which is what makes ``X`` Calabi-Yau, ``c_2 = 10 J^2``
    and ``c_3 = -40 J^3``. With ``int J^3 = 5``, the degree, that gives:
    - ``chi = -200``
    - ``c2.J = 50``
    - ``h^(2,1) = 101``, from ``chi = 2 (h11 - h21)`` with ``h11 = 1``

    ``h21`` is also counted directly: the 126 quintic monomials, less the 25
    of ``GL(5)``, which only change coordinates.
    """
    ambient = [comb(5, j) for j in range(4)]  # (1 + J)^5 to J^3
    normal_inverse = [(-5) ** j for j in range(4)]  # 1 / (1 + 5 J)
    c = [sum(ambient[i] * normal_inverse[j - i] for i in range(j + 1)) for j in range(4)]
    degree = 5
    h11 = 1
    euler = c[3] * degree
    h21 = h11 - euler // 2
    monomials = comb(5 + 5 - 1, 5)
    if monomials - 25 != h21:
        raise ArithmeticError("the monomial count and the Euler number disagree about h21")
    return {
        "c1": c[1],
        "c2": c[2],
        "c3": c[3],
        "intersection": degree,
        "euler": euler,
        "c2_dot_J": c[2] * degree,
        "h11": h11,
        "h21": h21,
    }


def _polylog(order: int, x: complex, terms: int = 400) -> complex:
    """``Li_order(x) = sum x^m / m^order`` for ``|x| < 1``."""
    m = np.arange(1, terms + 1)
    return complex(np.sum(x**m / m**order))


@dataclass(frozen=True)
class QuinticModuli:
    """The Kahler moduli space of the quintic, with worldsheet instantons.

    ``degrees`` instanton numbers are kept. They grow roughly as ``e^(8.05 d)``,
    so the sum converges only for ``Im t`` above about 1.3, and the truncation
    error is of order ``n_(degrees+1) |q|^(degrees+1)``. ``degrees = 10``
    leaves that below double precision for ``Im t >= 2``.
    """

    degrees: int = 10
    instantons: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "instantons", instanton_numbers(self.degrees))

    def _check(self, t: complex) -> complex:
        t = complex(t)
        if t.imag <= 0.0:
            raise ValueError(f"Im t is a volume and must be positive, got {t}")
        return t

    def prepotential(self, t: complex) -> tuple[complex, complex, complex, complex]:
        """``(F, F_t, F_tt, F_ttt)`` at ``t``."""
        t = self._check(t)
        topology = chern_numbers()
        kappa = topology["intersection"]
        linear = topology["c2_dot_J"] / 24
        constant = ZETA3 * topology["euler"] / (2 * (2j * np.pi) ** 3)
        F = -kappa / 6 * t**3 + linear * t + constant
        Ft = -kappa / 2 * t**2 + linear
        Ftt = -kappa * t
        Fttt = complex(-kappa)
        q = np.exp(2j * np.pi * t)
        tau = 2j * np.pi
        for d, n in enumerate(self.instantons, start=1):
            x = q**d
            F -= n * _polylog(3, x) / tau**3
            Ft -= n * d * _polylog(2, x) / tau**2
            Ftt -= n * d**2 * _polylog(1, x) / tau
            Fttt -= n * d**3 * x / (1 - x)
        return complex(F), complex(Ft), complex(Ftt), complex(Fttt)

    def exp_minus_kahler(self, t: complex) -> float:
        """``e^-K = (4/3) 5 (Im t)^3 - chi zeta(3) / (4 pi^3) + instantons``."""
        F, Ft, _, _ = self.prepotential(t)
        t = complex(t)
        value = 1j * (2 * (F - F.conjugate()) - (t - t.conjugate()) * (Ft + Ft.conjugate()))
        return float(value.real)

    def kahler_potential(self, t: complex) -> float:
        return -float(np.log(self.exp_minus_kahler(t)))

    def metric(self, t: complex) -> float:
        """``G_(t tbar) = d_t d_tbar K``, from ``F``'s derivatives in closed form."""
        F, Ft, Ftt, _ = self.prepotential(t)
        t = complex(t)
        f = self.exp_minus_kahler(t)
        f_t = 1j * (Ft - Ft.conjugate() - (t - t.conjugate()) * Ftt)
        f_tt_bar = -2.0 * Ftt.imag
        return float((abs(f_t) ** 2 - f * f_tt_bar) / f**2)

    def yukawa(self, t: complex) -> complex:
        """``kappa_ttt = -F_ttt = 5 + sum n_d d^3 q^d / (1 - q^d)``."""
        return -self.prepotential(t)[3]

    def normalized_yukawa(self, t: complex) -> float:
        """``e^K |kappa_ttt| G^(-3/2)``: invariant, and ``2/sqrt(3)`` at large volume."""
        return float(np.exp(self.kahler_potential(t)) * abs(self.yukawa(t)) / self.metric(t) ** 1.5)


__all__ = [
    "QuinticModuli",
    "ZETA3",
    "chern_numbers",
    "fundamental_period",
    "instanton_numbers",
    "mirror_map",
    "yukawa_expansion",
]
