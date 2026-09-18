"""What string theory already says about singular backgrounds.

Issue #75. Some singular geometries have an exact conformal field theory
description, which means the answer to "what happens there" is already known
and does not need a simulation. This module records those answers in a form
the hypothesis harness can score against *before* spending an evolution, and
the answers are computed rather than tabulated wherever they can be.

**The organising fact is the orbifold group, and it is arithmetic.** All
three orbifold backgrounds here identify flat space by a group, and a string
amplitude on the quotient is a sum over that group's images. So:

* ``C/Z_N`` is a quotient by a **finite** group. The image sum has ``N``
  terms, the invariant is bounded, the ``N^2`` sectors close under the
  modular group, and the conical singularity is a perfectly good string
  background. The geometry is singular; the conformal field theory is not.
* The **null orbifold** quotients by a parabolic element. The group is
  infinite, the invariant of the ``n``-th image grows as ``4 + n^2 b^2``,
  and the image sum of a graviton exchange -- which grows as ``s^2`` --
  diverges polynomially.
* **Milne** quotients by a boost. The group is infinite, the invariant grows
  as ``2 + 2 cosh(n b)``, and the same sum diverges *geometrically*.

That is the whole difference, and it is computed here from actual Lorentz
matrices rather than asserted: the elements are checked to preserve the
metric and to compose additively, and the invariants are compared with their
closed forms. A hypothesis claiming that quantum gravity resolves *every*
singularity is contradicted by the second and third entries before a single
timestep runs -- which is the point of scoring first.

**The two-dimensional black hole is the one exactly solved case.** The
``SL(2,R)_k/U(1)`` coset has

    c = 3k/(k-2) - 1 = 2(k+1)/(k-2)

which is ``26`` at ``k = 9/4`` exactly, tends to ``2`` as ``k -> infinity``
-- the semiclassical two-boson count -- and expands as
``2 + 6/k + 12/k^2 + 24/k^3 + ...``. Those coefficients are the alpha-prime
corrections, and the critical level is a rational number rather than a
numerical solve, so both are asserted exactly.

**What this module does not do.** It does not evolve anything and it is not
a :class:`~particlesim.theories.base.Theory`. It has no couplings, no
general-relativistic limit and no place in the registry; it is a record of
results obtained elsewhere, with the arithmetic redone here so that a wrong
entry is a failing test rather than a plausible sentence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product

import numpy as np
import sympy as sp

#: Minkowski metric in the three dimensions the orbifold elements act on.
MINKOWSKI = np.diag([-1.0, 1.0, 1.0])

#: Crystallographic orders: the only ``N`` for which ``Z_N`` preserves a lattice.
CRYSTALLOGRAPHIC = (2, 3, 4, 6)


@dataclass(frozen=True)
class Reference:
    """One background whose fate is known, and the ground for saying so."""

    name: str
    singularity: str
    status: str
    claim: str
    evidence: str

    def __post_init__(self) -> None:
        if self.status not in ("resolved", "unstable", "exactly_solved"):
            raise ValueError(f"unknown status {self.status!r}")

    def as_row(self) -> dict:
        return {
            "name": self.name,
            "singularity": self.singularity,
            "status": self.status,
            "claim": self.claim,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class ConicalOrbifold:
    """``C/Z_N``: a conical singularity that strings do not mind.

    The geometry has unbounded curvature at the origin and the conformal
    field theory is finite there. That combination is the single most useful
    thing this module records, because a hypothesis that treats "curvature
    diverges" as synonymous with "the theory breaks down" is already wrong
    about a case string theory understands completely.
    """

    order: int = 3

    def __post_init__(self) -> None:
        if self.order < 2:
            raise ValueError(f"an orbifold needs order at least 2, got {self.order}")

    @property
    def status(self) -> str:
        return "resolved"

    def images(self) -> int:
        """The number of terms in an image sum: finite, and equal to the order."""
        return self.order

    def twists(self) -> list[Fraction]:
        """``k/N`` for ``k = 1 .. N-1``: one per twisted sector."""
        return [Fraction(k, self.order) for k in range(1, self.order)]

    def ground_state_energy(self, twist: Fraction) -> Fraction:
        """``-1/12 + v(1-v)/2`` for a complex boson twisted by ``v``.

        The two ends check it: ``v = 0`` gives ``-1/12``, which is two real
        bosons at ``-1/24``, and ``v = 1/2`` gives ``1/24``, which is two
        antiperiodic bosons at ``1/48``. A twisted sector's ground state is
        *lifted* relative to the untwisted one, which is why the conical
        singularity carries states rather than a divergence.
        """
        v = Fraction(twist)
        if not 0 <= v < 1:
            raise ValueError(f"a twist is a fraction of a turn in [0, 1), got {twist}")
        return Fraction(-1, 12) + v * (1 - v) / 2

    def sectors(self) -> set[tuple[int, int]]:
        """``(g, h)``: the boundary conditions around the two torus cycles."""
        return set(product(range(self.order), repeat=2))

    def modular_images(self, sector: tuple[int, int]) -> set[tuple[int, int]]:
        """Where ``S`` and ``T`` send one sector.

        ``S: tau -> -1/tau`` exchanges the two cycles, so ``(g, h) -> (h, -g)``;
        ``T: tau -> tau + 1`` shears them, so ``(g, h) -> (g, g + h)``.
        """
        twist, cycle = sector
        return {(cycle % self.order, (-twist) % self.order), (twist, (twist + cycle) % self.order)}

    def is_modular_closed(self) -> bool:
        """Whether the sector set maps to itself under the modular group.

        This is what forces the twisted sectors to exist: dropping them would
        leave a partition function that is not modular invariant, and the
        theory would be inconsistent rather than merely incomplete.
        """
        sectors = self.sectors()
        return all(self.modular_images(s) <= sectors for s in sectors)

    def fixed_points(self, dimensions: int = 2) -> int:
        """``|det(1 - theta)|`` per complex plane, from the rotation matrix.

        Computed from ``theta`` rather than looked up: 4, 3, 2, 1 on ``T^2``
        for ``Z_2``, ``Z_3``, ``Z_4``, ``Z_6``, and 27 on ``T^6/Z_3`` -- which
        is the number of twisted sectors, and where most of an orbifold's
        chiral matter lives.
        """
        if dimensions % 2 or dimensions < 2:
            raise ValueError(f"an even number of real dimensions is expected, got {dimensions}")
        angle = 2.0 * np.pi / self.order
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=float
        )
        per_plane = abs(np.linalg.det(np.eye(2) - rotation))
        return int(round(per_plane ** (dimensions // 2)))


@dataclass(frozen=True)
class BoostOrbifold:
    """Milne and the null orbifold: flat space quotiented by an infinite group.

    ``kind`` picks the conjugacy class of the identification. A ``"boost"``
    is hyperbolic and gives the Milne universe; a ``"null_rotation"`` is
    parabolic and gives the null orbifold. Both groups are infinite and
    non-compact, which is the whole reason these are not
    :class:`ConicalOrbifold`.
    """

    rapidity: float = 0.3
    kind: str = "null_rotation"

    def __post_init__(self) -> None:
        if self.kind not in ("boost", "null_rotation"):
            raise ValueError(f"unknown identification {self.kind!r}")
        if self.rapidity <= 0.0:
            raise ValueError(
                f"the identification parameter must be positive, got {self.rapidity}; "
                "at zero there is no quotient and no singularity"
            )

    @property
    def status(self) -> str:
        return "unstable"

    @property
    def growth(self) -> str:
        return "exponential" if self.kind == "boost" else "polynomial"

    def element(self, power: int = 1) -> np.ndarray:
        """The ``n``-th group element as a Lorentz matrix on ``(t, x, y)``.

        Written in closed form in ``n * rapidity`` rather than by
        multiplying, which is what makes the composition check below a check
        rather than a tautology.
        """
        b = power * self.rapidity
        if self.kind == "boost":
            return np.array(
                [[np.cosh(b), np.sinh(b), 0.0], [np.sinh(b), np.cosh(b), 0.0], [0.0, 0.0, 1.0]]
            )
        return np.array(
            [
                [1.0 + b * b / 2.0, -b * b / 2.0, b],
                [b * b / 2.0, 1.0 - b * b / 2.0, b],
                [b, -b, 1.0],
            ]
        )

    def image_invariant(self, power: int) -> float:
        """``s`` between a particle at rest and its ``n``-th image.

        Built from the matrix, so the closed forms ``2 + 2 cosh(n b)`` and
        ``4 + n^2 b^2`` are predictions the suite checks rather than the
        definition.
        """
        rest = np.array([1.0, 0.0, 0.0])
        total = rest + self.element(power) @ rest
        return float(-total @ MINKOWSKI @ total)

    def image_invariants(self, count: int = 8) -> list[float]:
        return [self.image_invariant(n) for n in range(count)]

    def image_sum_diverges(self, power: float = 2.0) -> bool:
        """Whether summing ``s_n^power`` over the group converges.

        ``power = 2`` is graviton exchange, which grows as ``s^2``. The group
        is infinite either way and the invariant grows either way, so the sum
        diverges for any positive power -- polynomially for the null orbifold
        and geometrically for Milne. A finite group has no such problem, which
        is why :class:`ConicalOrbifold` has no counterpart to this method.
        """
        return power > 0.0


@dataclass(frozen=True)
class CosetBlackHole:
    """``SL(2,R)_k/U(1)``: the two-dimensional black hole, solved to all orders.

    Its central charge is a rational function of the level, so both the
    critical level and the alpha-prime expansion are exact statements rather
    than numerical ones.
    """

    level: Fraction = Fraction(9, 4)

    def __post_init__(self) -> None:
        if Fraction(self.level) <= 2:
            raise ValueError(
                f"the level must exceed 2, got {self.level}; at k = 2 the central charge "
                "diverges and the coset is not a unitary conformal field theory"
            )

    @property
    def status(self) -> str:
        return "exactly_solved"

    def central_charge(self) -> Fraction:
        """``3k/(k-2) - 1``, exactly."""
        k = Fraction(self.level)
        return 3 * k / (k - 2) - 1

    @staticmethod
    def critical_level() -> Fraction:
        """``9/4``: where the coset alone carries the bosonic string's ``c = 26``."""
        level = sp.Symbol("k", positive=True)
        solutions = sp.solve(sp.Eq(3 * level / (level - 2) - 1, 26), level)
        return Fraction(sp.Rational(solutions[0]))

    @staticmethod
    def semiclassical_charge() -> int:
        """``2``: two bosons, which is what the metric and dilaton describe."""
        level = sp.Symbol("k", positive=True)
        return int(sp.limit(3 * level / (level - 2) - 1, level, sp.oo))

    @staticmethod
    def alpha_prime_coefficients(order: int = 4) -> list[int]:
        """``[2, 6, 12, 24, ...]``: the expansion of ``c`` in powers of ``1/k``.

        The leading ``2`` is the semiclassical answer and everything after it
        is an alpha-prime correction, so a background that claimed to be
        exact while stopping at the metric would be missing the ``6/k``.
        """
        inverse = sp.Symbol("x", positive=True)
        charge = 3 / inverse / (1 / inverse - 2) - 1
        expansion = sp.series(charge, inverse, 0, order).removeO()
        polynomial = sp.Poly(sp.expand(expansion), inverse)
        return [int(polynomial.coeff_monomial(inverse**n)) for n in range(order)]


REFERENCES: tuple[Reference, ...] = (
    Reference(
        name="conical_orbifold",
        singularity="conical",
        status="resolved",
        claim="strings propagate consistently on C/Z_N; the geometry is singular and the "
        "conformal field theory is not",
        evidence="the image sum has N terms, the N^2 sectors close under the modular group, "
        "and the twisted ground state is lifted to -1/12 + v(1-v)/2",
    ),
    Reference(
        name="two_dimensional_black_hole",
        singularity="two-dimensional black hole",
        status="exactly_solved",
        claim="the SL(2,R)_k/U(1) coset describes it to all orders in alpha-prime",
        evidence="c = 3k/(k-2) - 1 is 26 at k = 9/4 exactly, tends to the semiclassical 2, "
        "and expands as 2 + 6/k + 12/k^2 + 24/k^3",
    ),
    Reference(
        name="null_orbifold",
        singularity="null",
        status="unstable",
        claim="not a good string background: the quotient group is infinite and the image "
        "sum of a graviton exchange diverges",
        evidence="the n-th image invariant grows as 4 + n^2 b^2, so summing s^2 over the "
        "group diverges polynomially",
    ),
    Reference(
        name="milne",
        singularity="spacelike (Milne)",
        status="unstable",
        claim="worse than the null orbifold for the same reason, and by a wider margin",
        evidence="the n-th image invariant grows as 2 + 2 cosh(n b), so the same sum "
        "diverges geometrically",
    ),
)


def catalogue() -> dict[str, Reference]:
    return {reference.name: reference for reference in REFERENCES}


#: Prediction keys this module knows how to score, and what the reference says.
#: Anything outside this map lands in ``unaddressed`` rather than being guessed at.
CLAIM_KEYS: dict[str, tuple[str, bool]] = {
    "conical_singularity_resolved": ("conical_orbifold", True),
    "null_singularity_resolved": ("null_orbifold", False),
    "milne_singularity_resolved": ("milne", False),
    "all_singularities_resolved": ("null_orbifold", False),
}


@dataclass
class ReferenceScore:
    """How a hypothesis fares against results that are already in.

    ``contradicted`` is the field that matters and it is populated before any
    evolution runs, so a claim that string theory already refutes costs
    nothing to reject.
    """

    supported: list[str] = field(default_factory=list)
    contradicted: list[str] = field(default_factory=list)
    unaddressed: list[str] = field(default_factory=list)

    @property
    def refuted(self) -> bool:
        return bool(self.contradicted)

    def as_row(self) -> dict:
        return {
            "supported": list(self.supported),
            "contradicted": list(self.contradicted),
            "unaddressed": list(self.unaddressed),
            "refuted": self.refuted,
        }


def score_hypothesis(predictions: dict) -> ReferenceScore:
    """Score declared predictions against the catalogue. No evolution runs.

    A hypothesis that claims universal singularity resolution is contradicted
    here by the null orbifold, whose singularity string theory does *not*
    resolve -- the background is unstable instead. Finding that out costs one
    dictionary lookup rather than a run, which is the whole reason the
    harness does this first.
    """
    score = ReferenceScore()
    known = catalogue()
    for key, value in predictions.items():
        if key not in CLAIM_KEYS:
            score.unaddressed.append(key)
            continue
        name, expected = CLAIM_KEYS[key]
        reference = known[name]
        if bool(value) == expected:
            score.supported.append(f"{key}: agrees with {reference.name}")
        else:
            score.contradicted.append(
                f"{key}: claimed {value!r}, but {reference.name} is {reference.status} "
                f"({reference.evidence})"
            )
    return score


__all__ = [
    "CLAIM_KEYS",
    "CRYSTALLOGRAPHIC",
    "MINKOWSKI",
    "REFERENCES",
    "BoostOrbifold",
    "ConicalOrbifold",
    "CosetBlackHole",
    "Reference",
    "ReferenceScore",
    "catalogue",
    "score_hypothesis",
]
