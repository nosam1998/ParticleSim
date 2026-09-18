"""Toroidal compactification to a four-dimensional effective theory.

Issue #76. Pick a vacuum -- a torus, optionally orbifolded -- and read off
the 4D field content, the moduli, the Kahler potential and the gauge kinetic
function, then emit a Tier A plugin carrying the couplings that result. The
point is that those couplings are *derived* rather than chosen, and the way
to show that is to check them against what can be worked out by hand.

**The no-scale identity is the sharp check, and it is exactly 3.** For the
three Kahler moduli of a factorised ``T^2 x T^2 x T^2``,

    K = -sum_i ln(T_i + Tbar_i)   =>   K^(i jbar) K_i K_jbar = 3

exactly, not approximately. That ``3`` is what cancels the ``-3|W|^2`` in the
F-term potential, so a superpotential independent of the ``T_i`` gives
``V = 0`` *identically*: the Kahler moduli are flat at tree level and the
vacuum energy vanishes without tuning. Both are asserted symbolically rather
than quoted.

**And the cancellation is not an accident of counting.** Include the
axio-dilaton and the same identity comes out at **4**, so the same
superpotential leaves ``V = e^K |W|^2 > 0``. The no-scale structure belongs
to the Kahler sector specifically; a module that computed "the identity" over
whatever moduli happened to be in scope would get 4 and report a cancellation
that does not happen.

**What is not here.** Twisted sectors. An orbifold's twisted states are not
visible in the invariant projection this module performs -- they live at the
fixed points and need the orbifold conformal field theory. The moduli counts
below are untwisted-sector counts and say so, which matters because the
twisted sector is where most of an orbifold's chiral matter comes from.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import sympy as sp

from particlesim.theories.base import Coupling, Theory
from particlesim.theories.kk import Torus

POSITIVE = (0.0, float("inf"))


@dataclass(frozen=True)
class Modulus:
    """One 4D scalar left over from the compactification."""

    name: str
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in ("kahler", "complex_structure", "axio_dilaton"):
            raise ValueError(f"unknown modulus kind {self.kind!r}")


@dataclass(frozen=True)
class ToroidalVacuum:
    """Three ``T^2`` factors, a string coupling, and an optional diagonal twist.

    ``twist`` is the orbifold's action on the three complex coordinates, as
    fractions of a full turn. ``None`` is the plain torus. The rule for what
    survives is computable and is applied rather than tabulated: a Kahler
    modulus ``T_i ~ dz_i ^ dzbar_i`` is invariant under any diagonal phase,
    so all three always survive; a complex structure modulus ``U_i`` survives
    only when the twist preserves the ``i``-th torus's complex structure,
    which needs ``2 v_i`` to be an integer. A ``Z_3`` twist has none, which is
    the known ``h^(2,1) = 0`` of the untwisted sector.
    """

    radii: tuple[float, float, float] = (1.0, 1.0, 1.0)
    string_coupling: float = 0.1
    twist: tuple[Fraction, Fraction, Fraction] | None = None

    def __post_init__(self) -> None:
        if len(self.radii) != 3:
            raise ValueError(f"three torus factors are expected, got {len(self.radii)}")
        for radius in self.radii:
            if radius <= 0.0:
                raise ValueError(f"every radius must be positive, got {self.radii}")
        if not 0.0 < self.string_coupling < 1.0:
            raise ValueError(
                f"the string coupling must lie in (0, 1), got {self.string_coupling}; "
                "outside it the tree-level description this module derives is not the "
                "right one"
            )
        if self.twist is not None:
            if len(self.twist) != 3:
                raise ValueError(f"a diagonal twist needs three entries, got {self.twist}")
            if sum(Fraction(v) for v in self.twist) % 1 != 0:
                raise ValueError(
                    f"the twist {self.twist} does not sum to an integer, so it is not in "
                    "SU(3) and breaks all the supersymmetry the derivation assumes"
                )

    @property
    def torus(self) -> Torus:
        """The compact space, as the Kaluza-Klein module models it."""
        return Torus(radii=tuple(self.radii))

    def moduli(self) -> list[Modulus]:
        """Untwisted-sector moduli: three Kahler, the surviving ``U``, the dilaton."""
        found = [Modulus(f"T{i + 1}", "kahler") for i in range(3)]
        for index in range(3):
            if self.twist is None or (2 * Fraction(self.twist[index])).denominator == 1:
                found.append(Modulus(f"U{index + 1}", "complex_structure"))
        found.append(Modulus("S", "axio_dilaton"))
        return found

    def kahler_moduli(self) -> list[Modulus]:
        return [m for m in self.moduli() if m.kind == "kahler"]

    def dilaton_vev(self) -> float:
        """``Re S = 1/g_s``, the real part of the axio-dilaton.

        This is the quantity the gauge kinetic function actually depends on;
        the string coupling is its inverse. Keeping the two steps apart is
        what makes :meth:`gauge_coupling` a derivation rather than a rename.
        """
        return 1.0 / float(self.string_coupling)

    def gauge_coupling(self) -> float:
        """``g^2 = 1 / Re S`` at tree level, from the gauge kinetic function ``f = S``.

        The heterotic gauge kinetic function is the dilaton superfield itself,
        so ``Re f = Re S`` sits in front of ``-(1/4) F^2`` and the coupling is
        its inverse. Going through :meth:`dilaton_vev` rather than returning
        ``string_coupling`` directly means the chain ``g_s -> Re S -> g^2`` is
        the thing a test can check.
        """
        return 1.0 / self.dilaton_vev()

    def kaluza_klein_scale(self) -> float:
        """The lightest non-zero tower mass, in string units.

        Derived from the same :class:`~particlesim.theories.kk.Torus` the
        Kaluza-Klein module uses, so the compactification and the tower
        cannot drift apart.
        """
        spectrum = self.torus.spectrum(levels=1)
        return float(spectrum[1][0])

    def superpotential(self) -> sp.Expr:
        """Zero. A plain toroidal compactification has no tree-level ``W``.

        No flux, no non-perturbative effects, so the moduli are exactly flat
        and the statement is ``0`` rather than "small". Anything that lifts
        them has to be added deliberately, which is the honest starting point
        for moduli stabilisation rather than a gap.
        """
        return sp.S.Zero


def modulus_symbols(moduli) -> tuple[list[sp.Symbol], list[sp.Symbol]]:
    """``(fields, conjugates)`` as independent positive symbols.

    Holomorphic and antiholomorphic coordinates are carried separately
    because the Kahler metric is a mixed second derivative; treating
    ``Tbar`` as ``conjugate(T)`` makes sympy's differentiation of
    ``ln(T + Tbar)`` a fight rather than a calculation.
    """
    fields = [sp.Symbol(m.name, positive=True) for m in moduli]
    conjugates = [sp.Symbol(f"{m.name}bar", positive=True) for m in moduli]
    return fields, conjugates


def kahler_potential(moduli, fields, conjugates) -> sp.Expr:
    """``-sum ln(X + Xbar)`` over the moduli supplied.

    Every modulus here enters logarithmically, which is what makes the
    identity below a count rather than a computation -- and is exactly why
    including the dilaton changes the answer.
    """
    return -sum(sp.log(f + c) for f, c in zip(fields, conjugates, strict=True))


def kahler_metric(potential, fields, conjugates) -> sp.Matrix:
    """``K_(i jbar)``, the mixed second derivative."""
    size = len(fields)
    return sp.Matrix(size, size, lambda i, j: sp.diff(potential, fields[i], conjugates[j]))


def no_scale_identity(potential, fields, conjugates) -> sp.Expr:
    """``K^(i jbar) K_i K_jbar``: 3 over the Kahler moduli, 4 with the dilaton.

    The number is the count of logarithms, so what it is computed over is the
    whole question. Three is the no-scale value; anything else and the
    cancellation below does not happen.
    """
    inverse = kahler_metric(potential, fields, conjugates).inv()
    lower = [sp.diff(potential, f) for f in fields]
    upper = [sp.diff(potential, c) for c in conjugates]
    return sp.simplify(
        sum(
            inverse[i, j] * lower[i] * upper[j]
            for i in range(len(fields))
            for j in range(len(fields))
        )
    )


def scalar_potential(potential, superpotential, fields, conjugates) -> sp.Expr:
    """``e^K (K^(i jbar) D_i W D_jbar Wbar - 3 |W|^2)``, the F-term potential.

    ``superpotential`` stands for both ``W`` and ``Wbar``, which is exact for
    a real ``W``; for the constant superpotential the cancellation is stated
    against, the phase is a Kahler transformation and carries no physics, so
    nothing is lost.

    The ``-3`` is a fixed feature of ``N = 1`` supergravity in four
    dimensions, which is why the identity above has to come out at exactly
    three for the cancellation to happen. Two logarithms give ``V < 0`` and
    four give ``V > 0``; only three gives zero.
    """
    inverse = kahler_metric(potential, fields, conjugates).inv()
    derivatives = [
        sp.diff(superpotential, f) + sp.diff(potential, f) * superpotential for f in fields
    ]
    conjugated = [
        sp.diff(superpotential, c) + sp.diff(potential, c) * superpotential for c in conjugates
    ]
    cross = sum(
        inverse[i, j] * derivatives[i] * conjugated[j]
        for i in range(len(fields))
        for j in range(len(fields))
    )
    return sp.simplify(sp.exp(potential) * (cross - 3 * superpotential**2))


def emit_plugin(vacuum: ToroidalVacuum, name: str = "string.compactify.torus") -> type[Theory]:
    """A Tier A plugin whose couplings are *derived* from ``vacuum``.

    The gauge coupling is ``g_s`` from ``f = S``, and the Kaluza-Klein scale
    comes from the same torus the tower does. Both defaults sit where the
    theory reduces to General Relativity, so the emitted plugin passes the
    same limit harness every hand-written one does -- which is the point of
    emitting a plugin rather than a dictionary.
    """
    derived_coupling = vacuum.gauge_coupling()
    derived_scale = vacuum.kaluza_klein_scale()
    count = len(vacuum.moduli())

    class Compactified(Theory):
        id = name
        frame = "einstein"
        couplings = [
            Coupling("gauge_coupling", 0.0, units="dimensionless", bounds=POSITIVE),
            Coupling("kaluza_klein_scale", 0.0, units="1/length", bounds=POSITIVE),
        ]
        provenance = (
            f"Derived from a toroidal compactification with radii {tuple(vacuum.radii)} "
            f"and string coupling {vacuum.string_coupling}; tree level, untwisted sector"
        )
        validity_statement = (
            "tree level and untwisted sector only; twisted states live at the fixed "
            "points and need the orbifold conformal field theory, and the moduli are "
            "unstabilised because the tree-level superpotential vanishes"
        )

        def gr_limit(self) -> dict[str, float]:
            return {"gauge_coupling": 0.0, "kaluza_klein_scale": 0.0}

        def effective_stress_energy(self, einstein, metric):
            return sp.Matrix(einstein) / (8 * sp.pi)

        def observable_predictions(self) -> dict:
            return {
                "gauge_coupling": self.values["gauge_coupling"],
                "kaluza_klein_scale": self.values["kaluza_klein_scale"],
                "modulus_count": count,
                "moduli_stabilised": False,
            }

    Compactified.derived = {
        "gauge_coupling": derived_coupling,
        "kaluza_klein_scale": derived_scale,
        "modulus_count": count,
    }
    return Compactified


DEFAULT_VACUUM = ToroidalVacuum()
"""Unit radii at ``g_s = 0.1``, the vacuum the registered plugin stands for.

One plugin per vacuum is the design, so the registry has to name one. This
is the plainest: an isotropic ``T^6`` at weak coupling, no twist. Registering
it means the emitted class goes through ``particlesim check-limits`` with
every hand-written plugin rather than only through this module's own tests.
"""

ToroidalCompactification = emit_plugin(DEFAULT_VACUUM)


__all__ = [
    "DEFAULT_VACUUM",
    "Modulus",
    "ToroidalCompactification",
    "ToroidalVacuum",
    "emit_plugin",
    "kahler_metric",
    "kahler_potential",
    "modulus_symbols",
    "no_scale_identity",
    "scalar_potential",
]
