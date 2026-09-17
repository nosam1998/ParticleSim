"""String-inspired inflaton potentials (design doc Sections 3.1, 4.4).

Four classes of potential that string constructions produce, as
:class:`~particlesim.cosmo.potentials.Potential` subclasses that the exact
mode solver in :mod:`particlesim.cosmo.perturbations` consumes unchanged.
Each carries a ``provenance`` naming where the form comes from and a
``truncation`` naming what has been dropped from it, because every one of
these potentials is an expansion of something larger and the terms left out
are the reason two papers on the same model can disagree.

**What is and is not claimed.** These are the potentials, not the
compactifications. Nothing here derives a Kaehler potential, stabilises a
modulus, or checks that a given set of parameters is attainable in a
consistent vacuum; the parameters are inputs, and a caller can set them to
values no known construction realises. What is checked is that the
potentials reproduce the observable predictions their source papers quote,
which is a statement about this pipeline rather than about string theory.

**The plateau family and why one prediction is universal.** Three of the
models here (and Starobinsky, which is not a string model at all) are
plateaux: at large field

    V = V_0 (1 - C exp(-(phi/f)^p))

with the correction exponentially small. For *any* ``C``, ``f`` and ``p``
that gives ``n_s - 1 = -2/N`` to leading order, which is the "robust,
model-independent" tilt Kaehler moduli inflation is quoted for. The
derivation is short enough to give here. With ``u = (phi/f)^p`` and
``C exp(-u) << 1``,

    sqrt(2 eps) = V'/V = C e^(-u) u',      eta = -C e^(-u) u'^2
    N = integral dphi/sqrt(2 eps) = (f^2/p^2 C) e^u u^(2/p - 2) + ...

and multiplying the last two gives ``eta N = -1`` whatever the parameters
are. What the parameters *do* change is the tensor ratio,

    r = 16 eps = 8 eta^2/u'^2 = (8 f^2/p^2) u^(2/p - 2) / N^2

which for ``p = 1`` is exactly ``8 f^2/N^2``: Starobinsky's ``12/N^2`` at
``f^2 = 3/2``, fibre inflation's ``24/N^2`` at ``f^2 = 3``, and an
unobservably small number for a blow-up modulus, whose ``p = 4/3`` brings a
``1/sqrt(u)`` suppression. So the tilt is where these models agree and the
tensor ratio is where they can be told apart, which is the whole reason the
tensor ratio is worth measuring.

Provenance in one place, since several classes share a source:

* Fibre inflation: Cicoli, Burgess and Quevedo, JCAP 0903:013 (2009),
  arXiv:0808.0691. The exponent ``1/sqrt(3)`` and the predictions
  ``eps ~= (3/2) eta^2``, ``r ~= 6 (n_s-1)^2`` and ``r ~= 0.005`` to
  ``0.01`` are that paper's.
* Kaehler moduli inflation: Conlon and Quevedo, JHEP 0601:146 (2006),
  hep-th/0509012, quoted for a model-independent ``n_s = 1 - 2/N_e`` of
  0.960 to 0.967 over ``N_e = 50`` to 60.
* Axion monodromy: Silverstein and Westphal, PRD 78:106003 (2008) for the
  mechanism; McAllister, Silverstein and Westphal, PRD 82:046003 (2010)
  for the linear potential and its ``r ~= 0.07``. The oscillatory term is
  the instanton modulation of Flauger, McAllister, Pajer, Westphal and Xu,
  JCAP 1006:009 (2010).
* D-brane inflation: the Coulomb potential of Kachru, Kallosh, Linde,
  Maldacena, McAllister and Trivedi, JCAP 0310:013 (2003).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from particlesim.cosmo.potentials import Potential

#: ``1/sqrt(3)``, the exponent of the fibre-inflation plateau.
FIBRE_EXPONENT = 1.0 / math.sqrt(3.0)


@dataclass(frozen=True)
class PlateauInflation(Potential):
    """``V = A (1 - C exp(-(phi/f)^p))``: the generic moduli plateau.

    The family whose universal ``n_s = 1 - 2/N`` is derived in the module
    docstring. It is worth having in its own right rather than only through
    its members, because the interesting question about a plateau model is
    which of its predictions survive changing ``(C, f, p)`` -- and with the
    family in hand that is a parameter sweep instead of an argument.

    Inflation runs at large ``phi``, where the exponential is small, and
    ends near the origin where it is not.
    """

    id: ClassVar[str] = "string.cosmo.plateau"
    provenance: ClassVar[str] = (
        "Generic large-field moduli plateau, the schematic form quoted in "
        "string-inflation reviews; p = 1 for a fibration or volume modulus "
        "and p = 4/3 for a blow-up modulus, where canonical normalisation "
        "of a local divisor volume gives tau ~ phi^(4/3)"
    )
    truncation: ClassVar[str] = (
        "one exponential only: the further alpha'-suppressed exponentials "
        "and the loop-suppressed positive exponential that eventually bounds "
        "the plateau at large field are dropped, so the plateau here extends "
        "indefinitely and a pivot placed arbitrarily far out is not physical"
    )

    amplitude: float = 1e-10
    coefficient: float = 1.0
    decay_constant: float = 1.0
    exponent: float = 1.0
    domain: ClassVar[tuple[float, float]] = (0.0, math.inf)

    def __post_init__(self) -> None:
        if self.coefficient <= 0.0:
            raise ValueError(
                f"coefficient must be positive, got {self.coefficient}: a negative one "
                "turns the plateau into a runaway and there is no inflation on it"
            )
        if self.decay_constant <= 0.0:
            raise ValueError(f"decay_constant must be positive, got {self.decay_constant}")
        if self.exponent <= 0.0:
            raise ValueError(f"exponent must be positive, got {self.exponent}")

    @property
    def typical_field(self) -> float:
        # Where the exponent u = (phi/f)^p is five, so the correction is
        # e^-5 and the field is unambiguously on the plateau.
        return self.decay_constant * 5.0 ** (1.0 / self.exponent)

    def _exponent_and_derivatives(self, phi):
        phi = np.asarray(phi, dtype=float)
        p, f = self.exponent, self.decay_constant
        scaled = phi / f
        u = scaled**p
        first = (p / f) * scaled ** (p - 1.0)
        second = (p * (p - 1.0) / f**2) * scaled ** (p - 2.0)
        third = (p * (p - 1.0) * (p - 2.0) / f**3) * scaled ** (p - 3.0)
        return u, first, second, third

    def value(self, phi):
        u, _, _, _ = self._exponent_and_derivatives(phi)
        return self.amplitude * (1.0 - self.coefficient * np.exp(-u))

    def gradient(self, phi):
        u, first, _, _ = self._exponent_and_derivatives(phi)
        return self.amplitude * self.coefficient * np.exp(-u) * first

    def curvature(self, phi):
        u, first, second, _ = self._exponent_and_derivatives(phi)
        return self.amplitude * self.coefficient * np.exp(-u) * (second - first**2)

    def third_derivative(self, phi):
        u, first, second, third = self._exponent_and_derivatives(phi)
        return (
            self.amplitude
            * self.coefficient
            * np.exp(-u)
            * (third - 3.0 * first * second + first**3)
        )


@dataclass(frozen=True)
class KahlerModuli(PlateauInflation):
    """Blow-up Kaehler moduli inflation: the plateau with ``p = 4/3``.

    The inflaton is the volume of a blow-up cycle, and canonical
    normalisation of a local divisor volume is power-law rather than
    logarithmic -- ``tau ~ phi^(4/3)`` -- which is where the exponent comes
    from. The consequence is the model's signature: the tilt is the
    universal ``1 - 2/N`` but the tensor ratio carries an extra
    ``u^(2/p - 2) = 1/sqrt(u)`` and is far below anything observable.
    """

    id: ClassVar[str] = "string.cosmo.kahler"
    provenance: ClassVar[str] = (
        "Conlon and Quevedo, JHEP 0601:146 (2006), hep-th/0509012: Kaehler "
        "moduli inflation in type IIB large-volume compactifications, quoted "
        "for a model-independent n_s = 1 - 2/N_e = 0.960 to 0.967 at "
        "N_e = 50 to 60"
    )
    truncation: ClassVar[str] = (
        "the leading exponential of the blow-up potential only, with the "
        "prefactor absorbed into the coefficient; the volume-mode direction, "
        "the racetrack structure and the uplift are not modelled, and no "
        "check is made that the parameters correspond to a stabilised vacuum"
    )

    exponent: float = 4.0 / 3.0


@dataclass(frozen=True)
class FibreInflation(Potential):
    """``V = A (3 - 4 e^(-c phi) + e^(-4 c phi))`` with ``c = 1/sqrt(3)``.

    The inflaton is a fibration modulus of a K3- or T^4-fibred Calabi-Yau,
    and the potential is generated by string loop corrections. The
    exponential structure is what makes the model predictive: writing
    ``y = (4/3) e^(-c phi)`` so that ``V = 3A(1 - y)`` up to the quartic
    term,

        sqrt(2 eps) = c y/(1-y),    eta = -c^2 y/(1-y)   =>   eps = eta^2/(2c^2)

    so with ``c^2 = 1/3`` the source paper's ``eps ~= (3/2) eta^2`` and
    ``r ~= 6 (n_s - 1)^2`` follow, and the tensor ratio is pinned to the
    tilt rather than being a free parameter. At the observed tilt that is
    ``r ~= 0.007``, inside the paper's quoted 0.005 to 0.01.

    The quartic term is utterly negligible at the pivot -- ``e^(-4 c phi)``
    is 2e-6 there against 3 -- and matters only in that it makes the
    potential vanish quadratically at the origin, ``V -> 2 A phi^2``, which
    is where inflation ends.
    """

    id: ClassVar[str] = "string.cosmo.fibre"
    provenance: ClassVar[str] = (
        "Cicoli, Burgess and Quevedo, JCAP 0903:013 (2009), arXiv:0808.0691: "
        "fibre inflation from string loop corrections in type IIB large-volume "
        "compactifications. The exponent 1/sqrt(3) and the predictions "
        "eps ~= (3/2) eta^2, r ~= 6 (n_s-1)^2 and r ~= 0.005 to 0.01 are that "
        "paper's; the numerical prefactors here are absorbed into one amplitude"
    )
    truncation: ClassVar[str] = (
        "the three alpha'- and loop-generated negative exponentials only. The "
        "loop-suppressed positive exponential that bounds the plateau at large "
        "field is dropped, so this potential inflates for unboundedly many "
        "e-folds where the full model does not; the volume and the other moduli "
        "are taken as frozen"
    )

    amplitude: float = 1e-10
    domain: ClassVar[tuple[float, float]] = (0.0, math.inf)

    @property
    def typical_field(self) -> float:
        return 6.0

    def _plateau(self, phi):
        return np.exp(-FIBRE_EXPONENT * np.asarray(phi, dtype=float))

    def value(self, phi):
        x = self._plateau(phi)
        return self.amplitude * (3.0 - 4.0 * x + x**4)

    def gradient(self, phi):
        x = self._plateau(phi)
        return 4.0 * self.amplitude * FIBRE_EXPONENT * (x - x**4)

    def curvature(self, phi):
        x = self._plateau(phi)
        return 4.0 * self.amplitude * FIBRE_EXPONENT**2 * (4.0 * x**4 - x)

    def third_derivative(self, phi):
        x = self._plateau(phi)
        return 4.0 * self.amplitude * FIBRE_EXPONENT**3 * (x - 16.0 * x**4)


@dataclass(frozen=True)
class AxionMonodromy(Potential):
    """``V = A phi^p + L cos(phi/f)``: a monomial from monodromy, plus instantons.

    Wrapping a brane gives an axion a monodromy, which turns its compact
    field range into an unbounded one and leaves a monomial potential
    behind. The exponent depends on the construction: ``p = 1`` for the
    linear case, ``p = 2/3`` and ``p = 3/2`` for others, and ``p = 2`` is
    ordinary chaotic inflation. Because the underlying field is still
    periodic, instantons add an oscillation on top, and that term is what
    distinguishes monodromy from a monomial written down by hand: it puts
    resonant features in the spectrum.

    The monomial part is exactly solvable in slow roll -- it is the
    ``PowerLaw`` family -- so ``n_s = 1 - (2p+4)/(4N+p)`` and
    ``r = 16p/(4N+p)``. At ``p = 1`` and sixty e-folds that is
    ``n_s = 0.9751`` and ``r = 0.0664``, the ``r ~= 0.07`` the linear model
    is quoted for.

    The modulation is kept small enough that ``V'`` does not change sign;
    at larger amplitudes the field traps and this is the wrong solver for
    it, which is why :func:`particlesim.cosmo.inflation.efolds` checks the
    sign of the gradient rather than assuming a monotonic roll.
    """

    id: ClassVar[str] = "string.cosmo.monodromy"
    provenance: ClassVar[str] = (
        "Silverstein and Westphal, PRD 78:106003 (2008) for monodromy-extended "
        "axion inflation; McAllister, Silverstein and Westphal, PRD 82:046003 "
        "(2010) for the linear potential and its r ~= 0.07; Flauger, McAllister, "
        "Pajer, Westphal and Xu, JCAP 1006:009 (2010) for the instanton "
        "modulation"
    )
    truncation: ClassVar[str] = (
        "one monomial and one harmonic. Flattening of the monomial by "
        "backreaction (which lowers the effective p at large field), the "
        "drift of the modulation amplitude with field, and multi-instanton "
        "harmonics are all dropped"
    )

    amplitude: float = 1e-10
    exponent: float = 1.0
    modulation: float = 0.0
    decay_constant: float = 0.1
    domain: ClassVar[tuple[float, float]] = (0.0, math.inf)

    def __post_init__(self) -> None:
        if self.exponent <= 0.0:
            raise ValueError(f"exponent must be positive, got {self.exponent}")
        if self.decay_constant <= 0.0:
            raise ValueError(f"decay_constant must be positive, got {self.decay_constant}")

    @property
    def typical_field(self) -> float:
        return 15.0 * math.sqrt(self.exponent / 2.0) if self.exponent > 0.5 else 12.0

    def value(self, phi):
        phi = np.asarray(phi, dtype=float)
        return self.amplitude * phi**self.exponent + self.modulation * np.cos(
            phi / self.decay_constant
        )

    def gradient(self, phi):
        phi = np.asarray(phi, dtype=float)
        f = self.decay_constant
        return self.amplitude * self.exponent * phi ** (self.exponent - 1.0) - (
            self.modulation / f
        ) * np.sin(phi / f)

    def curvature(self, phi):
        phi = np.asarray(phi, dtype=float)
        p, f = self.exponent, self.decay_constant
        return self.amplitude * p * (p - 1.0) * phi ** (p - 2.0) - (
            self.modulation / f**2
        ) * np.cos(phi / f)

    def third_derivative(self, phi):
        phi = np.asarray(phi, dtype=float)
        p, f = self.exponent, self.decay_constant
        return self.amplitude * p * (p - 1.0) * (p - 2.0) * phi ** (p - 3.0) + (
            self.modulation / f**3
        ) * np.sin(phi / f)


@dataclass(frozen=True)
class DBraneInflation(Potential):
    """``V = A (1 - (mu/phi)^4)``: the Coulomb potential of a brane pair.

    A D3-brane in a warped throat is attracted to an anti-D3-brane at the
    tip, and the inflaton is their separation. Slow roll is again exactly
    solvable at ``phi >> mu``: ``sqrt(2 eps) = 4 mu^4/phi^5`` gives
    ``N = phi^6/(24 mu^4)``, so ``eta = -20 mu^4/phi^6 = -5/(6N)`` and

        n_s - 1 = 2 eta - 6 eps = -5/(3N) - 6 eps

    The tilt is therefore robust -- ``0.972`` at sixty e-folds -- and the
    tensor ratio is *not*: ``eps = 8 mu^(4/3)/(24N)^(5/3)`` still carries
    the scale ``mu``, so ``r`` is a free parameter of the model rather than
    a prediction of it, and in the warped regime where the construction is
    controlled it is tiny. That asymmetry is the honest content of the
    model and the reason it is here next to fibre inflation, where the
    opposite is true.
    """

    id: ClassVar[str] = "string.cosmo.dbrane"
    provenance: ClassVar[str] = (
        "Kachru, Kallosh, Linde, Maldacena, McAllister and Trivedi, JCAP "
        "0310:013 (2003): the D3-anti-D3 Coulomb potential "
        "V = V_0 (1 - (M_Pl Delta/phi)^4) in a warped throat"
    )
    truncation: ClassVar[str] = (
        "the Coulomb term alone. The moduli-stabilisation mass term that "
        "causes the eta problem in this model, and the compactification "
        "corrections that are tuned against it to restore slow roll, are both "
        "dropped; inflation here ends at epsilon = 1 rather than at brane "
        "annihilation"
    )

    amplitude: float = 1e-10
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError(f"scale must be positive, got {self.scale}")

    @property
    def domain(self) -> tuple[float, float]:
        # Below the brane separation scale the Coulomb term dominates and
        # the truncated potential is negative.
        return (self.scale, math.inf)

    @property
    def typical_field(self) -> float:
        return 4.0 * self.scale

    def value(self, phi):
        phi = np.asarray(phi, dtype=float)
        return self.amplitude * (1.0 - (self.scale / phi) ** 4)

    def gradient(self, phi):
        phi = np.asarray(phi, dtype=float)
        return 4.0 * self.amplitude * self.scale**4 / phi**5

    def curvature(self, phi):
        phi = np.asarray(phi, dtype=float)
        return -20.0 * self.amplitude * self.scale**4 / phi**6

    def third_derivative(self, phi):
        phi = np.asarray(phi, dtype=float)
        return 120.0 * self.amplitude * self.scale**4 / phi**7
