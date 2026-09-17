"""Inflaton potentials (design doc Section 3.1, Level C1).

Everything here is in **reduced Planck units**, ``M_p = (8 pi G)^(-1/2) = 1``.
That is the convention every inflationary formula in the literature is
quoted in: the slow-roll parameters are dimensionless in it, the Starobinsky
exponent is ``sqrt(2/3)`` rather than carrying a mass, and a field
excursion of "ten" means ten reduced Planck masses. It is *not* the
convention :mod:`particlesim.cosmo.dynamics` uses, which is geometric
``G = c = 1`` because that is what the Tier B plugins are written in. The
two are kept apart rather than reconciled, because a single module trying to
serve both would have to guess which one a caller meant.

A potential supplies its value and the first three derivatives
analytically. The first two are what the equations of motion and the
slow-roll parameters need; the third only enters the running of the spectral
index, so :meth:`Potential.third_derivative` has a finite-difference
fallback for potentials where it is painful to write down. The built-ins all
override it, and a test checks every one of them against a central
difference, because an analytic derivative that is wrong is worse than a
numerical one that is approximate.

Each potential also declares ``domain`` -- the field interval on which it
describes inflation -- and ``typical_field``, a point inside the inflating
region. Those exist so that the searches in
:mod:`particlesim.cosmo.inflation` have somewhere to start and somewhere to
stop: natural inflation lives on ``(0, pi f)`` and a bracket that wanders
outside it finds a negative potential rather than an error.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

#: Step used by the finite-difference fallback for the third derivative.
#:
#: A central difference of an analytic second derivative has error
#: ``O(h^2 V'''') + O(eps_machine / h^2)``; at ``h = 1e-4`` both terms sit
#: near ``1e-8`` relative, which is four orders below the running of the
#: spectral index it is used to compute.
FD_STEP = 1e-4

#: ``sqrt(2/3)``, the exponent of the Starobinsky plateau.
STAROBINSKY_EXPONENT = math.sqrt(2.0 / 3.0)


class Potential(ABC):
    """A single-field inflaton potential ``V(phi)`` in reduced Planck units."""

    #: Field interval on which this potential describes inflation.
    domain: ClassVar[tuple[float, float]] = (-math.inf, math.inf)
    #: Registry id, keyed as in :func:`particlesim.theories.registry.list_inflaton_potentials`.
    id: ClassVar[str] = ""
    #: Where this form comes from, with a reference where there is one.
    provenance: ClassVar[str] = ""
    #: What has been dropped from the larger expression this truncates.
    #:
    #: Empty means nothing was: a quadratic is a quadratic. It is the
    #: string-inspired potentials in
    #: :mod:`particlesim.cosmo.string_potentials` that are expansions of
    #: something larger, and there the terms left out are the reason two
    #: papers on the same model can disagree, so they are named.
    truncation: ClassVar[str] = ""

    @property
    @abstractmethod
    def typical_field(self) -> float:
        """A field value inside the inflating region, used to seed searches."""

    @abstractmethod
    def value(self, phi):
        """``V(phi)``."""

    @abstractmethod
    def gradient(self, phi):
        """``dV/dphi``."""

    @abstractmethod
    def curvature(self, phi):
        """``d^2 V / dphi^2``."""

    def third_derivative(self, phi):
        """``d^3 V / dphi^3``, by central difference unless overridden."""
        phi = np.asarray(phi, dtype=float)
        h = FD_STEP * np.maximum(1.0, np.abs(phi))
        return (self.curvature(phi + h) - self.curvature(phi - h)) / (2.0 * h)

    def contains(self, phi) -> np.ndarray:
        low, high = self.domain
        phi = np.asarray(phi, dtype=float)
        return (phi > low) & (phi < high)

    def epsilon(self, phi):
        """Potential slow-roll parameter ``(1/2) (V'/V)^2``."""
        return 0.5 * (self.gradient(phi) / self.value(phi)) ** 2

    def eta(self, phi):
        """Potential slow-roll parameter ``V''/V``."""
        return self.curvature(phi) / self.value(phi)

    def xi_squared(self, phi):
        """Third slow-roll parameter ``V' V''' / V^2``, which sets the running."""
        return self.gradient(phi) * self.third_derivative(phi) / self.value(phi) ** 2

    def roll_direction(self, phi) -> float:
        """Sign of ``dphi/dN`` on the slow-roll trajectory at ``phi``.

        The field rolls downhill, ``dphi/dN = -V'/V`` with ``V > 0``, so the
        direction is fixed by the gradient and no potential has to declare
        it. A quadratic rolls towards the origin, natural inflation rolls
        away from it, and the searches that need to know which way the end
        of inflation lies ask this rather than assuming.
        """
        slope = float(self.gradient(phi))
        if slope == 0.0:
            raise ValueError(
                f"{type(self).__name__}: the potential is flat at phi = {float(phi):g}, "
                "so there is no rolling direction. Seed the search elsewhere"
            )
        return -math.copysign(1.0, slope)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}()"


@dataclass(frozen=True)
class Quadratic(Potential):
    """``V = (1/2) m^2 phi^2``: chaotic inflation, and the reference case.

    Slow roll gives ``n_s = 1 - 2/N`` and ``r = 8/N`` in closed form (the
    ``p = 2`` member of :class:`PowerLaw`), which is what makes it the first
    thing to check a numerical pipeline against.
    """

    id: ClassVar[str] = "inflation.quadratic"
    provenance: ClassVar[str] = (
        "Chaotic inflation, Linde 1983 (Phys. Lett. B 129, 177); the p = 2 "
        "member of the exactly solvable power-law family"
    )

    mass: float = 1e-5
    domain: ClassVar[tuple[float, float]] = (0.0, math.inf)

    @property
    def typical_field(self) -> float:
        return 15.0

    def value(self, phi):
        return 0.5 * self.mass**2 * np.asarray(phi, dtype=float) ** 2

    def gradient(self, phi):
        return self.mass**2 * np.asarray(phi, dtype=float)

    def curvature(self, phi):
        return np.full_like(np.asarray(phi, dtype=float), self.mass**2)

    def third_derivative(self, phi):
        return np.zeros_like(np.asarray(phi, dtype=float))


@dataclass(frozen=True)
class PowerLaw(Potential):
    """``V = A phi^p``.

    Slow roll is exactly solvable: ``epsilon = p/(4N+p)``, so

        n_s = 1 - (2p + 4)/(4N + p),   r = 16 p/(4N + p)

    with ``N`` counted from the pivot to ``epsilon_V = 1``. Both are used as
    benchmarks, and having a whole family rather than one case means a
    numerical spectrum can be checked against an analytic one at several
    exponents instead of at a single lucky point.
    """

    id: ClassVar[str] = "inflation.power_law"
    provenance: ClassVar[str] = (
        "Monomial large-field inflation; the closed forms for n_s and r are "
        "standard slow-roll algebra and are derived in this docstring"
    )

    amplitude: float = 1e-10
    exponent: float = 2.0
    domain: ClassVar[tuple[float, float]] = (0.0, math.inf)

    def __post_init__(self) -> None:
        if self.exponent <= 0.0:
            raise ValueError(
                f"exponent must be positive, got {self.exponent}: a negative power "
                "diverges at the origin and does not describe large-field inflation"
            )

    @property
    def typical_field(self) -> float:
        return 15.0 * math.sqrt(self.exponent / 2.0)

    def value(self, phi):
        return self.amplitude * np.asarray(phi, dtype=float) ** self.exponent

    def gradient(self, phi):
        p = self.exponent
        return self.amplitude * p * np.asarray(phi, dtype=float) ** (p - 1.0)

    def curvature(self, phi):
        p = self.exponent
        return self.amplitude * p * (p - 1.0) * np.asarray(phi, dtype=float) ** (p - 2.0)

    def third_derivative(self, phi):
        p = self.exponent
        return (
            self.amplitude * p * (p - 1.0) * (p - 2.0) * np.asarray(phi, dtype=float) ** (p - 3.0)
        )


@dataclass(frozen=True)
class Exponential(Potential):
    """``V = A exp(-lambda phi)``: power-law inflation, and the exact test.

    ``epsilon_V = lambda^2 / 2`` is constant, so inflation never ends and
    the Mukhanov-Sasaki equation has a closed-form Hankel solution. The
    spectrum is an exact power law with

        n_s - 1 = -2 epsilon / (1 - epsilon),    r = 16 epsilon

    both *exact*, not first order in ``epsilon``. That makes this the one
    benchmark here that distinguishes a correct mode solver from the
    slow-roll formula it is supposed to improve on: at ``lambda = 0.4`` the
    exact index is ``-0.1739`` where slow roll says ``-0.16``.
    """

    id: ClassVar[str] = "inflation.exponential"
    provenance: ClassVar[str] = (
        "Power-law inflation, Lucchin and Matarrese 1985 (PRD 32, 1316). The "
        "exact Hankel-function spectrum is what makes it the reference case "
        "for a mode solver"
    )

    amplitude: float = 1e-10
    rate: float = 0.4

    @property
    def typical_field(self) -> float:
        return 0.0

    def value(self, phi):
        return self.amplitude * np.exp(-self.rate * np.asarray(phi, dtype=float))

    def gradient(self, phi):
        return -self.rate * self.value(phi)

    def curvature(self, phi):
        return self.rate**2 * self.value(phi)

    def third_derivative(self, phi):
        return -(self.rate**3) * self.value(phi)


@dataclass(frozen=True)
class Starobinsky(Potential):
    """``V = A (1 - exp(-sqrt(2/3) phi))^2``: R^2 inflation in the Einstein frame.

    This is the potential the gate benchmark is stated on. The plateau at
    large ``phi`` makes ``epsilon`` exponentially small there, so the
    spectrum is close to scale invariant and the leading-order prediction is
    ``n_s = 1 - 2/N``, ``r = 12/N^2``.

    Provenance: Starobinsky 1980 (Phys. Lett. B 91, 99) for the model; the
    Einstein-frame potential and the ``n_s = 1 - 2/N`` asymptotics are
    standard, e.g. Planck 2018 inflation (A&A 641, A10).
    """

    id: ClassVar[str] = "inflation.starobinsky"
    provenance: ClassVar[str] = (
        "Starobinsky 1980 (Phys. Lett. B 91, 99), Einstein-frame potential of "
        "R + R^2 gravity; the n_s = 1 - 2/N, r = 12/N^2 asymptotics are standard"
    )

    amplitude: float = 1e-10
    domain: ClassVar[tuple[float, float]] = (0.0, math.inf)

    @property
    def typical_field(self) -> float:
        return 5.5

    def _plateau(self, phi):
        return np.exp(-STAROBINSKY_EXPONENT * np.asarray(phi, dtype=float))

    def value(self, phi):
        return self.amplitude * (1.0 - self._plateau(phi)) ** 2

    def gradient(self, phi):
        x = self._plateau(phi)
        return 2.0 * self.amplitude * STAROBINSKY_EXPONENT * x * (1.0 - x)

    def curvature(self, phi):
        x = self._plateau(phi)
        return 2.0 * self.amplitude * STAROBINSKY_EXPONENT**2 * x * (2.0 * x - 1.0)

    def third_derivative(self, phi):
        x = self._plateau(phi)
        return 2.0 * self.amplitude * STAROBINSKY_EXPONENT**3 * x * (1.0 - 4.0 * x)


@dataclass(frozen=True)
class Natural(Potential):
    """``V = A (1 + cos(phi/f))``: natural inflation on a pseudo-scalar axion.

    Inflation runs from the maximum at ``phi = 0`` down towards the minimum
    at ``phi = pi f``, so this is the one built-in whose field *increases*
    and the reason nothing here assumes a rolling direction. The decay
    constant has to exceed roughly one reduced Planck mass for sixty e-folds
    to fit, which is the well-known tension with axions in string
    compactifications.

    Provenance: Freese, Frieman and Olinto 1990 (PRL 65, 3233).
    """

    id: ClassVar[str] = "inflation.natural"
    provenance: ClassVar[str] = (
        "Freese, Frieman and Olinto 1990 (PRL 65, 3233): a pseudo-Nambu-Goldstone "
        "inflaton with a single instanton-generated harmonic"
    )

    amplitude: float = 1e-10
    decay_constant: float = 5.0

    def __post_init__(self) -> None:
        if self.decay_constant <= 0.0:
            raise ValueError(f"decay_constant must be positive, got {self.decay_constant}")

    @property
    def domain(self) -> tuple[float, float]:
        return (0.0, math.pi * self.decay_constant)

    @property
    def typical_field(self) -> float:
        # Half way up the inflating branch, which ends where
        # tan(phi/2f) = sqrt(2) f. Taking ``f`` itself would sit past the end
        # of inflation for a sub-Planckian decay constant -- exactly the
        # regime where the model is interesting because it fails.
        return self.decay_constant * math.atan(math.sqrt(2.0) * self.decay_constant)

    def value(self, phi):
        return self.amplitude * (1.0 + np.cos(np.asarray(phi, dtype=float) / self.decay_constant))

    def gradient(self, phi):
        f = self.decay_constant
        return -(self.amplitude / f) * np.sin(np.asarray(phi, dtype=float) / f)

    def curvature(self, phi):
        f = self.decay_constant
        return -(self.amplitude / f**2) * np.cos(np.asarray(phi, dtype=float) / f)

    def third_derivative(self, phi):
        f = self.decay_constant
        return (self.amplitude / f**3) * np.sin(np.asarray(phi, dtype=float) / f)


class MultiFieldPotential(ABC):
    """A potential on several fields, with the field index as the last axis.

    ``phi`` has shape ``(..., fields)`` throughout, so a single point is a
    one-dimensional array and a batch of points is two-dimensional. The
    gradient matches that shape and the Hessian adds one axis.
    """

    @property
    @abstractmethod
    def fields(self) -> int:
        """Number of fields."""

    @abstractmethod
    def value(self, phi):
        """``V(phi)``, shape ``phi.shape[:-1]``."""

    @abstractmethod
    def gradient(self, phi):
        """``dV/dphi_i``, shape ``phi.shape``."""

    @abstractmethod
    def hessian(self, phi):
        """``d^2 V / dphi_i dphi_j``, shape ``phi.shape + (fields,)``."""

    def epsilon(self, phi):
        """``(1/2) sum_i (V_i/V)^2``."""
        phi = np.asarray(phi, dtype=float)
        gradient = self.gradient(phi)
        value = self.value(phi)
        return 0.5 * np.sum((gradient / value[..., None]) ** 2, axis=-1)


@dataclass(frozen=True)
class SeparableSum(MultiFieldPotential):
    """``V = sum_i V_i(phi_i)``, one single-field potential per field.

    Sum-separable is not an arbitrary restriction: it is the class for which
    the number of e-folds along a slow-roll trajectory has a closed form,
    which is what makes the ``delta N`` spectrum in
    :mod:`particlesim.cosmo.perturbations` checkable against an analytic
    answer instead of only against itself. Coupled potentials are perfectly
    welcome as subclasses of :class:`MultiFieldPotential`; they just do not
    come with that check.
    """

    parts: tuple[Potential, ...]

    def __post_init__(self) -> None:
        if len(self.parts) < 1:
            raise ValueError("a separable sum needs at least one part")

    @property
    def fields(self) -> int:
        return len(self.parts)

    def value(self, phi):
        phi = np.asarray(phi, dtype=float)
        return sum(part.value(phi[..., i]) for i, part in enumerate(self.parts))

    def gradient(self, phi):
        phi = np.asarray(phi, dtype=float)
        return np.stack([part.gradient(phi[..., i]) for i, part in enumerate(self.parts)], axis=-1)

    def hessian(self, phi):
        phi = np.asarray(phi, dtype=float)
        diagonal = np.stack(
            [part.curvature(phi[..., i]) for i, part in enumerate(self.parts)], axis=-1
        )
        out = np.zeros(diagonal.shape + (self.fields,), dtype=float)
        index = np.arange(self.fields)
        out[..., index, index] = diagonal
        return out


def as_multifield(potential: Potential) -> SeparableSum:
    """Wrap a single-field potential as a one-field multi-field potential.

    Exists so the vector machinery can be run against the scalar machinery
    on the same physics: if the two disagree, one of them is wrong, and that
    is a sharper test than checking either against a formula.
    """
    return SeparableSum((potential,))
