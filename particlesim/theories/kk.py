"""Kaluza-Klein reduction: the tower, the lattice that has to reproduce it, moduli.

Issue #73. Compactify ``D = 4 + n`` on a torus or a simple orbifold and the
higher-dimensional field becomes a 4D tower whose masses are set by the
compactification geometry. The spectrum is the whole observable content, and
it is known in closed form, which is what makes the acceptance checkable.

**The tower.** On a circle of radius ``R`` a field's dependence on the extra
coordinate is a Fourier series, and each mode is a 4D field of mass ``|n|/R``.
On a torus with radii ``R_i``,

    m^2(n) = sum_i (n_i / R_i)^2

with the zero mode massless and everything else at the compactification
scale. The radii are *moduli*: they are not put in by hand but are 4D scalar
fields, and the whole tower moves when they do. That is the content of
:meth:`Torus.radion_coupling` -- the tower's scale is the modulus.

**Why a lattice at all, and why it is the real test.** A closed-form spectrum
compared against itself proves nothing. So :func:`lattice_spectrum` puts the
extra dimensions on a grid, builds the nearest-neighbour Laplacian and
*diagonalises* it. The eigenvalues are not the continuum ones: a lattice has
its own dispersion, so a mode at level ``k`` on ``N`` points comes out at

    m^2 = (4 / h^2) sin^2(pi k / N)

which sits *below* the continuum ``(k/R)^2`` and converges to it at second
order in ``1/N``. That is the same dispersion that governs
:mod:`particlesim.solvers.lattice.realtime`, and asserting the convergence
rather than a tolerance is what distinguishes "the reduction is right" from
"the grid was fine enough".

**The orbifold removes half the tower, and sometimes the zero mode.** On
``S^1/Z_2`` the circle is folded to an interval and fields are classified by
parity. Even fields keep ``cos(n y / R)`` for ``n >= 0`` -- including the
massless zero mode -- and odd fields keep ``sin(n y / R)`` for ``n >= 1``,
which has none. Both towers lose the two-fold degeneracy of ``+n`` and ``-n``,
because the projection identifies them. A zero mode that survives for one
parity and not the other is the mechanism the model-building literature uses
to break symmetries by boundary conditions, and it is a sharp thing to test:
the count changes by exactly one state.

**The cost model is why ``n <= 2``.** Diagonalising the lattice Laplacian
costs ``(points^n)^3``. At ``n = 1`` and 64 points that is 2.6e5 operations;
at ``n = 3`` it is 7.0e13, which is not a slow test but an impossible one.
:func:`lattice_cost` reports the number before it is paid, and
:func:`lattice_spectrum` refuses ``n > 2`` rather than appearing to hang.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import numpy as np

from particlesim.theories.base import Coupling, Theory

POSITIVE = (0.0, float("inf"))
LATTICE_DIMENSION_LIMIT = 2


@dataclass(frozen=True)
class Torus:
    """A flat ``n``-torus of circumferences ``2 pi R_i``, the compact space."""

    radii: tuple[float, ...] = (1.0,)

    def __post_init__(self) -> None:
        if not self.radii:
            raise ValueError("a compactification needs at least one extra dimension")
        for radius in self.radii:
            if radius <= 0.0:
                raise ValueError(f"every radius must be positive, got {self.radii}")

    @property
    def dimensions(self) -> int:
        return len(self.radii)

    @property
    def volume(self) -> float:
        """``prod (2 pi R_i)``, which sets the 4D Newton constant."""
        return float(np.prod([2.0 * np.pi * radius for radius in self.radii]))

    def mass_squared(self, modes) -> float:
        """``sum (n_i / R_i)^2`` for one set of mode numbers."""
        numbers = np.atleast_1d(np.asarray(modes, dtype=float))
        if numbers.size != self.dimensions:
            raise ValueError(
                f"expected {self.dimensions} mode numbers for this torus, got {numbers.size}"
            )
        return float(np.sum((numbers / np.asarray(self.radii)) ** 2))

    def radion_coupling(self) -> tuple[float, ...]:
        """``d m / d R`` at the first level: the tower rides on the moduli.

        Negative in every direction -- growing a dimension lowers its tower --
        which is the statement that the radii are dynamical fields whose
        motion is visible as the whole spectrum moving, not a set of
        parameters chosen once.
        """
        return tuple(-1.0 / radius**2 for radius in self.radii)

    def spectrum(self, levels: int = 3) -> list[tuple[float, int]]:
        """``(mass, degeneracy)`` up to ``levels`` in each direction, ascending.

        Degeneracies are counted rather than assumed: on a circle every
        non-zero level is two-fold from ``+n`` and ``-n``, but on a square
        torus level one is four-fold and a rectangular one splits it back
        into two pairs. Getting that structure right is most of what
        "reproduces the spectrum" means.
        """
        if levels < 0:
            raise ValueError(f"the level count must not be negative, got {levels}")
        counts: dict[float, int] = {}
        span = range(-levels, levels + 1)
        for modes in product(span, repeat=self.dimensions):
            mass = math.sqrt(self.mass_squared(modes))
            key = round(mass, 12)
            counts[key] = counts.get(key, 0) + 1
        return sorted(counts.items())


def orbifold_spectrum(
    torus: Torus, levels: int = 3, parity: str = "even"
) -> list[tuple[float, int]]:
    """``S^1/Z_2`` on the first direction: half the tower, and a zero mode or not.

    ``parity`` selects the fields kept by the projection. Even ones are
    cosines and start at ``n = 0``; odd ones are sines and start at ``n = 1``,
    so an odd field has *no* massless mode. Both are non-degenerate, because
    identifying ``y`` with ``-y`` identifies ``+n`` with ``-n``.
    """
    if parity not in ("even", "odd"):
        raise ValueError(f"parity must be 'even' or 'odd', got {parity!r}")
    if levels < 0:
        raise ValueError(f"the level count must not be negative, got {levels}")

    start = 0 if parity == "even" else 1
    counts: dict[float, int] = {}
    others = [range(-levels, levels + 1) for _ in range(torus.dimensions - 1)]
    for first in range(start, levels + 1):
        for rest in product(*others):
            mass = math.sqrt(torus.mass_squared((first, *rest)))
            key = round(mass, 12)
            counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items())


def lattice_cost(torus: Torus, points: int) -> dict[str, float]:
    """Sites and dense-diagonalisation operations, before either is paid."""
    if points < 2:
        raise ValueError(f"a lattice needs at least two points per side, got {points}")
    sites = float(points**torus.dimensions)
    return {
        "dimensions": float(torus.dimensions),
        "sites": sites,
        "operations": sites**3,
    }


def lattice_laplacian(torus: Torus, points: int) -> np.ndarray:
    """The nearest-neighbour Laplacian on the compact space, as a dense matrix.

    Built explicitly and diagonalised rather than evaluated from the
    analytic lattice dispersion, because the point is to *check* that
    dispersion rather than to assume it.
    """
    cost = lattice_cost(torus, points)
    if torus.dimensions > LATTICE_DIMENSION_LIMIT:
        raise ValueError(
            f"lattice mode is limited to {LATTICE_DIMENSION_LIMIT} extra dimensions; "
            f"{torus.dimensions} would need {cost['operations']:.1e} operations to "
            "diagonalise, which is not a slow test but an impossible one"
        )

    size = int(cost["sites"])
    matrix = np.zeros((size, size))
    shape = (points,) * torus.dimensions
    spacings = [2.0 * np.pi * radius / points for radius in torus.radii]

    for flat in range(size):
        index = np.unravel_index(flat, shape)
        for axis, spacing in enumerate(spacings):
            matrix[flat, flat] -= 2.0 / spacing**2
            for step in (-1, 1):
                neighbour = list(index)
                neighbour[axis] = (neighbour[axis] + step) % points
                matrix[flat, int(np.ravel_multi_index(tuple(neighbour), shape))] += 1.0 / spacing**2
    return matrix


def lattice_spectrum(torus: Torus, points: int) -> np.ndarray:
    """``m^2`` of every mode on the lattice, ascending, from diagonalisation.

    ``-lap`` is positive semi-definite, so the eigenvalues are the squared
    masses directly. Tiny negative values from round-off are clipped: the
    zero mode is exactly massless and a ``-1e-17`` would become a nan under
    a square root.
    """
    eigenvalues = np.linalg.eigvalsh(-lattice_laplacian(torus, points))
    return np.sort(np.maximum(eigenvalues, 0.0))


def lattice_masses(torus: Torus, points: int) -> np.ndarray:
    """Lattice masses, ``sqrt`` of :func:`lattice_spectrum`."""
    return np.sqrt(lattice_spectrum(torus, points))


def continuum_limit(level: int, points: int) -> float:
    """``sinc(pi k / N)``: how far a lattice level sits below the continuum.

    ``(4/h^2) sin^2(pi k/N)`` against ``(k/R)^2`` is exactly this squared, so
    a test can predict the lattice's answer rather than tolerate it.
    """
    if points < 2:
        raise ValueError(f"a lattice needs at least two points per side, got {points}")
    angle = np.pi * level / points
    return 1.0 if level == 0 else float(np.sin(angle) / angle)


class KaluzaKlein(Theory):
    """``D = 4 + n`` on a torus, with the inverse radius carrying the limit.

    ``inverse_radius`` rather than ``radius``, so four-dimensional physics
    sits at **zero** and a test can evaluate it. Decompactification is
    ``R -> infinity``, which no test can set; the tower's mass ``n/R`` is
    linear in the inverse radius, so the same statement becomes
    ``inverse_radius = 0`` and the whole tower goes to zero mass -- or, read
    the other way, becomes infinitely heavy in units of the tower spacing
    and decouples, leaving 4D gravity. This is the same convention that puts
    Maxwell at zero in the electromagnetic sector, and for the same reason: a
    limit you cannot evaluate is a claim, not a check.
    """

    id = "string.kk"
    couplings = [
        Coupling("inverse_radius", 0.0, units="1/length", bounds=POSITIVE),
        Coupling("dimensions", 1.0, units="count", bounds=POSITIVE),
    ]
    provenance = (
        "Kaluza 1921, Sitzungsber. Preuss. Akad. Wiss. 966; Klein 1926, "
        "Z. Phys. 37, 895; the modern tower and moduli as in any "
        "compactification review"
    )
    validity_statement = (
        "flat toroidal compactification with no warping and no flux; the moduli "
        "are treated as fixed backgrounds rather than stabilised"
    )

    def gr_limit(self) -> dict[str, float]:
        return {"inverse_radius": 0.0}

    def effective_stress_energy(self, einstein, metric):
        """``G / 8 pi``, exactly, and at every radius rather than only the limit.

        That is a statement rather than a shortcut. A *flat, unwarped* torus
        with fixed moduli and no flux has zero internal curvature and no
        potential, so the reduction gives four-dimensional Einstein gravity
        untouched: the compact space contributes nothing to the 4D vacuum
        stress-energy. The observable content of the compactification is the
        **tower**, which is matter one can excite, not a modified vacuum
        action -- see :meth:`torus` and :func:`Torus.spectrum`.

        The limitations that would break this are exactly the ones
        ``validity_statement`` names: warping, flux, a stabilising potential
        for the moduli, and the Casimir energy of the tower itself, which is
        a quantum effect and not present at tree level.
        """
        import sympy as sp

        return sp.Matrix(einstein) / (8 * sp.pi)

    def torus(self) -> Torus:
        """The compact space these couplings describe."""
        inverse = self.values["inverse_radius"]
        if inverse <= 0.0:
            raise ValueError(
                "a decompactified theory has no torus: inverse_radius is zero, "
                "which is the four-dimensional limit rather than a geometry"
            )
        count = int(self.values["dimensions"])
        return Torus(radii=(1.0 / inverse,) * count)

    def observable_predictions(self) -> dict:
        inverse = self.values["inverse_radius"]
        return {
            "tower_spacing": inverse,
            "extra_dimensions": int(self.values["dimensions"]),
            "lowest_massive_mode": inverse,
            "decoupled": inverse == 0.0,
        }


__all__ = [
    "KaluzaKlein",
    "Torus",
    "continuum_limit",
    "lattice_cost",
    "lattice_laplacian",
    "lattice_masses",
    "lattice_spectrum",
    "orbifold_spectrum",
]
