"""The NS-NS sector in four dimensions, and what dualising the B-field costs.

Issue #70. The bosonic string's massless sector is the metric, the dilaton
and the Kalb-Ramond two-form. Reduced to four dimensions its action is, in
the *string* frame,

    sqrt(-g_S) e^(-2 phi) [ R_S + 4 (grad phi)^2 - (1/12) H^2 ]

with ``H = dB``. Three things have to be got right to turn that into
something a solver can use, and each is checked here rather than quoted.

**The frame change is a total derivative, not an identity.** Setting
``g_S = e^(2 phi) g_E`` does *not* map the string-frame density onto

    sqrt(-g_E) [ R_E - 2 (grad phi)^2 - (1/12) e^(-4 phi) H^2 ]

term by term. The two differ by the divergence of ``-6 sqrt(-g) grad^mu
phi``, and only because that is a total derivative do they define the same
theory. On a Friedmann metric the difference comes out as ``d/dt(6 a^3
phidot)`` exactly, which is what :func:`frame_boundary_current` returns and
what the tests subtract. A reduction that merely got the coefficients right
would be a different action with the same terms in it.

**The dilaton's exponent flips under dualisation, and the factor is 3!.**
In four dimensions a three-form is dual to a one-form, so ``H`` carries the
same information as an axion ``chi``. Substituting
``H^(mu nu rho) = e^(4 phi) eps^(mu nu rho sigma) grad_sigma chi`` and
contracting,

    eps^(mu nu rho sigma) eps_(mu nu rho lambda) = 3! delta^sigma_lambda

turns ``-(1/12) e^(-4 phi) H^2`` into a kinetic term with coefficient
``1/2`` and, crucially, ``e^(+4 phi)``. The sign of the exponent reverses:
the axion is *strongly* coupled where the two-form was weakly coupled. That
is the whole reason the dual description is useful and the easiest thing to
get backwards, so :func:`levi_civita_contraction` computes the ``6`` from
the tensor rather than asserting it.

**A dualisation is defined by what happens to the Bianchi identity.** The
two-form's ``dH = 0`` holds identically, being ``d(dB)``; after the
substitution it becomes the axion's *equation of motion*,
``grad_mu(e^(4 phi) grad^mu chi) = 0``. And the two-form's equation of
motion becomes ``d(d chi) = 0``, which is trivial. Swapping a Bianchi
identity for a field equation is what a duality *is*, and
:func:`exterior_derivative` against :func:`axion_equation` checks it
exactly rather than by inspection.
"""

from __future__ import annotations

from itertools import permutations

import sympy as sp

from particlesim.theories.base import Coupling, Theory

POSITIVE = (0.0, float("inf"))


def _determinant_root(metric) -> sp.Expr:
    return sp.sqrt(-sp.Matrix(metric).det())


def _gradient_squared(metric, coords, field) -> sp.Expr:
    inverse = sp.Matrix(metric).inv()
    size = len(coords)
    return sum(
        inverse[i, j] * sp.diff(field, coords[i]) * sp.diff(field, coords[j])
        for i in range(size)
        for j in range(size)
    )


def string_frame_density(metric, coords, dilaton, three_form_squared=sp.S.Zero) -> sp.Expr:
    """``sqrt(-g) e^(-2 phi) [R + 4 (grad phi)^2 - H^2/12]`` in the string frame."""
    from particlesim.symbolic.curvature import MetricGeometry

    scalar = MetricGeometry(sp.Matrix(metric), list(coords), simplify=True).ricci_scalar
    return (
        _determinant_root(metric)
        * sp.exp(-2 * dilaton)
        * (scalar + 4 * _gradient_squared(metric, coords, dilaton) - three_form_squared / 12)
    )


def einstein_frame_density(metric, coords, dilaton, axion_squared=sp.S.Zero) -> sp.Expr:
    """``sqrt(-g) [R - 2 (grad phi)^2 - e^(4 phi) (grad chi)^2 / 2]``.

    The axion's exponent is ``+4``, not ``-4``: see the module docstring, and
    :func:`axion_kinetic_coefficient` for where the ``1/2`` comes from.
    """
    from particlesim.symbolic.curvature import MetricGeometry

    scalar = MetricGeometry(sp.Matrix(metric), list(coords), simplify=True).ricci_scalar
    return _determinant_root(metric) * (
        scalar
        - 2 * _gradient_squared(metric, coords, dilaton)
        - sp.exp(4 * dilaton) * axion_squared / 2
    )


def frame_boundary_current(metric, coords, dilaton) -> list[sp.Expr]:
    """``-6 sqrt(-g) grad^mu phi``: what separates the two densities.

    Its divergence is exactly the difference between them. Returned as the
    current rather than as the difference so a test can verify the
    *statement* -- that the two actions differ by a total derivative -- which
    is what makes them the same theory, rather than verifying a number.
    """
    inverse = sp.Matrix(metric).inv()
    root = _determinant_root(metric)
    size = len(coords)
    return [
        -6 * root * sum(inverse[i, j] * sp.diff(dilaton, coords[j]) for j in range(size))
        for i in range(size)
    ]


def divergence(current, coords) -> sp.Expr:
    """``d_mu V^mu`` for a vector *density*, so no Christoffels are needed."""
    return sum(sp.diff(component, coord) for component, coord in zip(current, coords, strict=True))


# --- the dualisation ------------------------------------------------------


def levi_civita_contraction(dimension: int = 4) -> int:
    """``eps^(mu nu rho sigma) eps_(mu nu rho lambda) / delta^sigma_lambda``: ``3!``.

    Summed from the symbol rather than quoted, because this factor is the
    entire difference between the two-form's ``1/12`` and the axion's
    ``1/2``, and quoting it would make the coefficient an assumption.
    """
    free = dimension - 1
    total = 0
    for indices in permutations(range(dimension), free):
        total += sp.LeviCivita(*indices, 0) * sp.LeviCivita(*indices, 0)
    return int(total)


def axion_kinetic_coefficient(dimension: int = 4) -> sp.Rational:
    """``(1/12) * 3! = 1/2``, the coefficient the dual kinetic term carries."""
    return sp.Rational(levi_civita_contraction(dimension), 12)


def dual_three_form(metric, coords, dilaton, axion) -> dict[tuple[int, int, int], sp.Expr]:
    """``H_(mu nu rho) = e^(4 phi) sqrt(-g) eps_(mu nu rho sigma) grad^sigma chi``.

    Returned as the independent components keyed by ascending index triples;
    the rest follow by antisymmetry and carrying them would invite a sign
    error in the bookkeeping rather than in the physics.
    """
    inverse = sp.Matrix(metric).inv()
    root = _determinant_root(metric)
    size = len(coords)

    components: dict[tuple[int, int, int], sp.Expr] = {}
    for triple in permutations(range(size), 3):
        if list(triple) != sorted(triple):
            continue
        total = sp.S.Zero
        for sigma in range(size):
            symbol = sp.LeviCivita(*triple, sigma)
            if symbol == 0:
                continue
            raised = sum(inverse[sigma, tau] * sp.diff(axion, coords[tau]) for tau in range(size))
            total += symbol * raised
        components[triple] = sp.simplify(sp.exp(4 * dilaton) * root * total)
    return components


def exterior_derivative(components, coords) -> sp.Expr:
    """``(dH)_(0123)``, the single independent component of a 4-form in 4D.

    ``d_0 H_(123) - d_1 H_(023) + d_2 H_(013) - d_3 H_(012)``.
    """
    signs = {(1, 2, 3): 1, (0, 2, 3): -1, (0, 1, 3): 1, (0, 1, 2): -1}
    return sp.simplify(
        sum(
            sign * sp.diff(components[triple], coords[[c for c in range(4) if c not in triple][0]])
            for triple, sign in signs.items()
        )
    )


def axion_equation(metric, coords, dilaton, axion) -> sp.Expr:
    """``d_mu(sqrt(-g) e^(4 phi) g^(mu nu) d_nu chi)``, the dual field's equation.

    The same object :func:`exterior_derivative` produces from the two-form's
    Bianchi identity, which is the content of the duality.
    """
    inverse = sp.Matrix(metric).inv()
    root = _determinant_root(metric)
    size = len(coords)
    current = [
        root
        * sp.exp(4 * dilaton)
        * sum(inverse[i, j] * sp.diff(axion, coords[j]) for j in range(size))
        for i in range(size)
    ]
    return sp.simplify(divergence(current, coords))


class Dilaton(Theory):
    """The NS-NS sector as a four-dimensional plugin.

    ``coupling`` multiplies everything the dilaton and axion do, so General
    Relativity sits at zero where a test can evaluate it -- the convention
    the electromagnetic sector and :class:`~particlesim.theories.kk.KaluzaKlein`
    already use.

    Like the Kaluza-Klein plugin, its ``effective_stress_energy`` is ``G/8pi``:
    the *Einstein*-frame action has a canonical Einstein-Hilbert term by
    construction, and the dilaton and axion are matter rather than a
    modification of gravity. In the string frame that would not be true, and
    the frame this plugin declares is the reason it is.
    """

    id = "string.eft4d.dilaton"
    frame = "einstein"
    couplings = [Coupling("coupling", 0.0, units="dimensionless", bounds=POSITIVE)]
    provenance = (
        "The NS-NS sector of the bosonic string; Kalb and Ramond 1974, "
        "Phys. Rev. D 9, 2273 for the two-form, and the standard "
        "three-form-to-axion dualisation in four dimensions"
    )
    validity_statement = (
        "tree level and lowest order in alpha'; the dilaton is massless and "
        "unstabilised, which is a known problem of the sector rather than of "
        "this implementation"
    )

    def gr_limit(self) -> dict[str, float]:
        return {"coupling": 0.0}

    def effective_stress_energy(self, einstein, metric):
        return sp.Matrix(einstein) / (8 * sp.pi)

    def observable_predictions(self) -> dict:
        strength = self.values["coupling"]
        return {
            "axion_dilaton_coupling": strength,
            "dilaton_decoupled": strength == 0.0,
            "axion_kinetic_exponent": 4,
            "two_form_kinetic_exponent": -4,
        }


__all__ = [
    "Dilaton",
    "axion_equation",
    "axion_kinetic_coefficient",
    "divergence",
    "dual_three_form",
    "einstein_frame_density",
    "exterior_derivative",
    "frame_boundary_current",
    "levi_civita_contraction",
    "string_frame_density",
]
