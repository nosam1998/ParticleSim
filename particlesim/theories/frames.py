"""Conformal frames: the map between them, and the mistakes it makes possible.

Issue #74. A scalar-tensor theory can be written with the scalar multiplying
the Ricci scalar (the *Jordan* frame) or with a canonical Einstein-Hilbert
term and the scalar moved into the matter sector (the *Einstein* frame). The
two are related by ``g~ = Omega^2 g`` and describe the same physics, which is
exactly what makes mixing them dangerous: nothing goes wrong loudly.

**The transformation is checked against the curvature machinery, not
asserted.** In ``D`` dimensions,

    R~ = Omega^-2 [ R - 2(D-1) box(ln Omega)
                      - (D-1)(D-2) (grad ln Omega)^2 ]

with ``box`` and ``grad`` taken in the *original* metric. Evaluated on a
conformally flat four-metric and compared against
:class:`~particlesim.symbolic.curvature.MetricGeometry` computing ``R~``
directly from ``g~``, the difference is exactly zero -- so the sign
convention here is the repository's own rather than a textbook's that might
not match it.

**Where the error actually lives: the matter coupling.** Matter minimally
coupled in the Jordan frame is *not* minimally coupled in the Einstein
frame; the conformal factor reappears as a direct coupling between the
scalar and matter, which is what fifth-force experiments constrain. So a
theory has two frames, not one: the frame its gravitational action is
written in, and the frame its matter couples minimally to. They usually
agree, and when they do not it is the interesting case rather than a
mistake.

An electromagnetic sector's constitutive relation is written against a
particular metric. Composing one with a gravity plugin whose *matter* frame
differs means the relation is being evaluated on the wrong metric --
silently, because both are valid theories and the stack has no other way to
notice. :class:`~particlesim.theories.base.TheoryStack` therefore compares
``matter_frame`` as well as ``frame``, and a plugin that leaves it unset
inherits its own frame, so nothing existing changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from particlesim.theories.base import Frame


def conformal_metric(metric, factor) -> sp.Matrix:
    """``g~ = Omega^2 g``."""
    return sp.Matrix(metric) * sp.together(factor**2)


def conformal_ricci_scalar(metric, coords, factor, ricci_scalar=None) -> sp.Expr:
    """``R~`` from the closed form, in the original metric's derivatives.

    ``ricci_scalar`` is the original ``R``; left out, it is computed. Passing
    it in matters when the original is flat and known to be zero, because
    computing it symbolically is the expensive part and getting zero back is
    not informative.
    """
    metric = sp.Matrix(metric)
    coordinates = list(coords)
    dimension = metric.shape[0]
    if len(coordinates) != dimension:
        raise ValueError(
            f"expected {dimension} coordinates for a {dimension}x{dimension} metric, "
            f"got {len(coordinates)}"
        )

    if ricci_scalar is None:
        from particlesim.symbolic.curvature import MetricGeometry

        ricci_scalar = MetricGeometry(metric, coordinates, simplify=True).ricci_scalar

    logarithm = sp.log(factor)
    inverse = metric.inv()
    box = sum(
        inverse[a, b] * sp.diff(logarithm, coordinates[a], coordinates[b])
        for a in range(dimension)
        for b in range(dimension)
    )
    gradient = sum(
        inverse[a, b] * sp.diff(logarithm, coordinates[a]) * sp.diff(logarithm, coordinates[b])
        for a in range(dimension)
        for b in range(dimension)
    )
    return factor**-2 * (
        ricci_scalar - 2 * (dimension - 1) * box - (dimension - 1) * (dimension - 2) * gradient
    )


@dataclass(frozen=True)
class FrameMap:
    """``source -> target`` by ``g~ = factor^2 g``.

    The factor is required to be positive rather than merely non-zero: a
    conformal transformation has to preserve the signature, and a negative
    ``Omega^2`` would flip every sign in the metric while still squaring to
    something real. Sympy cannot always decide positivity of a symbol, so a
    symbol declared ``positive=True`` is accepted and a bare one is refused
    with the reason.
    """

    source: Frame
    target: Frame
    factor: sp.Expr

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise ValueError(f"a frame map must change frame; both ends are {self.source!r}")
        positive = sp.ask(sp.Q.positive(self.factor))
        if positive is False:
            raise ValueError(f"the conformal factor must be positive, got {self.factor}")
        if positive is None and not sp.S(self.factor).is_positive:
            raise ValueError(
                f"cannot establish that the conformal factor {self.factor} is positive; "
                "declare the symbol with positive=True, since a negative Omega^2 would "
                "flip the signature rather than rescale it"
            )

    def inverse(self) -> FrameMap:
        """The map back, with factor ``1/Omega``."""
        return FrameMap(source=self.target, target=self.source, factor=1 / self.factor)

    def apply(self, metric) -> sp.Matrix:
        return conformal_metric(metric, self.factor)


def einstein_frame_potential(potential, coupling) -> sp.Expr:
    """``V~ = V / F^2`` in four dimensions, where ``Omega^2 = F``.

    The reason a Jordan-frame potential that is flat at large field can
    become a plateau in the Einstein frame, which is most of why
    scalar-tensor inflation works at all.
    """
    return sp.together(potential / coupling**2)


def canonical_field_factor(coupling, kinetic, field) -> sp.Expr:
    """``(d phi_hat / d phi)^2 = omega/F + (3/2) (F'/F)^2``.

    What it costs to make the Einstein-frame scalar canonical. The second
    term comes from the conformal transformation itself and is there even
    when the Jordan-frame scalar has no kinetic term at all -- which is why
    ``f(R)`` gravity has a propagating scalar despite its action containing
    no ``(grad phi)^2``.
    """
    derivative = sp.diff(coupling, field)
    return sp.together(kinetic / coupling + sp.Rational(3, 2) * (derivative / coupling) ** 2)


def brans_dicke_canonical_factor(parameter, field) -> sp.Expr:
    """``(2 omega + 3) / (2 phi^2)``: the closed form, for checking the general one.

    Brans-Dicke is ``F = phi`` with ``omega(phi) = omega_BD / phi``, and the
    canonical field is ``sqrt((2 omega + 3)/2) ln phi``. The general
    expression above has to reproduce this, and the ``+3`` is the whole
    content of the check -- it is the conformal term, and dropping it gives a
    theory whose scalar decouples at the wrong value of ``omega``.
    """
    return sp.together((2 * parameter + 3) / (2 * field**2))


__all__ = [
    "FrameMap",
    "brans_dicke_canonical_factor",
    "canonical_field_factor",
    "conformal_metric",
    "conformal_ricci_scalar",
    "einstein_frame_potential",
]
