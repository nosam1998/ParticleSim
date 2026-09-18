"""Conformal frames: the transformation against the curvature machinery.

Issue #74's acceptance -- mismatched frames raise at construction -- was
already met by ``TheoryStack``. What was missing is the two tasks behind it:
validating the conformal map itself, and catching the *matter* coupling
mismatch, which is the one a frame comparison alone does not see.
"""

from __future__ import annotations

import pytest
import sympy as sp

from particlesim.symbolic.curvature import MetricGeometry
from particlesim.theories.base import Theory, TheoryStack
from particlesim.theories.frames import (
    FrameMap,
    brans_dicke_canonical_factor,
    canonical_field_factor,
    conformal_metric,
    conformal_ricci_scalar,
    einstein_frame_potential,
)
from particlesim.theories.registry import get_theory

COORDS = list(sp.symbols("t x y z", real=True))
FLAT = sp.diag(-1, 1, 1, 1)


# --- the transformation, against an independent computation ---------------


def test_the_conformal_ricci_scalar_matches_the_curvature_machinery():
    """Closed form against ``MetricGeometry`` computing ``R~`` from ``g~`` directly.

    Two different calculations: one differentiates the transformed metric
    through Christoffels and Riemann, the other applies

        R~ = Omega^-2 [R - 2(D-1) box ln Omega - (D-1)(D-2)(grad ln Omega)^2]

    to the original. The difference is *exactly* zero, which also pins the
    sign convention to this repository's rather than to a textbook's.
    """
    sigma = sp.Function("sigma")(COORDS[0], COORDS[1])
    factor = sp.exp(sigma)

    direct = MetricGeometry(conformal_metric(FLAT, factor), COORDS, simplify=True).ricci_scalar
    closed = conformal_ricci_scalar(FLAT, COORDS, factor, ricci_scalar=sp.S.Zero)
    assert sp.simplify(direct - closed) == 0


def test_a_constant_factor_just_rescales_the_curvature():
    """``R~ = R/Omega^2`` when the derivative terms drop out.

    Convention-free, so it holds whatever the sign of the Riemann tensor,
    and it is the one case where the answer can be read off by eye.
    """
    factor = sp.Symbol("c", positive=True)
    metric = sp.diag(-1, 1, 1, sp.Symbol("a", positive=True) ** 2)
    scalar = MetricGeometry(metric, COORDS, simplify=True).ricci_scalar

    closed = conformal_ricci_scalar(metric, COORDS, factor, ricci_scalar=scalar)
    assert sp.simplify(closed - scalar / factor**2) == 0


def test_conformal_ricci_refuses_mismatched_coordinates():
    with pytest.raises(ValueError, match="expected 4 coordinates"):
        conformal_ricci_scalar(FLAT, COORDS[:2], sp.Symbol("w", positive=True))


# --- the map --------------------------------------------------------------


def test_a_round_trip_returns_the_original_metric():
    """Omega then 1/Omega is the identity, exactly."""
    factor = sp.Symbol("Omega", positive=True)
    forward = FrameMap(source="jordan", target="einstein", factor=factor)
    back = forward.inverse()

    assert back.source == "einstein" and back.target == "jordan"
    assert sp.simplify(back.apply(forward.apply(FLAT)) - FLAT) == sp.zeros(4, 4)


def test_a_map_must_change_frame():
    with pytest.raises(ValueError, match="must change frame"):
        FrameMap(source="jordan", target="jordan", factor=sp.Symbol("w", positive=True))


def test_a_negative_conformal_factor_is_refused():
    """``Omega^2 < 0`` flips the signature rather than rescaling it."""
    with pytest.raises(ValueError, match="must be positive"):
        FrameMap(source="jordan", target="einstein", factor=sp.Integer(-2))


def test_an_undecidable_factor_is_refused_with_the_reason():
    """A bare symbol could be negative, and sympy cannot rule it out.

    Refused rather than assumed, because the failure it would cause -- a
    signature flip -- is not one that shows up as a wrong number.
    """
    with pytest.raises(ValueError, match="declare the symbol with positive=True"):
        FrameMap(source="jordan", target="einstein", factor=sp.Symbol("maybe"))


# --- what the map does to the action --------------------------------------


def test_the_einstein_frame_potential_is_v_over_f_squared():
    field = sp.Symbol("phi", positive=True)
    potential = sp.Symbol("Lambda", positive=True) * field**4
    assert (
        sp.simplify(
            einstein_frame_potential(potential, field)
            - sp.Symbol("Lambda", positive=True) * field**2
        )
        == 0
    )


def test_the_canonical_factor_reproduces_brans_dicke():
    """The general expression against the closed form, and the ``+3`` is the point.

    Brans-Dicke is ``F = phi`` with ``omega(phi) = omega_BD/phi``, whose
    canonical factor is ``(2 omega + 3)/(2 phi^2)``. The ``3`` comes from the
    conformal transformation rather than from the Jordan-frame kinetic term,
    so dropping it gives a theory whose scalar decouples at the wrong
    ``omega`` -- and at ``omega = -3/2`` the field is not canonical at all,
    which is exactly where this vanishes.
    """
    field = sp.Symbol("phi", positive=True)
    parameter = sp.Symbol("omega_BD", real=True)

    general = canonical_field_factor(coupling=field, kinetic=parameter / field, field=field)
    assert sp.simplify(general - brans_dicke_canonical_factor(parameter, field)) == 0
    assert sp.simplify(general.subs(parameter, sp.Rational(-3, 2))) == 0


def test_f_of_r_gets_a_scalar_from_the_conformal_term_alone():
    """No Jordan-frame kinetic term, and still a propagating scalar.

    ``f(R)`` in scalar-tensor form has ``omega = 0``, so the whole canonical
    factor is the ``(3/2)(F'/F)^2`` piece the transformation itself
    contributes. That the result is non-zero is why ``f(R)`` has an extra
    degree of freedom at all.
    """
    field = sp.Symbol("phi", positive=True)
    factor = canonical_field_factor(coupling=field, kinetic=sp.S.Zero, field=field)
    assert sp.simplify(factor - sp.Rational(3, 2) / field**2) == 0
    assert factor != 0


# --- the consistency error the frame comparison alone misses --------------


class _EinsteinActionJordanMatter(Theory):
    """Canonical gravity, matter still coupled to the Jordan metric.

    Not a contrivance: this is what a scalar-tensor theory looks like after
    the conformal transformation, and the reason fifth-force experiments
    constrain it.
    """

    id = "test.einstein_action_jordan_matter"
    frame = "einstein"
    matter_frame = "jordan"


class _PlainEinstein(Theory):
    id = "test.plain_einstein"
    frame = "einstein"


def test_matter_frame_defaults_to_the_action_frame():
    """So every existing plugin means what it already meant."""
    assert _PlainEinstein().effective_matter_frame == "einstein"
    assert get_theory("gr").effective_matter_frame == "einstein"


def test_a_matter_frame_mismatch_raises_though_the_frames_agree():
    """Both plugins say "einstein", and the stack is still wrong.

    This is the case a frame comparison alone cannot see: the constitutive
    relation is written against one metric and would be evaluated on the
    other, with both theories individually valid, so nothing else would
    notice.
    """
    gravity = _EinsteinActionJordanMatter()
    sector = _PlainEinstein()
    assert gravity.frame == sector.frame  # the old check passes

    with pytest.raises(ValueError, match="couples matter in the"):
        TheoryStack(gravity=gravity, em=sector)


def test_agreeing_matter_frames_compose():
    class _Matching(Theory):
        id = "test.matching"
        frame = "einstein"
        matter_frame = "jordan"

    stack = TheoryStack(gravity=_EinsteinActionJordanMatter(), em=_Matching())
    assert stack.em is not None


def test_the_original_frame_mismatch_still_raises():
    """The acceptance of issue #74, which was already met; kept under test."""

    class _Jordan(Theory):
        id = "test.jordan_sector"
        frame = "jordan"

    with pytest.raises(ValueError, match="frame"):
        TheoryStack(gravity=get_theory("gr"), em=_Jordan())
