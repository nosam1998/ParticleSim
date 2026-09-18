"""Einstein-Maxwell-dilaton-axion, and the inner surface it does not have.

Issue #71. The four-dimensional string effective action couples the dilaton
to the Maxwell field exponentially,

    R - 2 (grad phi)^2 - e^(-2 a phi) F^2

with ``a = 1`` for the heterotic string, ``a = sqrt(3)`` for the
Kaluza-Klein reduction of five-dimensional gravity, and ``a = 0`` for
ordinary Einstein-Maxwell. Its static charged solution is a one-parameter
deformation of Reissner-Nordstrom, and the deformation is not gentle.

**Where the parameter actually bites.** Write the solution as

    ds^2 = -f dt^2 + f^(-1) dr^2 + R(r)^2 dOmega^2
    f = (1 - r_+/r)(1 - r_-/r)^b,    R^2 = r^2 (1 - r_-/r)^(1-b)
    b = (1 - a^2)/(1 + a^2)

At ``a = 0`` the exponent ``b`` is 1, ``R = r``, and ``r = r_-`` is the
Cauchy horizon of Reissner-Nordstrom: a regular null surface with finite
curvature and a surface gravity. For **any** ``a > 0`` the exponent drops
below 1, the areal radius ``R`` collapses to zero at ``r_-``, and the
surface is a curvature singularity instead.

**And the transition is discontinuous.** The Kretschmann scalar diverges
there as

    K ~ (r - r_-)^(-(2 + 4 a^2/(1 + a^2)))

whose exponent tends to **2** as ``a -> 0`` while the value *at* ``a = 0`` is
finite. So the Cauchy horizon's regularity is not a continuous property of
the dilaton coupling: an arbitrarily small ``a`` destroys it outright. This
is why the comparison the issue asks for is not "mass inflation at a
different rate" but "no arena for mass inflation at all", and it is measured
in :mod:`particlesim.scenarios.singularity.interior` rather than asserted.

**The extremality bound moves too.** ``r_+ = M + sqrt(M^2 - (1-a^2)Q^2)``, so
the bound is ``M^2 >= (1 - a^2) Q^2`` and dissolves entirely at ``a = 1``: a
dilaton black hole can carry charge that would over-extremalise a
Reissner-Nordstrom hole of the same mass.

**What is not here.** The axion. Static, spherically symmetric EMDA has a
constant axion -- the Kerr-Sen family is where it becomes dynamical, and that
is rotating. The module says so in ``validity_statement`` rather than
implying the axion sector has been exercised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import sympy as sp

from particlesim.theories.base import Coupling, Theory

NON_NEGATIVE = (0.0, float("inf"))


@dataclass(frozen=True)
class DilatonBlackHole:
    """The static charged solution at dilaton coupling ``a``.

    ``dilaton_coupling = 0`` is Reissner-Nordstrom exactly, ``1`` is the
    Gibbons-Maeda-Garfinkle-Horowitz-Strominger solution of the heterotic
    string, and ``sqrt(3)`` is the Kaluza-Klein black hole.
    """

    mass: float = 1.0
    charge: float = 0.9
    dilaton_coupling: float = 0.0

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            raise ValueError(f"the mass must be positive, got {self.mass}")
        if self.charge < 0.0:
            raise ValueError(f"the charge must not be negative, got {self.charge}")
        if self.dilaton_coupling < 0.0:
            raise ValueError(
                f"the dilaton coupling must not be negative, got {self.dilaton_coupling}; "
                "its sign is a convention absorbed into the sign of the dilaton"
            )
        bound = (1.0 - self.dilaton_coupling**2) * self.charge**2
        if self.mass**2 < bound:
            raise ValueError(
                f"M^2 = {self.mass**2} is below the extremality bound "
                f"(1 - a^2) Q^2 = {bound}, so there is no horizon; note the bound itself "
                "depends on the dilaton coupling and vanishes at a = 1"
            )
        if self.charge > 0.0 and self.inner_radius >= self.outer_radius:
            raise ValueError(
                f"the inner surface at {self.inner_radius} is not inside the horizon at "
                f"{self.outer_radius}; this solution is not a black hole"
            )

    @property
    def exponent(self) -> float:
        """``b = (1 - a^2)/(1 + a^2)``, the exponent everything turns on."""
        a2 = self.dilaton_coupling**2
        return (1.0 - a2) / (1.0 + a2)

    @property
    def outer_radius(self) -> float:
        """``r_+ = M + sqrt(M^2 - (1-a^2) Q^2)``, the event horizon."""
        bound = (1.0 - self.dilaton_coupling**2) * self.charge**2
        return self.mass + math.sqrt(self.mass**2 - bound)

    @property
    def inner_radius(self) -> float:
        """``r_- = (1+a^2) Q^2 / r_+``: a Cauchy horizon only at ``a = 0``."""
        return (1.0 + self.dilaton_coupling**2) * self.charge**2 / self.outer_radius

    @property
    def has_cauchy_horizon(self) -> bool:
        """True only for an ordinary charged hole with a genuine inner horizon.

        The areal radius at ``r_-`` is ``0`` for any ``a > 0``, so the surface
        is singular rather than a horizon. A charge of zero leaves
        Schwarzschild, which has no inner surface at all.
        """
        return self.dilaton_coupling == 0.0 and self.charge > 0.0

    def metric_function(self, radius: float) -> float:
        """``f(r)``, written factorised so it stays exact next to ``r_-``.

        The expanded form ``1 - 2M/r + Q^2/r^2`` loses every significant
        figure as ``r -> r_-``, which is precisely the region the interior
        diagnostics live in, so it is never used.
        """
        return self.metric_function_from_gap(radius - self.inner_radius)

    def metric_function_from_gap(self, gap: float) -> float:
        """``f`` as a function of ``r - r_-``, which is the only exact way in.

        Near the inner surface the gap is the small quantity, and passing a
        radius instead throws it away: at ``r - r_- = 1e-70`` the ratio
        ``r_-/r`` rounds to exactly one and ``f`` comes back as zero. The
        interior diagnostics reach that regime within a hundred units of
        advanced time, so they carry the gap throughout and call this.
        """
        radius = self.inner_radius + gap
        return (1.0 - self.outer_radius / radius) * (gap / radius) ** self.exponent

    def areal_radius(self, radius: float) -> float:
        """``R(r)``: the radius of the sphere, which is not ``r`` unless ``a = 0``."""
        base = max(1.0 - self.inner_radius / radius, 0.0)
        return radius * base ** ((1.0 - self.exponent) / 2.0)

    def outer_surface_gravity(self) -> float:
        """``kappa_+ = (1 - r_-/r_+)^b / (2 r_+)``, reducing to the usual form at ``a = 0``."""
        outer, inner = self.outer_radius, self.inner_radius
        return (1.0 - inner / outer) ** self.exponent / (2.0 * outer)

    def inner_surface_gravity(self) -> float:
        """``kappa_- = (r_+ - r_-)/(2 r_-^2)``, and only when there is a horizon there.

        For ``a > 0`` the derivative of ``f`` at ``r_-`` is infinite because
        the exponent is below one, which is the analytic shadow of the
        surface not being a horizon. Returning a number anyway would be the
        error worth guarding against, so this raises instead.
        """
        if not self.has_cauchy_horizon:
            raise ValueError(
                f"at dilaton coupling {self.dilaton_coupling} and charge {self.charge} there "
                "is no inner horizon to assign a surface gravity to; the surface at r_- is a "
                "curvature singularity for any a > 0 and absent for Q = 0"
            )
        outer, inner = self.outer_radius, self.inner_radius
        return (outer - inner) / (2.0 * inner**2)

    def kretschmann_exponent(self) -> float:
        """The power of ``(r - r_-)^-1`` in ``K`` at the inner surface.

        ``2 + 4 a^2/(1 + a^2)`` for ``a > 0``, and ``0`` at ``a = 0`` because
        the Cauchy horizon is regular. The two do not join: the formula tends
        to ``2`` as ``a -> 0`` while the value there is ``0``. That
        discontinuity is the whole content of the comparison with general
        relativity, and it is measured against a symbolic Kretschmann scalar
        rather than taken on trust.
        """
        if self.has_cauchy_horizon or self.charge == 0.0:
            return 0.0
        return 2.0 + 2.0 * (1.0 - self.exponent)

    def symbolic_metric(self, radius: sp.Symbol, polar: sp.Symbol) -> sp.Matrix:
        """The metric with ``r`` and ``theta`` symbolic, for curvature work."""
        outer = sp.nsimplify(self.outer_radius, rational=True)
        inner = sp.nsimplify(self.inner_radius, rational=True)
        b = sp.nsimplify(self.exponent, rational=True)
        f = (1 - outer / radius) * (1 - inner / radius) ** b
        areal_squared = radius**2 * (1 - inner / radius) ** (1 - b)
        return sp.diag(-f, 1 / f, areal_squared, areal_squared * sp.sin(polar) ** 2)


class EMDA(Theory):
    """Einstein-Maxwell-dilaton-axion as a Tier A plugin."""

    id = "string.eft4d.emda"
    frame = "einstein"
    couplings = [
        Coupling("dilaton_coupling", 0.0, units="dimensionless", bounds=NON_NEGATIVE),
        Coupling("charge", 0.0, units="mass", bounds=NON_NEGATIVE),
    ]
    provenance = (
        "Einstein-Maxwell-dilaton-axion, the four-dimensional string effective action "
        "with an exponentially coupled Maxwell field; a = 1 is heterotic, a = sqrt(3) "
        "is the Kaluza-Klein reduction of five-dimensional gravity"
    )
    validity_statement = (
        "static and spherically symmetric only, where the axion is constant; the "
        "dynamical axion of the Kerr-Sen family needs rotation and is not covered"
    )

    def gr_limit(self) -> dict[str, float]:
        """No charge, no dilaton coupling: Schwarzschild.

        Setting the charge to zero is enough on its own -- ``r_- = 0`` then
        for every ``a`` -- and that is itself a check the suite makes. Both
        are declared here because the theory's content is the pair.
        """
        return {"dilaton_coupling": 0.0, "charge": 0.0}

    def effective_stress_energy(self, einstein: sp.Matrix, metric: sp.Matrix) -> sp.Matrix:
        """The Einstein-frame action has a canonical Einstein-Hilbert term.

        So the dilaton and the Maxwell field are matter, not a modification
        of gravity, and the split is the general-relativistic one. In the
        string frame that statement would be false, which is why the declared
        frame is what makes it checkable.
        """
        return sp.Matrix(einstein) / (8 * sp.pi)

    def metric_family(self, params: dict[str, float]) -> sp.Matrix:
        """The static charged solution at the plugin's couplings."""
        radius, polar = sp.symbols("r theta", positive=True)
        hole = DilatonBlackHole(
            mass=float(params["mass"]),
            charge=self.values["charge"],
            dilaton_coupling=self.values["dilaton_coupling"],
        )
        return hole.symbolic_metric(radius, polar)

    def observable_predictions(self) -> dict:
        hole = DilatonBlackHole(
            mass=1.0,
            charge=self.values["charge"],
            dilaton_coupling=self.values["dilaton_coupling"],
        )
        return {
            "dilaton_coupling": self.values["dilaton_coupling"],
            "charge": self.values["charge"],
            "horizon_radius": hole.outer_radius,
            "inner_radius": hole.inner_radius,
            "has_cauchy_horizon": hole.has_cauchy_horizon,
            "mass_inflation": hole.has_cauchy_horizon,
        }


__all__ = ["EMDA", "DilatonBlackHole"]
