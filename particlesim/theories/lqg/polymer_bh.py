"""The Ashtekar-Olmedo-Singh polymerised black-hole interior (design doc Section 4.5).

Inside the horizon, Schwarzschild is a homogeneous Kantowski-Sachs
cosmology, and loop quantum gravity's effective dynamics apply to it as they
do to the Friedmann case in :mod:`~particlesim.theories.lqg.lqc`. Ashtekar,
Olmedo and Singh (2018, PRL 121, 241301; PRD 98, 126003) write the interior
metric as

    ds^2 = -N^2 dT^2 + (p_b^2 / (L_o^2 p_c)) dx^2 + p_c dOmega^2

and replace the connection components ``b`` and ``c`` in the Hamiltonian by
``sin(delta_b b)/delta_b`` and ``sin(delta_c c)/delta_c``. With the lapse
``N = gamma delta_b sqrt(p_c) / sin(delta_b b)`` the two pairs decouple, and
the equations of motion are solved in closed form. ``T = 0`` is the
black-hole horizon, and ``T`` runs towards the classical singularity.

    tan(delta_c c / 2) = K exp(-2T),       K = gamma L_o delta_c / (8 m)
    p_c = 4 m^2 (exp(2T) + K^2 exp(-2T))
    cos(delta_b b) = b_o tanh(b_o T / 2 + artanh(1 / b_o)),   b_o = sqrt(1 + gamma^2 delta_b^2)
    p_b = -2 gamma m L_o (sin(delta_b b)/delta_b) / (sin^2(delta_b b)/delta_b^2 + gamma^2)

``p_c`` no longer reaches zero. It has a minimum at ``exp(4T) = K^2``: the
**transition surface**, where the black-hole interior turns into a
white-hole interior. The white-hole horizon is at ``cos(delta_b b) = -1``,
a finite ``T``.

**The polymerisation parameters are fixed by the area gap, not chosen.**
AOS require the physical area of two plaquettes at the transition surface to
equal the area gap ``Delta``: ``2 pi delta_c delta_b |p_b| = Delta`` and
``4 pi delta_b^2 p_c = Delta``. For a macroscopic mass the second gives
``p_c = m gamma L_o delta_c`` there, and the two together put
``delta_b b = pi/2`` on the surface, whence

    delta_b = (sqrt(Delta) / (sqrt(2 pi) gamma^2 m))^(1/3)
    L_o delta_c = (1/2) (gamma Delta^2 / (4 pi^2 m))^(1/3)

-- which is AOS's result, re-derived here from their conditions and the
solution above. At that order the two conditions are *tangent*: their ratio
is ``(X^2 + 1)/(2X)`` with ``X = 2 K^(1/4) / (gamma delta_b)``, which reaches
one only at ``X = 1``. So the closed form is a double root, and at finite
mass the exact conditions have two solutions, one either side of it, closing
in as ``m^(-1/3)`` (:meth:`PolymerBlackHole.plaquette_polymerisation`).

**What follows from the prescription, and is the benchmark.** The
transition surface's radius is ``sqrt(m gamma L_o delta_c)``, proportional to
``m^(1/3)``, so the Kretschmann scalar there, of order ``m^2/r^6``, is
**independent of the mass**: 82188.36 in Planck units, the same to 2e-08
from ``m = 10^6`` to ``10^12``, and the curvature's maximum along the
interior. Curvature is bounded by a universal value rather than one that
grows with the hole, which is the scaling AOS report. The white hole comes
out with the black hole's mass less a relative ``eps (ln(eps/4) + 1)``,
``eps = gamma^2 delta_b^2``: 7e-02 at ``m = 100``, 5.5e-06 at ``10^9``.

Units: ``G = hbar = c = 1``, so masses, lengths and the area gap are in
Planck units. ``L_o`` is a fiducial length that drops out of everything
physical; it is set to one, and ``delta_c`` below means ``L_o delta_c``.

This is the effective interior only, not the quantum theory, and it says
nothing about perturbations, Hawking radiation or the exterior beyond what
the same metric family continues to.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp

from particlesim.theories.base import Coupling, FieldSpec, Theory

#: The Barbero-Immirzi parameter from black-hole entropy.
BARBERO_IMMIRZI = 0.2375
#: The area gap of loop quantum gravity, ``4 sqrt(3) pi gamma``, in Planck areas.
AREA_GAP = 4.0 * np.sqrt(3.0) * np.pi * BARBERO_IMMIRZI


class PolymerBlackHole(Theory):
    """AOS effective Schwarzschild interior as a Tier B plugin."""

    id = "lqg.polymer_bh"
    tier = "B"
    dimension = 4
    fields = [FieldSpec("g", "metric", rank=2)]
    couplings = [
        Coupling("area_gap", AREA_GAP, units="planck_area", bounds=(0.0, 1e6)),
        Coupling("immirzi", BARBERO_IMMIRZI, units="dimensionless", bounds=(1e-3, 10.0)),
    ]
    frame = "einstein"
    formulation = "standard"
    provenance = (
        "Ashtekar, Olmedo and Singh 2018 (PRL 121, 241301; PRD 98, 126003): "
        "holonomy-corrected Kantowski-Sachs interior with the polymerisation "
        "fixed by the area gap at the transition surface. Effective dynamics only."
    )
    validity_statement = (
        "the homogeneous interior between the black- and white-hole horizons, "
        "for masses well above the Planck mass; effective, not the quantum theory"
    )

    # --- parameters -------------------------------------------------------

    @property
    def gamma(self) -> float:
        return float(self.values["immirzi"])

    def polymerisation(self, mass: float) -> tuple[float, float]:
        """``(delta_b, L_o delta_c)`` from AOS's large-mass solution of the area conditions."""
        _positive(mass)
        gap, gamma = self.values["area_gap"], self.gamma
        delta_b = (np.sqrt(gap) / (np.sqrt(2.0 * np.pi) * gamma**2 * mass)) ** (1.0 / 3.0)
        delta_c = 0.5 * (gamma * gap**2 / (4.0 * np.pi**2 * mass)) ** (1.0 / 3.0)
        return float(delta_b), float(delta_c)

    def plaquette_polymerisation(
        self, mass: float
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Both ``(delta_b, L_o delta_c)`` pairs that solve the two area conditions exactly.

        ``4 pi delta_b^2 p_c = Delta`` at the transition surface is explicit
        in ``delta_c`` once ``delta_b`` is known, since ``p_c`` there is
        ``m gamma delta_c``. What is left is one equation in ``delta_b``, and
        it has **two** roots, one either side of the closed form.

        At leading order in ``1/m`` the ratio of the two conditions is
        ``(X^2 + 1)/(2X)`` with ``X = 2 K^(1/4) / (gamma delta_b)``, which
        touches one only at ``X = 1``: a double root, and AOS's
        :meth:`polymerisation` is that point. At finite mass the next order
        pushes the minimum of the mismatch just below zero and the double
        root splits. The minimum's depth is ``O(m^(-2/3))``, so the roots
        close in on the closed form as ``m^(-1/3)``.
        """
        from scipy.optimize import brentq, minimize_scalar

        _positive(mass)
        gap, gamma = self.values["area_gap"], self.gamma
        if gap == 0.0:
            return (0.0, 0.0), (0.0, 0.0)

        def delta_c_of(delta_b: float) -> float:
            return gap / (4.0 * np.pi * gamma * mass * delta_b**2)

        def mismatch(log_delta_b: float) -> float:
            delta_b = float(np.exp(log_delta_b))
            delta_c = delta_c_of(delta_b)
            time = self._transition_time(mass, delta_c)
            if time <= self._white_hole_time(delta_b):
                return float("inf")
            state = self.interior(mass, time, (delta_b, delta_c))
            return float(np.log(2.0 * np.pi * delta_c * delta_b * abs(state["p_b"]) / gap))

        centre = float(np.log(self.polymerisation(mass)[0]))
        lowest = minimize_scalar(
            mismatch,
            bounds=(centre - 0.5, centre + 0.5),
            method="bounded",
            options={"xatol": 1e-12},
        )
        if not lowest.fun < 0.0:
            raise ValueError(f"the area conditions have no solution at m = {mass}")
        roots = []
        for side in (-1.0, 1.0):
            far = lowest.x + side * 0.5
            roots.append(brentq(mismatch, *sorted((lowest.x, far)), xtol=1e-15, rtol=1e-15))
        pairs = tuple((float(np.exp(x)), delta_c_of(float(np.exp(x)))) for x in roots)
        return pairs[0], pairs[1]

    # --- the interior -----------------------------------------------------

    def interior(
        self, mass: float, time, polymerisation: tuple[float, float] | None = None
    ) -> dict[str, np.ndarray]:
        """The closed-form solution at ``T = time``, with ``L_o = 1``.

        Returns ``b, p_b, c, p_c``; ``sin_b = sin(delta_b b)/delta_b``,
        ``cos_b = cos(delta_b b)`` and the same for ``c``, which stay regular
        as the polymerisation goes to zero; and the metric components
        ``lapse`` (``N``), ``g_xx`` and ``g_thth``.

        ``sin_b`` is computed from ``b_o`` and ``tanh`` without forming
        ``1 - cos^2``. Near the transition surface of a large hole
        ``b_o - 1`` and ``1 + tanh`` are both of order ``m^(-2/3)``, and
        subtracting numbers near one would lose most of the digits.
        """
        _positive(mass)
        delta_b, delta_c = self.polymerisation(mass) if polymerisation is None else polymerisation
        gamma = self.gamma
        time = np.asarray(time, dtype=float)

        k = gamma * delta_c / (8.0 * mass)
        p_c = 4.0 * mass**2 * (np.exp(2.0 * time) + k**2 * np.exp(-2.0 * time))
        ratio = k * np.exp(-2.0 * time)  # tan(delta_c c / 2)
        cos_c = (1.0 - ratio**2) / (1.0 + ratio**2)
        sin_c = gamma * np.exp(-2.0 * time) / (4.0 * mass * (1.0 + ratio**2))
        c = (
            2.0 * np.arctan(ratio) / delta_c
            if delta_c > 0.0
            else gamma * np.exp(-2.0 * time) / (4.0 * mass)
        )

        b_o = np.sqrt(1.0 + gamma**2 * delta_b**2)
        excess = gamma**2 * delta_b**2 / (b_o + 1.0)  # b_o - 1
        grow = np.exp(b_o * time)
        tanh = (grow - 1.0) / (grow + 1.0)
        rise = 2.0 * grow / (1.0 + grow)  # 1 + tanh
        cos_b = b_o * (b_o * rise - excess) / (excess + rise)
        # sin^2(delta_b b)/delta_b^2 = gamma^2 t (-2 b_o - t (1 + b_o^2)) / (b_o + t)^2 with
        # t = tanh, regular at delta_b = 0, where it is gamma^2 (exp(-T) - 1). The middle
        # factor is written as (b_o - 1)^2 - (1 + t)(1 + b_o^2): near the white-hole
        # horizon both terms are tiny, and the textbook form subtracts two numbers near 2.
        sin_b_squared = gamma**2 * tanh * (excess**2 - rise * (1.0 + b_o**2)) / (excess + rise) ** 2
        sin_b = np.sqrt(np.maximum(sin_b_squared, 0.0))
        b = (
            np.arccos(np.clip(cos_b, -1.0, 1.0)) / delta_b
            if delta_b > 0.0
            else gamma * np.sqrt(np.maximum(np.exp(-time) - 1.0, 0.0))
        )
        p_b = -2.0 * gamma * mass * sin_b / (sin_b_squared + gamma**2)
        with np.errstate(divide="ignore"):
            lapse = gamma * np.sqrt(p_c) / sin_b
        return {
            "b": b,
            "p_b": p_b,
            "c": c,
            "p_c": p_c,
            "sin_b": sin_b,
            "cos_b": cos_b,
            "sin_c": sin_c,
            "cos_c": cos_c,
            "lapse": lapse,
            "g_xx": p_b**2 / p_c,
            "g_thth": p_c,
        }

    def hamilton_equations(self, mass: float, polymerisation: tuple[float, float] | None = None):
        """``d(b, p_b, c, p_c)/dT`` from the effective Hamiltonian, for an ODE solver.

        The independent route to the same solution: nothing here uses the
        closed form, only ``{c, p_c} = 2 gamma``, ``{b, p_b} = gamma`` and the
        Hamiltonian with AOS's lapse.
        """
        _positive(mass)
        delta_b, delta_c = self.polymerisation(mass) if polymerisation is None else polymerisation
        if delta_b <= 0.0 or delta_c <= 0.0:
            raise ValueError("the polymerised equations need both delta_b and delta_c positive")
        gamma = self.gamma

        def rhs(_time, state):
            b, p_b, c, p_c = state
            sine = np.sin(delta_b * b)
            return [
                -0.5 * (sine / delta_b + gamma**2 * delta_b / sine),
                0.5 * p_b * np.cos(delta_b * b) * (1.0 - gamma**2 * delta_b**2 / sine**2),
                -2.0 * np.sin(delta_c * c) / delta_c,
                2.0 * p_c * np.cos(delta_c * c),
            ]

        return rhs

    def masses(self, state, polymerisation: tuple[float, float]) -> tuple[float, float]:
        """AOS's ``O_b`` and ``O_c``, each equal to the mass on a solution."""
        b, p_b, c, p_c = state
        delta_b, delta_c = polymerisation
        gamma = self.gamma
        sin_b = np.sin(delta_b * b) / delta_b
        o_b = -(sin_b + gamma**2 / sin_b) * p_b / (2.0 * gamma)
        o_c = np.sin(delta_c * c) / delta_c * p_c / gamma
        return o_b, o_c

    # --- the transition surface and the white hole ------------------------

    def _transition_time(self, mass: float, delta_c: float) -> float:
        k = self.gamma * delta_c / (8.0 * mass)
        return float(0.5 * np.log(k)) if k > 0.0 else float("-inf")

    def _white_hole_time(self, delta_b: float) -> float:
        if delta_b == 0.0:
            return float("-inf")
        b_o = np.sqrt(1.0 + self.gamma**2 * delta_b**2)
        # artanh(1/b_o) = log((b_o + 1)/(b_o - 1)) / 2, with b_o - 1 formed stably.
        b_o_less_one = self.gamma**2 * delta_b**2 / (b_o + 1.0)
        return float(-2.0 * np.log((b_o + 1.0) / b_o_less_one) / b_o)

    def transition_time(self, mass: float) -> float:
        """``T`` of the transition surface, where ``p_c`` is smallest."""
        _positive(mass)
        return self._transition_time(mass, self.polymerisation(mass)[1])

    def transition_radius(self, mass: float) -> float:
        """Areal radius of the transition surface, ``sqrt(m gamma L_o delta_c)``."""
        _positive(mass)
        return float(np.sqrt(mass * self.gamma * self.polymerisation(mass)[1]))

    def white_hole_time(self, mass: float) -> float:
        """``T`` of the white-hole horizon, ``cos(delta_b b) = -1``."""
        _positive(mass)
        return self._white_hole_time(self.polymerisation(mass)[0])

    def white_hole_mass(self, mass: float) -> float:
        """Half the areal radius of the white-hole horizon: the mass it presents outside."""
        _positive(mass)
        delta_b, delta_c = self.polymerisation(mass)
        if delta_b == 0.0:
            return float("nan")
        time = self._white_hole_time(delta_b)
        k = self.gamma * delta_c / (8.0 * mass)
        return float(mass * np.sqrt(np.exp(2.0 * time) + k**2 * np.exp(-2.0 * time)))

    # --- curvature ---------------------------------------------------------

    def kretschmann(
        self, mass: float, time, polymerisation: tuple[float, float] | None = None
    ) -> np.ndarray:
        """``R_abcd R^abcd`` along the interior, from the phase-space variables.

        For ``-N^2 dT^2 + A^2 dx^2 + B^2 dOmega^2`` it is a sum of squares in
        proper-time derivatives,

            4 [2 (B''/B)^2 + ((1 + B'^2)/B^2)^2 + (A''/A)^2 + 2 (A' B' / (A B))^2]

        and Hamilton's equations give every derivative in closed form:
        ``B' = sin_b cos_c / gamma`` with ``B = sqrt(p_c)``, and
        ``A'/A = (sin_b / (gamma B)) [cos_b (1 - gamma^2/sin_b^2)/2 - cos_c]``.
        Nothing is differenced numerically. At zero polymerisation it is
        ``48 m^2 / p_c^3`` to round-off.

        ``A'/A`` diverges at either horizon, where ``A`` vanishes, and
        ``A''/A`` is the difference of two such terms; within about 1e-3 of a
        horizon in ``T`` the result loses digits. The interior between them
        is where this is meant to be read.
        """
        delta_b, delta_c = self.polymerisation(mass) if polymerisation is None else polymerisation
        gamma = self.gamma
        state = self.interior(mass, time, (delta_b, delta_c))
        sin_b, cos_b = state["sin_b"], state["cos_b"]
        sin_c_scaled, cos_c = state["sin_c"], state["cos_c"]
        sin_c = sin_c_scaled * delta_c  # sin(delta_c c), zero at zero polymerisation
        radius = np.sqrt(state["p_c"])
        per_time = sin_b / (gamma * radius)  # d/dtau = per_time * d/dT

        # d/dT of sin_b and of cos_c, from Hamilton's equations.
        d_sin_b = -0.5 * cos_b * (sin_b + gamma**2 / sin_b)
        d_cos_c = 2.0 * sin_c**2

        b_dot = sin_b * cos_c / gamma
        b_ddot = per_time * (d_sin_b * cos_c + sin_b * d_cos_c) / gamma

        q = 0.5 * cos_b * (1.0 - gamma**2 / sin_b**2) - cos_c
        # d/dT of q; delta_b^2 (sin_b^2 + gamma^2) is (1 - cos_b^2) + gamma^2 delta_b^2.
        d_q = (
            0.25 * (1.0 - cos_b**2 + gamma**2 * delta_b**2) * (1.0 - gamma**2 / sin_b**2)
            + gamma**2 * cos_b * d_sin_b / sin_b**3
            - d_cos_c
        )
        a_rate = per_time * q  # A'/A
        # (A'/A)' = per_time d/dT (per_time q),
        # with d(per_time)/dT = per_time (d_sin_b / sin_b - cos_c).
        a_rate_dot = per_time * (per_time * (d_sin_b / sin_b - cos_c) * q + per_time * d_q)
        a_ddot = a_rate_dot + a_rate**2  # A''/A

        return 4.0 * (
            2.0 * (b_ddot / radius) ** 2
            + ((1.0 + b_dot**2) / radius**2) ** 2
            + a_ddot**2
            + 2.0 * (a_rate * b_dot / radius) ** 2
        )

    def maximum_kretschmann(self, mass: float) -> tuple[float, float]:
        """``(T, K)`` at the largest curvature between the two horizons."""
        from scipy.optimize import minimize_scalar

        _positive(mass)
        start, end = self.white_hole_time(mass), 0.0
        if not np.isfinite(start):
            raise ValueError("no transition surface: the curvature is unbounded")
        grid = np.linspace(start, end, 4001)[40:-40]
        values = self.kretschmann(mass, grid)
        index = int(np.nanargmax(np.where(np.isfinite(values), values, np.nan)))
        result = minimize_scalar(
            lambda t: -float(self.kretschmann(mass, t)),
            bounds=(grid[max(index - 1, 0)], grid[min(index + 1, grid.size - 1)]),
            method="bounded",
            options={"xatol": 1e-12},
        )
        return float(result.x), float(-result.fun)

    # --- the Tier B contract ---------------------------------------------

    def metric_family(self, params: dict[str, float]) -> sp.Matrix:
        """The black-hole side in areal radius, ``(x, r, theta, phi)``.

        Between the horizon and the transition surface ``r = sqrt(p_c)`` is a
        good time coordinate, and the closed form can be written in it:
        ``exp(2T)`` is the larger root of ``4 m^2 (u + K^2/u) = r^2``. With
        ``x`` in the role of Schwarzschild's ``t`` this is directly
        comparable with :meth:`particlesim.theories.gr.GR.metric_family`,
        which it becomes, exactly, when the area gap is zero.
        """
        mass = sp.nsimplify(params["mass"], rational=True)
        gamma = sp.nsimplify(self.gamma, rational=True)
        # Exact zeros at the GR limit, so the harness's simplification is exact;
        # floats otherwise, since rationalising them only makes 30-digit integers.
        delta_b, delta_c = (
            sp.Float(value) if value else sp.Integer(0)
            for value in self.polymerisation(float(mass))
        )
        r, th = sp.symbols("r theta", positive=True)

        k = gamma * delta_c / (8 * mass)
        exp_2t = (r**2 + sp.sqrt(r**4 - 64 * mass**4 * k**2)) / (8 * mass**2)
        b_o = sp.sqrt(1 + gamma**2 * delta_b**2)
        grow = exp_2t ** (b_o / 2)
        tanh = (grow - 1) / (grow + 1)
        sin_b_squared = gamma**2 * tanh * (-2 * b_o - tanh * (1 + b_o**2)) / (b_o + tanh) ** 2
        g_xx = 4 * gamma**2 * mass**2 * sin_b_squared / ((sin_b_squared + gamma**2) ** 2 * r**2)
        dr_dt = 4 * mass**2 * (exp_2t - k**2 / exp_2t) / r
        g_rr = -(gamma**2) * r**2 / (sin_b_squared * dr_dt**2)
        return sp.diag(g_xx, g_rr, r**2, r**2 * sp.sin(th) ** 2)

    def gr_limit(self) -> dict[str, float]:
        return {"area_gap": 0.0}

    def observable_predictions(self) -> dict[str, Any]:
        quantum = self.values["area_gap"] > 0.0
        return {
            "singularity_resolved": quantum,
            "transition_surface": quantum,
            "white_hole_interior": quantum,
            "curvature_bound_independent_of_mass": quantum,
        }


def _positive(mass: float) -> None:
    if not mass > 0.0:
        raise ValueError(f"the mass must be positive, got {mass}")


__all__ = ["AREA_GAP", "BARBERO_IMMIRZI", "PolymerBlackHole"]
