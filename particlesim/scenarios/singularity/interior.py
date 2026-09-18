"""The charged interior: mass inflation, and the geometry that has to host it.

Issue #71. A charged black hole's Cauchy horizon is where general relativity
stops predicting, and it is unstable: an infalling perturbation, however
weak, is blueshifted without bound on the way there, and the mass function
measured just inside diverges. The issue asks for that to be reported and
compared with Einstein-Maxwell-dilaton-axion. The comparison turns out not
to be between two rates.

**What is measured, in general relativity.** An outgoing null ray inside the
hole obeys ``dr/dv = f(r)/2`` and is attracted to ``r_-``. Integrating that
in ``ln(r - r_-)`` -- never in ``r``, where the approach destroys every
significant figure -- gives an approach rate that agrees with the closed-form
inner surface gravity ``kappa_- = (r_+ - r_-)/(2 r_-^2)`` to machine
precision. Feeding the resulting crossing radius through the Dray-'t
Hooft-Redmount relation for two crossing null shells,

    f_A f_D = f_B f_C

makes the mass function behind them grow like ``v^(-p) e^(kappa_- v)``. So
its logarithmic derivative is ``kappa_- - p/v``, and both constants come out
of one integration at once: the *rate* is the inner surface gravity and the
*correction* is the Price-law exponent of the decaying tail that drove it.
That the perturbation decays while the mass it produces diverges is the
whole phenomenon, and the numbers here are measured rather than restated.

The model is the cross-flow one and is stated rather than implied: a
Reissner-Nordstrom background of mass ``M``, an ingoing perturbation
``A v^(-p)``, an outgoing shell of mass ``Delta m``, and the crossing radius
taken from the integrated ray. The full coupled Einstein-Maxwell-scalar
evolution is a much larger computation and is not what is claimed. Within the
cross-flow model the rate is pure geometry: both amplitudes and both signs can
move without shifting it, which the suite checks.

**The crossing relation earns its keep at large radius.** Away from any
horizon ``f_A -> 1`` and the relation degenerates to ``m_D = m_B + m_C -
m_A``: the masses simply add and nothing inflates. Mass inflation is
entirely the ``1/f_A`` factor switching on as the crossing slides down to
the inner horizon, and :func:`crossing_mass` is checked against both ends.

**What the comparison actually is.** Under EMDA there is no Cauchy horizon to
inflate at. For any dilaton coupling ``a > 0`` the areal radius collapses at
``r_-``, the Kretschmann scalar diverges as ``(r - r_-)^(-(2 + 4a^2/(1+a^2)))``
and the outgoing ray arrives at *finite* advanced time instead of
asymptoting. The exponent that governs the approach is ``b = (1-a^2)/(1+a^2)``,
and ``int dr (r - r_-)^(-b)`` converges for every ``b < 1``: the borderline is
exactly ``a = 0``. So the honest report is not a modified rate but an absent
mechanism, and :func:`crossing_mass` refuses the EMDA case rather than
returning a number for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import quad, solve_ivp

from particlesim.theories.emda import DilatonBlackHole

#: Where the outgoing ray starts, as a fraction of the way from ``r_-`` to ``r_+``.
DEFAULT_START = 0.5


@dataclass(frozen=True)
class PriceTail:
    """The decaying ingoing perturbation, ``delta m(v) = A v^(-p)``.

    Price's law fixes ``p = 4 l + 4`` for a multipole ``l``, so ``p = 12`` is
    the quadrupole. The amplitude is irrelevant to the rate and is carried
    only so the reported mass function has a scale.
    """

    amplitude: float = 0.05
    exponent: float = 12.0

    def __post_init__(self) -> None:
        if self.exponent <= 0.0:
            raise ValueError(
                f"the tail exponent must be positive, got {self.exponent}; a tail that "
                "does not decay is not the perturbation whose blueshift is at issue"
            )

    def mass_excess(self, advanced_time: float) -> float:
        return self.amplitude * advanced_time**-self.exponent


def outgoing_ray(hole: DilatonBlackHole, until: float = 140.0, start: float = DEFAULT_START):
    """Integrate ``dr/dv = f/2`` in ``ln(r - r_-)``, returning the dense solution.

    The logarithmic variable is not a convenience. The approach is
    exponential, so by ``v = 60`` the gap is below ``1e-35``; carrying ``r``
    itself would have lost every digit of it long before, and ``f`` computed
    as ``1 - 2M/r + Q^2/r^2`` would already be exactly zero from cancellation.
    In ``u = ln(r - r_-)`` the right-hand side is ``(r - r_+)/(2 r^2)``
    exactly, with no subtraction of nearly equal numbers anywhere.
    """
    if not hole.has_cauchy_horizon:
        raise ValueError(
            "an outgoing ray only asymptotes to an inner horizon when one exists; "
            "use advance_to_inner_surface for the dilaton case, where the ray reaches "
            "the singular surface at finite advanced time"
        )
    outer, inner = hole.outer_radius, hole.inner_radius

    def rhs(_: float, state):
        radius = inner + math.exp(state[0])
        return [(radius - outer) / (2.0 * radius**2)]

    return solve_ivp(
        rhs,
        (0.0, until),
        [math.log(start * (outer - inner))],
        rtol=1e-13,
        atol=1e-15,
        dense_output=True,
    )


def blueshift_rate(hole: DilatonBlackHole, window: tuple[float, float] = (80.0, 120.0)) -> float:
    """``-d ln(r - r_-)/dv`` along the ray, which should be ``kappa_-``.

    Measured from the integration rather than differentiated from the closed
    form, so agreeing with :meth:`DilatonBlackHole.inner_surface_gravity` is
    a check on both.
    """
    solution = outgoing_ray(hole, until=window[1] + 10.0)
    times = np.linspace(window[0], window[1], 400)
    return float(-np.polyfit(times, solution.sol(times)[0], 1)[0])


def advance_to_inner_surface(hole: DilatonBlackHole, start: float = DEFAULT_START) -> float:
    """Advanced time for the outgoing ray to reach ``r_-``: finite iff ``a > 0``.

    ``dv = 2 dr / |f|`` and near ``r_-`` the integrand behaves as
    ``(r - r_-)^(-b)``. That integral converges for every ``b < 1`` and
    diverges logarithmically at ``b = 1``, so the answer is infinite exactly
    when the dilaton coupling vanishes. Returning ``inf`` rather than a large
    number is the point: the Cauchy horizon is not reached at any finite
    advanced time, which is what makes the blueshift unbounded.
    """
    outer, inner = hole.outer_radius, hole.inner_radius
    if hole.charge == 0.0:
        return math.inf
    if hole.exponent >= 1.0:
        return math.inf
    begin = inner + start * (outer - inner)
    # |f| = |1 - r_+/r| (r - r_-)^b r^-b, so the endpoint singularity is algebraic
    # and is handed to the quadrature as a weight rather than integrated through.
    # Plain adaptive quadrature reports a roundoff failure here for small a, where
    # the exponent is closest to the divergent case.
    value, _ = quad(
        lambda r: 2.0 * r**hole.exponent / abs(1.0 - outer / r),
        inner,
        begin,
        weight="alg",
        wvar=(-hole.exponent, 0.0),
        limit=400,
    )
    return float(value)


def kretschmann_exponent_measured(
    hole: DilatonBlackHole, gaps: tuple[float, float] = (1e-6, 1e-10), digits: int = 50
) -> float:
    """The divergence power of ``K`` at ``r_-``, from the curvature tensor itself.

    Builds the metric symbolically, takes the Kretschmann scalar through
    :class:`~particlesim.symbolic.curvature.MetricGeometry`, and reads the
    slope of ``ln K`` against ``ln(r - r_-)`` between two gaps four decades
    apart. The comparison with the closed form
    :meth:`DilatonBlackHole.kretschmann_exponent` is what turns that formula
    from a claim into a measurement -- including at ``a = 0``, where the
    measured slope is zero because the Cauchy horizon is regular.

    Evaluated in ``mpmath`` at ``digits`` places, because the gaps are small
    enough that double precision would report the base as exactly ``r_-``.
    """
    import mpmath as mp
    import sympy as sp

    from particlesim.symbolic.curvature import MetricGeometry

    time, radius, polar, azimuth = sp.symbols("t r theta phi")
    metric = hole.symbolic_metric(radius, polar)
    scalar = MetricGeometry(metric, [time, radius, polar, azimuth]).kretschmann
    inner = sp.nsimplify(hole.inner_radius, rational=True)
    numeric = sp.lambdify(radius, scalar.subs(polar, sp.pi / 2), "mpmath")

    with mp.workdps(digits):
        base = mp.mpf(str(sp.Rational(inner)))
        values = [abs(mp.mpf(numeric(base + mp.mpf(str(g))))) for g in gaps]
        if min(values) == 0:
            return 0.0
        return float(mp.log(values[0] / values[1]) / mp.log(mp.mpf(str(gaps[1] / gaps[0]))))


def crossing_mass(
    hole: DilatonBlackHole, gap: float, ingoing_excess: float, outgoing_excess: float
) -> float:
    """Mass behind two crossing null shells, from ``f_A f_D = f_B f_C``.

    The crossing point is given as ``gap = r - r_-`` rather than as a radius,
    because by the time the ray is deep enough for the mass to inflate the
    gap is around ``1e-70`` and a radius has no room left to hold it.

    ``ingoing_excess`` and ``outgoing_excess`` are the mass each shell adds
    to the background ``hole``. Their metric functions come from the
    background's by the exact identity ``f(r, m) = f(r, M) - 2(m - M)/r``, so
    nothing is ever evaluated through the expanded polynomial that cancels to
    zero near ``r_-``.

    Raises for a dilaton black hole. The relation is a statement about the
    Reissner-Nordstrom class, and applying it where the inner surface is a
    curvature singularity would produce a number describing nothing.
    """
    if not hole.has_cauchy_horizon:
        raise ValueError(
            f"the crossing relation needs a Cauchy horizon, and at dilaton coupling "
            f"{hole.dilaton_coupling} there is none; the surface at r_- is a curvature "
            "singularity, so there is no mass inflation to report"
        )
    radius = hole.inner_radius + gap
    background = hole.metric_function_from_gap(gap)
    ingoing = background - 2.0 * ingoing_excess / radius
    outgoing = background - 2.0 * outgoing_excess / radius
    behind = ingoing * outgoing / background
    return (1.0 + hole.charge**2 / radius**2 - behind) * radius / 2.0


@dataclass
class MassInflationReport:
    """What the interior does, in the same fields whatever the theory."""

    theory: str
    inflates: bool
    arrival_time: float
    kretschmann_exponent: float
    surface_gravity: float | None = None
    measured_rate: float | None = None
    tail_exponent: float | None = None
    final_mass: float | None = None
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "theory": self.theory,
            "inflates": self.inflates,
            "arrival_time": self.arrival_time,
            "kretschmann_exponent": self.kretschmann_exponent,
            "surface_gravity": self.surface_gravity,
            "measured_rate": self.measured_rate,
            "tail_exponent": self.tail_exponent,
            "final_mass": self.final_mass,
            "notes": list(self.notes),
        }


def mass_function(
    hole: DilatonBlackHole, tail: PriceTail, shell_mass: float, solution, advanced_time: float
) -> float:
    """The inflating mass at one advanced time, from the integrated ray."""
    gap = math.exp(float(solution.sol(advanced_time)[0]))
    return crossing_mass(hole, gap, tail.mass_excess(advanced_time), shell_mass)


def report_mass_inflation(
    hole: DilatonBlackHole,
    tail: PriceTail | None = None,
    shell_mass: float = 1e-3,
    window: tuple[float, float] = (80.0, 120.0),
) -> MassInflationReport:
    """Run the interior diagnostic and say what happened, including "nothing".

    The report is deliberately able to come back with ``inflates = False`` and
    a finite ``arrival_time``; a diagnostic that could only find mass
    inflation would be a demonstration rather than a comparison.
    """
    tail = tail or PriceTail()
    exponent = hole.kretschmann_exponent()
    if not hole.has_cauchy_horizon:
        arrival = advance_to_inner_surface(hole)
        note = (
            "no Cauchy horizon: the areal radius collapses at r_- and the outgoing ray "
            f"reaches the singular surface at finite advanced time {arrival:.4f}"
            if hole.charge > 0.0
            else "uncharged, so there is no inner surface at all"
        )
        return MassInflationReport(
            theory=f"EMDA a={hole.dilaton_coupling}",
            inflates=False,
            arrival_time=arrival,
            kretschmann_exponent=exponent,
            notes=[note],
        )

    solution = outgoing_ray(hole, until=window[1] + 10.0)
    step = 1e-3
    rates = []
    for advanced_time in np.linspace(window[0], window[1], 5):
        ahead = math.log(abs(mass_function(hole, tail, shell_mass, solution, advanced_time + step)))
        behind = math.log(
            abs(mass_function(hole, tail, shell_mass, solution, advanced_time - step))
        )
        rates.append((ahead - behind) / (2.0 * step) + tail.exponent / advanced_time)
    final = mass_function(hole, tail, shell_mass, solution, window[1])
    return MassInflationReport(
        theory="general relativity",
        inflates=True,
        arrival_time=math.inf,
        kretschmann_exponent=exponent,
        surface_gravity=hole.inner_surface_gravity(),
        measured_rate=float(np.mean(rates)),
        tail_exponent=tail.exponent,
        final_mass=float(final),
        notes=[
            "the mass function diverges although the perturbation driving it decays; "
            "d ln m/dv = kappa_- - p/v, so one integration measures both constants"
        ],
    )


__all__ = [
    "DEFAULT_START",
    "MassInflationReport",
    "PriceTail",
    "advance_to_inner_surface",
    "blueshift_rate",
    "crossing_mass",
    "kretschmann_exponent_measured",
    "mass_function",
    "outgoing_ray",
    "report_mass_inflation",
]
