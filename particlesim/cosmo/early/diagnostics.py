"""Bounce and contraction diagnostics (design doc Section 3.1).

Three questions get asked of every contracting or bouncing background, and
all three have answers that are theorems rather than matters of taste.

**Does the contraction dilute anisotropy or amplify it?** A homogeneous
anisotropy in a contracting universe grows as ``a^-6`` -- that is the shear
term in the generalised Friedmann equation, and it is what drives the
Belinski-Khalatnikov-Lifshitz chaos a contraction ends in. A component with
``epsilon_H`` constant grows as ``a^(-2 epsilon)``. Since ``a`` is
decreasing, the component outgrows the shear if and only if
``2 epsilon > 6``, so ``epsilon > 3`` is the dividing line between a
contraction that smooths and one that shatters. That single inequality is
why ekpyrotic models want a steep potential and why a dust or radiation
contraction is not an alternative to inflation.

**Did it actually bounce, and what paid for it?** In general relativity on
a flat slice, ``H_dot = -(1/2)(rho + p)``. A bounce needs ``H = 0`` with
``H_dot > 0``, so it needs ``rho + p < 0``: the null energy condition must
be violated. There is no way around that in general relativity, which
means a bounce in a GR background is a statement about the matter, and a
bounce in a *modified* background is a statement about the theory. The
report says which, by measuring ``rho + p`` at the turning point rather
than assuming. Running a loop-quantum-cosmology bounce through it returns
``null_energy_violated = False`` -- the bounce came from the corrected
Friedmann equation, not from exotic matter -- and that is the check worth
having.

**How close did it get?** The minimum scale factor and the maximum density,
read at the turning point the integrator located rather than at the nearest
output sample, because a sharp peak between samples is understated by tens
of per cent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Growth exponent of a homogeneous anisotropy: ``rho_shear ~ a^-6``.
#:
#: Two powers from the shear tensor and four from the volume, or
#: equivalently ``epsilon = 3`` for a stiff fluid with ``w = 1``, which is
#: what a pure shear behaves as.
ANISOTROPY_EXPONENT = 6.0


@dataclass(frozen=True)
class ContractionReport:
    """What a contraction with a given ``epsilon_H`` does to anisotropy."""

    epsilon: float

    @property
    def equation_of_state(self) -> float:
        """``w = (2/3) epsilon - 1``."""
        return 2.0 * self.epsilon / 3.0 - 1.0

    @property
    def scaling_exponent(self) -> float:
        """``p`` in ``a ~ (-t)^p``, which is ``1/epsilon``."""
        return 1.0 / self.epsilon

    @property
    def dilutes_anisotropy(self) -> bool:
        """Does the dominant component outgrow the ``a^-6`` shear?"""
        return 2.0 * self.epsilon > ANISOTROPY_EXPONENT

    @property
    def ekpyrotic(self) -> bool:
        """The same condition, under the name the literature uses."""
        return self.dilutes_anisotropy

    def anisotropy_ratio(self, contraction: float) -> float:
        """Shear-to-component density ratio after contracting by ``contraction``.

        ``contraction`` is ``a_initial/a_final``, so it is larger than one.
        The ratio goes as ``contraction^(6 - 2 epsilon)``: it falls for
        ``epsilon > 3`` and runs away below it.
        """
        if contraction <= 0.0:
            raise ValueError(f"contraction factor must be positive, got {contraction}")
        return float(contraction ** (ANISOTROPY_EXPONENT - 2.0 * self.epsilon))

    def summary(self) -> dict[str, float | bool]:
        return {
            "epsilon": self.epsilon,
            "equation_of_state": self.equation_of_state,
            "scaling_exponent": self.scaling_exponent,
            "dilutes_anisotropy": self.dilutes_anisotropy,
        }


def contraction_report(epsilon: float) -> ContractionReport:
    """Diagnostics for a contraction with expansion parameter ``epsilon_H``."""
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError(
            f"epsilon must be positive, got {epsilon}: a contraction has "
            "H_dot < 0 relative to H^2 and epsilon = -H_dot/H^2 > 0"
        )
    return ContractionReport(epsilon=epsilon)


@dataclass(frozen=True)
class BounceReport:
    """Whether a background bounced, and what made it possible."""

    bounced: bool
    bounce_time: float | None
    minimum_scale_factor: float
    maximum_density: float
    hubble_slope: float | None
    null_energy_sum: float | None

    @property
    def null_energy_violated(self) -> bool | None:
        """``rho + p < 0`` at the turning point, or ``None`` if there was none.

        In general relativity this is *required* for a bounce. Where it is
        false and the universe bounced anyway, the Friedmann equation being
        integrated is not the general-relativistic one -- which is the
        whole point of a Tier B plugin, and is worth reading off a run
        rather than taking on trust.
        """
        if self.null_energy_sum is None:
            return None
        return bool(self.null_energy_sum < 0.0)

    def summary(self) -> dict[str, float | bool | None]:
        return {
            "bounced": self.bounced,
            "bounce_time": self.bounce_time,
            "minimum_scale_factor": self.minimum_scale_factor,
            "maximum_density": self.maximum_density,
            "hubble_slope": self.hubble_slope,
            "null_energy_violated": self.null_energy_violated,
        }


def bounce_report(
    time, scale_factor, hubble, density=None, pressure=None, bounce_time: float | None = None
) -> BounceReport:
    """Locate a turning point in ``hubble`` and report what happened there.

    ``bounce_time`` may be supplied by a solver that found the event
    exactly; otherwise it is interpolated from the sign change of ``H``,
    which is accurate to the output spacing.

    ``maximum_density`` is sample-limited and will understate a sharp peak
    by tens of per cent -- take the peak value from the solver's own event
    (``BackgroundRun.bounce_density``) when that is the question. The
    null-energy verdict is not sample-limited in the same way, because it
    is the *sign* of ``rho + p``: for ordinary matter both terms are
    positive at every sample and no amount of resolution changes that, so
    the verdict is robust where the peak value is not.
    """
    time = np.asarray(time, dtype=float)
    scale_factor = np.asarray(scale_factor, dtype=float)
    hubble = np.asarray(hubble, dtype=float)
    if not (time.shape == scale_factor.shape == hubble.shape):
        raise ValueError(
            f"time, scale_factor and hubble must have the same shape, got "
            f"{time.shape}, {scale_factor.shape}, {hubble.shape}"
        )

    rising = np.where((hubble[:-1] < 0.0) & (hubble[1:] >= 0.0))[0]
    bounced = bool(rising.size)
    slope = None
    energy_sum = None
    if bounced:
        index = int(rising[0])
        if bounce_time is None:
            span = hubble[index + 1] - hubble[index]
            weight = 0.0 if span == 0.0 else -hubble[index] / span
            bounce_time = float(time[index] + weight * (time[index + 1] - time[index]))
        step = time[index + 1] - time[index]
        slope = float((hubble[index + 1] - hubble[index]) / step) if step != 0.0 else None
        if density is not None and pressure is not None:
            density = np.asarray(density, dtype=float)
            pressure = np.asarray(pressure, dtype=float)
            energy_sum = float(
                np.interp(bounce_time, time, density) + np.interp(bounce_time, time, pressure)
            )
    else:
        bounce_time = None

    return BounceReport(
        bounced=bounced,
        bounce_time=bounce_time,
        minimum_scale_factor=float(scale_factor.min()),
        maximum_density=float(np.max(density)) if density is not None else float("nan"),
        hubble_slope=slope,
        null_energy_sum=energy_sum,
    )


def scaling_exponent(time, scale_factor, singular_time: float = 0.0) -> float:
    """Fit ``p`` in ``a ~ |t - t_singular|^p`` by least squares in logarithms.

    The defining solutions of every model here are power laws in cosmic
    time, so the check on a numerical background is a slope. Fitting rather
    than differencing at one point because a transient that has not quite
    died leaves a curvature in the log-log plot, and a fit's residual makes
    that visible where a two-point slope hides it.
    """
    time = np.asarray(time, dtype=float)
    scale_factor = np.asarray(scale_factor, dtype=float)
    gap = np.abs(time - singular_time)
    if np.any(gap <= 0.0) or np.any(scale_factor <= 0.0):
        raise ValueError(
            "the fit needs |t - t_singular| > 0 and a > 0 at every sample: "
            "trim the run before the singular time"
        )
    slope, _ = np.polyfit(np.log(gap), np.log(scale_factor), 1)
    return float(slope)
